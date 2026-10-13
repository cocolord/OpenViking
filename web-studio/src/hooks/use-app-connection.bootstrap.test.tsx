// @vitest-environment jsdom
import { createServer } from 'node:http'
import type { RequestListener } from 'node:http'
import { once } from 'node:events'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, expect, it, vi } from 'vitest'

import {
  AppConnectionProvider,
  createIdentityScopeKey,
  useAppConnection,
} from './use-app-connection'
import type { ConnectionDraft } from './use-app-connection'
import { fetchServerHealth } from './use-server-mode'
import { ovClient } from '#/lib/ov-client'

vi.mock('@tanstack/react-router', () => ({
  useNavigate: () => vi.fn(),
  useRouterState: () => '/home',
}))

const servers: Array<ReturnType<typeof createServer>> = []
const queryClients: Array<QueryClient> = []

afterEach(async () => {
  cleanup()
  for (const queryClient of queryClients.splice(0)) queryClient.clear()
  localStorage.removeItem('ov_console_connection')
  await Promise.all(
    servers
      .splice(0)
      .map(
        (server) =>
          new Promise<void>((resolve) => server.close(() => resolve())),
      ),
  )
  vi.restoreAllMocks()
})

async function startHealthServer(handler: RequestListener): Promise<string> {
  const server = createServer((req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader(
      'Access-Control-Allow-Headers',
      'X-OpenViking-Account, X-OpenViking-User, X-API-Key',
    )
    if (req.method === 'OPTIONS') {
      res.end()
      return
    }
    res.setHeader('Content-Type', 'application/json')
    handler(req, res)
  })
  servers.push(server)
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  return 'http://127.0.0.1:' + (server.address() as { port: number }).port
}

function renderConnection(connection: ConnectionDraft) {
  let current: ReturnType<typeof useAppConnection> | undefined
  function Snapshot() {
    current = useAppConnection()
    return null
  }
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  queryClients.push(queryClient)
  localStorage.setItem('ov_console_connection', JSON.stringify(connection))
  render(
    <QueryClientProvider client={queryClient}>
      <AppConnectionProvider>
        <Snapshot />
      </AppConnectionProvider>
    </QueryClientProvider>,
  )
  return { current: () => current!, queryClient }
}

it.each(['', 'root-key'])(
  'detects trusted mode on cold startup and resolves identity only with a key: %s',
  async (rootKey) => {
    const observed: Array<
      [string | undefined, string | undefined, string | undefined]
    > = []
    const baseUrl = await startHealthServer((req, res) => {
      const account = req.headers['x-openviking-account'] as string | undefined
      const user = req.headers['x-openviking-user'] as string | undefined
      const apiKey = req.headers['x-api-key'] as string | undefined
      observed.push([account, user, apiKey])
      if (req.url?.startsWith('/api/v1/admin')) {
        res.statusCode = 403
        res.end(JSON.stringify({ error: { code: 'PERMISSION_DENIED' } }))
        return
      }
      if (account && user && apiKey !== 'root-key') {
        res.statusCode = 401
        res.end(JSON.stringify({ error: { code: 'UNAUTHENTICATED' } }))
        return
      }
      res.end(
        JSON.stringify({
          status: 'ok',
          auth_mode: 'trusted',
          root_api_key_required: true,
          ...(account && user && apiKey === 'root-key'
            ? { account_id: account, user_id: user, role: 'user' }
            : {}),
        }),
      )
    })
    const { current } = renderConnection({
      baseUrl,
      accountId: 'account-a',
      userId: 'alice',
      apiKey: '',
      adminApiKey: rootKey,
    })
    await waitFor(() => {
      expect(current().serverMode).toBe('trusted')
      expect(current().isConnectionRoleLoading).toBe(false)
      expect(current().connectionRole).toBe(rootKey ? 'user' : 'unknown')
    })
    expect(current().connection).toMatchObject({
      accountId: 'account-a',
      userId: 'alice',
    })
    if (rootKey) {
      expect(observed).toContainEqual(['account-a', 'alice', rootKey])
    } else {
      expect(observed).toEqual([[undefined, undefined, undefined]])
    }
  },
)

it.each([
  {
    accountId: 'default',
    userId: 'default',
    apiKey: '',
    missingIdentity: false,
  },
  {
    accountId: 'server-team',
    userId: 'server-user',
    apiKey: 'stale-key',
    missingIdentity: false,
  },
  {
    accountId: 'legacy-account',
    userId: 'claude',
    apiKey: '',
    missingIdentity: true,
  },
])(
  'applies a dev identity only when the server returns it: $accountId/$userId',
  async ({ accountId, userId, apiKey, missingIdentity }) => {
    const observed: Array<Record<string, unknown>> = []
    const baseUrl = await startHealthServer((req, res) => {
      observed.push(req.headers)
      res.end(
        JSON.stringify({
          status: 'ok',
          auth_mode: 'dev',
          ...(missingIdentity
            ? {}
            : {
                role: 'root',
                account_id: req.headers['x-openviking-account'] || accountId,
                user_id: req.headers['x-openviking-user'] || userId,
              }),
        }),
      )
    })
    const { current, queryClient } = renderConnection({
      baseUrl,
      accountId: 'legacy-account',
      userId: 'claude',
      apiKey,
      adminApiKey: apiKey,
    })
    const clearSpy = vi.spyOn(queryClient, 'clear')
    const identity = { accountId, userId }

    await waitFor(() => {
      expect(current().serverMode).toBe('dev')
      expect(current().connection).toMatchObject(identity)
    })
    expect(ovClient.getConnection()).toMatchObject({
      ...identity,
      identityHeaders: false,
    })
    expect(current().identityScopeKey).toBe(
      createIdentityScopeKey(current().connection, 'dev'),
    )
    expect(
      JSON.parse(localStorage.getItem('ov_console_connection') || '{}'),
    ).toMatchObject(identity)
    expect(clearSpy).toHaveBeenCalledTimes(missingIdentity ? 0 : 1)
    if (!apiKey) expect(observed).toHaveLength(1)
    expect(
      observed.some(
        (headers) =>
          headers['x-api-key'] === (apiKey || undefined) &&
          !headers['x-openviking-account'] &&
          !headers['x-openviking-user'],
      ),
    ).toBe(true)
  },
)

it.each(['api_key', 'trusted', 'dev'] as const)(
  'keeps configured credentials when a gateway protects health in %s mode',
  async (mode) => {
    const keys: Array<string | undefined> = []
    const apiKey = 'gateway-user-key'
    const adminApiKey = mode === 'dev' ? 'gateway-control-key' : ''
    const baseUrl = await startHealthServer((req, res) => {
      const key = req.headers['x-api-key'] as string | undefined
      keys.push(key)
      if (key !== apiKey && (!adminApiKey || key !== adminApiKey)) {
        res.statusCode = 401
        res.end(JSON.stringify({ error: { code: 'UNAUTHENTICATED' } }))
        return
      }
      res.end(
        JSON.stringify({
          auth_mode: mode,
          role: mode === 'dev' ? 'root' : 'user',
          account_id:
            mode === 'dev'
              ? req.headers['x-openviking-account'] || 'default'
              : 'tenant-a',
          user_id:
            mode === 'dev'
              ? req.headers['x-openviking-user'] || 'default'
              : 'alice',
        }),
      )
    })
    const { current } = renderConnection({
      baseUrl,
      accountId: 'tenant-a',
      userId: 'alice',
      apiKey,
      adminApiKey,
    })
    await waitFor(() => {
      expect(current().serverMode).toBe(mode)
      expect(current().connectionRole).toBe(mode === 'dev' ? 'root' : 'user')
      expect(current().connection).toMatchObject(
        mode === 'dev'
          ? { accountId: 'default', userId: 'default' }
          : { accountId: 'tenant-a', userId: 'alice' },
      )
    })
    expect(keys).not.toContain(undefined)
    expect(keys).toContain(apiKey)
    if (adminApiKey) expect(keys[0]).toBe(adminApiKey)
  },
)

it.each(['trusted', 'api_key'] as const)(
  'preserves the target identity when switching from dev to %s',
  async (mode) => {
    const devUrl = await startHealthServer((_req, res) => {
      res.end(
        JSON.stringify({
          auth_mode: 'dev',
          role: 'root',
          account_id: 'default',
          user_id: 'default',
        }),
      )
    })
    const targetUrl = await startHealthServer((req, res) => {
      if (req.url === '/api/v1/admin/accounts') {
        res.end(JSON.stringify({ status: 'ok', result: [] }))
        return
      }
      const identity = req.headers['x-api-key']
        ? {
            role: mode === 'trusted' ? 'user' : 'root',
            account_id:
              mode === 'trusted'
                ? req.headers['x-openviking-account']
                : 'default',
            user_id:
              mode === 'trusted' ? req.headers['x-openviking-user'] : 'default',
          }
        : {}
      res.end(JSON.stringify({ auth_mode: mode, ...identity }))
    })
    const { current } = renderConnection({
      baseUrl: devUrl,
      accountId: 'default',
      userId: 'default',
      apiKey: '',
      adminApiKey: '',
    })
    await waitFor(() => expect(current().serverMode).toBe('dev'))
    const target = {
      baseUrl: targetUrl,
      accountId: 'tenant-b',
      userId: 'alice',
      apiKey: '',
      adminApiKey: 'target-root-key',
    }
    act(() => current().saveConnection(target))
    expect(current().serverMode).toBe('checking')
    expect(current().connection).toEqual(target)
    await waitFor(() => {
      expect(current().serverMode).toBe(mode)
      expect(current().isConnectionRoleLoading).toBe(false)
    })
    expect(current().connection).toEqual(target)
    expect(ovClient.getConnection()).toMatchObject({
      accountId: target.accountId,
      userId: target.userId,
      identityHeaders: mode === 'trusted',
    })
    expect(
      JSON.parse(localStorage.getItem('ov_console_connection') || '{}'),
    ).toEqual(target)
  },
)

it.each(['discovery', 'data'] as const)(
  'ignores an obsolete dev %s response after the target connection changes',
  async (phase) => {
    let releaseDev: (() => void) | undefined
    const devUrl = await startHealthServer((req, res) => {
      if (phase === 'data' && req.headers['x-openviking-account']) {
        res.end(
          JSON.stringify({
            auth_mode: 'dev',
            role: 'root',
            account_id: req.headers['x-openviking-account'],
            user_id: req.headers['x-openviking-user'],
          }),
        )
        return
      }
      releaseDev = () =>
        res.end(
          JSON.stringify({
            auth_mode: 'dev',
            role: 'root',
            account_id: 'old-team',
            user_id: 'old-user',
          }),
        )
    })
    const targetUrl = await startHealthServer((_req, res) => {
      res.end(JSON.stringify({ auth_mode: 'trusted' }))
    })
    const { current } = renderConnection({
      baseUrl: devUrl,
      accountId: 'legacy-account',
      userId: 'claude',
      apiKey: phase === 'data' ? 'old-key' : '',
      adminApiKey: '',
    })
    await waitFor(() => expect(releaseDev).toBeDefined())
    const obsoleteProbe = fetchServerHealth(
      devUrl,
      phase === 'data' ? { 'X-API-Key': 'old-key' } : {},
    )
    const target = {
      ...current().connection,
      baseUrl: targetUrl,
      accountId: 'tenant-b',
      userId: 'alice',
    }
    act(() => current().saveConnection(target))
    await waitFor(() => expect(current().serverMode).toBe('trusted'))
    await act(async () => {
      releaseDev!()
      await obsoleteProbe
    })
    expect(current().serverMode).toBe('trusted')
    expect(current().connection).toEqual(target)
    expect(ovClient.getConnection()).toMatchObject({
      accountId: target.accountId,
      userId: target.userId,
    })
    expect(
      JSON.parse(localStorage.getItem('ov_console_connection') || '{}'),
    ).toEqual(target)
  },
)
