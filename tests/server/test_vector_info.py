# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from openviking.server.auth import get_request_context
from openviking.server.identity import RequestContext, ResolvedIdentity, Role, UserIdentifier
from openviking.server.routers import debug
from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.viking_vector_index_backend import VikingVectorIndexBackend
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig
from tests.server.test_auth import _build_auth_http_test_app


@pytest.fixture
def vector_info_app(monkeypatch):
    app = FastAPI()
    app.include_router(debug.router)
    app.dependency_overrides[get_request_context] = lambda: RequestContext(
        user=UserIdentifier("account_a", "user_a"), role=Role(Role.USER)
    )
    service = SimpleNamespace(vikingdb_manager=None)
    monkeypatch.setattr(debug, "get_service", lambda: service)
    return app, service


async def test_vector_info_resolves_current_accounts_index(tmp_path, vector_info_app):
    app, service = vector_info_app
    configs = {
        account: VectorDBBackendConfig(
            backend="local",
            path=str(tmp_path / account),
            name=account,
            dimension=4,
            distance_metric=metric,
        )
        for account, metric in [("account_a", "cosine"), ("account_b", "ip")]
    }
    manager = VikingVectorIndexBackend(configs["account_a"])

    async def resolve(account_id):
        return SimpleNamespace(vectordb=configs[account_id], dedicated_vectordb=True)

    manager.set_vector_config_resolver(SimpleNamespace(resolve=resolve))
    service.vikingdb_manager = manager
    try:
        for account in configs:
            backend = await manager.get_account_backend(account)
            assert await backend.create_collection(
                account, CollectionSchemas.context_collection(account, 4)
            )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            for account, scale in [
                ("account_a", "cosine_affine_0_1"),
                ("account_b", "inner_product"),
            ]:
                app.dependency_overrides[get_request_context] = lambda account=account: (
                    RequestContext(user=UserIdentifier(account, "user"), role=Role(Role.USER))
                )
                response = await client.get("/api/v1/debug/vector/info")
                assert response.status_code == 200
                body = response.json()
                assert body["status"] == "ok"
                assert body["result"]["collection_name"] == account
                assert body["result"]["distance_metric"] == configs[account].distance_metric
                assert body["result"]["dense_score"]["scale"] == scale
    finally:
        await manager.close()


@pytest.mark.parametrize(
    "has_manager,status_code,error_code",
    [(False, 503, "NO_VECTOR_DB"), (True, 404, "NOT_FOUND")],
)
async def test_vector_info_missing_storage_error_envelope(
    vector_info_app, has_manager, status_code, error_code
):
    app, service = vector_info_app
    if has_manager:
        service.vikingdb_manager = SimpleNamespace(
            get_account_backend=AsyncMock(
                return_value=SimpleNamespace(get_vector_info=AsyncMock(return_value=None))
            )
        )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/debug/vector/info")
    assert response.status_code == status_code
    assert response.json()["status"] == "error"
    assert response.json()["error"]["code"] == error_code


async def test_vector_info_rejects_root_api_key_before_accessing_storage(monkeypatch):
    app = _build_auth_http_test_app(
        ResolvedIdentity(role=Role(Role.ROOT), account_id="default", user_id="default"),
        auth_enabled=True,
    )
    app.include_router(debug.router)

    def unexpected_service_access():
        pytest.fail("ROOT API keys must be rejected before reading tenant vector metadata")

    monkeypatch.setattr(debug, "get_service", unexpected_service_access)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/debug/vector/info")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "PERMISSION_DENIED"
