// Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: AGPL-3.0
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace vectordb {

// Shared by index recall and the batch scorer used after model reranking.
class TimeDecayScorer {
 public:
  TimeDecayScorer(double origin, double offset, double scale, double decay)
      : origin_(origin), offset_(offset) {
    if (!std::isfinite(origin) || !std::isfinite(offset) ||
        !std::isfinite(scale) || !std::isfinite(decay) || offset < 0 ||
        scale <= 0 || decay <= 0 || decay >= 1) {
      throw std::invalid_argument("invalid time-decay parameters");
    }
    rate_ = std::log(decay) / scale;
  }

  double operator()(double timestamp) const {
    if (!std::isfinite(timestamp)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return std::exp(rate_ * std::max(0.0, std::abs(origin_ - timestamp) - offset_));
  }

 private:
  double origin_;
  double offset_;
  double rate_;
};

// NaN denotes a missing time factor and leaves the semantic/model score intact.
// Keep the input score precision: float for index recall, double for model scores.
template <typename Score>
std::vector<size_t> fuse_and_rank_time_decay(
    std::vector<Score>& scores, const std::vector<double>& time_scores,
    size_t limit, bool fuse = true) {
  if (scores.size() != time_scores.size()) {
    throw std::invalid_argument("scores and time scores must have equal lengths");
  }
  for (size_t i = 0; i < scores.size(); ++i) {
    if (!std::isfinite(scores[i])) {
      throw std::invalid_argument("scores must be finite");
    }
    if (std::isnan(time_scores[i])) continue;
    if (!std::isfinite(time_scores[i]) || time_scores[i] < 0 || time_scores[i] > 1) {
      throw std::invalid_argument("time scores must be in [0, 1]");
    }
    if (fuse) scores[i] *= time_scores[i];
  }
  std::vector<size_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return scores[a] > scores[b];
  });
  order.resize(std::min(order.size(), limit));
  return order;
}

}  // namespace vectordb
