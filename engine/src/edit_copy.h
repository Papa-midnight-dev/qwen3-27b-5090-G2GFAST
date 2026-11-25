// SPDX-License-Identifier: Apache-2.0
// Edit copy: learn a short replacement the output makes again and again while it copies its input (a renamed key or
// variable), and draft the input with the replacement applied. Ported verbatim (namespace and two accessors added)
// from yasakani-dev/yasakani src/edit_copy.hpp (v0.2.0+, Apache-2.0) for a research patch of 25 Nov 2026. It only
// proposes tokens: the target verifies every one, so the outputs do not change.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace q27_editcopy {
// Learn a short edit only after the target's committed output rejoins the input.
// Two different source locations must supply the same evidence before use.
// The caller owns target verification. This object only produces token proposals.
class EditCopy {
  // kept_ratio: a rule is proposed while (unchanged copies of its source text) * kept_ratio <= (edits seen), so a replacement that
  // holds only at some places (some records of a file) stays in use where it pays, and one that is mostly not applied stops.
  static constexpr int gap = 6, anchor = 4, rule_limit = 12, kept_ratio = 4, miss_limit = 2;
  struct Rule {
    std::array<int32_t, gap + 1> from{}, to{};
    int nf = 0, nt = 0, score = 0, last = -1;
    int n_edit = 0, n_kept = 0;   // places where the target applied the edit / copied the source text unchanged
    int miss_run = 0, miss_src = -1;   // proposals of this rule rejected at its edit since its last hit (an observed edit takes one back),
                                       // and the source place of the last one (a place counts once)
    bool usable() const { return score >= 2 && n_kept * kept_ratio <= n_edit && miss_run < miss_limit; }
  };
  std::array<Rule, rule_limit> rules_{};
  int used_ = 0, input_end_, source_ = -1, output_ = -1, literal_ = 0, run_ = 0;   // run_: tokens copied unchanged in a row
  mutable int prop_rule_ = -1, prop_src_ = -1;   // the rule (and its source place) that gave the first edited token of the last proposal

  int match(const std::vector<int32_t>& ids, int at) const {
    int best = -1;
    for (int i = 0; i < used_; ++i) {
      const Rule& r = rules_[i];
      if (!r.usable() || r.nf > input_end_ - at || ids[at] != r.from[0]) continue;
      if (std::equal(r.from.begin(), r.from.begin() + r.nf, ids.begin() + at) &&
          (best < 0 || r.nf > rules_[best].nf || (r.nf == rules_[best].nf && r.score > rules_[best].score))) best = i;
    }
    return best;
  }

  void observe_edit(const std::vector<int32_t>& ids, int ns, int no) {
    // For a pure insertion or deletion, retain the first shared token on both sides.
    if (ns == 0 || no == 0) { ++ns; ++no; }
    int slot = -1;
    for (int i = 0; i < used_; ++i) {
      Rule& r = rules_[i];
      if (r.nf != ns || !std::equal(r.from.begin(), r.from.begin() + ns, ids.begin() + source_)) continue;
      if (r.nt == no && std::equal(r.to.begin(), r.to.begin() + no, ids.begin() + output_)) {
        if (r.last != source_) { r.score = std::min(7, r.score + 1); ++r.n_edit; r.miss_run = std::max(0, r.miss_run - 1); }
        r.last = source_;
        return;
      }
      if (r.last != source_) r.score = std::max(0, r.score - 1);
    }
    if (used_ < rule_limit) slot = used_++;
    else {
      for (int i = 0; i < used_; ++i) if (rules_[i].score < 2) { slot = i; break; }
      if (slot < 0) return;
    }
    Rule& r = rules_[slot];
    r.nf = ns; r.nt = no; r.score = 1; r.last = source_; r.n_edit = 1; r.n_kept = 0; r.miss_run = 0; r.miss_src = -1;
    std::copy_n(ids.begin() + source_, ns, r.from.begin());
    std::copy_n(ids.begin() + output_, no, r.to.begin());
  }

  // The target copied the source text of a rule unchanged at this place, so the replacement does not hold everywhere (for example
  // only some records of a file change). Count it when the unchanged copy has passed the whole source text (so a copy that spans
  // two model passes is seen too); the rule stops being proposed once such places outweigh the edits (kept_ratio).
  void observe_kept(const std::vector<int32_t>& ids) {
    for (int i = 0; i < used_; ++i) {
      Rule& r = rules_[i];
      if (run_ < anchor + r.nf || source_ - r.nf < 0) continue;
      if (std::equal(r.from.begin(), r.from.begin() + r.nf, ids.begin() + (source_ - r.nf))) ++r.n_kept;
    }
  }

public:
  explicit EditCopy(int input_end) : input_end_(input_end) {}
  // (q27 port) An alignment is held and a rule could be proposed: the next step may be an edit-copy step.
  bool active() const {
    return source_ >= 0 && std::any_of(rules_.begin(), rules_.begin() + used_, [](const Rule& r) { return r.usable(); });
  }
  int input_end() const { return input_end_; }

  // ids[0..input_end_) is immutable input. ids[input_end_..end) is observed output.
  void advance(const std::vector<int32_t>& ids, int end, int raw_next) {
    if (source_ >= 0) {
      while (source_ < input_end_ && output_ < end) {
        if (ids[source_] == ids[output_]) {
          ++source_; ++output_; literal_ = std::min(anchor, literal_ + 1); ++run_;
          if (used_ > 0) observe_kept(ids);
          continue;
        }
        int ds = -1, dout = -1;
        for (int cost = 1; cost <= gap * 2 && ds < 0; ++cost) {
          for (int s = std::max(0, cost - gap); s <= std::min(gap, cost); ++s) {
            const int o = cost - s;
            if (source_ + s + anchor > input_end_ || output_ + o + anchor > end) continue;
            if (std::equal(ids.begin() + source_ + s, ids.begin() + source_ + s + anchor,
                           ids.begin() + output_ + o)) { ds = s; dout = o; break; }
          }
        }
        if (ds >= 0) {
          if (literal_ >= anchor) observe_edit(ids, ds, dout);
          source_ += ds; output_ += dout; literal_ = 0; run_ = 0;
          continue;
        }
        if (end - output_ < gap + anchor) return;  // need more committed evidence
        source_ = -1;
        break;
      }
      if (source_ >= 0 && source_ < input_end_) return;
    }
    source_ = -1;
    if (end - input_end_ >= anchor && raw_next >= anchor && raw_next < input_end_) {
      source_ = raw_next; output_ = end; literal_ = anchor; run_ = anchor;
    }
  }

  // Feedback from the target on the last proposal. missed(): it rejected the first edited token, so the rule does not hold there.
  // A rule stops after miss_limit such misses with no hit between them (for example when the replacement differs from record to
  // record); a hit, or each edit of the rule seen in the output, takes one back.
  void missed() {
    if (prop_rule_ < 0 || prop_rule_ >= used_) return;
    Rule& r = rules_[prop_rule_];
    if (r.miss_src != prop_src_) { ++r.miss_run; r.miss_src = prop_src_; r.last = prop_src_; }
    prop_rule_ = -1;
  }
  void hit() {
    if (prop_rule_ >= 0 && prop_rule_ < used_) rules_[prop_rule_].miss_run = 0;
    prop_rule_ = -1;
  }

  // Never copy from uncommitted output. If alignment spans the last model call,
  // replay our proposal across the observed suffix before proposing any new token.
  int propose(const std::vector<int32_t>& ids, int end, int limit, std::vector<int32_t>& out) const {
    out.clear(); prop_rule_ = -1;
    if (source_ < 0 || limit <= 0 || std::none_of(rules_.begin(), rules_.begin() + used_,
                                                   [](const Rule& r) { return r.usable(); })) return -1;
    int s = source_, o = output_, first_edit = -1;
    while (s < input_end_ && static_cast<int>(out.size()) < limit) {
      const int ri = match(ids, s);
      const Rule* rule = ri >= 0 ? &rules_[ri] : nullptr;
      const int32_t* data = rule ? rule->to.data() : &ids[s];
      const int count = rule ? rule->nt : 1;
      for (int j = 0; j < count && static_cast<int>(out.size()) < limit; ++j) {
        if (o < end) {
          if (ids[o++] != data[j]) { out.clear(); return -1; }
        } else {
          if (rule && first_edit < 0) { first_edit = static_cast<int>(out.size()); prop_rule_ = ri; prop_src_ = s; }
          out.push_back(data[j]);
        }
      }
      s += rule ? rule->nf : 1;
    }
    return o == end ? first_edit : -1;
  }
};
} // namespace q27_editcopy
