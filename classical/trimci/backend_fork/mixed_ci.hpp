// ============================================================================
//  mixed_ci.hpp — Tier-2 C++ port of NuQu's mixed-state H_ij oracle.
//
//  Header-only C++ implementation of `classical/trimci/hij.py::connections`:
//  given a mixed fermion-boson Fock state, apply every ladder-operator term of
//  the Hamiltonian and return the connected states + matrix elements. This is
//  the kernel a TrimCI selection engine would call through an `HijProvider`
//  hook instead of its hardwired Slater-Condon-from-FCIDUMP routine (see
//  backend_fork/README.md Tier 2).
//
//  CORRECTNESS CONTRACT: byte-for-byte the same conventions as the Python:
//   * fermion sign = (-1)^(occupied modes strictly below p), computed BEFORE
//     flipping the bit (OpenFermion convention);
//   * boson ladder b|n> = sqrt(n)|n-1>, b†|n> = sqrt(n+1)|n+1>, with the hard
//     cutoff dropping any element that would leave [0, N_f-1];
//   * each term applies its ops right-to-left; fermion and boson factors act on
//     disjoint registers and multiply.
//  Validated against the Python reference in test_mixed_ci_cpp.py.
//
//  Representation: fermion occupations in an ARBITRARY-WIDTH bitmask (a
//  std::vector<uint64_t> of W = ceil(n_ferm_modes/64) words — exactly as wide as
//  the system needs, no fixed cap, RAM-proportional to the actual mode count),
//  boson occupations in a std::vector<uint16_t> (N_f<=65536). This scales to any
//  lattice the host RAM can hold: L=3 in 3d is 108 modes (W=2), L=4 in 3d is 256
//  (W=4), and larger. All bit manipulation routes through the `Ferm` typedef +
//  the fbit/fflip/fpopcount_below helpers, so the representation is the single
//  chokepoint. The production form packs both registers into a TrimCI
//  determinant container — swap MixedDet without touching the logic.
// ============================================================================
#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mixedci {

using cdouble = std::complex<double>;

// Fermion occupation bitmask: W = ceil(n_ferm_modes/64) little-endian 64-bit
// words (word w holds modes [64w, 64w+64)). Width is fixed for a given run and
// carried by the state itself, so the kernel supports an arbitrary mode count.
using Ferm = std::vector<uint64_t>;

inline bool fbit(const Ferm& m, int p) {
    return (m[p >> 6] >> (p & 63)) & 1ULL;
}
inline void fflip(Ferm& m, int p) {
    m[p >> 6] ^= (1ULL << (p & 63));
}
// (-1)^(occupied modes strictly below p) — the OpenFermion/JW parity.
inline int fpopcount_below(const Ferm& m, int p) {
    const int w = p >> 6, b = p & 63;
    int c = 0;
    for (int i = 0; i < w; ++i) c += __builtin_popcountll(m[i]);
    if (b) c += __builtin_popcountll(m[w] & ((1ULL << b) - 1));
    return c;
}
inline int fpopcount(const Ferm& m) {
    int c = 0;
    for (uint64_t w : m) c += __builtin_popcountll(w);
    return c;
}

// One ladder operator: (mode index, action) with action 1 = creation (dagger),
// 0 = annihilation — exactly the OpenFermion term encoding our MixedH uses.
struct LadderOp {
    int mode;
    int action;
};

// coeff * (product of ferm_ops) * (product of bos_ops). Empty ops => constant.
struct Term {
    cdouble coeff;
    std::vector<LadderOp> ferm_ops;
    std::vector<LadderOp> bos_ops;
};

// A mixed Fock basis state: fermion bitmask (W words) + boson occupation vector.
struct MixedDet {
    Ferm ferm;                  // W = ceil(n_ferm_modes/64) little-endian words
    std::vector<uint16_t> bos;  // length = n_boson_modes, each in [0, N_f), N_f<=65536

    bool operator==(const MixedDet& o) const {
        return ferm == o.ferm && bos == o.bos;
    }
};

struct MixedDetHash {
    size_t operator()(const MixedDet& d) const {
        // splitmix64 finalizer folded over every fermion word (std::hash<uint64_t>
        // is identity on libc++/libstdc++, so states sharing a fermion det would
        // bucket-clump), then the boson bytes. Hash values are internal to the
        // C++ maps — they need not match Python.
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (uint64_t w : d.ferm) {
            uint64_t x = w + 0x9e3779b97f4a7c15ULL;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            x ^= (x >> 31);
            h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        for (uint16_t b : d.bos) {
            h ^= (uint64_t)b + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return (size_t)h;
    }
};

using ConnMap = std::unordered_map<MixedDet, cdouble, MixedDetHash>;

// --- Apply fermion ladder ops (right-to-left) to a bitmask. ----------------
// Returns false if annihilated; else updates occ and sets sign (+/-1).
inline bool apply_fermion_ops(const std::vector<LadderOp>& ops,
                              Ferm& occ, int& sign) {
    sign = 1;
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
        const int p = it->mode;
        const bool occupied = fbit(occ, p);
        if (it->action == 0) {                 // annihilation a_p
            if (!occupied) return false;
            if (fpopcount_below(occ, p) & 1) sign = -sign;
            fflip(occ, p);
        } else {                               // creation a_p^dagger
            if (occupied) return false;
            if (fpopcount_below(occ, p) & 1) sign = -sign;
            fflip(occ, p);
        }
    }
    return true;
}

// --- Apply boson ladder ops (right-to-left) to an occupation vector. -------
// Returns false if annihilated / pushed past the cutoff; else updates bos and
// multiplies amp by the sqrt factors.
inline bool apply_boson_ops(const std::vector<LadderOp>& ops,
                            std::vector<uint16_t>& bos, int N_f, double& amp) {
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
        const int m = it->mode;
        const int n = bos[m];
        if (it->action == 0) {                 // annihilation b_m
            if (n == 0) return false;
            amp *= std::sqrt((double)n);
            bos[m] = (uint16_t)(n - 1);
        } else {                               // creation b_m^dagger
            if (n + 1 >= N_f) return false;    // hard Fock cutoff
            amp *= std::sqrt((double)(n + 1));
            bos[m] = (uint16_t)(n + 1);
        }
    }
    return true;
}

// --- All connections of |state>: {connected state -> H_ij}. ----------------
// Includes the diagonal; prunes near-zero (cancelling) entries. This IS the
// column oracle the TrimCI expansion/trim and Davidson matvec consume.
inline ConnMap connections(const std::vector<Term>& terms,
                           const MixedDet& state, int N_f) {
    ConnMap out;
    out.reserve(terms.size());
    Ferm occ;                                   // scratch, reused across terms
    std::vector<uint16_t> bos;                  // scratch, reused across terms
    for (const Term& t : terms) {
        occ = state.ferm;                       // copy-assign reuses capacity
        int sign;
        if (!apply_fermion_ops(t.ferm_ops, occ, sign)) continue;
        bos = state.bos;                        // copy; boson register is disjoint
        double amp = 1.0;
        if (!apply_boson_ops(t.bos_ops, bos, N_f, amp)) continue;
        out[MixedDet{occ, bos}] += t.coeff * (double)sign * amp;
    }
    for (auto it = out.begin(); it != out.end();) {
        if (std::abs(it->second) <= 1e-14) it = out.erase(it);
        else ++it;
    }
    return out;
}

// --- Diagonal-term machinery (for the Epstein-Nesbet H_aa denominators). ---
// Mirrors classical/trimci/pt2.py::_is_diagonal / diagonal_element. A ladder
// product is diagonal iff the NET occupation change on every mode is zero (each
// mode carries equal creations and annihilations), so it maps |a> -> scalar|a>.
// The full-connections diagonal (neighbors(a)[a]) equals this sum but pays for
// EVERY connection of a; the PT2 external sum touches O(|V| x fan-out) distinct
// external states, so evaluating H_aa from the diagonal sub-list only is the
// difference between affordable and not.
inline bool is_diagonal_term(const Term& t) {
    std::unordered_map<int, int> net;
    for (const auto& op : t.ferm_ops) net[op.mode] += (op.action == 1 ? 1 : -1);
    for (const auto& kv : net) if (kv.second != 0) return false;
    net.clear();
    for (const auto& op : t.bos_ops) net[op.mode] += (op.action == 1 ? 1 : -1);
    for (const auto& kv : net) if (kv.second != 0) return false;
    return true;
}

// <a|H|a> from the diagonal terms only. Each diagonal term maps a -> a with a
// state-dependent scalar (fermion number/sign x boson sqrt product), summed.
inline cdouble diagonal_only(const std::vector<Term>& diag_terms,
                             const MixedDet& a, int N_f) {
    cdouble val(0.0, 0.0);
    Ferm occ;                                   // scratch, reused across terms
    std::vector<uint16_t> bos;                  // scratch, reused across terms
    for (const Term& t : diag_terms) {
        occ = a.ferm;
        int sign;
        if (!apply_fermion_ops(t.ferm_ops, occ, sign)) continue;
        bos = a.bos;
        double amp = 1.0;
        if (!apply_boson_ops(t.bos_ops, bos, N_f, amp)) continue;
        val += t.coeff * (double)sign * amp;    // diagonal term => target == a
    }
    return val;
}

// --- Provider: the interface a forked TrimCI selection kernel calls. -------
// `neighbors` = connections (edges + self); `diagonal` = H_ii. A forked
// run_trim / run_expansion / compute_diagonals dispatches through this instead
// of compute_H_ij(...,h1,eri) / generate_excitations(...,n_orb).
class MixedHijProvider {
public:
    // RAM safeguard. The connection cache is bounded by BOTH a state count
    // (cache_cap) and an estimated byte budget (max_cache_bytes), and is cleared
    // wholesale when either is exceeded — it can't leak, which matters at large L
    // where each cached state holds a Hamiltonian-sized ConnMap (n_terms grows
    // with the lattice, so 100k cached states could be hundreds of GB). The byte
    // budget is the real guard for big systems; the count is a cheap secondary.
    // cache_cap = 0 OR max_cache_bytes = 0 disables that bound (the other still
    // applies; both 0 => unbounded, caller's risk).
    explicit MixedHijProvider(std::vector<Term> terms, int N_f,
                              size_t cache_cap = 100000,
                              size_t max_cache_bytes = (size_t)1 << 30)
        : terms_(std::move(terms)), N_f_(N_f), cache_cap_(cache_cap),
          max_cache_bytes_(max_cache_bytes) {
        // Pre-filter the diagonal sub-list once (used by the PT2 H_aa sum).
        for (const Term& t : terms_)
            if (is_diagonal_term(t)) diag_terms_.push_back(t);
    }

    // Cached connections of |d>. Returns a const reference valid until the next
    // neighbors() call that triggers a clear — callers consume it immediately
    // (before their next call), so the reference is always live in use.
    // unordered_map element references survive rehash, so an emplace here does
    // not invalidate a reference returned by a prior call.
    const ConnMap& neighbors(const MixedDet& d) const {
        auto it = cache_.find(d);
        if (it != cache_.end()) return it->second;
        if ((cache_cap_ > 0 && cache_.size() >= cache_cap_) ||
            (max_cache_bytes_ > 0 && cache_bytes_ >= max_cache_bytes_)) {
            cache_.clear();
            cache_bytes_ = 0;
        }
        ConnMap c = connections(terms_, d, N_f_);
        cache_bytes_ += entry_bytes(d, c);
        auto ins = cache_.emplace(d, std::move(c));
        return ins.first->second;
    }

    cdouble diagonal(const MixedDet& d) const {
        const ConnMap& c = neighbors(d);
        auto it = c.find(d);
        return it != c.end() ? it->second : cdouble(0.0, 0.0);
    }

    // <d|H|d> from the diagonal sub-list only — cheap, and does NOT populate
    // the connection cache (the PT2 external sum visits far more distinct
    // states than would fit, so caching their full ConnMaps would blow RAM).
    cdouble diagonal_fast(const MixedDet& d) const {
        return diagonal_only(diag_terms_, d, N_f_);
    }

    const std::vector<Term>& terms() const { return terms_; }
    int N_f() const { return N_f_; }
    size_t cache_size() const { return cache_.size(); }
    size_t cache_bytes() const { return cache_bytes_; }
    void clear_cache() const { cache_.clear(); cache_bytes_ = 0; }

private:
    // Conservative byte estimate for a cached (key -> ConnMap) entry: the key,
    // plus each connection's MixedDet + value + a typical unordered_map node/bucket
    // overhead (~56 B on libc++). Used only to bound RAM, so over-estimating is safe.
    size_t entry_bytes(const MixedDet& key, const ConnMap& c) const {
        const size_t det = key.ferm.size() * 8 + key.bos.size() * 2;
        const size_t per_conn = det + sizeof(cdouble) + 56;
        return det + 32 + c.size() * per_conn;
    }

    std::vector<Term> terms_;
    std::vector<Term> diag_terms_;   // net-zero-per-mode sub-list (for H_aa)
    int N_f_;
    size_t cache_cap_;
    size_t max_cache_bytes_;
    mutable size_t cache_bytes_ = 0;
    mutable std::unordered_map<MixedDet, ConnMap, MixedDetHash> cache_;
};

}  // namespace mixedci
