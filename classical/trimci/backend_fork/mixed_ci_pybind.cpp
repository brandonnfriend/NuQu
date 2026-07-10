// ============================================================================
//  mixed_ci_pybind.cpp — pybind11 wrapper around mixed_ci.hpp.
//
//  Exposes the C++ `MixedHijProvider` so the port can be (a) validated against
//  the Python `hij.connections`, and (b) called as the matrix-element provider.
//  This is a STANDALONE extension (no TrimCI dependency) — built in-project with
//  build_mixed_ci.sh. In the real fork, mixed_ci.hpp is compiled into the TrimCI
//  engine and wired to its run_trim/run_expansion; this module is the test/driver.
//
//  Python boundary:
//    MixedProvider(terms, N_f)  where terms = list of
//        (coeff: complex, ferm_ops: list[(mode,action)], bos_ops: list[(mode,action)])
//    .connections(ferm: int, bos: list[int]) ->
//        list of ((ferm: int, bos: tuple[int]), value: complex)
//    .diagonal(ferm, bos) -> complex
//    .build_context(ferm: (N,W) uint64, bos: (N,n_bos) uint16) -> SubspaceContext
//
//  SubspaceContext — complex CSC of the projected H over a fixed state set.
//    Built once from the provider's cached connections; subsequent .matvec()
//    calls iterate the CSC arrays directly (no hash lookups, no matrix rebuild).
//    Used by backend.cpp_diagonalize_matfree via scipy eigsh + LinearOperator:
//      ctx = prov.build_context(ferm, bos)
//      op  = LinearOperator((N, N), matvec=lambda v: ctx.matvec(v.real, v.imag))
//      E0, vecs = eigsh(op, k=1, which='SA')
//    Advantages over build_real_embedded_coo + davidson_solve_sparse:
//      * No 2x real embedding (native complex) -> 2x fewer entries, 2x less memory
//      * No scipy COO->CSR conversion (was ~30% of hot-path time)
//      * CSC built once, reused for all eigsh iterations (no per-call rebuild)
//      * Connections cached in provider across calls (cross-round reuse)
// ============================================================================
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "mixed_ci.hpp"

namespace py = pybind11;
using namespace mixedci;

// (mode, action) pair list straight from Python.
using PyLadder = std::vector<std::pair<int, int>>;
// (coeff, ferm_ops, bos_ops)
using PyTerm = std::tuple<cdouble, PyLadder, PyLadder>;

static std::vector<LadderOp> to_ops(const PyLadder& in) {
    std::vector<LadderOp> out;
    out.reserve(in.size());
    for (const auto& mp : in) out.push_back(LadderOp{mp.first, mp.second});
    return out;
}

// Fermion masks cross the Python boundary as an (N, W) uint64 array — W little-
// endian 64-bit words per state (numpy has no native >64-bit integer, and W is
// arbitrary so the mode count is unbounded). read_ferm pulls row k into a W-word
// Ferm; write_ferm stores a Ferm into row k of an output (M, W) array.
template <class Ref>
static inline Ferm read_ferm(const Ref& f, py::ssize_t k, int W) {
    Ferm m(W);
    for (int w = 0; w < W; ++w) m[w] = f(k, w);
    return m;
}
template <class Mut>
static inline void write_ferm(Mut& out, py::ssize_t k, const Ferm& m) {
    for (size_t w = 0; w < m.size(); ++w) out(k, (py::ssize_t)w) = m[w];
}

template <class T>
static py::array_t<T> to_np(const std::vector<T>& v) {
    return py::array_t<T>(static_cast<py::ssize_t>(v.size()), v.data());
}

// ============================================================================
//  SubspaceContext — complex CSC of the projected H over a fixed state set.
//
//  Stores the projected Hamiltonian H|_{states x states} in complex CSC format
//  (column-major: for each column j=state, all rows i with H_ij != 0).
//  The CSC is built ONCE, single-pass, directly into the flat arrays in the
//  constructor (append per column, record indptr) — no transient
//  vector-of-columns (which would double peak nnz memory), and the input state
//  list / index map are consumed at build time and NOT retained (they are dead
//  after the CSC exists; matvec/diagonal read only the CSC).
//
//  Memory (steady state): row indices are int32 (states < 2^31), values are two
//  doubles per non-zero => 4 + 16 = 20 bytes/nnz, plus 8 bytes/column for indptr.
//  At N=400k, k_conn~30 => nnz~12M => ~240 MB. No 2N real embedding, no retained
//  states, no per-column vectors.
// ============================================================================
class SubspaceContext {
public:
    // Build the CSC in one pass. `prov` is used only during construction;
    // `states`/`index` are read but not stored.
    SubspaceContext(MixedHijProvider& prov,
                    const std::vector<MixedDet>& states,
                    const std::unordered_map<MixedDet, int, MixedDetHash>& index) {
        const int64_t N = (int64_t)states.size();
        // int32 row indices: guards the laptop/HPC scale we target (2.1e9 states
        // is far beyond any single-node core). nnz can still exceed int32, so
        // indptr stays int64.
        if (N > (int64_t)INT32_MAX)
            throw std::runtime_error(
                "SubspaceContext: N exceeds int32 row-index range (2^31)");
        N_ = (int)N;
        indptr_.resize(N_ + 1);
        indptr_[0] = 0;
        // Rough reserve to cut reallocation spikes (grows if under-estimated).
        const size_t guess = (size_t)N_ * 16;
        row_idx_.reserve(guess);
        data_re_.reserve(guess);
        data_im_.reserve(guess);
        for (int j = 0; j < N_; ++j) {
            const ConnMap& c = prov.neighbors(states[j]);
            for (const auto& kv : c) {
                auto it = index.find(kv.first);
                if (it != index.end()) {
                    row_idx_.push_back(it->second);      // < N_ < 2^31
                    data_re_.push_back(kv.second.real());
                    data_im_.push_back(kv.second.imag());
                }
            }
            indptr_[j + 1] = (int64_t)row_idx_.size();
        }
    }

    int size() const { return N_; }
    int64_t nnz() const { return indptr_.empty() ? 0 : indptr_.back(); }

    // Complex H @ v: out[i] += sum_j H_ij * v_j, using the stored CSC.
    // vr = Re(v), vi = Im(v), both length N. Returns (out_re, out_im).
    // No hash lookups — pure sequential CSC array traversal.
    py::tuple matvec(py::array_t<double> vr_arr, py::array_t<double> vi_arr) const {
        auto vr = vr_arr.unchecked<1>();
        auto vi = vi_arr.unchecked<1>();
        std::vector<double> or_(N_, 0.0), oi_(N_, 0.0);
        for (int j = 0; j < N_; ++j) {
            const double vjr = vr(j), vji = vi(j);
            // Skip exact zeros to avoid touching cold cache lines in or_/oi_
            // (CSC column j might scatter to many rows).
            if (vjr == 0.0 && vji == 0.0) continue;
            for (int64_t k = indptr_[j]; k < indptr_[j + 1]; ++k) {
                const int i = row_idx_[k];
                // (a + ib)(x + iy) = (ax-by) + i(ay+bx)
                or_[i] += data_re_[k] * vjr - data_im_[k] * vji;
                oi_[i] += data_re_[k] * vji + data_im_[k] * vjr;
            }
        }
        return py::make_tuple(to_np(or_), to_np(oi_));
    }

    // Real part of the diagonal H_jj for each state j. Used as the initial
    // guess seed for eigsh (lowest-diagonal unit vector).
    py::array_t<double> diagonal() const {
        std::vector<double> diag(N_, 0.0);
        for (int j = 0; j < N_; ++j) {
            for (int64_t k = indptr_[j]; k < indptr_[j + 1]; ++k) {
                if (row_idx_[k] == j) {
                    diag[j] = data_re_[k];
                    break;  // diagonal is unique per column in H
                }
            }
        }
        return to_np(diag);
    }

private:
    int N_ = 0;
    // Complex CSC: column j spans row_idx_[indptr_[j]..indptr_[j+1]).
    std::vector<int64_t> indptr_;   // length N+1 (nnz may exceed int32)
    std::vector<int32_t> row_idx_;  // row index of each non-zero (< N < 2^31)
    std::vector<double>  data_re_;  // Re(H_ij) for each non-zero
    std::vector<double>  data_im_;  // Im(H_ij) for each non-zero
};


class PyMixedProvider {
public:
    PyMixedProvider(const std::vector<PyTerm>& py_terms, int N_f,
                    size_t cache_cap = 100000,
                    size_t max_cache_bytes = (size_t)1 << 30) {
        std::vector<Term> terms;
        terms.reserve(py_terms.size());
        int maxmode = -1;
        for (const auto& pt : py_terms) {
            const auto fops = to_ops(std::get<1>(pt));
            for (const auto& op : fops) maxmode = std::max(maxmode, op.mode);
            terms.push_back(Term{std::get<0>(pt), fops, to_ops(std::get<2>(pt))});
        }
        // word count needed to hold every fermion mode the terms reference —
        // used to size masks for the single-state debug interface.
        ferm_words_ = (maxmode < 0) ? 1 : (maxmode / 64 + 1);
        // shared_ptr so SubspaceContext can share ownership (keeps provider alive
        // even if the Python PyMixedProvider wrapper is garbage-collected while a
        // SubspaceContext is still in use).
        provider_ = std::make_shared<MixedHijProvider>(std::move(terms), N_f,
                                                        cache_cap, max_cache_bytes);
    }

    size_t cache_size() const { return provider_->cache_size(); }
    size_t cache_bytes() const { return provider_->cache_bytes(); }

    // Single-state debug/validation interface (<=64 fermion modes): ferm is a
    // plain Python int. The hot path uses the array methods below, which carry
    // the full arbitrary-width mask. Returns [((ferm, (bos...)), value), ...].
    py::list connections(uint64_t ferm, const std::vector<int>& bos_in) {
        MixedDet d;
        d.ferm.assign(ferm_words_, 0);
        d.ferm[0] = ferm;
        d.bos.assign(bos_in.begin(), bos_in.end());
        const ConnMap& c = provider_->neighbors(d);
        py::list out;
        for (const auto& kv : c) {
            py::tuple bos_t = py::cast(std::vector<int>(kv.first.bos.begin(),
                                                        kv.first.bos.end()));
            out.append(py::make_tuple(py::make_tuple(kv.first.ferm[0], bos_t),
                                      kv.second));
        }
        return out;
    }

    cdouble diagonal(uint64_t ferm, const std::vector<int>& bos_in) {
        MixedDet d;
        d.ferm.assign(ferm_words_, 0);
        d.ferm[0] = ferm;
        d.bos.assign(bos_in.begin(), bos_in.end());
        return provider_->diagonal(d);
    }

    // Build the projected complex H over `states` as COO (rows, cols, re, im),
    // entirely in C++. States: ferm (uint64, N x W words) + bos (uint16, N x n_bos).
    // This is the matrix-build inner loop for every subspace diagonalization.
    py::tuple build_coo(py::array_t<uint64_t> ferm_arr,
                        py::array_t<uint16_t> bos_arr) {
        auto f = ferm_arr.unchecked<2>();
        auto b = bos_arr.unchecked<2>();
        const size_t N = f.shape(0);
        const int W = static_cast<int>(f.shape(1));
        const int n_bos = static_cast<int>(b.shape(1));

        std::vector<MixedDet> states(N);
        // int64 indices: keep correct past 2^31 states/nonzeros (review #3).
        std::unordered_map<MixedDet, int64_t, MixedDetHash> index;
        index.reserve(N * 2);
        for (size_t k = 0; k < N; ++k) {
            MixedDet d;
            d.ferm = read_ferm(f, k, W);
            d.bos.resize(n_bos);
            for (int m = 0; m < n_bos; ++m) d.bos[m] = b(k, m);
            index.emplace(d, static_cast<int64_t>(k));
            states[k] = std::move(d);
        }

        std::vector<int64_t> rows, cols;
        std::vector<double> re, im;
        for (size_t j = 0; j < N; ++j) {
            const ConnMap& c = provider_->neighbors(states[j]);
            for (const auto& kv : c) {
                auto it = index.find(kv.first);
                if (it != index.end()) {
                    rows.push_back(it->second);
                    cols.push_back(static_cast<int64_t>(j));
                    re.push_back(kv.second.real());
                    im.push_back(kv.second.imag());
                }
            }
        }
        return py::make_tuple(to_np(rows), to_np(cols), to_np(re), to_np(im));
    }

    // Build the 2N x 2N REAL symmetric embedding M = [[Re,-Im],[Im,Re]] of the
    // projected complex-Hermitian H directly as COO (rows, cols, data), so the
    // Python side does ONE real csr_matrix call — no complex CSR, no
    // Hermitization, no bmat (those were ~30% of the solve). Returns
    // (rows, cols, data, n2 = 2N).
    py::tuple build_real_embedded_coo(py::array_t<uint64_t> ferm_arr,
                                      py::array_t<uint16_t> bos_arr) {
        auto f = ferm_arr.unchecked<2>();
        auto b = bos_arr.unchecked<2>();
        const int64_t N = static_cast<int64_t>(f.shape(0));
        const int W = static_cast<int>(f.shape(1));
        const int n_bos = static_cast<int>(b.shape(1));

        std::vector<MixedDet> states(N);
        std::unordered_map<MixedDet, int64_t, MixedDetHash> index;
        index.reserve(N * 2);
        for (int64_t k = 0; k < N; ++k) {
            MixedDet d;
            d.ferm = read_ferm(f, k, W);
            d.bos.resize(n_bos);
            for (int m = 0; m < n_bos; ++m) d.bos[m] = b(k, m);
            index.emplace(d, k);
            states[k] = std::move(d);
        }

        std::vector<int64_t> rows, cols;
        std::vector<double> data;
        for (int64_t j = 0; j < N; ++j) {
            const ConnMap& c = provider_->neighbors(states[j]);
            for (const auto& kv : c) {
                auto it = index.find(kv.first);
                if (it == index.end()) continue;
                const int64_t i = it->second;
                const double a = kv.second.real();
                const double bb = kv.second.imag();
                rows.push_back(i);     cols.push_back(j);     data.push_back(a);
                rows.push_back(i + N); cols.push_back(j + N); data.push_back(a);
                if (bb != 0.0) {
                    rows.push_back(i);     cols.push_back(j + N); data.push_back(-bb);
                    rows.push_back(i + N); cols.push_back(j);     data.push_back(bb);
                }
            }
        }
        return py::make_tuple(to_np(rows), to_np(cols), to_np(data),
                              static_cast<int64_t>(2 * N));
    }

    // Build the 2N real symmetric embedding M = [[Re,-Im],[Im,Re]] of the
    // projected complex-Hermitian H DIRECTLY in CSC (indptr, indices, data) +
    // its diagonal, so Python passes the arrays straight to davidson_solve_sparse
    // with NO scipy construction at all (the COO->CSR sort was ~30% of the
    // solve). M is symmetric so CSC == CSR for the matvec. The two passes (for
    // M-columns j and j+N) both read connections(state_j); the cache makes the
    // second pass free. Returns (indptr, indices, data, diag, n2 = 2N).
    py::tuple build_real_embedded_csc(py::array_t<uint64_t> ferm_arr,
                                      py::array_t<uint16_t> bos_arr) {
        auto f = ferm_arr.unchecked<2>();
        auto b = bos_arr.unchecked<2>();
        const int N = static_cast<int>(f.shape(0));
        const int W = static_cast<int>(f.shape(1));
        const int n_bos = static_cast<int>(b.shape(1));

        std::vector<MixedDet> states(N);
        std::unordered_map<MixedDet, int, MixedDetHash> index;
        index.reserve(N * 2);
        for (int k = 0; k < N; ++k) {
            MixedDet d;
            d.ferm = read_ferm(f, k, W);
            d.bos.resize(n_bos);
            for (int m = 0; m < n_bos; ++m) d.bos[m] = b(k, m);
            index.emplace(d, k);
            states[k] = std::move(d);
        }

        const int n2 = 2 * N;
        std::vector<int> indptr(n2 + 1, 0);
        std::vector<int> indices;
        std::vector<double> data;
        std::vector<double> diag(n2, 0.0);

        // Pass 1 — M-columns 0..N-1: row i -> Re, row i+N -> Im.
        for (int j = 0; j < N; ++j) {
            const ConnMap& c = provider_->neighbors(states[j]);
            for (const auto& kv : c) {
                auto it = index.find(kv.first);
                if (it == index.end()) continue;
                const int i = it->second;
                const double a = kv.second.real(), bb = kv.second.imag();
                indices.push_back(i);     data.push_back(a);
                if (bb != 0.0) { indices.push_back(i + N); data.push_back(bb); }
                if (i == j) diag[j] = a;
            }
            indptr[j + 1] = static_cast<int>(indices.size());
        }
        // Pass 2 — M-columns N..2N-1: row i -> -Im, row i+N -> Re.
        for (int j = 0; j < N; ++j) {
            const ConnMap& c = provider_->neighbors(states[j]);
            for (const auto& kv : c) {
                auto it = index.find(kv.first);
                if (it == index.end()) continue;
                const int i = it->second;
                const double a = kv.second.real(), bb = kv.second.imag();
                if (bb != 0.0) { indices.push_back(i); data.push_back(-bb); }
                indices.push_back(i + N);  data.push_back(a);
                if (i == j) diag[N + j] = a;
            }
            indptr[N + j + 1] = static_cast<int>(indices.size());
        }
        return py::make_tuple(to_np(indptr), to_np(indices), to_np(data),
                              to_np(diag), n2);
    }

    // Expansion: given a core (ferm, bos, coeff_re, coeff_im), return the
    // neighbor states NOT in the core with their max |H_ij * c_j| score.
    // Python ranks/top-k from there. The neighbor-generation inner loop is C++.
    py::tuple expand(py::array_t<uint64_t> ferm_arr,
                     py::array_t<uint16_t> bos_arr,
                     py::array_t<double> cr_arr,
                     py::array_t<double> ci_arr) {
        auto f = ferm_arr.unchecked<2>();
        auto b = bos_arr.unchecked<2>();
        auto cr = cr_arr.unchecked<1>();
        auto ci = ci_arr.unchecked<1>();
        const size_t N = f.shape(0);
        const int W = static_cast<int>(f.shape(1));
        const int n_bos = static_cast<int>(b.shape(1));

        std::vector<MixedDet> core(N);
        std::unordered_map<MixedDet, char, MixedDetHash> core_set;
        core_set.reserve(N * 2);
        for (size_t k = 0; k < N; ++k) {
            MixedDet d;
            d.ferm = read_ferm(f, k, W);
            d.bos.resize(n_bos);
            for (int m = 0; m < n_bos; ++m) d.bos[m] = b(k, m);
            core_set.emplace(d, 1);
            core[k] = std::move(d);
        }

        std::unordered_map<MixedDet, double, MixedDetHash> scores;
        for (size_t j = 0; j < N; ++j) {
            const cdouble cj(cr(j), ci(j));
            const ConnMap& c = provider_->neighbors(core[j]);
            for (const auto& kv : c) {
                if (core_set.count(kv.first)) continue;
                const double s = std::abs(kv.second * cj);
                auto it = scores.find(kv.first);
                if (it == scores.end()) scores.emplace(kv.first, s);
                else if (s > it->second) it->second = s;
            }
        }

        const size_t M = scores.size();
        // cf as (M, W) uint64 — the arbitrary-width fermion mask, one row per
        // candidate (numpy has no native >64-bit int, so we ship the words).
        py::array_t<uint64_t> cf_arr({(py::ssize_t)M, (py::ssize_t)W});
        auto cf_mut = cf_arr.mutable_unchecked<2>();
        std::vector<uint16_t> cb(M * (size_t)n_bos);
        std::vector<double> sc(M);
        size_t k = 0;
        for (const auto& kv : scores) {
            write_ferm(cf_mut, k, kv.first.ferm);
            for (int m = 0; m < n_bos; ++m) cb[k * n_bos + m] = kv.first.bos[m];
            sc[k] = kv.second;
            ++k;
        }
        py::array_t<uint16_t> cb_arr({(py::ssize_t)M, (py::ssize_t)n_bos});
        std::memcpy(cb_arr.mutable_data(), cb.data(), cb.size() * sizeof(uint16_t));
        return py::make_tuple(cf_arr, cb_arr, to_np(sc));
    }

    // Expansion + top-k in ONE C++ call: score the neighbors NOT in the core
    // exactly as expand() does, but return only the top-`keep` by score
    // (selected in C++ via nth_element). This is the Tier-2 array-native path:
    // the plain expand() returns ALL unique candidates (millions at large core),
    // whose (M, n_bos) arrays would be shipped to Python only to be immediately
    // top-k'd there. Selecting in C++ ships only `keep` rows — bounded RAM +
    // no giant transient. The returned candidates are guaranteed unique (map
    // keys) and disjoint from the core, so the Python side can concatenate them
    // onto the core with NO further deduplication.
    py::tuple expand_topk(py::array_t<uint64_t> ferm_arr,
                          py::array_t<uint16_t> bos_arr,
                          py::array_t<double> cr_arr,
                          py::array_t<double> ci_arr,
                          size_t keep) {
        auto f = ferm_arr.unchecked<2>();
        auto b = bos_arr.unchecked<2>();
        auto cr = cr_arr.unchecked<1>();
        auto ci = ci_arr.unchecked<1>();
        const size_t N = f.shape(0);
        const int W = static_cast<int>(f.shape(1));
        const int n_bos = static_cast<int>(b.shape(1));

        std::vector<MixedDet> core(N);
        std::unordered_map<MixedDet, char, MixedDetHash> core_set;
        core_set.reserve(N * 2);
        for (size_t k = 0; k < N; ++k) {
            MixedDet d;
            d.ferm = read_ferm(f, k, W);
            d.bos.resize(n_bos);
            for (int m = 0; m < n_bos; ++m) d.bos[m] = b(k, m);
            core_set.emplace(d, 1);
            core[k] = std::move(d);
        }

        std::unordered_map<MixedDet, double, MixedDetHash> scores;
        for (size_t j = 0; j < N; ++j) {
            const cdouble cj(cr(j), ci(j));
            const ConnMap& c = provider_->neighbors(core[j]);
            for (const auto& kv : c) {
                if (core_set.count(kv.first)) continue;
                const double s = std::abs(kv.second * cj);
                auto it = scores.find(kv.first);
                if (it == scores.end()) scores.emplace(kv.first, s);
                else if (s > it->second) it->second = s;
            }
        }

        // Rank by score. Point into the (stable) map keys to avoid copying
        // MixedDet during the partial sort; unordered_map keys don't move after
        // the map is fully built.
        std::vector<std::pair<double, const MixedDet*>> ranked;
        ranked.reserve(scores.size());
        for (const auto& kv : scores) ranked.push_back({kv.second, &kv.first});
        const size_t K = std::min(keep, ranked.size());
        if (K < ranked.size()) {
            std::nth_element(
                ranked.begin(), ranked.begin() + K, ranked.end(),
                [](const std::pair<double, const MixedDet*>& a,
                   const std::pair<double, const MixedDet*>& b_) {
                    return a.first > b_.first;   // largest scores first
                });
        }

        // Materialize the top-K candidates into (K, W) / (K, n_bos) arrays.
        py::array_t<uint64_t> cf_arr({(py::ssize_t)K, (py::ssize_t)W});
        auto cf_mut = cf_arr.mutable_unchecked<2>();
        std::vector<uint16_t> cb(K * (size_t)n_bos);
        std::vector<double> sc(K);
        for (size_t k = 0; k < K; ++k) {
            const MixedDet& d = *ranked[k].second;
            write_ferm(cf_mut, (py::ssize_t)k, d.ferm);
            for (int m = 0; m < n_bos; ++m) cb[k * n_bos + m] = d.bos[m];
            sc[k] = ranked[k].first;
        }
        py::array_t<uint16_t> cb_arr({(py::ssize_t)K, (py::ssize_t)n_bos});
        if (K > 0)
            std::memcpy(cb_arr.mutable_data(), cb.data(),
                        cb.size() * sizeof(uint16_t));
        return py::make_tuple(cf_arr, cb_arr, to_np(sc));
    }

    // -------------------------------------------------------------------------
    //  pt2_accumulate: Epstein-Nesbet PT2 pass-1 in C++ (the bottleneck the
    //  pure-Python epstein_nesbet_pt2 hits at large #terms / large core).
    //
    //  Given the variational core (ferm, bos) and its NORMALIZED amplitudes
    //  (cr, ci), do ONE pass over the core's connections and split each row into
    //   * internal (target in core): accumulate the Rayleigh quotient
    //     E_ray = <psi|H|psi> = sum_{a,j in V} conj(c_a) <a|H|j> c_j;
    //   * external (target not in core): accumulate the coherent first-order
    //     amplitude A_a = sum_j <a|H|j> c_j into a per-external-state map.
    //  Then evaluate the Epstein-Nesbet diagonal H_aa = <a|H|a> for each DISTINCT
    //  external state from the diagonal sub-list only (diagonal_fast — no cache
    //  blow-up). The O(M) reduction dE = sum_a |A_a|^2 / (E_var - H_aa), with the
    //  intruder/amp guards, is done vectorized on the Python side.
    //
    //  This mirrors classical/trimci/pt2.py::epstein_nesbet_pt2 pass 1+2 exactly
    //  (same connection convention, same diagonal classification), so C++ and
    //  Python agree to floating-point round-off. Returns
    //    (amp_re (M,), amp_im (M,), Haa (M,), e_ray_re, e_ray_im)
    //  where M is the number of distinct external determinants.
    // -------------------------------------------------------------------------
    py::tuple pt2_accumulate(py::array_t<uint64_t> ferm_arr,
                             py::array_t<uint16_t> bos_arr,
                             py::array_t<double> cr_arr,
                             py::array_t<double> ci_arr) {
        auto f = ferm_arr.unchecked<2>();
        auto b = bos_arr.unchecked<2>();
        auto cr = cr_arr.unchecked<1>();
        auto ci = ci_arr.unchecked<1>();
        const size_t N = f.shape(0);
        const int W = static_cast<int>(f.shape(1));
        const int n_bos = static_cast<int>(b.shape(1));

        std::vector<MixedDet> core(N);
        // MixedDet -> row index in (cr, ci): lets the internal branch recover
        // c_a for the Rayleigh sum without a second array.
        std::unordered_map<MixedDet, int64_t, MixedDetHash> index;
        index.reserve(N * 2);
        for (size_t k = 0; k < N; ++k) {
            MixedDet d;
            d.ferm = read_ferm(f, k, W);
            d.bos.resize(n_bos);
            for (int m = 0; m < n_bos; ++m) d.bos[m] = b(k, m);
            index.emplace(d, static_cast<int64_t>(k));
            core[k] = std::move(d);
        }

        std::unordered_map<MixedDet, cdouble, MixedDetHash> ext_amp;
        cdouble e_ray(0.0, 0.0);
        for (size_t j = 0; j < N; ++j) {
            const cdouble cj(cr(j), ci(j));
            const ConnMap& c = provider_->neighbors(core[j]);
            for (const auto& kv : c) {
                auto it = index.find(kv.first);
                if (it != index.end()) {
                    const cdouble ca(cr(it->second), ci(it->second));
                    e_ray += std::conj(ca) * kv.second * cj;   // <a|H|j> c_j c_a*
                } else {
                    ext_amp[kv.first] += kv.second * cj;        // A_a += <a|H|j> c_j
                }
            }
        }

        const size_t M = ext_amp.size();
        std::vector<double> amp_re(M), amp_im(M), Haa(M);
        size_t k = 0;
        for (const auto& kv : ext_amp) {
            amp_re[k] = kv.second.real();
            amp_im[k] = kv.second.imag();
            Haa[k] = provider_->diagonal_fast(kv.first).real();
            ++k;
        }
        return py::make_tuple(to_np(amp_re), to_np(amp_im), to_np(Haa),
                              e_ray.real(), e_ray.imag());
    }

    // -------------------------------------------------------------------------
    //  build_context: build a SubspaceContext over the given state set.
    //
    //  Converts (ferm_arr, bos_arr) -> C++ MixedDet objects + a state->index
    //  map, then constructs a SubspaceContext that holds the complex CSC of the
    //  projected H. The states/index are LOCAL — consumed at construction and
    //  freed on return (the CSC is self-contained; the context retains neither
    //  the states nor the provider). The CSC is built from provider.neighbors(),
    //  which populates the provider cache along the way.
    //
    //  Typical use (matrix-free eigensolver):
    //    ctx = prov.build_context(ferm, bos)
    //    op  = LinearOperator((N, N), matvec=lambda v: ctx.matvec(...), dtype=complex)
    //    E0, vecs = eigsh(op, k=1, which='SA', tol=1e-10, v0=v0)
    // -------------------------------------------------------------------------
    std::shared_ptr<SubspaceContext>
    build_context(py::array_t<uint64_t> ferm_arr,
                  py::array_t<uint16_t> bos_arr) {
        auto f = ferm_arr.unchecked<2>();
        auto b = bos_arr.unchecked<2>();
        const int N = static_cast<int>(f.shape(0));
        const int W = static_cast<int>(f.shape(1));
        const int n_bos = static_cast<int>(b.shape(1));

        std::vector<MixedDet> states(N);
        std::unordered_map<MixedDet, int, MixedDetHash> index;
        index.reserve(N * 2);
        for (int k = 0; k < N; ++k) {
            MixedDet d;
            d.ferm = read_ferm(f, k, W);
            d.bos.resize(n_bos);
            for (int m = 0; m < n_bos; ++m) d.bos[m] = b(k, m);
            index.emplace(d, k);
            states[k] = std::move(d);
        }
        return std::make_shared<SubspaceContext>(*provider_, states, index);
    }

private:
    std::shared_ptr<MixedHijProvider> provider_;
    int ferm_words_ = 1;   // word count for the single-state debug interface
};

PYBIND11_MODULE(mixed_ci, m) {
    m.doc() = "NuQu Tier-2 mixed fermion-boson H_ij provider (C++ port of hij.connections)";

    py::class_<SubspaceContext, std::shared_ptr<SubspaceContext>>(m, "SubspaceContext")
        .def("size", &SubspaceContext::size,
             "Number of states N in the subspace.")
        .def("nnz", &SubspaceContext::nnz,
             "Number of non-zeros in the complex CSC of the projected H.")
        .def("matvec", &SubspaceContext::matvec,
             py::arg("vr"), py::arg("vi"),
             "Complex H@v: (vr=Re(v), vi=Im(v)) -> (out_re, out_im). "
             "Pure C++ CSC traversal — no hash lookups, safe to call in a loop.")
        .def("diagonal", &SubspaceContext::diagonal,
             "Real diagonal of H over the subspace (H_jj.real for j=0..N-1). "
             "Used to seed the eigsh initial guess.");

    py::class_<PyMixedProvider>(m, "MixedProvider")
        .def(py::init<const std::vector<PyTerm>&, int, size_t, size_t>(),
             py::arg("terms"), py::arg("N_f"), py::arg("cache_cap") = 100000,
             py::arg("max_cache_bytes") = (size_t)1 << 30)
        .def("cache_size", &PyMixedProvider::cache_size)
        .def("cache_bytes", &PyMixedProvider::cache_bytes)
        .def("connections", &PyMixedProvider::connections,
             py::arg("ferm"), py::arg("bos"))
        .def("diagonal", &PyMixedProvider::diagonal,
             py::arg("ferm"), py::arg("bos"))
        .def("build_coo", &PyMixedProvider::build_coo,
             py::arg("ferm"), py::arg("bos"),
             "Projected complex H over `states` as COO (rows, cols, re, im).")
        .def("build_real_embedded_coo", &PyMixedProvider::build_real_embedded_coo,
             py::arg("ferm"), py::arg("bos"),
             "2N real symmetric embedding of H as COO (rows, cols, data, n2).")
        .def("build_real_embedded_csc", &PyMixedProvider::build_real_embedded_csc,
             py::arg("ferm"), py::arg("bos"),
             "2N real symmetric embedding of H as CSC (indptr, indices, data, diag, n2).")
        .def("expand", &PyMixedProvider::expand,
             py::arg("ferm"), py::arg("bos"), py::arg("cr"), py::arg("ci"),
             "Scored neighbor candidates (cand_ferm, cand_bos, scores).")
        .def("expand_topk", &PyMixedProvider::expand_topk,
             py::arg("ferm"), py::arg("bos"), py::arg("cr"), py::arg("ci"),
             py::arg("keep"),
             "Top-`keep` scored candidates NOT in core (cand_ferm, cand_bos, "
             "scores), selected in C++. Unique + disjoint from core; concatenate "
             "onto the core with no dedup. The Tier-2 array-native expansion.")
        .def("build_context", &PyMixedProvider::build_context,
             py::arg("ferm"), py::arg("bos"),
             "Build a SubspaceContext (complex CSC + fast matvec) over the given "
             "state set. Used by cpp_diagonalize_matfree via eigsh + LinearOperator.")
        .def("pt2_accumulate", &PyMixedProvider::pt2_accumulate,
             py::arg("ferm"), py::arg("bos"), py::arg("cr"), py::arg("ci"),
             "Epstein-Nesbet PT2 pass-1 in C++: coherent external amplitudes + "
             "diagonal H_aa over distinct external determinants, plus the internal "
             "Rayleigh quotient. Returns (amp_re, amp_im, Haa, e_ray_re, e_ray_im); "
             "the Python side does the vectorized dE reduction.");
}
