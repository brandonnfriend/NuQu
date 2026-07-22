// ============================================================================
//  NuQu mixed-CI fork — Tier 1: sparse (CSR) Davidson entry point.
//
//  WHERE THIS GOES: paste this lambda registration inside
//      void bind_fast_expansion(py::module& m) { ... }
//  in  cpp/trimci_core/bindings/bind_fast_expansion.cpp,
//  immediately AFTER the existing `davidson_solve_dense` registration
//  (it is in the same scope, so `fe_mod`, `davidson_solve`, `DavidsonParams`,
//   and `DavidsonResult` are all already visible — no extra includes).
//
//  WHY: the stock `davidson_solve_dense` requires the full N×N matrix
//  (O(N^2) memory). Our hybrid feeds the projected H of a selected
//  determinant set; at scale (and doubled by the complex→real 2N embedding)
//  that dense matrix is the memory wall. This variant takes the H as a CSR
//  sparse matrix (data/indices/indptr) and runs the SAME C++ `davidson_solve`
//  over a sparse matvec — O(nnz) memory, OpenMP-parallel. Nothing else in the
//  backend changes; no CMake edit (same translation unit).
//
//  Python caller builds the CSR (e.g. via scipy.sparse from our `connections`
//  oracle, including the 2N real embedding of the complex-Hermitian H) and
//  passes data (float64), indices (int32), indptr (int32), diag (float64).
// ============================================================================

    // ========================================================================
    // davidson_solve_sparse (CSR H) — NuQu mixed-CI hybrid addition
    // ========================================================================
    fe_mod.def("davidson_solve_sparse", [](
        py::array_t<double, py::array::c_style | py::array::forcecast> data_arr,
        py::array_t<int,    py::array::c_style | py::array::forcecast> indices_arr,
        py::array_t<int,    py::array::c_style | py::array::forcecast> indptr_arr,
        py::array_t<double, py::array::c_style | py::array::forcecast> diag_arr,
        DavidsonParams params,
        py::array_t<double, py::array::c_style | py::array::forcecast> initial_guess_arr)
        -> DavidsonResult
    {
        auto d = diag_arr.unchecked<1>();
        size_t N = d.shape(0);
        std::vector<double> diag(N);
        for (size_t i = 0; i < N; ++i) diag[i] = d(i);

        // Borrow the CSR buffers (kept alive by the captured py::array handles).
        const double* data    = data_arr.data();
        const int*    indices = indices_arr.data();
        const int*    indptr  = indptr_arr.data();

        // Sparse CSR matvec: sigma = H @ v.
        auto matvec_fn =
            [data, indices, indptr](const double* v, double* sigma, size_t n) {
                #pragma omp parallel for schedule(static)
                for (long long i = 0; i < static_cast<long long>(n); ++i) {
                    double s = 0.0;
                    for (int p = indptr[i]; p < indptr[i + 1]; ++p)
                        s += data[p] * v[indices[p]];
                    sigma[i] = s;
                }
            };

        // Optional initial guess (same convention as davidson_solve_dense).
        std::vector<std::vector<double>> guess;
        if (initial_guess_arr.size() > 0) {
            auto ig = initial_guess_arr.unchecked<2>();
            int n_guess = ig.shape(0);
            for (int s = 0; s < n_guess; ++s) {
                guess.emplace_back(N);
                for (size_t i = 0; i < N; ++i) guess.back()[i] = ig(s, i);
            }
        }

        DavidsonResult result;
        {
            py::gil_scoped_release release;
            result = davidson_solve(matvec_fn, diag, N, params, guess);
        }
        return result;
    },
    py::arg("data"), py::arg("indices"), py::arg("indptr"), py::arg("diag"),
    py::arg("params") = DavidsonParams{},
    py::arg("initial_guess") = py::array_t<double>(),
    "Solve H x = lambda x via Davidson with a CSR sparse H "
    "(data/indices/indptr/diag). O(nnz) memory; OpenMP-parallel matvec. "
    "NuQu mixed-CI hybrid addition.");
