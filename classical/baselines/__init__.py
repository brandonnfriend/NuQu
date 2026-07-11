"""
classical/baselines/ — "rule out the cheaper classical methods" demonstrations.

These are the computational demos behind the claim that a TrimCI-style *selected
CI* is the least-bad classical ground-state method for the dynamical-pion EFT.
Each module turns one method's literature verdict (see
`claude/research/classical_baselines/`) into a defensible, laptop-scale number.

Modules
-------
cc_reduction   : D-CC. Coupled cluster (pyscf CCSD/CCSD(T)) on the *CC-friendliest*
                 fermionic reduction of the model (pions frozen out -> a pure
                 2-body 4-component Hubbard-like H). Shows CC is accurate only
                 where the problem is trivially easy and breaks at the physical
                 attractive contact, while exact FCI / selected CI stay controlled.
sign_structure : D-AFQMC-sign. Exact, term-resolved sign-problem diagnostics for
                 the auxiliary-field / QMC family (verifies the model is
                 sign-problematic; pinpoints the spin-isospin terms responsible).
"""
