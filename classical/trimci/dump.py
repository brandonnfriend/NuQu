"""
Generalized dump writer for the mixed fermion-boson EFT Hamiltonian.

Standard FCIDUMP (Knowles & Handy 1989) serializes a purely fermionic
one-/two-body Hamiltonian (h1, eri) — which is all the released TrimCI can
read. Our dynamical-pion H additionally has bosonic modes and mixed
fermion-boson couplings, so we serialize the full operator as a typed
record list (the spec sketched in
`claude/research/trimci/01_hamiltonian_form.md` Sec. 4). This is the
ground-truth, round-trippable artifact; emitting a *standard* FCIDUMP for a
fermionized encoding (to drive the released package) is a separate
integration route tracked in TODO.md.

File format (human-readable, FCIDUMP-flavored):

    &MIXEDFCI N_FERM=4,N_BOS=3,N_F=4,N_ELEC=1,
      CONSTANT_RE=202.5,CONSTANT_IM=0.0,
    &END
    # TYPE  coeff_re coeff_im  <ops>
    # ops are (mode action) pairs, action 1=creation(dag), 0=annihilation
    # F: fermion only | B: boson only | M: fermion ops '|' boson ops
    F   -14.79570901 0.0   0 1 0 0
    B    135.0 0.0   0 1 0 0
    M    5.78e-05 0.0   0 1 1 0 | 1 0 2 1
    ...

Constant (identity) terms are folded into CONSTANT_RE/IM in the header
rather than emitted as records.
"""

from __future__ import annotations

from .hamiltonian import MixedH, OperatorTerm


def _ops_to_str(ops):
    return " ".join(f"{m} {a}" for (m, a) in ops)


def _str_to_ops(tokens):
    it = iter(tokens)
    return tuple((int(m), int(a)) for m, a in zip(it, it))


def write_dump(H, path, n_elec=None):
    """Write a MixedH to a generalized-FCIDUMP text file."""
    const = H.constant()
    lines = []
    lines.append(
        f"&MIXEDFCI N_FERM={H.n_ferm_modes},N_BOS={H.n_bos_modes},"
        f"N_F={H.N_f},N_ELEC={'' if n_elec is None else n_elec},"
    )
    lines.append(f"  CONSTANT_RE={const.real!r},CONSTANT_IM={const.imag!r},")
    lines.append("&END")
    lines.append("# TYPE  coeff_re coeff_im  <ops: (mode action) pairs, 1=dag 0=ann>")
    lines.append("# F fermion-only | B boson-only | M fermion '|' boson")

    n_records = 0
    for t in H.terms:
        if not t.ferm_ops and not t.bos_ops:
            continue  # constant folded into header
        if t.ferm_ops and not t.bos_ops:
            lines.append(f"F   {t.coeff.real!r} {t.coeff.imag!r}   {_ops_to_str(t.ferm_ops)}")
        elif t.bos_ops and not t.ferm_ops:
            lines.append(f"B   {t.coeff.real!r} {t.coeff.imag!r}   {_ops_to_str(t.bos_ops)}")
        else:
            lines.append(
                f"M   {t.coeff.real!r} {t.coeff.imag!r}   "
                f"{_ops_to_str(t.ferm_ops)} | {_ops_to_str(t.bos_ops)}"
            )
        n_records += 1

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return {"path": path, "n_records": n_records, "constant": const}


def _parse_header(header_lines):
    fields = {}
    blob = " ".join(header_lines).replace("&MIXEDFCI", "").replace("&END", "")
    for tok in blob.split(","):
        tok = tok.strip()
        if "=" in tok:
            k, v = tok.split("=", 1)
            fields[k.strip()] = v.strip()
    return fields


def read_dump(path):
    """Read a generalized-FCIDUMP text file back into a MixedH."""
    with open(path) as f:
        raw = [ln.rstrip("\n") for ln in f]

    header_lines, body_start = [], 0
    for i, ln in enumerate(raw):
        header_lines.append(ln)
        if ln.strip().startswith("&END"):
            body_start = i + 1
            break
    fields = _parse_header(header_lines)

    n_ferm = int(fields["N_FERM"])
    n_bos = int(fields["N_BOS"])
    N_f = int(fields["N_F"])
    const_re = float(fields.get("CONSTANT_RE", 0.0))
    const_im = float(fields.get("CONSTANT_IM", 0.0))

    terms = []
    if abs(const_re) > 0 or abs(const_im) > 0:
        terms.append(OperatorTerm(complex(const_re, const_im), (), ()))

    for ln in raw[body_start:]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        typ = parts[0]
        coeff = complex(float(parts[1]), float(parts[2]))
        rest = parts[3:]
        if typ == "F":
            terms.append(OperatorTerm(coeff, _str_to_ops(rest), ()))
        elif typ == "B":
            terms.append(OperatorTerm(coeff, (), _str_to_ops(rest)))
        elif typ == "M":
            bar = rest.index("|")
            terms.append(OperatorTerm(
                coeff, _str_to_ops(rest[:bar]), _str_to_ops(rest[bar + 1:])
            ))
        else:
            raise ValueError(f"unknown record type {typ!r}")

    meta = {"n_ferm_modes": n_ferm, "n_bos_modes": n_bos, "N_f": N_f,
            "source": path}
    return MixedH(terms=terms, n_ferm_modes=n_ferm, n_bos_modes=n_bos,
                  N_f=N_f, meta=meta)


def summarize(H):
    """Per-block term counts for a quick sanity read of a built Hamiltonian."""
    n_const = sum(1 for t in H.terms if not t.ferm_ops and not t.bos_ops)
    n_ferm = sum(1 for t in H.terms if t.ferm_ops and not t.bos_ops)
    n_bos = sum(1 for t in H.terms if t.bos_ops and not t.ferm_ops)
    n_mixed = sum(1 for t in H.terms if t.ferm_ops and t.bos_ops)
    return {
        "n_ferm_modes": H.n_ferm_modes,
        "n_bos_modes": H.n_bos_modes,
        "N_f": H.N_f,
        "constant": H.constant(),
        "n_terms_total": len(H.terms),
        "n_constant": n_const,
        "n_fermion": n_ferm,
        "n_boson": n_bos,
        "n_mixed": n_mixed,
    }
