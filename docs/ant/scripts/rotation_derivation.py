"""Derive the ZNE -> ZRT rotation of a nine-component CCF tensor with SymPy.

Writes the equations shown on docs/ant/rotation.md as LaTeX snippets into
docs/ant/generated/ (the page includes them), so no rotation formula on that
page is typed by hand. It then checks that the symbolic result is what
`seisforge.ant.rotation.rotate_zne_ccf_to_zrt` computes. Run it after changing
the rotation:

    PYTHONPATH=<worktree>/src python docs/ant/scripts/rotation_derivation.py

Conventions (Lin, Moschetti & Ritzwoller 2008, Fig. 2): C_ab correlates
component a at the virtual source A with component b at the receiver B;
theta is the azimuth at A towards B, psi the back azimuth at B towards A,
both clockwise from north; R points from A towards B at both stations, and T
is R turned 90 degrees clockwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sympy as sp

from seisforge.ant.rotation import ZRT_COMPONENTS, rotate_zne_ccf_to_zrt

OUT_DIR = Path(__file__).resolve().parents[1] / "generated"
ZNE, ZRT = "ZNE", "ZRT"

theta, psi = sp.symbols("theta psi", real=True)
C = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"C_{{{ZNE[i]}{ZNE[j]}}}"))  # rows: A, columns: B

# Rows are the new unit vectors (Z, R, T) written in (Z, N, E).
M_s = sp.Matrix([  # virtual source A: R = (cos theta, sin theta) points towards B
    [1, 0, 0],
    [0, sp.cos(theta), sp.sin(theta)],
    [0, -sp.sin(theta), sp.cos(theta)],
])
M_r = sp.Matrix([  # receiver B: R = -(cos psi, sin psi) points away from A
    [1, 0, 0],
    [0, -sp.cos(psi), -sp.sin(psi)],
    [0, sp.sin(psi), -sp.cos(psi)],
])
for M in (M_s, M_r):  # proper rotations
    assert sp.simplify(M * M.T - sp.eye(3)) == sp.zeros(3) and sp.simplify(M.det()) == 1

# A CCF is bilinear in the two recordings: projecting A on u and B on v gives
# sum_ij u_i C_ij v_j. With the new axes as rows of M_s and M_r:
C_zrt = sp.expand(M_s * C * M_r.T)

# Flat Earth (psi = theta + pi): both stations use the same rotation.
flat = {psi: theta + sp.pi}
assert sp.simplify(M_r.subs(flat) - M_s) == sp.zeros(3)
C_flat = sp.expand(C_zrt.subs(flat))


def _latex(expr) -> str:
    return sp.latex(expr, fold_func_brackets=True)


def _combination(expr) -> str:
    """LaTeX of a linear combination of the C_ij, terms in Z, N, E order."""
    terms = []
    for i in range(3):
        for j in range(3):
            coefficient = sp.factor(expr.coeff(C[i, j]))
            if coefficient == 0:
                continue
            text = {1: "", -1: "-"}.get(coefficient, _latex(coefficient) + r"\,")
            terms.append(text + _latex(C[i, j]))
    out = terms[0]
    for term in terms[1:]:
        out += " - " + term[1:].lstrip() if term.startswith("-") else " + " + term
    return out


def _aligned(matrix, components) -> str:
    rows = [
        rf"C_{{{c}}} &= {_combination(matrix[ZRT.index(c[0]), ZRT.index(c[1])])}"
        for c in components
    ]
    return "$$\n\\begin{aligned}\n" + "\\\\\n".join(rows) + "\n\\end{aligned}\n$$\n"


OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "rotation_matrices.md").write_text(
    f"$$\nM_s = {_latex(M_s)},\\qquad M_r = {_latex(M_r)}\n$$\n", encoding="utf-8"
)
(OUT_DIR / "rotation_components.md").write_text(
    _aligned(C_zrt, [c for c in ZRT_COMPONENTS if c != "ZZ"]), encoding="utf-8"
)
(OUT_DIR / "rotation_flat.md").write_text(
    _aligned(C_flat, ["ZR", "RR", "TT"]), encoding="utf-8"
)

# Same numbers as the package: random angles and tensors. Each entry is a sum
# of at most four products of order-one numbers, so 1e-12 is generous.
evaluate = sp.lambdify((theta, psi, *list(C)), C_zrt, "numpy")
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(200):
    azimuth, back_azimuth = rng.uniform(0.0, 360.0, 2)
    values = rng.normal(size=9)
    ccf = {ZNE[i] + ZNE[j]: values[3 * i + j] for i in range(3) for j in range(3)}
    package = rotate_zne_ccf_to_zrt(ccf, azimuth, back_azimuth)
    symbolic = np.asarray(evaluate(np.deg2rad(azimuth), np.deg2rad(back_azimuth), *values), float)
    for c in ZRT_COMPONENTS:
        worst = max(worst, abs(package[c] - symbolic[ZRT.index(c[0]), ZRT.index(c[1])]))
assert worst < 1e-12, worst
print(f"wrote {OUT_DIR}; symbolic vs package, largest difference over 200 cases: {worst:.1e}")
