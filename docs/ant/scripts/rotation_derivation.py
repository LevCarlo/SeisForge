# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Rotation derivation (executable)
#
# This page is the executed form of `docs/ant/scripts/rotation_derivation.py`,
# a [jupytext](https://jupytext.readthedocs.io) percent-format script. It
# derives the ZNE → ZRT rotation of the nine-component cross-correlation
# tensor symbolically with SymPy and then checks, numerically, that the result
# is what `seisforge.ant.rotation.rotate_zne_ccf_to_zrt` computes. The
# reasoning behind each step is on the
# [rotation page](../rotation.md); here the algebra is done by the computer.
#
# Run it locally with `python docs/ant/scripts/rotation_derivation.py`, or open
# it as a notebook with `jupytext --to notebook`.

# %%
import numpy as np
import sympy as sp
from IPython.display import Math, display

sp.init_printing()

# %% [markdown]
# ## Symbols
#
# $\theta$ is the azimuth at the virtual source A (clockwise from north, the
# direction from A towards B); $\psi$ is the back azimuth at the receiver B
# (the direction from B towards A). The CCF tensor has the virtual-source
# component as its **row** index and the receiver component as its **column**
# index: $C_{ab}(t)$ correlates component $a$ at A with component $b$ at B.

# %%
theta, psi = sp.symbols(r"theta psi", real=True)
zne = "ZNE"
C = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"C_{{{zne[i]}{zne[j]}}}"))
C

# %% [markdown]
# ## The two bases
#
# Each row of a rotation matrix is one new unit vector written in the old
# (Z, N, E) components. At the virtual source the radial unit vector is
# $\hat R_A = (-\cos\theta, -\sin\theta)$ in (N, E): it points *away* from B.
# At the receiver $\hat R_B = (\cos\psi, \sin\psi)$ points *towards* A. In
# both cases $\hat T$ is $\hat R$ turned by +90° (clockwise seen from above,
# north → east). These are the vectors the package code uses (see the
# docstring of `rotate_zrt_ccf_to_zne`).

# %%
S = sp.Matrix([
    [1, 0, 0],
    [0, -sp.cos(theta), -sp.sin(theta)],
    [0, sp.sin(theta), -sp.cos(theta)],
])  # rows: Z, R_A, T_A (virtual source A)
R = sp.Matrix([
    [1, 0, 0],
    [0, sp.cos(psi), sp.sin(psi)],
    [0, -sp.sin(psi), sp.cos(psi)],
])  # rows: Z, R_B, T_B (receiver B)
display(Math(r"S = " + sp.latex(S) + r",\qquad \mathsf{R} = " + sp.latex(R)))

# %% [markdown]
# Both are proper rotations: orthogonal with determinant +1. The source block
# is a rotation of the (N, E) axes by $\theta + \pi$, the receiver block a
# rotation by $\psi$.

# %%
assert sp.simplify(S * S.T - sp.eye(3)) == sp.zeros(3)
assert sp.simplify(R * R.T - sp.eye(3)) == sp.zeros(3)
assert sp.simplify(S.det()) == 1 and sp.simplify(R.det()) == 1
print("S and R are orthogonal with determinant +1")

# %% [markdown]
# ## Forward rotation
#
# A component of the correlation at A along a unit vector $\hat u$ and at B
# along $\hat v$ is bilinear: $C_{uv} = \sum_{ij} u_i\,C_{ij}\,v_j$. Stacking
# the new unit vectors as rows of $S$ (source) and $\mathsf R$ (receiver)
# gives
#
# $$C' = S\,C\,\mathsf{R}^{\mathsf T}.$$

# %%
Cp = sp.expand(S * C * R.T)
zrt = "ZRT"
for i in range(3):
    for j in range(3):
        display(Math(rf"C_{{{zrt[i]}{zrt[j]}}} = " + sp.latex(sp.collect(Cp[i, j], list(C)))))

# %% [markdown]
# ## Same numbers as the package
#
# The expressions above, evaluated for random angles and random tensors, agree
# with `rotate_zne_ccf_to_zrt` to rounding error (each entry is a sum of at
# most four products of numbers of order one, so `1e-12` is generous).

# %%
from seisforge.ant.rotation import ZRT_COMPONENTS, rotate_zne_ccf_to_zrt  # noqa: E402

rng = np.random.default_rng(0)
f = sp.lambdify((theta, psi, *list(C)), Cp, "numpy")
worst = 0.0
for _ in range(200):
    az, baz = rng.uniform(0, 360, 2)
    values = rng.normal(size=9)
    ccf = {zne[i] + zne[j]: values[3 * i + j] for i in range(3) for j in range(3)}
    package = rotate_zne_ccf_to_zrt(ccf, az, baz)
    symbolic = np.asarray(f(np.deg2rad(az), np.deg2rad(baz), *values), dtype=float)
    for comp in ZRT_COMPONENTS:
        i, j = zrt.index(comp[0]), zrt.index(comp[1])
        worst = max(worst, abs(package[comp] - symbolic[i, j]))
print(f"largest difference over 200 random cases: {worst:.1e}")
assert worst < 1e-12

# %% [markdown]
# ## Inverse rotation
#
# Because $S$ and $\mathsf R$ are orthogonal, the inverse is
# $C = S^{\mathsf T} C' \mathsf R$. Substituting the forward rotation returns
# the original tensor exactly.

# %%
assert sp.simplify(S.T * Cp * R - C) == sp.zeros(3)
print("S^T (S C R^T) R == C")

# %% [markdown]
# ## Invariants (independent checks)
#
# Rotations cannot create or destroy correlation energy. Three identities hold
# at every lag for any $\theta, \psi$:
#
# 1. the vertical-horizontal pairs keep their length,
#    $C_{ZR}^2 + C_{ZT}^2 = C_{ZN}^2 + C_{ZE}^2$ and
#    $C_{RZ}^2 + C_{TZ}^2 = C_{NZ}^2 + C_{EZ}^2$;
# 2. the Frobenius norm of the horizontal 2×2 block is unchanged;
# 3. its determinant is unchanged, $C_{RR}C_{TT} - C_{RT}C_{TR} =
#    C_{NN}C_{EE} - C_{NE}C_{EN}$, because both rotations have determinant +1.

# %%
H, Hp = C[1:, 1:], Cp[1:, 1:]
checks = {
    "ZR^2+ZT^2": Cp[0, 1] ** 2 + Cp[0, 2] ** 2 - (C[0, 1] ** 2 + C[0, 2] ** 2),
    "RZ^2+TZ^2": Cp[1, 0] ** 2 + Cp[2, 0] ** 2 - (C[1, 0] ** 2 + C[2, 0] ** 2),
    "Frobenius": sum(x**2 for x in Hp) - sum(x**2 for x in H),
    "determinant": Hp.det() - H.det(),
}
for name, expr in checks.items():
    assert sp.simplify(sp.expand_trig(expr)) == 0, name
    print(f"{name}: invariant")

# %% [markdown]
# On a flat Earth $\psi = \theta + \pi$, and the two horizontal blocks become
# the *same* rotation, $Q$. Then $H' = Q H Q^{\mathsf T}$ is a similarity
# transform, so the trace and the antisymmetric part are also invariant:
# $C_{RR} + C_{TT} = C_{NN} + C_{EE}$ and $C_{RT} - C_{TR} = C_{NE} - C_{EN}$.
# On a sphere $\psi \ne \theta + \pi$ in general and these two do **not**
# hold; the three identities above still do.

# %%
flat = {psi: theta + sp.pi}
assert sp.simplify(S[1:, 1:] - R[1:, 1:].subs(flat)) == sp.zeros(2)
trace = sp.simplify((Hp.trace() - H.trace()).subs(flat))
antisym = sp.simplify(((Hp[0, 1] - Hp[1, 0]) - (H[0, 1] - H[1, 0])).subs(flat))
assert trace == 0 and antisym == 0
print("flat Earth: trace and antisymmetric part invariant")
display(Math(r"\text{in general: } (C_{RR}+C_{TT}) - (C_{NN}+C_{EE}) = "
             + sp.latex(sp.simplify(Hp.trace() - H.trace()))))

# %% [markdown]
# ## Angle errors leak energy into the mixed components
#
# Differentiating with respect to the angles gives exact identities. An error
# $\delta\psi$ moves $\delta\psi\,\partial_\psi C'$ into each component, so
# with zero true RT the measured RT is $-\delta\psi\,C_{RR}$ to first order
# (and likewise for the source angle $\theta$).

# %%
leaks = {
    "d/dpsi RT = -RR": sp.diff(Cp[1, 2], psi) + Cp[1, 1],
    "d/dpsi ZT = -ZR": sp.diff(Cp[0, 2], psi) + Cp[0, 1],
    "d/dtheta TR = -RR": sp.diff(Cp[2, 1], theta) + Cp[1, 1],
    "d/dtheta TZ = -RZ": sp.diff(Cp[2, 0], theta) + Cp[1, 0],
}
for name, expr in leaks.items():
    assert sp.simplify(expr) == 0, name
    print(name)

# %% [markdown]
# ## Where the radial axes point
#
# With $\hat x = (\cos\theta, \sin\theta)$ the unit vector from A towards B
# (flat Earth), both radial axes equal $-\hat x$ and both transverse axes are
# equal: the package's R points from the receiver back towards the virtual
# source at *both* stations.

# %%
x_hat = sp.Matrix([sp.cos(theta), sp.sin(theta)])
R_A, T_A = S[1, 1:].T, S[2, 1:].T
R_B, T_B = R[1, 1:].T.subs(flat), R[2, 1:].T.subs(flat)
assert sp.simplify(R_A + x_hat) == sp.zeros(2, 1)
assert sp.simplify(R_B + x_hat) == sp.zeros(2, 1)
assert sp.simplify(T_A - T_B) == sp.zeros(2, 1)
print("R_A = R_B = -x_hat,  T_A = T_B")
