"""Component and phase-correction rules for AFTAN CCF measurements.

``pi_over_4`` = p adds ``p * pi / 4`` to the picked FTAN phase (core.py). With
the Fourier kernel ``exp(+i omega t)``, a branch ``cos(w (t - r / c) + phi0)``
is measured at ``c`` when ``p = -4 phi0 / pi`` (mod 8).

What is measured
----------------
AFTAN measures the cross-correlation ``C(t)`` itself; it does not convert it
to the empirical Green's function ``G(t) = -dC/dt`` first (Lin, Moschetti &
Ritzwoller 2008, sec. 2). The time derivative is a phase shift of pi/2, and
it is contained in the constants below: for ZZ the diffuse-field CCF is
``J0(kr)`` with far-field phase ``phi0 = +pi/4`` (p = -1), while ``-dC/dt``
has ``phi0 = -pi/4`` (p = +1). Run on ``-dC/dt``, the ZZ constant would have
to be +1 (checked numerically, tests/test_ant_aftan_components.py).

Constants (far field of the 2-D diffuse field, retrograde fundamental-mode
Rayleigh waves, Love waves on TT; derivation in docs/ant/dispersion.md, section 6)
----------------------------------------------------------------------------
They belong to the **stored** component and are the same on both lag branches
(the acausal half of ``J_n`` carries the same ``phi0`` after time reversal).
They assume the R/T axes of ``seisforge.ant.rotation`` (Lin et al. 2008:
R along the propagation at both stations), where ZR = +eps J1 and
RZ = -eps J1 (``J1 ~ cos(kr - 3 pi / 4)``):

=========  ====
component   p
=========  ====
ZZ, RR, TT  -1
ZR          -3
RZ          +1
=========  ====

ZR/RZ rotated by SeisForge before 0.2.0 (R and T reversed) have the opposite
sign, so these constants are pi off for them; such files must be rotated
again. ZT, TZ, RT, TR have no automatic value.

Negative branch and reciprocity: the time-reversed negative lag of ``ZR`` of
pair A_B is the physical ``RZ`` of pair B_A *with a sign*,
``ZR_AB(-t) = -RZ_BA(t)``, because the radial axis at B points away from A
when B is the receiver and towards A when it is the source. The labels
returned by `physical_component_for_branch` ignore that sign; the phase
constant is never taken from the reciprocal component.
"""

from __future__ import annotations


_ZRT_RECIPROCAL_COMPONENTS = {
    "ZZ": "ZZ",
    "ZR": "RZ",
    "ZT": "TZ",
    "RZ": "ZR",
    "RR": "RR",
    "RT": "TR",
    "TZ": "ZT",
    "TR": "RT",
    "TT": "TT",
}

# Stored component -> pi_over_4, both lag branches (module docstring).
AUTOMATIC_PI_OVER_4 = {"ZZ": -1.0, "RR": -1.0, "TT": -1.0, "ZR": -3.0, "RZ": 1.0}
_PHASE_PAGE = "docs/ant/dispersion.md#6-other-components-open-question"


def phase_constant_warning(
    component: str | None,
    pi_over_4: float,
    *,
    component_note: str = "",
) -> str | None:
    """Warning text when ``pi_over_4`` is not known to fit the stored component.

    ``component`` is the stored (file) component, or None when it is unknown
    (``component_note`` then says why). Returns None when the value matches
    `AUTOMATIC_PI_OVER_4`. Pure diagnostic: it changes no measurement.
    """
    unaffected = "group velocity is unaffected"
    if component is None:
        note = f" ({component_note})" if component_note else ""
        return (
            f"component unknown{note}: pi_over_4={pi_over_4:g} was not checked "
            "against it; phase velocity is right only if it fits the component "
            "(ZZ/RR/TT -1, ZR -3, RZ +1); " + unaffected
        )
    component = normalize_component(component)
    expected = AUTOMATIC_PI_OVER_4.get(component)
    if expected is None:
        return (
            f"pi_over_4 for {component} has not been checked; phase velocity "
            f"is unverified; {unaffected}"
        )
    remainder = (pi_over_4 - expected) % 8.0
    if not (abs(remainder) < 1e-9 or abs(remainder - 8.0) < 1e-9):
        return (
            f"pi_over_4={pi_over_4:g} for stored {component} differs from the "
            f"derived {expected:g} ({_PHASE_PAGE}): phase velocity is wrong; {unaffected}"
        )
    return None


def normalize_component(component: str) -> str:
    """Return a normalized two-letter ZRT component code."""
    normalized = component.strip().upper()
    if normalized not in _ZRT_RECIPROCAL_COMPONENTS:
        raise ValueError(
            f"Unsupported AFTAN component {component!r}; expected one of "
            f"{sorted(_ZRT_RECIPROCAL_COMPONENTS)}."
        )
    return normalized


def reciprocal_zrt_component(component: str) -> str:
    """Return the reciprocal ZRT component for a reversed CCF branch."""
    return _ZRT_RECIPROCAL_COMPONENTS[normalize_component(component)]


def physical_component_for_branch(component: str, branch: str) -> str:
    """Label of the physical component represented by a selected CCF branch.

    The negative branch of ``ab`` for pair A_B is labelled ``ba`` of pair
    B_A. For ZR/RZ/ZT/TZ the two differ by a sign (module docstring); the
    label is used for bookkeeping only, never to choose ``pi_over_4``.
    """
    normalized = normalize_component(component)
    if branch == "negative":
        return reciprocal_zrt_component(normalized)
    if branch in {"positive", "stack"}:
        return normalized
    raise ValueError(f"Unsupported AFTAN branch {branch!r}.")


def physical_station_pair_for_branch(
    source_station: str | None,
    receiver_station: str | None,
    branch: str,
) -> tuple[str | None, str | None]:
    """Return the physical propagation station pair for a selected CCF branch."""
    if branch == "negative":
        if source_station is None or receiver_station is None:
            return source_station, receiver_station
        return receiver_station, source_station
    if branch in {"positive", "stack"}:
        return source_station, receiver_station
    raise ValueError(f"Unsupported AFTAN branch {branch!r}.")


def physical_branch_for_branch(branch: str) -> str:
    """Return the branch label after interpreting negative lag as reciprocal lag."""
    if branch == "negative":
        return "positive"
    if branch in {"positive", "stack"}:
        return branch
    raise ValueError(f"Unsupported AFTAN branch {branch!r}.")


def rayleigh_pi_over_4(component: str) -> float:
    """Automatic phase correction (units of pi/4) of a stored component."""
    normalized = normalize_component(component)
    try:
        return AUTOMATIC_PI_OVER_4[normalized]
    except KeyError as exc:
        raise ValueError(
            "Automatic pi_over_4 is only defined for components "
            f"{sorted(AUTOMATIC_PI_OVER_4)}; got {normalized!r}. "
            "Set aftan.pi_over_4 manually for this component."
        ) from exc


def automatic_pi_over_4(component: str, branch: str) -> tuple[str, float]:
    """Physical component label of the branch and the automatic pi/4 coefficient.

    The coefficient is that of the stored ``component`` on every branch.
    """
    physical_component = physical_component_for_branch(component, branch)
    return physical_component, rayleigh_pi_over_4(component)
