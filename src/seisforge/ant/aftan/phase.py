"""Component and phase-correction rules for AFTAN CCF measurements.

``pi_over_4`` = p adds ``p * pi / 4`` to the picked FTAN phase (core.py), so
a causal branch ``cos(w (t - r / c) + phi0)`` is measured at ``c`` when
``p = -4 phi0 / pi``.

Known issue (docs/aftan_phase_constants.md): `_RAYLEIGH_PI_OVER_4` together
with the negative-branch ZR <-> RZ swap is not consistent with the package
rotation (`seisforge.ant.rotation`). An independent diffuse-field check
(`seisforge.ant.synthetic.synthetic_rayleigh_ccf_zne`, Bessel closed forms)
gives `_CHECKED_PI_OVER_4`: ZR needs +1 and RZ -3 on both lag branches. The
table is left unchanged until a separate task fixes it; `phase_constant_warning`
flags the affected measurements without changing any number.
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

_RAYLEIGH_PI_OVER_4 = {
    "ZZ": -1.0,
    "ZR": 1.0,
    "RZ": -1.0,
    "RR": -1.0,
    "TT": -1.0,
}


# Values confirmed by the independent check, for the stored component on
# either lag branch (docs/aftan_phase_constants.md). Not used to measure.
_CHECKED_PI_OVER_4 = {
    "ZZ": -1.0,
    "RR": -1.0,
    "TT": -1.0,
    "ZR": 1.0,
    "RZ": -3.0,
}
_PHASE_ISSUE_PAGE = "docs/aftan_phase_constants.md"


def phase_constant_warning(
    component: str | None,
    pi_over_4: float,
    *,
    component_note: str = "",
) -> str | None:
    """Warning text when ``pi_over_4`` is not known to fit the stored component.

    ``component`` is the stored (file) component, or None when it is unknown
    (``component_note`` then says why). Returns None when the value matches
    the independent check for ZZ, RR or TT. Pure diagnostic: it changes no
    measurement.
    """
    unaffected = "group velocity is unaffected"
    if component is None:
        note = f" ({component_note})" if component_note else ""
        return (
            f"component unknown{note}: pi_over_4={pi_over_4:g} was not checked "
            "against it; phase velocity is right only if it fits the component "
            "(checked: ZZ/RR/TT -1, ZR +1, RZ -3); " + unaffected
        )
    component = normalize_component(component)
    checked = _CHECKED_PI_OVER_4.get(component)
    if checked is None:
        return (
            f"pi_over_4 for {component} has not been checked; phase velocity "
            f"is unverified; {unaffected}"
        )
    remainder = (pi_over_4 - checked) % 8.0
    if not (abs(remainder) < 1e-9 or abs(remainder - 8.0) < 1e-9):
        return (
            f"pi_over_4={pi_over_4:g} for stored {component} differs from the "
            f"independently checked {checked:g} (known issue, {_PHASE_ISSUE_PAGE}): "
            f"phase velocity is wrong; {unaffected}"
        )
    if component in {"ZR", "RZ"}:
        return (
            f"ZR/RZ phase constants are under revision ({_PHASE_ISSUE_PAGE}); "
            f"pi_over_4={pi_over_4:g} agrees with the independent check for {component}"
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
    """Return the physical component represented by a selected CCF branch."""
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
    """Return the Rayleigh-wave phase correction in units of pi/4."""
    normalized = normalize_component(component)
    try:
        return _RAYLEIGH_PI_OVER_4[normalized]
    except KeyError as exc:
        raise ValueError(
            "Automatic pi_over_4 is only defined for Rayleigh Z/R and "
            f"self components {sorted(_RAYLEIGH_PI_OVER_4)}; got {normalized!r}. "
            "Set aftan.pi_over_4 manually for this component."
        ) from exc


def automatic_pi_over_4(component: str, branch: str) -> tuple[str, float]:
    """Return physical component and automatic pi/4 coefficient."""
    physical_component = physical_component_for_branch(component, branch)
    return physical_component, rayleigh_pi_over_4(physical_component)
