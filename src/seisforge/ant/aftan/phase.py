"""Component and phase-correction rules for AFTAN CCF measurements."""

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
