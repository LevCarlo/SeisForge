"""Shared helpers for strict YAML configuration sections.

Two small tools that every workflow's config layer can reuse:

- `check_keys` rejects unknown keys in a mapping, naming where the mapping is
  and the closest allowed key, so a misspelled option (``burnin`` for
  ``burn_in``) fails loudly instead of silently falling back to a default.
- `rename_legacy_sections` / `warn_legacy_sections` move deprecated top-level
  section names (``Sampler``) to their current spelling (``sampler``) and emit
  one `ConfigDeprecationWarning` per load listing every rename.

These helpers only inspect and rename keys; they never change values.
"""

from __future__ import annotations

import difflib
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "ConfigDeprecationWarning",
    "check_keys",
    "rename_legacy_sections",
    "warn_legacy_sections",
]


class ConfigDeprecationWarning(FutureWarning):
    """A configuration file uses a deprecated spelling.

    Subclasses `FutureWarning` so it is shown to users by default (plain
    `DeprecationWarning` is hidden outside ``__main__``).
    """


def check_keys(mapping: Any, allowed: Iterable[str], *, where: str) -> None:
    """Raise `ValueError` if ``mapping`` is not a mapping or has unknown keys.

    Parameters
    ----------
    mapping
        The parsed YAML mapping to check.
    allowed
        Every key the reader of this mapping understands, aliases included.
    where
        Location used in the message, e.g. ``"sampler.tempering"``.
    """
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{where} must be a mapping, got {type(mapping).__name__}.")
    allowed = sorted(set(allowed))
    unknown = [key for key in mapping if key not in allowed]
    if not unknown:
        return
    details = []
    for key in unknown:
        close = difflib.get_close_matches(str(key), allowed, n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        details.append(f"{key!r}{hint}")
    noun = "key" if len(unknown) == 1 else "keys"
    raise ValueError(
        f"Unknown {noun} in {where}: {', '.join(details)}. "
        f"Allowed keys: {', '.join(allowed)}."
    )


def rename_legacy_sections(
    config: Mapping[str, Any],
    renames: Mapping[str, str],
    *,
    where: str = "configuration",
) -> tuple[dict, list[tuple[str, str]]]:
    """Return a copy of ``config`` with legacy top-level section names renamed.

    ``renames`` maps each legacy name to its current name; several legacy names
    may map to the same current name (aliases). A section given under both a
    legacy and the current name, or under two legacy aliases, is an error.
    The input is not modified; nested values are shared, not copied.

    Returns the renamed mapping and the ``(legacy, current)`` pairs used, for
    `warn_legacy_sections`.
    """
    if not isinstance(config, Mapping):
        raise ValueError(f"{where} must be a mapping, got {type(config).__name__}.")
    output: dict = {}
    sources: dict[str, str] = {}
    used: list[tuple[str, str]] = []
    for key, value in config.items():
        target = renames.get(key, key)
        if target in sources:
            raise ValueError(
                f"{where} gives section {target!r} twice "
                f"(as {sources[target]!r} and {key!r}); keep only {target!r}."
            )
        sources[target] = key
        output[target] = value
        if target != key:
            used.append((key, target))
    return output, used


def warn_legacy_sections(
    used: Iterable[tuple[str, str]],
    *,
    where: str = "configuration",
    stacklevel: int = 3,
) -> None:
    """Emit one `ConfigDeprecationWarning` listing all legacy section renames."""
    used = list(dict.fromkeys(used))
    if not used:
        return
    mapping = ", ".join(f"{old} -> {new}" for old, new in used)
    warnings.warn(
        f"Deprecated section names in {where}: {mapping}. "
        "Use the lowercase names; the old names will be removed in a future release.",
        ConfigDeprecationWarning,
        stacklevel=stacklevel,
    )
