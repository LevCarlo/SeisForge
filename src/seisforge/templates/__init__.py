"""Config templates of the examples, copied by `seisforge init <workflow> [directory]`.

The files under this package are copies of the config files in the
repository's `examples/` folders (one source of truth: `examples/`;
`tools/sync_templates.py` copies them here and
`tests/test_templates.py` checks that both are identical). Paths inside the
templates point to the example's `data/` layout; edit the lines marked
`# change this` for your own data.
"""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

__all__ = ["WORKFLOWS", "copy_template", "template_files"]

# workflow name -> (example folder, files copied by `seisforge init`)
WORKFLOWS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ccf-rotate": ("01_ccf_rotate", ("config.yml",)),
    "aftan": ("02_aftan", ("config.yml",)),
    "eikonal": ("03_eikonal", ("config.yml",)),
    "rf": ("04_rf_qc_hk", ("rf_qc.yml", "hkseq.yml")),
    "hvsr-forward": ("05_hvsr_forward", ("config.yml", "hvsr_forward.py")),
    "inversion": ("06_inversion", ("inv.yaml", "obs.yaml")),
    "full-chain": ("full_chain", ("aftan.yml", "eikonal.yml", "inv.yaml", "obs.yaml", "run.sh", "glue.py")),
}


def template_files(workflow: str) -> tuple[str, ...]:
    """File names of one workflow's template; raises ValueError for an unknown workflow."""
    if workflow not in WORKFLOWS:
        raise ValueError(f"Unknown workflow {workflow!r}; choose from {', '.join(WORKFLOWS)}.")
    return WORKFLOWS[workflow][1]


def copy_template(workflow: str, directory: str | Path = ".", *, overwrite: bool = False) -> list[Path]:
    """Copy a workflow's template files into ``directory`` (created if missing).

    Refuses (FileExistsError, nothing written) if any target file exists,
    unless ``overwrite``. Returns the written paths.
    """
    names = template_files(workflow)
    target = Path(directory)
    destinations = [target / name for name in names]
    existing = [str(path) for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {', '.join(existing)}; pass --overwrite.")
    target.mkdir(parents=True, exist_ok=True)
    source = resources.files(__package__) / workflow.replace("-", "_")
    for name, destination in zip(names, destinations, strict=True):
        with resources.as_file(source / name) as path:
            shutil.copyfile(path, destination)
    return destinations
