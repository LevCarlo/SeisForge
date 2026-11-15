"""`seisforge init` templates: identical to the examples, copied without overwriting."""

from __future__ import annotations

from pathlib import Path

import pytest

from seisforge.cli.main import main
from seisforge.templates import WORKFLOWS, copy_template

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("workflow", list(WORKFLOWS))
def test_template_is_identical_to_the_example(tmp_path, workflow):
    example, names = WORKFLOWS[workflow]
    written = copy_template(workflow, tmp_path)
    assert [p.name for p in written] == list(names)
    for name in names:
        # Run tools/sync_templates.py after editing an example's config.
        assert (tmp_path / name).read_bytes() == (ROOT / "examples" / example / name).read_bytes()


def test_init_refuses_to_overwrite_without_flag(tmp_path, capsys):
    assert main(["init", "aftan", str(tmp_path)]) == 0
    (tmp_path / "config.yml").write_text("edited\n")
    assert main(["init", "aftan", str(tmp_path)]) == 1
    assert (tmp_path / "config.yml").read_text() == "edited\n"
    assert "--overwrite" in capsys.readouterr().out
    assert main(["init", "aftan", str(tmp_path), "--overwrite"]) == 0
    assert (tmp_path / "config.yml").read_text() != "edited\n"


def test_unknown_workflow_is_rejected():
    with pytest.raises(SystemExit):
        main(["init", "no-such-workflow"])
