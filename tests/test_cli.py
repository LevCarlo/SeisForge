from seisforge.cli.main import main


def test_top_level_help(capsys):
    assert main([]) == 0

    output = capsys.readouterr().out
    assert "rf" in output
    assert "ant" in output
    assert "inv" in output


def test_rf_help(capsys):
    assert main(["rf"]) == 0

    output = capsys.readouterr().out
    assert "qc" in output
    assert "hkseq" in output


def test_ant_help(capsys):
    assert main(["ant"]) == 0

    output = capsys.readouterr().out
    assert "rotate-ccf" in output
    assert "aftan" in output


def test_ant_rotate_ccf_help_uses_obj_components(capsys):
    try:
        main(["ant", "rotate-ccf", "-h"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("rotate-ccf help should exit")

    output = capsys.readouterr().out
    assert "--obj-components" in output


def test_rf_qc_parser_requires_config():
    try:
        main(["rf", "qc"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("rf qc should require a config file")


def test_ant_rotate_ccf_parser_requires_config():
    try:
        main(["ant", "rotate-ccf"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("ant rotate-ccf should require a config file")


def test_ant_aftan_parser_requires_config():
    try:
        main(["ant", "aftan"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("ant aftan should require a config file")


def test_ant_aftan_help_includes_energy_map_options(capsys):
    try:
        main(["ant", "aftan", "-h"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("aftan help should exit")

    output = capsys.readouterr().out
    assert "--write-energy-map" in output
    assert "--plot-energy-map" in output
