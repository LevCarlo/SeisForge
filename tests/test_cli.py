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


def test_rf_qc_parser_requires_config():
    try:
        main(["rf", "qc"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("rf qc should require a config file")
