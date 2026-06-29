# SeisForge

A Python toolkit for seismic processing and analysis, including:
- Ambient Noise
- Receiver Functions
- Seafloor Compliance

## Installation

For local development:

```bash
conda activate seis
python -m pip install -e ".[dev]"
```

## Command Line Interface

SeisForge groups commands by scientific workflow:

```bash
seisforge --help
seisforge rf --help
seisforge rf qc -c examples/configs/rf_qc.yml --plot
seisforge rf hkseq -c examples/configs/hkseq.yml --mode both --plot
```

Legacy entry points are still available:

```bash
rfqc -c examples/configs/rf_qc.yml --plot
hkseq -c examples/configs/hkseq.yml --mode both --plot
```

## Examples And Docs

Example YAML configs live in `examples/configs/`.

Workflow notes live in `docs/`.
