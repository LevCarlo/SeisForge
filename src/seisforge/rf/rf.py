"""Compatibility imports for receiver-function QC tools.

New code should import from :mod:`seisforge.rf.qc`.
"""

from seisforge.rf.qc import RFstream, RFtrace, load_parse_args, main, rf_QC, run_rf_qc_config

__all__ = [
    "RFstream",
    "RFtrace",
    "load_parse_args",
    "main",
    "rf_QC",
    "run_rf_qc_config",
]
