from seisforge.rf.hk import HkParam as ModernHkParam
from seisforge.rf.hkSeq import HkParam as LegacyHkParam
from seisforge.rf.qc import RFstream as ModernRFstream
from seisforge.rf.rf import RFstream as LegacyRFstream


def test_legacy_rf_module_exports_modern_objects():
    assert LegacyRFstream is ModernRFstream


def test_legacy_hkseq_module_exports_modern_objects():
    assert LegacyHkParam is ModernHkParam
