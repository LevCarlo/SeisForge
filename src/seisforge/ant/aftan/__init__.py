"""AFTAN dispersion measurement for ambient-noise CCFs."""

from .config import build_station_aftan_config, _override_energy_map_config
from .core import apply_phase_cycle_shift
from .products import _align_to_period_grid
from .cycle import (
    PhaseCycleOverrideResult,
    infer_dispersion_dat_path,
    override_phase_cycle,
)
from .driver import run_aftan_config, run_aftan_file, run_station_aftan
from .models import (
    AFTANAlphaConfig,
    AFTANBasicConfig,
    AFTANConfig,
    AFTANEnergyMapConfig,
    AFTANPhaseCycleConfig,
    AFTANPMFConfig,
    AFTANPeriodSamplingConfig,
    AFTANQCConfig,
    AFTANResult,
    AFTANSNRConfig,
    AFTANShortDistanceGuardConfig,
    AFTANTimeTaperConfig,
    BranchTrace,
    StationAFTANConfig,
)
from .phase import (
    automatic_pi_over_4,
    physical_branch_for_branch,
    physical_component_for_branch,
    physical_station_pair_for_branch,
)
from .qc import _period_qc, _qc_warnings
from .snr import _aftan_diagram_snr, _snr

__all__ = [
    "AFTANAlphaConfig",
    "AFTANBasicConfig",
    "AFTANConfig",
    "AFTANEnergyMapConfig",
    "AFTANPhaseCycleConfig",
    "AFTANPMFConfig",
    "AFTANPeriodSamplingConfig",
    "AFTANQCConfig",
    "AFTANResult",
    "AFTANSNRConfig",
    "AFTANShortDistanceGuardConfig",
    "AFTANTimeTaperConfig",
    "BranchTrace",
    "StationAFTANConfig",
    "PhaseCycleOverrideResult",
    "apply_phase_cycle_shift",
    "build_station_aftan_config",
    "run_aftan_config",
    "run_aftan_file",
    "run_station_aftan",
    "infer_dispersion_dat_path",
    "override_phase_cycle",
    "automatic_pi_over_4",
    "physical_branch_for_branch",
    "physical_component_for_branch",
    "physical_station_pair_for_branch",
    "_aftan_diagram_snr",
    "_align_to_period_grid",
    "_override_energy_map_config",
    "_period_qc",
    "_qc_warnings",
    "_snr",
]
