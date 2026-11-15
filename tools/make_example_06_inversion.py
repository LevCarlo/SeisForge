"""Write the synthetic data of the inversion example (station STA001) from a stated truth model.

Run from the repository root:

    PYTHONPATH=src python tools/make_example_06_inversion.py [--overwrite]

Truth (all values assumed, chosen to look like a sedimentary basin over a
crust whose Vs increases with depth; not taken from a real station): the
parameters of `examples/06_inversion/inv.yaml` set to `TRUTH` below, i.e.

- sediment, 0-1 km: Vs from 1.70 km/s at the surface to 2.60 km/s at 1 km
  (linear gradient);
- crust, 1-15 km: cubic B-spline (8 coefficients, geometric knots,
  alpha 1.6) with coefficients 2.65, 2.85, 3.05, 3.20, 3.30, 3.45, 3.60,
  3.70 km/s; half-space below 15 km;
- Vp and density from Vs by Brocher (2005), as in `inv.yaml`.

The model is forwarded with the forward settings of `obs.yaml` (disba,
fundamental-mode Rayleigh phase velocity and ellipticity) but layered much
more finely than the inversion: uniform layers of `TRUTH_DZ_KM` = 1/32 km
(assumed; the reference layering of `seisforge.inv.layering` tests, at which
the STA001 predictions are converged to well below the noise), with the
segment boundaries and `zmax` of `inv.yaml`. Using the inversion's own
layering would hide its discretization error: the example would then agree
with itself whatever the layering. The truth must lie inside the prior: the
script stops if its log prior is not finite.

Noise (assumed): independent Gaussian, standard deviation `DISPERSION_SIGMA_KM_S`
= 0.03 km/s on the phase velocity (about 1 %) and `HV_SIGMA` = 0.02 on the
ellipticity, at the 14 periods `PERIODS_S` (0.8-7 s), drawn with
`numpy.random.default_rng(SEED)`, SEED = 20261115, dispersion first. The
written sigma is the true noise standard deviation.

Outputs (in `--out`, default `examples/06_inversion/data`): `vph_disp.dat`, `hv.dat`
(three columns: period, value, sigma) and `true_model.dat` (the finely layered
truth model the data were computed from: top depth, thickness, Vp, Vs, density;
the truth parameters in the header). Existing files are kept unless
`--overwrite` is given.

All formulas are in the package (`seisforge.inv.setup`, `seisforge.inv.forward`);
this script only sets the values and writes the files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from seisforge.inv.config import load_inversion_input, observation_specs_from_config
from seisforge.inv.forward import predict_dispersion, predict_rayleigh_hv
from seisforge.inv.io import dispersion_request_from_config, rayleigh_hv_request_from_config
from seisforge.inv.layering import layered_model_from_parameterized, reference_discretization
from seisforge.inv.setup import build_inversion_setup

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "06_inversion"

# Assumed truth, in the parameterization of examples/06_inversion/inv.yaml (km/s).
TRUTH = {
    "sediment.vs_top": 1.70,
    "sediment.vs_bottom": 2.60,
    "crust.c0": 2.65,
    "crust.c1": 2.85,
    "crust.c2": 3.05,
    "crust.c3": 3.20,
    "crust.c4": 3.30,
    "crust.c5": 3.45,
    "crust.c6": 3.60,
    "crust.c7": 3.70,
}
PERIODS_S = np.array([0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
DISPERSION_SIGMA_KM_S = 0.03  # assumed, about 1 % of the phase velocity
HV_SIGMA = 0.02  # assumed
SEED = 20261115
# Uniform layer thickness of the truth model (assumed, see the docstring);
# independent of the `discretization` the inversion uses.
TRUTH_DZ_KM = 1.0 / 32.0


def truth_theta(names) -> np.ndarray:
    """`TRUTH` in the order of the inversion parameters; names must match exactly."""
    if set(names) != set(TRUTH):
        raise ValueError(f"inv.yaml parameters {sorted(names)} do not match TRUTH {sorted(TRUTH)}.")
    return np.array([TRUTH[name] for name in names])


def synthetic_data(inv_file: Path, obs_file: Path):
    """Truth layered model, noise-free and noisy observations."""
    inversion_input = load_inversion_input(
        kind="disp_hv", inv_file=inv_file, obs_file=obs_file, require_observations=False
    )
    setup = build_inversion_setup(inversion_input.run_config)
    theta = truth_theta(setup.parameterization.names)
    prior = setup.prior.evaluate(theta)
    if not prior.success:
        raise ValueError(f"The truth lies outside the prior of {inv_file}: {prior.error}")
    discretization = reference_discretization(
        setup.prior.discretization, reference_dz_km=TRUTH_DZ_KM
    )
    model = layered_model_from_parameterized(
        setup.parameterization.vector_to_model(theta), discretization
    )

    specs = observation_specs_from_config(inversion_input.obs_config)
    dispersion = predict_dispersion(model, dispersion_request_from_config({**specs["dispersion"], "periods": PERIODS_S}))
    hv = predict_rayleigh_hv(model, rayleigh_hv_request_from_config({**specs["hv"], "periods": PERIODS_S}))
    if not (np.array_equal(dispersion.periods, PERIODS_S) and np.array_equal(hv.periods, PERIODS_S)):
        raise RuntimeError("The forward solver dropped periods; choose periods it can solve.")

    rng = np.random.default_rng(SEED)
    velocity = dispersion.velocity + rng.normal(0.0, DISPERSION_SIGMA_KM_S, PERIODS_S.size)
    ellipticity = hv.hv + rng.normal(0.0, HV_SIGMA, PERIODS_S.size)
    return setup, theta, model, velocity, ellipticity


def _header(lines) -> str:
    return "\n".join(lines)  # numpy.savetxt prefixes every line with "# "


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inv", type=Path, default=EXAMPLE / "inv.yaml")
    parser.add_argument("--obs", type=Path, default=EXAMPLE / "obs.yaml")
    parser.add_argument("--out", type=Path, default=EXAMPLE / "data")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args(argv)

    setup, theta, model, velocity, ellipticity = synthetic_data(args.inv, args.obs)
    files = {name: args.out / name for name in ("vph_disp.dat", "hv.dat", "true_model.dat")}
    existing = [str(p) for p in files.values() if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {', '.join(existing)}; pass --overwrite.")
    args.out.mkdir(parents=True, exist_ok=True)

    source = [
        "Synthetic data written by tools/make_example_06_inversion.py (see its docstring);",
        f"truth = {', '.join(f'{n}={v:.2f}' for n, v in zip(setup.parameterization.names, theta, strict=True))};",
        f"forwarded in uniform {TRUTH_DZ_KM:g} km layers; Gaussian noise, numpy default_rng({SEED}), "
        f"sigma {DISPERSION_SIGMA_KM_S} km/s (dispersion), {HV_SIGMA} (H/V).",
    ]
    np.savetxt(
        files["vph_disp.dat"],
        np.column_stack([PERIODS_S, velocity, np.full(PERIODS_S.size, DISPERSION_SIGMA_KM_S)]),
        fmt=["%.2f", "%.4f", "%.3f"],
        header=_header([*source, "period_s  rayleigh_phase_velocity_km_s  sigma_km_s"]),
        comments="# ",
    )
    np.savetxt(
        files["hv.dat"],
        np.column_stack([PERIODS_S, ellipticity, np.full(PERIODS_S.size, HV_SIGMA)]),
        fmt=["%.2f", "%.4f", "%.3f"],
        header=_header([*source, "period_s  rayleigh_hv  sigma"]),
        comments="# ",
    )
    np.savetxt(
        files["true_model.dat"],
        np.column_stack([model.layer_tops, model.thickness, model.vp, model.vs, model.rho]),
        fmt="%.6f",
        header=_header(
            [
                "Truth model of the STA001 synthetic data (tools/make_example_06_inversion.py), in the",
                f"uniform {TRUTH_DZ_KM:g} km layers used for the forward calculation (not the",
                "inversion layering); last row is the half-space (thickness 0).",
                source[1],
                "top_km  thickness_km  vp_km_s  vs_km_s  rho_g_cm3",
            ]
        ),
        comments="# ",
    )
    for path in files.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
