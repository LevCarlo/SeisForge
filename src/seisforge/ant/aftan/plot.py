"""Plotting helpers for AFTAN diagnostics."""

from __future__ import annotations

from pathlib import Path


def _write_aftan_summary_plot(dataset, path: Path, *, phase_dataset=None, snr_label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    amplitude_name = (
        "normalized_amplitude"
        if "normalized_amplitude" in dataset.data_vars
        else "amplitude"
    )
    has_phase = phase_dataset is not None
    height_ratios = [0.4, 0.4, 0.2] if has_phase else [0.7, 0.3]
    nrows = 3 if has_phase else 2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(7.4, 7.0 if has_phase else 5.4),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    group_ax = axes[0]
    snr_ax = axes[-1]
    group_mesh = group_ax.pcolormesh(
        dataset["target_period"],
        dataset["velocity"],
        dataset[amplitude_name].transpose("velocity", "target_period"),
        shading="auto",
        cmap="viridis",
    )
    group_ax.plot(
        dataset["target_period"],
        dataset["picked_group_velocity"],
        color="white",
        linewidth=1.4,
    )
    group_ax.set_ylabel("Group velocity (km/s)")
    group_ax.set_title(f"{dataset.attrs.get('stage', 'ftan')} FTAN summary")
    fig.colorbar(group_mesh, ax=group_ax, label=amplitude_name.replace("_", " "))

    if has_phase:
        phase_ax = axes[1]
        phase_mesh = phase_ax.pcolormesh(
            phase_dataset["target_period"],
            phase_dataset["phase_velocity"],
            phase_dataset["phase_candidate_amplitude"].transpose(
                "phase_velocity",
                "target_period",
            ),
            shading="auto",
            cmap="magma",
        )
        phase_ax.plot(
            phase_dataset["target_period"],
            phase_dataset["picked_phase_velocity"],
            color="white",
            linewidth=1.4,
        )
        phase_ax.set_ylabel("Phase velocity (km/s)")
        fig.colorbar(phase_mesh, ax=phase_ax, label="candidate amplitude")

    snr_ax.plot(
        dataset["target_period"],
        dataset["picked_snr"],
        color="black",
        linewidth=1.2,
    )
    snr_ax.set_ylabel(snr_label)
    snr_ax.set_xlabel("Period (s)")
    snr_ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)
