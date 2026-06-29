"""Plotting helpers for receiver-function streams."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12


def _normalize_abs(data: np.ndarray) -> np.ndarray:
    max_abs = np.max(np.abs(data))
    if max_abs == 0:
        return data
    return data / max_abs


def plot_rf_stream(
    rf_stream,
    by: str | None = "slow",
    window: tuple[float, float] | list[float] = (-5, 30),
    save: bool = False,
    save_path: str | None = None,
    show: bool = False,
):
    """Plot normalized RF traces and their average waveform."""
    if rf_stream.st is None:
        raise ValueError("Stream is not loaded. Please load data first.")
    if len(rf_stream.st) == 0:
        raise ValueError("Cannot plot an empty RF stream.")
    if save and save_path is None:
        raise ValueError("save_path is required when save=True.")

    b = rf_stream.st[0].stats.sac.b
    e = rf_stream.st[0].stats.sac.e
    npts = rf_stream.st[0].stats.npts
    t = np.linspace(b, e, npts)
    t_idx = (t >= window[0]) & (t <= window[1])

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(6, 8),
        height_ratios=[0.15, 1.0],
        sharex=True,
    )
    rf_data = []
    for i, tr in enumerate(rf_stream.st):
        data_norm = _normalize_abs(tr.data[t_idx])
        if by == "slow":
            yy = tr.stats.sac.user0
            shifted = data_norm * 1.2e-3 + yy
        elif by is None:
            yy = i
            shifted = data_norm + i
        else:
            raise ValueError("Invalid `by` argument. Use 'slow' or None.")

        ax[1].plot(t[t_idx], shifted, c="k", lw=0.5)
        ax[1].fill_between(
            t[t_idx],
            yy,
            shifted,
            where=(shifted > yy),
            facecolor="tab:red",
        )
        ax[1].fill_between(
            t[t_idx],
            yy,
            shifted,
            where=(shifted < yy),
            facecolor="blue",
        )
        rf_data.append(data_norm)

    ave_rf = np.mean(rf_data, axis=0)
    ax[0].plot(t[t_idx], ave_rf, c="k", lw=1.0)
    ax[0].fill_between(t[t_idx], ave_rf, where=(ave_rf > 0), facecolor="tab:red")
    ax[0].fill_between(t[t_idx], ave_rf, where=(ave_rf < 0), facecolor="blue")

    if by == "slow":
        ax[1].set_ylabel(
            "Slowness (s/km)",
            fontdict={"family": "Times New Roman", "weight": "bold"},
        )
    else:
        ax[1].set_ylabel(
            "Index (#)",
            fontdict={"family": "Times New Roman", "weight": "bold"},
        )

    ax[1].set_xlabel(
        "Time (s)",
        fontdict={"family": "Times New Roman", "weight": "bold"},
    )
    ax[1].set_xlim(window[0], window[1])
    ax[1].axvline(0, color="k", lw=1.0, ls="--")

    ax[0].set_ylabel(
        "Amplitude",
        fontdict={"family": "Times New Roman", "weight": "bold"},
    )
    ax[0].tick_params(top=True, labeltop=True)
    ax[0].axvline(0, color="k", lw=1.0, ls="--")
    fig.tight_layout()

    if save:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
