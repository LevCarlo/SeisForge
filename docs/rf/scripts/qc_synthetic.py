"""Figure qc_synthetic.svg (and -dark.svg): RF quality control on traces with known labels.

42 synthetic radial RFs of the crust of `_rf.CRUST`, built as in
`tools/make_example_04_rf.py` (a = 2.5, dt 0.1 s, -10 to 60 s, unit-peak
Gaussian, white noise of 3 % of the peak): 30 good traces (p uniform in
0.045-0.078 s/km, SNR header uniform in 8-30) and 3 each of four assumed
failure types: `lowsnr` (SNR header 2), `farslow` (p = 0.09 s/km), `flipped`
(negative polarity), `noisy` (noise 60 % of the peak).

The `RFstream` tests run in the order of `seisforge rf qc`, each on the
survivors of the previous one, with the settings of `examples/04_rf_qc_hk`
(SNR >= 3, 0.04 <= p <= 0.08, P peak in [-2, 2] s, MAD z <= 2.5, CC >= 0.5,
windows [-5, 30] s; F-test off). Known answer: exactly the 12 bad traces
are removed. (b), (c): the MAD and CC metrics of the traces entering
those tests. (d) F-test p-values of `RFstream.f_test_select`, cumulative:
on the 30 good traces (one model, but p spans 0.045-0.078 s/km, so Ps and
the multiples move by up to about 1 s), and on 30 traces of the same model
at one p = 0.06 s/km with the same noise, where the test's null hypothesis
(one common waveform plus independent noise) holds and the p-values are
uniform (Kolmogorov-Smirnov p-value above 0.01 is asserted).
"""

import numpy as np
from scipy import stats

import _rf
from _rf import _style, plt

SEED, NOISE_FRACTION = 20261115, 0.03

rng = np.random.default_rng(SEED)
events = [(f"good{i:02d}", rng.uniform(0.045, 0.078), rng.uniform(8, 30), 1.0, NOISE_FRACTION)
          for i in range(30)]
for n in range(3):
    events += [
        (f"lowsnr{n}", rng.uniform(0.05, 0.07), 2.0, 1.0, NOISE_FRACTION),
        (f"farslow{n}", 0.09, 15.0, 1.0, NOISE_FRACTION),
        (f"flipped{n}", rng.uniform(0.05, 0.07), 15.0, -1.0, NOISE_FRACTION),
        (f"noisy{n}", rng.uniform(0.05, 0.07), 15.0, 1.0, 0.6),
    ]
traces = []
for name, p, snr, polarity, noise in events:
    time_s, rf = _rf.synthetic_rf(p=p, gauss_a=2.5, dt_s=0.1)
    data = polarity * rf + rng.normal(0.0, noise * np.abs(rf).max(), rf.size)
    traces.append(_rf.sac_trace(time_s, data, p, snr=snr, name=name))
stream = _rf.rf_stream(traces)

window = (-5, 30)
STEPS = [
    ("SNR", lambda s: s.SNR_select(threshold=3.0, sac_head="user2")),
    ("slowness", lambda s: s.slowness_select(slow_min=0.04, slow_max=0.08, sac_head="user0")),
    ("P peak", lambda s: s.P_amp_select(tmin=-2.0, tmax=2.0, window=window)),
    ("MAD", lambda s: s.MAD_select(threshold=2.5, window=window)),
    ("CC", lambda s: s.CC_select(threshold=0.5, window=window)),
]
removed_by = {}
metrics = {}
current = stream
for step, select in STEPS:
    names_before = [tr.stats.sac.kevnm for tr in current.st]
    if step in ("MAD", "CC"):
        full = {"MAD": current.MAD_select(threshold=np.inf, window=window),
                "CC": current.CC_select(threshold=-np.inf, window=window)}[step]
        metrics[step] = dict(zip(names_before, full.qc_metrics))
    current = select(current)
    kept = {tr.stats.sac.kevnm for tr in current.st}
    for name in names_before:
        if name not in kept:
            removed_by[name] = step
bad = {name for name, *_ in events if not name.startswith("good")}
assert set(removed_by) == bad, sorted(set(removed_by) ^ bad)

good_stream = _rf.rf_stream([tr for tr in traces if tr.stats.sac.kevnm.startswith("good")])
f_values = np.array(good_stream.f_test_select(threshold=-np.inf, window=window).qc_metrics)
f_rejected = int(np.sum(f_values < 0.05))
time_s, rf_one_p = _rf.synthetic_rf(p=0.06, gauss_a=2.5, dt_s=0.1)
null_stream = _rf.rf_stream([
    _rf.sac_trace(time_s, rf_one_p + rng.normal(0.0, NOISE_FRACTION * np.abs(rf_one_p).max(),
                                                rf_one_p.size), 0.06, name=f"same{i:02d}")
    for i in range(30)
])
f_null = np.array(null_stream.f_test_select(threshold=-np.inf, window=window).qc_metrics)
f_null_rejected = int(np.sum(f_null < 0.05))
assert stats.kstest(f_null, "uniform").pvalue > 0.01


def draw():
    fig = plt.figure(figsize=(11.5, 3.9))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.5, 0.7, 0.7, 1])
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    colors = {"SNR": _style.EXTRA2, "slowness": _style.EXTRA, "P peak": _style.HIGHLIGHT,
              "MAD": _style.SECOND, "CC": _style.MEASURED}
    ax = axes[0]
    for row, tr in enumerate(traces):
        color = colors.get(removed_by.get(tr.stats.sac.kevnm), _style.ALTERNATIVE)
        t = tr.stats.sac.b + tr.stats.delta * np.arange(tr.stats.npts)
        ax.plot(t, row + 0.9 * tr.data / np.abs(tr.data).max(), color=color, lw=0.6)
    for step, color in colors.items():
        if step in removed_by.values():
            ax.plot([], [], color=color, label=step)
    ax.plot([], [], color=_style.ALTERNATIVE, label="kept")
    ax.set_xlim(-5, 30)
    ax.set_ylim(-1, len(traces))
    ax.set_yticks([])
    ax.set_xlabel("time after direct P (s)")
    ax.set_title("42 traces; the top 12 are bad")
    ax.legend(loc="lower right", fontsize=7, title="removed by", title_fontsize=7)

    for ax, step, threshold, title in ((axes[1], "MAD", 2.5, "MAD z (keep ≤ 2.5)"),
                                       (axes[2], "CC", 0.5, "CC (keep ≥ 0.5)")):
        names = list(metrics[step])
        values = np.array([metrics[step][n] for n in names])
        is_bad = np.array([n in bad for n in names])
        jitter = np.linspace(-0.3, 0.3, len(values))
        ax.plot(jitter[~is_bad], values[~is_bad], "o", color=_style.MEASURED, ms=3, label="good")
        ax.plot(jitter[is_bad], values[is_bad], "o", color=_style.HIGHLIGHT, ms=4, label="bad")
        ax.axhline(threshold, color=_style.INK, lw=1.0, ls="--")
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_title(title)
    axes[1].legend(loc="upper left", fontsize=7, bbox_to_anchor=(0.08, 1.0))

    ax = axes[3]
    cdf = (np.arange(30) + 1) / 30
    ax.step(np.sort(f_values), cdf, where="post", color=_style.MEASURED, lw=1.0,
            label=f"30 good, p 0.045-0.078 ({f_rejected} < 0.05)")
    ax.step(np.sort(f_null), cdf, where="post", color=_style.SECOND, lw=1.0,
            label=f"30 at one p ({f_null_rejected} < 0.05)")
    ax.plot([0, 1], [0, 1], color=_style.TRUTH, lw=1.0, ls=":", label="uniform")
    ax.axvline(0.05, color=_style.HIGHLIGHT, lw=0.8, ls="--", label="threshold 0.05")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F-test p-value")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("F-test p-values")
    ax.legend(loc="lower right", fontsize=7)
    for axis, label in zip(axes, "abcd"):
        _style.panel_label(axis, label)
    fig.tight_layout()
    return fig


_rf.save(draw, __file__)
for step, _ in STEPS:
    print(f"{step:9s} removed: {sorted(n for n, s in removed_by.items() if s == step)}")
print(f"F-test p on good traces: {f_rejected} of 30 below 0.05 "
      f"({int(np.sum(f_values < 0.01))} below 0.01)")
print(f"F-test p at one ray parameter: {f_null_rejected} of 30 below 0.05, "
      f"KS p-value {stats.kstest(f_null, 'uniform').pvalue:.2f}")
