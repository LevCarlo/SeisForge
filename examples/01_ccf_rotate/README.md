# 01 CCF rotation: ZNE to ZRT

Ambient-noise cross-correlations (CCFs) are computed between the vertical
(Z), north (N) and east (E) channels of two stations: nine combinations.
Surface-wave work needs them along the path instead: radial (R, along the
line between the stations) and transverse (T). This example rotates one
station pair. The data are synthetic: a known ZRT wavefield (Rayleigh waves
on ZZ, ZR, RZ, RR, a Love wave on TT, nothing on ZT, TZ, RT, TR) turned into
ZNE (`tools/make_example_01_ccf_rotate.py`), so rotating back must recover
it. Every config key: `examples/reference_configs/ccf_rotate.yml`.

## Input

| file | what it is |
|---|---|
| `data/ccf_zne/XX.A/XX.A_XX.B/XX.A_XX.B_<ZZ,ZN,ZE,NZ,NN,NE,EZ,EN,EE>.SAC` | the nine ZNE CCFs of the pair |
| `data/true_zrt.npz` | the truth (only for checking) |
| `config.yml` | settings |

Folder layout: `<root_datadir>/<source station>/<source>_<receiver>/`. The
first letter of a component is the virtual-source station's channel, the
second the receiver's (`NZ` = N at XX.A correlated with Z at XX.B). The
rotation angles come from SAC headers `az`/`baz`, or are computed from
`evla`/`evlo` (virtual source) and `stla`/`stlo` (receiver). From
`XX.A_XX.B_NE.SAC`:

```text
b: -100.0  delta: 0.2  npts: 1001  evlo: 110.0  evla: 30.0  stlo: 110.6  stla: 30.5  kcmpnm: NE
```

## Run

```bash
bash run.sh
```

Under 1 s. Delete `output/` before running again.

## Output

| file | what it is |
|---|---|
| `output/XX.A/XX.A_XX.B/XX.A_XX.B_<ZZ,ZR,ZT,RZ,RR,RT,TZ,TR,TT>.SAC` | the rotated CCFs (input names with the component replaced) |
| `output/XX.A/XX.A_XX.B/rotate_ccf.log` | angles used and where they came from, R/T convention |
| `result.png` | the nine rotated CCFs over the truth |

## How to tell the result is right

`result.png` should look like `expected_result.png`: energy on ZZ, ZR, RZ,
RR and TT, flat ZT, TZ, RT, TR. `run.sh` prints the largest difference from
the noise-free truth per component; on this data it is 0.029-0.040 of the
largest ZZ amplitude (the added noise has standard deviation 0.01, so about
3-4 noise standard deviations; acceptable below 0.05). The log reports
azimuth 46.02 and back azimuth 226.32 degrees.

Common wrong-looking results:

- **Large ZT, TZ, RT or TR** (comparable to ZR): wrong angles. Check the
  coordinates in the headers, that source and receiver are not swapped, and
  any `azimuth`/`back_azimuth` set in the config.
- **ZR and RZ with opposite sign to what you expect**: R points from the
  virtual source towards the receiver at *both* stations, along the
  propagation, and T is R turned 90 degrees clockwise, as in Lin, Moschetti &
  Ritzwoller (2008, Fig. 2). SeisForge before 0.2.0 reversed R and T, so ZR,
  RZ, ZT and TZ written by older versions have the opposite sign; rotate
  them again. See
  [the rotation method page](../../docs/ant/rotation.md).
- **"Failed to resolve required input component"**: a file is missing or
  `input_template` does not match the file names.

## Using your own data

1. `io.root_datadir`: the folder holding `<source>/<source>_<receiver>/`.
2. `io.source_station`, `io.receiver_station`: the pair. For many pairs use
   `jobs:` (see the reference config).
3. `io.input_template`: your file names, with `{component}` where the
   two-letter component is.
4. `io.output_root_datadir`: where the ZRT files go.
