# Ambient-Noise Workflows

!!! tip "Runnable example"
    `examples/01_ccf_rotate` runs this workflow on synthetic data with a known answer
    (`bash run.sh`); see the [examples overview](examples.md). The commands below
    use the full reference config in `examples/reference_configs/`.


## Rotate CCFs From ZNE To ZRT

`seisforge ant rotate-ccf` rotates the ZNE SAC cross-correlations of **one
station pair** to ZRT. The numerical rotation is
`seisforge.ant.rotation.rotate_zne_ccf_to_zrt`; `seisforge.ant.ccf_rotate`
only reads and writes the files.

```yaml
io:
  input_template: CCF_ZNE/NET.STA1_NET.STA2_{component}_pws.SAC
  output_template: CCF_ZRT/NET.STA1_NET.STA2_{component}_pws.SAC
  overwrite: false
rotation:
  components: [ZZ, ZR, ZT, RZ, RR, RT, TZ, TR, TT]   # default: all nine
  # azimuth: 120.0        # degrees, give both or neither
  # back_azimuth: 300.0
```

- **Files.** `input_template` is the path of the ZNE files with `{component}`
  (`ZZ`, `ZN`, ... `EE`); only the files the requested ZRT components need are
  read (ZR needs ZN and ZE, RR the four horizontal pairs, ...). Shell
  wildcards are allowed if exactly one file matches each component.
  `output_template` is the path of the ZRT files with `{component}`. Relative
  paths are relative to the config file. These are the only config keys
  besides `components`, `azimuth`, `back_azimuth` and `overwrite`; an unknown
  key is an error.
- **Angles.** Azimuth (station 1 to 2) and back azimuth (2 to 1) in degrees:
  from the config or `--azimuth`/`--back-azimuth` when both are given, else
  from SAC `az`/`baz` of the first input file, else computed on the WGS84
  ellipsoid from `evla/evlo` (virtual source) and `stla/stlo` (receiver).
  The log names the source.
- **Axes.** R points from the virtual source towards the receiver at both
  stations, along the propagation, as in Fig. 2 of Lin, Moschetti &
  Ritzwoller (2008), and T is R turned 90 degrees clockwise. SeisForge
  before 0.2.0 reversed R and T, so its ZR, RZ, ZT and TZ have the opposite
  sign (ZZ, RR, RT, TR, TT are bit for bit the same); such files are not
  recognised and must be rotated again. See
  [Rotation](ant/rotation.md#relation-to-lin-et-al-2008-and-to-seisforge-before-020).
- **Outputs.** Each ZRT file keeps the SAC header of its first ZNE input,
  with `kcmpnm` set to the component and `kuser0` to `lin2008` (a record).
  Existing files are not replaced unless `io.overwrite: true` or
  `--overwrite`. `rotate_ccf.log` next to the outputs lists the files read
  and written, the angles and their source, and the software versions.
- **Never the reciprocal pair.** A missing input is an error naming the file;
  the reciprocal correlation B-A cannot stand in for A-B, not even for `ZZ`,
  because its lags are reversed,
  $C^{BA}_{ij}(t) = C^{AB}_{ji}(-t)$ (see
  [Rotation, reversing source and receiver](ant/rotation.md#reversing-source-and-receiver)).

Many pairs are a loop in your own script, overriding the templates:

```bash
for pair in A_B A_C B_C; do
  seisforge ant rotate-ccf -c ccf_rotate.yml \
    --input-template "CCF_ZNE/${pair}_{component}_pws.SAC" \
    --output-template "CCF_ZRT/${pair}_{component}_pws.SAC"
done
```
