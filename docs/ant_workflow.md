# Ambient-Noise Workflows

!!! tip "Runnable example"
    `examples/01_ccf_rotate` runs this workflow on synthetic data with a known answer
    (`bash run.sh`); see the [examples overview](examples.md). The commands below
    use the full reference config in `examples/reference_configs/`.


## Rotate CCFs From ZNE To ZRT

Use the grouped CLI:

```bash
seisforge ant rotate-ccf -c examples/reference_configs/ccf_rotate.yml
```

The rotation workflow keeps the numerical rotation in
`seisforge.ant.rotation.rotate_zne_ccf_to_zrt` and puts file IO in
`seisforge.ant.ccf_rotate`. The YAML config describes persistent batch
settings: the CCF root directories, station pair, filename template, output
components, and overwrite behavior. CLI flags are intended as small run-time
overrides.

Minimal single-pair config:

```yaml
io:
  root_datadir: /path/to/CCF_ZNE
  output_root_datadir: /path/to/CCF_ZRT
  source_station: NET.STA1
  receiver_station: NET.STA2
  input_template: "*_{component}_pws.SAC"

obj_components: [ZZ, ZR, ZT, RZ, RR, RT, TZ, TR, TT]
```

SeisForge builds the primary input directory as
`root_datadir/source_station/source_station_receiver_station` and writes to the
same station-pair layout under `output_root_datadir`. Select the desired CCF
stack product directly in `input_template`. For example,
`input_template: "*_{component}_pws.SAC"` reads only the `pws` CCF products and
ignores other stack products such as `ls` or `linear`.

Output filenames preserve the matched input filename by default and replace only
the component token. For example,
`NET.STA1_NET.STA2_ZN_pws.SAC` becomes
`NET.STA1_NET.STA2_ZR_pws.SAC`. This is the default when `output_template` is
not set. To force a custom output name, set `output_template`; supported fields
are `{component}`, `{name}`, `{source_component}`, `{source_name}`, and
`{source_stem}`.

R and T follow `rotation.radial_convention` (CLI `--radial-convention`):
`lin2008` (default) points R from the virtual source towards the receiver at
both stations, along the propagation, as in Fig. 2 of Lin, Moschetti &
Ritzwoller (2008), with T turned 90 degrees clockwise from R; `legacy`
reproduces SeisForge before that change, where R and T were reversed, so
ZR, RZ, ZT and TZ have the opposite sign (ZZ, RR, RT, TR, TT are bit for bit
the same). The value is written to `rotate_ccf.log` and to SAC `kuser0` of
each output file, which AFTAN reads to pick the ZR/RZ phase constants. See
[Rotation](ant/rotation.md#relation-to-lin-et-al-2008-and-the-legacy-convention).

By default, azimuth and back azimuth are inferred from SAC headers. The workflow
first uses `az/baz` when both are present. If they are missing, it computes the
angles from coordinates, assuming `evla/evlo` describe the virtual-source
station and `stla/stlo` describe the receiver station. Set `azimuth` and/or
`back_azimuth` in the YAML or with CLI flags only when the headers are missing
or need to be overridden.

Each job writes `rotate_ccf.log` into its output directory. The log records the
input files, output files, resolved angles, whether angles came from config,
`az/baz`, or coordinate calculation, and any error raised during the job. When
angles are not specified, the log explicitly records the failed SAC HEADER
`az/baz` read before coordinate fallback is used.

For multiple station pairs, add `jobs`. Each job inherits the top-level `io`
and `rotation` defaults and can override stations, filename templates, and
azimuths.

Requested output ZRT `obj_components` automatically determine which input ZNE
components are read. The log records this inferred `required_input_components`
list for each job, so configs do not need a separate input-component list.

For input lookup, SeisForge reads only the source-receiver pair of the job,
`CC_ZNE/STA1/STA1_STA2`, and raises an error naming the missing file if a
required component is not there. It never falls back to the reciprocal
directory `CC_ZNE/STA2/STA2_STA1`, not even for `ZZ`, `NN` or `EE`: the
reciprocal correlation describes propagation in the opposite direction,
$C^{BA}_{ij}(t) = C^{AB}_{ji}(-t)$, so its lags are reversed and it cannot
stand in for the requested file (see
[Rotation, reversing source and receiver](ant/rotation.md#reversing-source-and-receiver)).
