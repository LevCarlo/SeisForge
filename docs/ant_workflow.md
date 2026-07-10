# Ambient-Noise Workflows

## Rotate CCFs From ZNE To ZRT

Use the grouped CLI:

```bash
seisforge ant rotate-ccf -c examples/configs/ccf_rotate.yml
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

For input lookup, SeisForge always searches the source-receiver pair first:
`CC_ZNE/STA1/STA1_STA2`. Only the diagonal ZNE components `ZZ`, `NN`, and `EE`
may fall back to the reciprocal pair `CC_ZNE/STA2/STA2_STA1`, because those
components are equivalent under station reversal in this storage convention.
Off-diagonal components such as `ZN`, `NZ`, `NE`, and `EN` must match the
requested source-receiver definition and are never read from the reciprocal
directory. Files read from the reciprocal directory are still written with the
current job's station-pair name in the output directory.
