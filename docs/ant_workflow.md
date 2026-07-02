# Ambient-Noise Workflows

## Rotate CCFs From ZNE To ZRT

Use the grouped CLI:

```bash
seisforge ant rotate-ccf -c examples/configs/ccf_rotate.yml
```

The rotation workflow keeps the numerical rotation in
`seisforge.ant.rotation.rotate_zne_ccf_to_zrt` and puts file IO in
`seisforge.ant.ccf`. The YAML config describes persistent batch settings:
input/output directories, filename templates, output components, and overwrite
behavior. CLI flags are intended as small run-time overrides.

Minimal single-pair config:

```yaml
io:
  input_dir: /path/to/CCF_ZNE/NET.STA1_NET.STA2
  output_dir: /path/to/CCF_ZRT/NET.STA1_NET.STA2
  input_template: "*_{component}_pws.SAC"

obj_components: [ZZ, ZR, ZT, RZ, RR, RT, TZ, TR, TT]
```

Output filenames preserve the matched input filename by default and replace only
the component token. For example,
`NET.STA1_NET.STA2_ZN_pws.SAC` becomes
`NET.STA1_NET.STA2_ZR_pws.SAC`. The legacy output template
`{component}.SAC` is treated the same way so old configs keep the full naming
pattern. To force a custom output name, set `output_template`; supported fields
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
and `rotation` defaults and can override paths, azimuths, and names.

Requested output ZRT `obj_components` automatically determine which input ZNE
components are read. The log records this inferred `required_input_components`
list for each job, so configs do not need a separate input-component list.
