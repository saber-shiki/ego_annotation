# Milk RL candidates — Rerun recordings

Five separate self-contained RRDs, generated from the existing zero-action/256/512/768/1024-step NPZs. No training, model inference or source NPZ edits.

- milk_zero.rrd: zero-action reference (objective1821.42)
- milk_step_0256.rrd: best trained checkpoint, still worse than zero (8900.22)
- milk_step_0512.rrd: checkpoint512 (209826.27)
- milk_step_0768.rrd: checkpoint768 (439834.37)
- milk_step_1024.rrd: checkpoint1024 (6259180.74)

Use Rerun0.37.2 or compatible newer viewer. Example on a machine with a GUI:

```sh
/mnt/user-home/kupingxin/ego_skrl/.venv/bin/rerun milk_step_1024.rrd
```

Each RRD embeds all150 saved JPEG frames and complete NPZ triangle geometry, source camera transforms and correctly resized intrinsics. It can be copied away from this directory without the original NPZ/PLY/JPEG dependencies. Timeline `frame` runs0–149 and `time_s` spans0–4.9666667s for this5-second30FPS clip. Use the timeline scrubber/play control; the 3D view supports rotation/zoom. The second default view is camera/source RGB. Solid mesh colors are not texture reconstruction; the RGB remains source imagery.

Full object mesh:138150vertices/270345triangles. Each hand:778vertices/1538triangles. Preserve NPZ display validity; object148valid frames and2explicit clears. Object geometry is relogged on reappearance so clears cannot leave stale or permanently hidden meshes. Both hand surfaces logged on all150frames.

Coordinate semantics: original source-world frame, camera RDF, world-from-camera/world-from-object transforms; no gravity alignment inferred. NPZ K was1408x1408; RGB is960x960, so project `resized_K` half-pixel formula is used. No points/boxes replace triangles. No learned contact marker or physical acceptance is invented.

All outputs remain uncertain policy/zero-action candidates. Training objective and visible quality did not improve; these files are for inspection, not ground truth or accepted physical annotation.

Each `.receipt.json` binds NPZ/raw-manifest/RRD hashes, frame and topology counts and conventions. `validation.json` records SDK CLI verification plus decoded entity row counts (camera, RGB, hands, poses and clears); this is file/content validation, not an interactive viewer or physical-quality pass.

Exporter source and launch logs:
/mnt/user-home/kupingxin/ego_skrl/policy_rrd_export/
