# MOVIN OSC Receiver & Viewer (Python)

Python project for receiving MOVIN OSC motion and point cloud packets and visualizing them in real time.

## Features

- Receives `/MOVIN/Frame` skeleton chunks
- Reassembles chunked motion frames
- Receives `/MOVIN/PointCloud` point cloud chunks
- Visualizes skeleton and point cloud together in one `pygame` + `PyOpenGL` window
- Supports multiple actors with different skeleton colors
- Shows per-joint local axes in RGB
- Handles non-UTF-8 OSC strings, including common Korean encodings

## OSC Formats

### `/MOVIN/Frame`

Header:

`[timestamp, actorName, frameIdx, numChunks, chunkIdx, totalBoneCount, chunkBoneCount]`

Per-bone payload:

`[boneIndex, parentIndex, boneName, px, py, pz, rqx, rqy, rqz, rqw, qx, qy, qz, qw, sx, sy, sz]`

Where:

- `px, py, pz` are the local position.
- `rqx, rqy, rqz, rqw` are the rest-pose local rotation.
- `qx, qy, qz, qw` are the local rotation.
- `sx, sy, sz` are the local scale.

### `/MOVIN/PointCloud`

Header:

`[frameIdx, totalPoints, chunkIdx, numChunks, chunkPointCount]`

Per-point payload:

`[x, y, z]`

Where:

- `x, y, z` are the world-coordinate position.

## Install With Conda

```powershell
conda env create -f environment.yml
conda activate movin-osc
```

If you prefer to update an existing environment:

```powershell
conda env update -f environment.yml --prune
conda activate movin-osc
```

## Install With venv

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

Optional flags:

```powershell
python main.py --host 0.0.0.0 --port 11235 --point-size 3.0 --fps 60 --axis-size 0.08
```

Default values:

- `--port 11235`
- `--fps 60`
- `--point-size 3.0`
- `--axis-size 0.08`
- `--timeout 1.0`

Viewer controls:

- Left mouse drag: rotate camera
- Mouse wheel: zoom
- `R`: reset camera
- Up/Down arrows: move camera target vertically
- `Esc`: exit

## Offline Testing With BVH

`tools/bvh_osc_sender.py` streams a BVH file as `/MOVIN/Frame` packets so you can exercise the full receiver pipeline without running the Unity sender. It parses the BVH, converts each frame to the wire format (Unity-space local TRS, chunked), and sends over UDP.

Run the viewer and sender in two terminals:

```powershell
# Terminal 1
python main.py

# Terminal 2
python tools/bvh_osc_sender.py
```

By default it streams `data/motion_actor_new.bvh` to `127.0.0.1:11235` as actor `TestActor`, at a **60 Hz output rate**. Two things are auto-detected from the BVH and do not need to be configured:

- **Native frame rate → 60 Hz resampling.** 120 Hz sources are downsampled (every other frame), 30 Hz sources are duplicated (each frame twice), and arbitrary ratios fall back to index-based nearest-frame pick. Use `--fps` to pick a different target rate — playback speed stays correct regardless.
- **Rotation channel order.** Any permutation of `Xrotation / Yrotation / Zrotation` is accepted; each joint's order is read from the BVH `CHANNELS` line and the quaternion is composed accordingly (e.g. ZXY vs. ZYX).
- **Unit scale.** If the BVH appears to be in centimeters (offsets / root height above ~10), a 0.01 cm→m scale is applied. Otherwise it is assumed to already be in meters. Override with `--scale 0.01` or `--scale 1.0` if the heuristic misfires.

Common options:

```powershell
python tools/bvh_osc_sender.py --bvh data/motion_actor_legacy.bvh
python tools/bvh_osc_sender.py --fps 30 --actor Bravo
python tools/bvh_osc_sender.py --chunk-size 7            # force multi-chunk mode
python tools/bvh_osc_sender.py --no-loop --frames 300    # send a fixed number of frames then exit
python tools/bvh_osc_sender.py --scale 1.0               # override auto-detected unit scale
python tools/bvh_osc_sender.py --self-test               # validate the pipeline without opening a socket
```

All flags:

- `--bvh <path>`: BVH file to stream (default `data/motion_actor_new.bvh`)
- `--host <ip>` / `--port <n>`: receiver address (default `127.0.0.1:11235`)
- `--actor <name>`: actor name sent in the header (default `TestActor`)
- `--fps <hz>`: target send rate in Hz (default `60`); BVH frames are resampled to match
- `--chunk-size <n>`: bones per OSC chunk (default `20`)
- `--scale <meters_per_unit>`: override unit scale (default: auto; 0.01 for cm, 1.0 for m)
- `--frames <n>`: stop after N sent frames (default: unlimited)
- `--loop` / `--no-loop`: loop the BVH when it ends (default: loop)
- `--self-test`: run the internal sanity check and exit (no network I/O)

Tested end-to-end with both bundled BVH files:

- `data/motion_actor_legacy.bvh` — 51 bones, 120 Hz, centimeters, ZXY rotation order
- `data/motion_actor_new.bvh` — 60 bones, 60 Hz, meters, ZYX rotation order

## License

Copyright 2025 MOVIN. All Rights Reserved.
