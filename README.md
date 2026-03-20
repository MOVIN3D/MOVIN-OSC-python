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

### `/MOVIN/PointCloud`

Header:

`[frameIdx, totalPoints, chunkIdx, numChunks, chunkPointCount]`

Per-point payload:

`[x, y, z]`

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

## Notes

- Motion drawing reconstructs world-space joint positions from transmitted local transforms.
- Axis colors are `X=red`, `Y=green`, `Z=blue`.
