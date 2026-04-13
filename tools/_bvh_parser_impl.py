from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class BoneMeta:
    index: int               # 0..N-1, hierarchy DFS order
    parent: int              # -1 for root
    name: str                # joint name
    offset: np.ndarray       # shape (3,), BVH offset in BVH units (cm), float32
    channel_offset: int      # starting column in the per-frame motion row
    num_channels: int        # 3 (rotation-only) or 6 (root w/ translation)
    has_translation: bool    # True iff num_channels == 6
    rot_axes: Tuple[str, str, str] = ("Z", "X", "Y")
    # rot_axes[k] is the axis (one of "X","Y","Z") that motion value k of the
    # rotation block applies to, in channel order. BVH channels are intrinsic,
    # so the composed rotation is q = q0 * q1 * q2 where qk is axis_angle(rot_axes[k], angle_k).


@dataclass
class ParsedBVH:
    bones: List[BoneMeta]
    frame_time: float
    num_frames: int
    motion: np.ndarray       # (num_frames, total_channels), float32
    total_channels: int


def _quat_mul(lhs, rhs):
    x1, y1, z1, w1 = lhs
    x2, y2, z2, w2 = rhs
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float32)


def _axis_angle(axis_xyz, angle_rad):
    half = angle_rad * 0.5
    s = np.sin(half)
    return np.array([axis_xyz[0]*s, axis_xyz[1]*s, axis_xyz[2]*s, np.cos(half)], dtype=np.float32)


_AXIS_VEC = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}


def euler_to_quat(rot_axes: Tuple[str, str, str], angles_deg: Tuple[float, float, float]) -> np.ndarray:
    """BVH intrinsic Euler -> unit quaternion [x,y,z,w] (float32).
    rot_axes is the channel-order axis sequence like ("Z","X","Y") or ("Z","Y","X");
    angles_deg are the motion values in the SAME order. BVH channels are applied
    as intrinsic rotations left-to-right, equivalent to the quaternion product
    q = q0 * q1 * q2 under Hamilton convention (right-to-left application)."""
    q0 = _axis_angle(_AXIS_VEC[rot_axes[0]], np.radians(angles_deg[0]))
    q1 = _axis_angle(_AXIS_VEC[rot_axes[1]], np.radians(angles_deg[1]))
    q2 = _axis_angle(_AXIS_VEC[rot_axes[2]], np.radians(angles_deg[2]))
    return _quat_mul(_quat_mul(q0, q1), q2)


def euler_zxy_to_quat(z_deg: float, x_deg: float, y_deg: float) -> np.ndarray:
    """Legacy wrapper for ZXY channel order. Prefer euler_to_quat for new code."""
    return euler_to_quat(("Z", "X", "Y"), (z_deg, x_deg, y_deg))


_POSITION_XYZ = ("Xposition", "Yposition", "Zposition")
_ROTATION_NAMES = {"Xrotation": "X", "Yrotation": "Y", "Zrotation": "Z"}


def _parse_channel_names(ch_names: Tuple[str, ...]) -> Tuple[bool, Tuple[str, str, str]]:
    """Given the CHANNELS token list, return (has_translation, rot_axes).
    Accepts:
      - 3 rotation channels in any permutation of {Xrotation, Yrotation, Zrotation}
      - 6 channels: Xposition Yposition Zposition + any rotation permutation
    """
    count = len(ch_names)
    if count == 6:
        if ch_names[:3] != _POSITION_XYZ:
            raise AssertionError(f"Expected XYZ position channels, got {ch_names[:3]}")
        rot_names = ch_names[3:]
        has_translation = True
    elif count == 3:
        rot_names = ch_names
        has_translation = False
    else:
        raise AssertionError(f"Unsupported channel count {count}: {ch_names}")
    try:
        rot_axes = tuple(_ROTATION_NAMES[n] for n in rot_names)
    except KeyError as e:
        raise AssertionError(f"Unknown rotation channel {e} in {ch_names}") from None
    if set(rot_axes) != {"X", "Y", "Z"}:
        raise AssertionError(f"Rotation channels must cover each of X/Y/Z exactly once: {rot_names}")
    return has_translation, rot_axes


def parse_bvh(path: str) -> ParsedBVH:
    """Parse a standard BVH file. Supports ROOT/JOINT/End Site blocks with
    CHANNELS lines. Only reads channel types we care about:
      - Root (6 channels): Xposition Yposition Zposition Zrotation Xrotation Yrotation
      - Joint (3 channels): Zrotation Xrotation Yrotation
    Asserts the channel order matches one of these two patterns and errors clearly otherwise.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    tokens = text.split()
    i = 0
    n = len(tokens)

    bones: List[BoneMeta] = []
    # stack entries: (bone_index or None for End Site)
    parent_stack: List[int] = []
    channel_cursor = 0
    in_end_site = False

    def peek(offset=0):
        return tokens[i + offset] if i + offset < n else None

    while i < n:
        tok = tokens[i]

        if tok == "ROOT" or tok == "JOINT":
            i += 1
            name = tokens[i]
            i += 1
            parent = parent_stack[-1] if parent_stack else -1
            bone_idx = len(bones)
            # placeholder — offset and channels set when encountered
            bones.append(BoneMeta(
                index=bone_idx,
                parent=parent,
                name=name,
                offset=np.zeros(3, dtype=np.float32),
                channel_offset=channel_cursor,
                num_channels=0,
                has_translation=False,
            ))
            parent_stack.append(bone_idx)
            continue

        if tok == "End":
            # End Site — consume "Site" token then push sentinel
            i += 1  # skip "Site"
            i += 1
            in_end_site = True
            parent_stack.append(-1)  # sentinel
            continue

        if tok == "{":
            i += 1
            continue

        if tok == "}":
            i += 1
            popped = parent_stack.pop() if parent_stack else None
            if popped == -1:
                in_end_site = False
            continue

        if tok == "OFFSET":
            i += 1
            ox, oy, oz = float(tokens[i]), float(tokens[i+1]), float(tokens[i+2])
            i += 3
            if not in_end_site and parent_stack and parent_stack[-1] != -1:
                bones[parent_stack[-1]].offset = np.array([ox, oy, oz], dtype=np.float32)
            continue

        if tok == "CHANNELS":
            i += 1
            count = int(tokens[i])
            i += 1
            ch_names = tuple(tokens[i + k] for k in range(count))
            i += count
            has_translation, rot_axes = _parse_channel_names(ch_names)
            if parent_stack and parent_stack[-1] != -1:
                bone = bones[parent_stack[-1]]
                bone.channel_offset = channel_cursor
                bone.num_channels = count
                bone.has_translation = has_translation
                bone.rot_axes = rot_axes
                channel_cursor += count
            continue

        if tok == "MOTION":
            i += 1
            break

        i += 1

    total_channels = channel_cursor

    # Parse "Frames: N"
    assert tokens[i] == "Frames:", f"Expected 'Frames:' got '{tokens[i]}'"
    i += 1
    num_frames = int(tokens[i])
    i += 1

    # Parse "Frame Time: dt"
    assert tokens[i] == "Frame" and tokens[i+1] == "Time:", \
        f"Expected 'Frame Time:' got '{tokens[i]} {tokens[i+1]}'"
    i += 2
    frame_time = float(tokens[i])
    i += 1

    # All remaining tokens are float data
    data = np.array(tokens[i:i + num_frames * total_channels], dtype=np.float32)
    motion = data.reshape(num_frames, total_channels)

    return ParsedBVH(
        bones=bones,
        frame_time=frame_time,
        num_frames=num_frames,
        motion=motion,
        total_channels=total_channels,
    )


if __name__ == "__main__":
    import sys, pathlib
    paths = sys.argv[1:] or [
        str(pathlib.Path(__file__).resolve().parent.parent / "data" / "motion_actor_legacy.bvh"),
        str(pathlib.Path(__file__).resolve().parent.parent / "data" / "motion_actor_new.bvh"),
    ]
    for path in paths:
        if not pathlib.Path(path).exists():
            print(f"skip (not found): {path}")
            continue
        parsed = parse_bvh(path)
        print(f"--- {pathlib.Path(path).name}")
        print(f"  {len(parsed.bones)} bones, {parsed.num_frames} frames @ {1.0/parsed.frame_time:.1f} Hz, "
              f"total_channels={parsed.total_channels}")
        root = parsed.bones[0]
        print(f"  root: {root.name} channels={root.num_channels} rot_axes={root.rot_axes}")
        print(f"  non-root sample: {parsed.bones[1].name} rot_axes={parsed.bones[1].rot_axes}")

    q = euler_zxy_to_quat(0.0, 0.0, 0.0)
    assert np.allclose(q, [0, 0, 0, 1], atol=1e-6), q
    q = euler_zxy_to_quat(0.0, 0.0, 90.0)
    expected_w = np.cos(np.pi/4)
    expected_y = np.sin(np.pi/4)
    assert np.allclose(q, [0, expected_y, 0, expected_w], atol=1e-5), q

    q_zxy = euler_to_quat(("Z", "X", "Y"), (10.0, 20.0, 30.0))
    q_zyx = euler_to_quat(("Z", "Y", "X"), (10.0, 30.0, 20.0))
    assert not np.allclose(q_zxy, q_zyx, atol=1e-3), "ZXY and ZYX must differ for mixed angles"
    q_any = euler_to_quat(("Z", "Y", "X"), (0.0, 90.0, 0.0))
    assert np.allclose(q_any, [0, np.sin(np.pi/4), 0, np.cos(np.pi/4)], atol=1e-5), q_any
    print("Self-test OK")
