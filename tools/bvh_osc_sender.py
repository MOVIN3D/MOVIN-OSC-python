"""Offline OSC test sender: stream a BVH file as /MOVIN/Frame packets.

Usage:
    python tools/bvh_osc_sender.py --bvh data/motion_actor_legacy.bvh
    python tools/bvh_osc_sender.py --chunk-size 7 --actor Bravo
    python tools/bvh_osc_sender.py --self-test
"""
import argparse
import os
import sys
import time
from typing import List

import numpy as np
from pythonosc.udp_client import SimpleUDPClient

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _bvh_parser_impl import ParsedBVH, BoneMeta, parse_bvh, euler_to_quat
from _coord_impl import bvh_pos_to_unity, bvh_quat_to_unity
from _osc_sender_impl import FrameBone, send_frame


_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
_UNIT_SCALE = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def detect_unit_scale(parsed: ParsedBVH) -> float:
    """Heuristically pick 0.01 (cm→m) or 1.0 (already in meters) based on data extent.
    Human skeletons in meters have offsets ≤ ~1; in cm they routinely exceed 10."""
    max_offset = 0.0
    for bone in parsed.bones:
        m = float(np.max(np.abs(bone.offset))) if bone.offset.size else 0.0
        if m > max_offset:
            max_offset = m
    root_y_abs = 0.0
    if parsed.bones and parsed.bones[0].has_translation and parsed.num_frames > 0:
        root_y_abs = abs(float(parsed.motion[0][1]))
    size_hint = max(max_offset, root_y_abs)
    return 0.01 if size_hint > 10.0 else 1.0


def build_frame_bones(parsed: ParsedBVH, frame_idx: int, scale: float) -> List[FrameBone]:
    row = parsed.motion[frame_idx]
    frame_bones: List[FrameBone] = []
    for meta in parsed.bones:
        col = meta.channel_offset
        if meta.has_translation:
            tx, ty, tz = float(row[col]), float(row[col + 1]), float(row[col + 2])
            angles = (float(row[col + 3]), float(row[col + 4]), float(row[col + 5]))
            local_pos_bvh = np.array([tx, ty, tz], dtype=np.float32)
        else:
            angles = (float(row[col]), float(row[col + 1]), float(row[col + 2]))
            local_pos_bvh = meta.offset

        local_pos_unity = bvh_pos_to_unity(local_pos_bvh, scale=scale)
        local_rot_bvh = euler_to_quat(meta.rot_axes, angles)
        local_rot_unity = bvh_quat_to_unity(local_rot_bvh)

        frame_bones.append(FrameBone(
            bone_index=meta.index,
            parent_index=meta.parent,
            name=meta.name,
            local_position=local_pos_unity,
            rest_rotation=_IDENTITY_QUAT,
            local_rotation=local_rot_unity,
            local_scale=_UNIT_SCALE,
        ))
    return frame_bones


def _resample_strategy(bvh_fps: float, target_fps: float) -> str:
    ratio = bvh_fps / target_fps
    if abs(ratio - 1.0) < 1e-3:
        return "passthrough (1:1)"
    if ratio > 1.0:
        return f"downsample {bvh_fps:.1f}→{target_fps:.1f}Hz (every {ratio:.3g}th BVH frame)"
    return f"upsample {bvh_fps:.1f}→{target_fps:.1f}Hz (each BVH frame repeated ~{1.0/ratio:.3g}×)"


def run(args: argparse.Namespace) -> None:
    parsed = parse_bvh(args.bvh)
    bvh_fps = 1.0 / parsed.frame_time
    target_fps = args.fps
    period = 1.0 / target_fps
    total_frames_to_send = args.frames if args.frames is not None else parsed.num_frames
    scale = args.scale if args.scale is not None else detect_unit_scale(parsed)

    client = SimpleUDPClient(args.host, args.port)
    print(
        f"Loaded {len(parsed.bones)} bones, {parsed.num_frames} frames @ "
        f"{bvh_fps:.1f} Hz → streaming as '{args.actor}' to "
        f"{args.host}:{args.port} @ {target_fps:.1f} Hz, chunk_size={args.chunk_size}"
    )
    unit_label = "cm→m" if abs(scale - 0.01) < 1e-6 else ("m→m" if abs(scale - 1.0) < 1e-6 else f"×{scale}")
    print(f"  resample: {_resample_strategy(bvh_fps, target_fps)}")
    print(f"  unit scale: {scale} ({unit_label}){' [auto]' if args.scale is None else ' [override]'}")

    frame_counter = 0
    wall_start = time.time()
    last_log = wall_start
    next_tick = wall_start

    try:
        while True:
            bvh_idx = int(frame_counter * bvh_fps / target_fps) % parsed.num_frames
            frame_bones = build_frame_bones(parsed, bvh_idx, scale)
            timestamp = f"{time.time():.6f}"
            send_frame(
                client=client,
                timestamp=timestamp,
                actor_name=args.actor,
                frame_idx=frame_counter,
                bones=frame_bones,
                chunk_size=args.chunk_size,
            )
            frame_counter += 1

            if not args.loop and frame_counter >= total_frames_to_send:
                break
            if args.frames is not None and frame_counter >= args.frames:
                break

            now = time.time()
            if now - last_log >= 1.0:
                elapsed = now - wall_start
                print(
                    f"  sent frame {frame_counter} (bvh idx {bvh_idx}) "
                    f"@ {frame_counter/elapsed:.1f} fps wall",
                    flush=True,
                )
                last_log = now

            next_tick += period
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.time()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")

    print(f"Done. Total frames sent: {frame_counter}")


def _self_test(bvh_path: str) -> None:
    parsed = parse_bvh(bvh_path)
    scale = detect_unit_scale(parsed)
    fb = build_frame_bones(parsed, 0, scale)
    assert len(fb) == len(parsed.bones), (len(fb), len(parsed.bones))
    root = fb[0]
    assert root.parent_index == -1, root.parent_index
    assert root.name == parsed.bones[0].name
    row0 = parsed.motion[0]
    expected_root_pos = np.array([-row0[0] * scale, row0[1] * scale, row0[2] * scale], dtype=np.float32)
    assert np.allclose(root.local_position, expected_root_pos, atol=1e-6), (
        root.local_position, expected_root_pos,
    )
    child = fb[1]
    off = parsed.bones[1].offset
    expected_child_pos = np.array([-off[0] * scale, off[1] * scale, off[2] * scale], dtype=np.float32)
    assert np.allclose(child.local_position, expected_child_pos, atol=1e-6), (
        child.local_position, expected_child_pos,
    )
    assert np.allclose(root.rest_rotation, _IDENTITY_QUAT)
    assert np.allclose(root.local_scale, _UNIT_SCALE)
    print(
        f"Self-test OK: {len(fb)} frame bones, rot_axes={parsed.bones[0].rot_axes}, "
        f"scale={scale}, root pos={root.local_position}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a BVH file as /MOVIN/Frame OSC packets.")
    default_bvh = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "motion_actor_new.bvh",
    )
    parser.add_argument("--bvh", default=default_bvh, help="Path to BVH file")
    parser.add_argument("--host", default="127.0.0.1", help="Receiver host")
    parser.add_argument("--port", type=int, default=11235, help="Receiver port")
    parser.add_argument("--actor", default="TestActor", help="Actor name sent in header")
    parser.add_argument("--fps", type=float, default=60.0,
                        help="Target send rate in Hz (default: 60). BVH frames are resampled to match: "
                             "120Hz source → every other frame; 30Hz source → each frame twice; etc.")
    parser.add_argument("--chunk-size", type=int, default=20, help="Bones per chunk")
    parser.add_argument("--scale", type=float, default=None,
                        help="Meters per BVH unit (default: auto-detect; 0.01 for cm, 1.0 for m)")
    parser.add_argument("--frames", type=int, default=None, help="Stop after N frames (default: unlimited)")
    parser.add_argument("--loop", dest="loop", action="store_true", help="Loop BVH when it ends (default)")
    parser.add_argument("--no-loop", dest="loop", action="store_false", help="Stop after one BVH pass")
    parser.set_defaults(loop=True)
    parser.add_argument("--self-test", action="store_true", help="Run internal sanity test and exit")
    args = parser.parse_args()

    if args.self_test:
        _self_test(args.bvh)
        return

    run(args)


if __name__ == "__main__":
    main()
