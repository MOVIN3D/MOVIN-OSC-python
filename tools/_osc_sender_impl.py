from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class FrameBone:
    bone_index: int
    parent_index: int
    name: str
    local_position: np.ndarray   # (3,) float
    rest_rotation: np.ndarray    # (4,) float, [x,y,z,w]
    local_rotation: np.ndarray   # (4,) float, [x,y,z,w]
    local_scale: np.ndarray      # (3,) float


def bone_to_osc_fields(bone: FrameBone) -> list:
    """Flatten one FrameBone into the 17-field list expected by /MOVIN/Frame."""
    p = bone.local_position
    rq = bone.rest_rotation
    q = bone.local_rotation
    s = bone.local_scale
    return [
        int(bone.bone_index),
        int(bone.parent_index),
        str(bone.name),
        float(p[0]), float(p[1]), float(p[2]),
        float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3]),
        float(q[0]), float(q[1]), float(q[2]), float(q[3]),
        float(s[0]), float(s[1]), float(s[2]),
    ]


def build_chunk_args(
    timestamp: str,
    actor_name: str,
    frame_idx: int,
    num_chunks: int,
    chunk_index: int,
    total_bone_count: int,
    chunk_bones: List[FrameBone],
) -> list:
    """Build the full ordered argument list for one /MOVIN/Frame OSC message."""
    args = [
        str(timestamp),
        str(actor_name),
        int(frame_idx),
        int(num_chunks),
        int(chunk_index),
        int(total_bone_count),
        int(len(chunk_bones)),
    ]
    for bone in chunk_bones:
        args.extend(bone_to_osc_fields(bone))
    return args


def send_frame(
    client,
    timestamp: str,
    actor_name: str,
    frame_idx: int,
    bones: List[FrameBone],
    chunk_size: int,
) -> int:
    """Split bones into chunks of size `chunk_size`, send each as a /MOVIN/Frame packet."""
    from pythonosc.osc_message_builder import OscMessageBuilder

    total = len(bones)
    chunks = [bones[i:i + chunk_size] for i in range(0, total, chunk_size)]
    # Edge case: empty bones list produces one empty chunk to satisfy callers expecting >=1 send.
    if not chunks:
        chunks = [[]]
    num_chunks = len(chunks)

    for chunk_index, chunk_bones in enumerate(chunks):
        args = build_chunk_args(
            timestamp, actor_name, frame_idx,
            num_chunks, chunk_index, total, chunk_bones,
        )
        builder = OscMessageBuilder(address="/MOVIN/Frame")
        for arg in args:
            if isinstance(arg, int):
                builder.add_arg(arg, "i")
            elif isinstance(arg, float):
                builder.add_arg(arg, "f")
            else:
                builder.add_arg(str(arg), "s")
        client.send(builder.build())

    return num_chunks


if __name__ == "__main__":
    import threading
    import time
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import ThreadingOSCUDPServer
    from pythonosc.udp_client import SimpleUDPClient

    received = []

    def handler(address, *args):
        received.append((address, args))

    disp = Dispatcher()
    disp.map("/MOVIN/Frame", handler)
    server = ThreadingOSCUDPServer(("127.0.0.1", 0), disp)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    client = SimpleUDPClient("127.0.0.1", port)
    bones = [
        FrameBone(
            bone_index=i,
            parent_index=i - 1,
            name=f"Bone{i}",
            local_position=np.array([float(i), 0.0, 0.0], dtype=np.float32),
            rest_rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            local_rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            local_scale=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        ) for i in range(5)
    ]
    n_sent = send_frame(client, "t0", "TestActor", 42, bones, chunk_size=3)
    assert n_sent == 2, n_sent

    time.sleep(0.2)
    server.shutdown()
    server.server_close()

    assert len(received) == 2, received
    addr, args = received[0]
    assert addr == "/MOVIN/Frame"
    assert args[0] == "t0"
    assert args[1] == "TestActor"
    assert args[2] == 42
    assert args[3] == 2             # num_chunks
    assert args[4] == 0             # chunk_index
    assert args[5] == 5             # total_bone_count
    assert args[6] == 3             # chunk_bone_count (first chunk has 3)
    assert args[7] == 0             # bone_index
    assert args[8] == -1            # parent_index
    assert args[9] == "Bone0"       # bone_name
    _, args2 = received[1]
    assert args2[6] == 2

    # Print example of build_chunk_args output for first chunk
    example = build_chunk_args("t0", "TestActor", 42, 2, 0, 5, bones[:3])
    print("build_chunk_args example (first chunk):", example)
    print("Self-test OK (sent", n_sent, "chunks, received", len(received), "messages)")
