import numpy as np

CM_TO_M: float = 0.01


def bvh_pos_to_unity(
    pos_bvh: np.ndarray,
    is_root_translation: bool = False,
    scale: float = CM_TO_M,
) -> np.ndarray:
    """Convert a BVH-space position (right-handed Y-up) to Unity space
    (left-handed Y-up, meters) as the receiver expects.

      unity.x = -bvh.x * scale
      unity.y =  bvh.y * scale
      unity.z =  bvh.z * scale

    `scale` is the number of meters per BVH unit. Default CM_TO_M assumes
    centimeter input; pass `scale=1.0` for BVH files already in meters.
    is_root_translation is accepted for API clarity only."""
    p = np.asarray(pos_bvh, dtype=np.float32)
    s = float(scale)
    return np.array([-p[0] * s, p[1] * s, p[2] * s], dtype=np.float32)


def bvh_quat_to_unity(quat_bvh: np.ndarray) -> np.ndarray:
    """Convert a BVH-space quaternion [x,y,z,w] (right-handed) to Unity space
    [x,y,z,w] (left-handed).
      unity = (q.x, -q.y, -q.z, q.w)
    Returns float32."""
    q = np.asarray(quat_bvh, dtype=np.float32)
    return np.array([q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _unity_to_opengl_pos(p):
    return np.array([-p[0], p[1], p[2]], dtype=np.float32)


def _unity_to_opengl_rot(q):
    return np.array([q[0], -q[1], -q[2], q[3]], dtype=np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for _ in range(100):
        pos_bvh_cm = rng.uniform(-200.0, 200.0, size=3).astype(np.float32)
        raw = rng.standard_normal(4).astype(np.float32)
        q_bvh = raw / np.linalg.norm(raw)

        pos_unity = bvh_pos_to_unity(pos_bvh_cm)
        q_unity = bvh_quat_to_unity(q_bvh)

        pos_gl = _unity_to_opengl_pos(pos_unity)
        q_gl = _unity_to_opengl_rot(q_unity)

        expected_pos_gl = pos_bvh_cm * CM_TO_M
        assert np.allclose(pos_gl, expected_pos_gl, atol=1e-6), (pos_gl, expected_pos_gl)
        assert np.allclose(q_gl, q_bvh, atol=1e-6), (q_gl, q_bvh)

    p = bvh_pos_to_unity(np.array([100.0, 50.0, 25.0], dtype=np.float32))
    p = _unity_to_opengl_pos(p)
    assert np.allclose(p, [1.0, 0.5, 0.25], atol=1e-6), p

    q = bvh_quat_to_unity(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    q = _unity_to_opengl_rot(q)
    assert np.allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1e-6), q

    print("Self-test OK")
