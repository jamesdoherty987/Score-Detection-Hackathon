"""
stereo_calibrate.py — Stereo projection / triangulation helpers for score_live.

World origin is the midpoint of the crossbar on the goal plane.
  X → toward the RIGHT post
  Y → UP
  Z → INTO THE FIELD  (posts at Z=0; cameras at Z=−behind)

Use: python capture/score_live.py
"""

from __future__ import annotations

import math
import sys

import cv2
import numpy as np

import stereo_config as cfg

BALL_RADIUS_M = cfg.BALL_RADIUS_M


def make_proj_matrix(
    cam_pos_world: np.ndarray,
    h_deg: float,
    v_deg: float,
    focal_px: float,
    img_w: int,
    img_h: int,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """
    Build a 3×4 OpenCV projection matrix P = K @ [R | t] for a camera
    positioned at cam_pos_world and pointing inward/upward.

    h_deg: horizontal aim (positive = look toward +X; ~90° = across goal)
    v_deg: vertical upward angle (positive = look upward)
    """
    h = math.radians(h_deg)
    v = math.radians(v_deg)

    cam_z = np.array([math.sin(h) * math.cos(v), math.sin(v), math.cos(h) * math.cos(v)])

    world_up = np.array([0.0, 1.0, 0.0])
    cam_x = np.cross(world_up, cam_z)
    norm = np.linalg.norm(cam_x)
    if norm < 1e-6:
        cam_x = np.array([1.0, 0.0, 0.0])
    else:
        cam_x /= norm

    cam_y_down = np.cross(cam_x, cam_z)
    cam_y_down /= np.linalg.norm(cam_y_down)

    if abs(roll_deg) > 1e-6:
        r = math.radians(roll_deg)
        c, s = math.cos(r), math.sin(r)
        cam_x2 = c * cam_x + s * cam_y_down
        cam_y2 = -s * cam_x + c * cam_y_down
        cam_x = cam_x2 / np.linalg.norm(cam_x2)
        cam_y_down = cam_y2 / np.linalg.norm(cam_y2)

    R_c2w = np.column_stack([cam_x, cam_y_down, cam_z])
    R_w2c = R_c2w.T
    t = R_w2c @ (-cam_pos_world)

    fx_spec, fy_spec = cfg.focal_axes(img_w, img_h)
    scale = (focal_px / fx_spec) if fx_spec > 1e-6 else 1.0
    fx = fx_spec * scale
    fy = fy_spec * scale
    cx, cy = img_w / 2.0, img_h / 2.0
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    Rt = np.hstack([R_w2c, t.reshape(3, 1)])
    return K @ Rt


def triangulate_3d(
    P1: np.ndarray,
    P2: np.ndarray,
    pt1: tuple[float, float],
    pt2: tuple[float, float],
) -> np.ndarray | None:
    """Triangulate a 3D world point from two pixel observations."""
    p1 = np.array([[pt1[0]], [pt1[1]]], dtype=np.float64)
    p2 = np.array([[pt2[0]], [pt2[1]]], dtype=np.float64)
    pts4d = cv2.triangulatePoints(P1, P2, p1, p2)
    w = float(np.asarray(pts4d[3]).reshape(-1)[0])
    if abs(w) < 1e-8:
        return None
    return (pts4d[:3] / w).flatten()


def project_world(P: np.ndarray, xyz: np.ndarray) -> tuple[float, float] | None:
    """Project a 3D world point with a 3×4 projection matrix → pixel (u, v)."""
    hom = P @ np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
    if abs(hom[2]) < 1e-8:
        return None
    return float(hom[0] / hom[2]), float(hom[1] / hom[2])


def calib_centres_from_dets(
    det_l: tuple,
    det_r: tuple,
    pos_l: np.ndarray,
    pos_r: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], str, np.ndarray | None]:
    """
    Nudge YOLO box centres from the facing hemisphere toward the true ball centre.
    Returns (pt_l, pt_r, correction_summary, rough_pos).
    """
    face_offset = BALL_RADIUS_M * (4.0 / (3.0 * math.pi))

    rough = triangulate_3d(P1, P2, (det_l[0], det_l[1]), (det_r[0], det_r[1]))
    if rough is None:
        return (
            (float(det_l[0]), float(det_l[1])),
            (float(det_r[0]), float(det_r[1])),
            "no rough 3D — skipped SW/SE correction",
            None,
        )

    P = rough.copy()
    if float(P[1]) < 0.05:
        P[1] = max(float(P[1]), 0.3)

    def correct_one(
        det: tuple, cam: np.ndarray, Proj: np.ndarray, face_name: str,
    ) -> tuple[float, float, str]:
        to_ball = P - cam
        dist = float(np.linalg.norm(to_ball))
        if dist < 1e-3:
            return float(det[0]), float(det[1]), f"{face_name}: skip"
        direction = to_ball / dist
        face_pt = P - face_offset * direction
        pix_centre = project_world(Proj, P)
        pix_face = project_world(Proj, face_pt)
        if pix_centre is None or pix_face is None:
            return float(det[0]), float(det[1]), f"{face_name}: project fail"
        du = pix_centre[0] - pix_face[0]
        dv = pix_centre[1] - pix_face[1]
        return float(det[0]) + du, float(det[1]) + dv, f"{face_name}: Δu={du:+.1f}px Δv={dv:+.1f}px"

    u_l, v_l, msg_l = correct_one(det_l, pos_l, P1, "L=SW")
    u_r, v_r, msg_r = correct_one(det_r, pos_r, P2, "R=SE")
    return (u_l, v_l), (u_r, v_r), f"{msg_l}; {msg_r}", rough


if __name__ == "__main__":
    print(
        "stereo_calibrate.py is a library now.\n"
        "Run:  python capture/score_live.py",
        file=sys.stderr,
    )
    raise SystemExit(2)
