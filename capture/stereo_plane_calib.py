"""
stereo_plane_calib.py — Shared helpers for score_live (plane, cams, ball).

Draws the goal plane, opens dual UVC cameras, aligns far-post clicks, and
runs YOLO + stereo triangulation for GOAL / WIDE calls.

Not a standalone app — run: python capture/score_live.py
"""

from __future__ import annotations

import json
import math
import platform
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from auto_exposure import try_get, try_set
import stereo_config as cfg
from stereo_calibrate import (
    project_world,
    triangulate_3d,
    calib_centres_from_dets,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLANE_PATH = PROJECT_ROOT / "exports" / "stereo" / "plane.json"

COCO_MODEL = "yolov8n.pt"
SPORTS_BALL_CLASS = 32
IMAGE_SIZE = 256              # small default (Pi-friendly); override with --imgsz
CONFIDENCE = 0.30

PLANE_CROSS_M = 0.05
POST_MARGIN_M = 0.02          # extra width beyond ±baseline/2 still called "between"
REPROJ_MAX_PX = 40.0          # reject 3D if it doesn't land on both detections
CALL_LATCH_S = 1.2
NEAR_PLANE_Z_M = 1.25         # |Z| below this → YOLO every NEAR_STRIDE frames
FAST_SPEED_M_S = 4.0          # 3D speed above this → treat like near-plane
IDLE_YOLO_STRIDE = 6          # deep-field / coasting: run every N frames
SEARCH_STRIDE = 3             # hunting after motion, before a 3D track exists
NEAR_STRIDE = 1               # at the posts: keep every frame for score calls
ARM_AFTER_MOTION = 12         # keep detector hot this many frames after motion
STALE_DET_FRAMES = 18         # drop held boxes if YOLO hasn't run this long
YOLO_DEVICE = "cpu"           # Pi has no CUDA; PCs can pass --device 0
COL_GRID = (80, 200, 80)
COL_POST = (0, 255, 255)
COL_BAR = (0, 180, 255)
COL_CLICK_FAR = (0, 255, 0)
COL_BALL = (0, 255, 255)
COL_STATUS = (220, 220, 220)
COL_DEBUG = (180, 255, 255)
COL_CALL_BLUE = (255, 90, 20)     # BGR — in field, not yet through
COL_CALL_GREEN = (0, 255, 80)     # through + between posts
COL_CALL_RED = (0, 0, 255)        # through + wide

def dist3(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def put_label(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color=COL_STATUS,
    scale: float = 0.42,
    thick: int = 1,
) -> None:
    x, y = org
    cv2.putText(
        img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
        scale, (0, 0, 0), thick + 2, cv2.LINE_AA,
    )
    cv2.putText(
        img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thick, cv2.LINE_AA,
    )

# ─── Camera helpers (same pattern as other stereo scripts) ───────────────────

# Dual UVC cams (e.g. Arducam OV9281) need MJPEG at modest res or V4L2 times out.
DEFAULT_CAM_W, DEFAULT_CAM_H = 640, 480
DEFAULT_CAM_FPS = 30.0

def _fourcc_mjpg() -> int:
    # Must stay int — float() corrupts FOURCC values above 2^24.
    return int(cv2.VideoWriter_fourcc(*"MJPG"))

def _fourcc_to_str(code: float | int) -> str:
    n = int(code)
    if n <= 0:
        return "?"
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))

def _is_capture_symlink_name(name: str) -> bool:
    """UVC capture nodes are usually *-video-index0; index1+ are often metadata."""
    lower = name.lower()
    if "index0" in lower:
        return True
    if "index" in lower:
        return False
    return True

def list_linux_camera_links() -> list[dict[str, str]]:
    """Stable udev links under /dev/v4l/by-id and by-path (Linux/Pi)."""
    rows: list[dict[str, str]] = []
    if platform.system() != "Linux":
        return rows
    for kind in ("by-id", "by-path"):
        root = Path("/dev/v4l") / kind
        if not root.is_dir():
            continue
        for link in sorted(root.iterdir()):
            try:
                target = str(link.resolve())
            except OSError:
                continue
            rows.append(
                {
                    "kind": kind,
                    "name": link.name,
                    "link": str(link),
                    "device": target,
                    "role": "capture" if _is_capture_symlink_name(link.name) else "metadata?",
                }
            )
    return rows

def resolve_camera_source(spec: str | int) -> tuple[str, object, int]:
    """
    Resolve a camera spec to (label, OpenCV source, backend).

    Accepts:
      0, "0"                         → /dev/video0 (Linux) or index
      /dev/video2                     → that node
      /dev/v4l/by-id/...              → stable serial/name link
      /dev/v4l/by-path/...            → stable USB-port link (best for identical cams)
      substring of a by-id/by-path name (must match exactly one capture link)
    """
    raw = str(spec).strip()
    system = platform.system()

    if system != "Linux":
        if raw.isdigit():
            idx = int(raw)
            if system == "Darwin":
                return (f"AVFoundation {idx}", idx, cv2.CAP_AVFOUNDATION)
            return (f"MSMF {idx}", idx, cv2.CAP_MSMF)
        raise RuntimeError(f"Camera path specs are Linux-only; got {raw!r}")

    # Explicit device node or udev link.
    if raw.startswith("/dev/"):
        path = Path(raw)
        if not path.exists():
            raise RuntimeError(f"Camera device not found: {raw}")
        return (f"V4L2 {raw}", str(path), cv2.CAP_V4L2)

    # Numeric index → prefer /dev/videoN path (capture node you asked for).
    if raw.isdigit():
        idx = int(raw)
        dev = f"/dev/video{idx}"
        if Path(dev).exists():
            return (f"V4L2 {dev}", dev, cv2.CAP_V4L2)
        return (f"V4L2 index {idx}", idx, cv2.CAP_V4L2)

    # Substring match against stable links (capture nodes only).
    links = [
        r for r in list_linux_camera_links()
        if r["role"] == "capture" and raw.lower() in r["name"].lower()
    ]
    # Prefer by-id when both match.
    links.sort(key=lambda r: (0 if r["kind"] == "by-id" else 1, r["name"]))
    if not links:
        raise RuntimeError(
            f"No capture camera matching {raw!r}. Run with --list to see by-id / by-path names."
        )
    if len(links) > 1:
        opts = ", ".join(r["link"] for r in links[:6])
        raise RuntimeError(
            f"Camera spec {raw!r} matches {len(links)} devices. Use a fuller path. Matches: {opts}"
        )
    link = links[0]["link"]
    return (f"V4L2 {link}", link, cv2.CAP_V4L2)

def configure_capture(
    cap: cv2.VideoCapture,
    width: int = DEFAULT_CAM_W,
    height: int = DEFAULT_CAM_H,
    fps: float = DEFAULT_CAM_FPS,
) -> None:
    """Force MJPEG + resolution before streaming (critical for dual USB on Pi)."""
    mjpg = _fourcc_mjpg()
    # FourCC before size — some UVC drivers ignore MJPG if set after.
    try_set(cap, cv2.CAP_PROP_FOURCC, mjpg)
    try_set(cap, cv2.CAP_PROP_FRAME_WIDTH, float(width))
    try_set(cap, cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    try_set(cap, cv2.CAP_PROP_FPS, float(fps))
    # Keep a few buffered frames; LatestFrameCam drains to the newest.
    try_set(cap, cv2.CAP_PROP_BUFFERSIZE, 3.0)
    # Re-assert after geometry — drivers often reset FOURCC on size change.
    try_set(cap, cv2.CAP_PROP_FOURCC, mjpg)

def open_camera(
    spec: str | int,
    width: int | None = None,
    height: int | None = None,
    quiet: bool = False,
) -> cv2.VideoCapture:
    w = DEFAULT_CAM_W if width is None else int(width)
    h = DEFAULT_CAM_H if height is None else int(height)
    name, source, backend = resolve_camera_source(spec)
    if not quiet:
        print(f"  Opening {name} (MJPG {w}×{h})...")
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        raise RuntimeError(
            f"Could not open {name}. "
            "Pass the real capture node (e.g. /dev/video4, /dev/video1) or a by-id/by-path "
            "*-video-index0 link. Run --list to see devices."
        )

    configure_capture(cap, width=w, height=h)

    ok, frame = False, None
    for attempt in range(40):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0 and frame.any():
            break
        if not quiet and attempt in (0, 9, 19, 39):
            print(f"    … waiting for first frame ({attempt + 1}/40)")
        time.sleep(0.025)

    if not (ok and frame is not None and getattr(frame, "size", 0) > 0 and frame.any()):
        try:
            cap.release()
        except Exception:
            pass
        raise RuntimeError(f"{name}: opened but no frames at MJPG {w}×{h}.")

    if not quiet:
        fh, fw = frame.shape[:2]
        fourcc_s = _fourcc_to_str(try_get(cap, cv2.CAP_PROP_FOURCC, 0.0))
        print(f"  OK — {fw}×{fh} fourcc={fourcc_s!r}  (single-cam read OK)")
    return cap


def list_working_cameras(
    width: int | None = None,
    height: int | None = None,
) -> list[tuple[str, str]]:
    """
    Return [(device, hint), ...] for capture nodes (no open — avoids V4L2 hangs).

    Prefer by-path. If by-id names look like Arducam/OV9281, keep only those
    (drops a random laptop/Pi webcam when three nodes show up).
    """
    _ = (width, height)
    # device -> best hint
    info: dict[str, str] = {}

    def _hint(r: dict[str, str]) -> str:
        name = r.get("name") or ""
        # by-path: ...-usb-0:1.2:1.0-video-index0 → usb-0:1.2
        if r.get("kind") == "by-path" and "usb-" in name:
            part = name.split("usb-", 1)[-1]
            part = part.split("-video", 1)[0]
            return f"usb-{part}"
        if r.get("kind") == "by-id":
            return name[:48]
        return r.get("kind") or ""

    rows = list_linux_camera_links()
    for kind in ("by-path", "by-id"):
        for r in rows:
            if r["kind"] != kind or r["role"] != "capture":
                continue
            device = r.get("device") or ""
            if not device.startswith("/dev/video") or not Path(device).exists():
                continue
            if device not in info or kind == "by-path":
                info[device] = _hint(r)

    if not info:
        meta = {
            r["device"]
            for r in rows
            if r["role"] != "capture" and r.get("device", "").startswith("/dev/video")
        }
        for i in range(16):
            dev = f"/dev/video{i}"
            if Path(dev).exists() and dev not in meta:
                info[dev] = ""

    items = sorted(info.items(), key=lambda kv: kv[0])

    # If we can spot the stereo pair by name, drop other webcams.
    stereoish = [
        (d, h) for d, h in items
        if any(tok in (h + d).lower() for tok in ("arducam", "ov9281", "b0332", "ln013"))
    ]
    if len(stereoish) >= 2:
        return stereoish

    return items

def read_cam(
    cap: cv2.VideoCapture,
    retries: int = 5,
    flush: int = 0,
) -> tuple[bool, np.ndarray | None]:
    """Read one frame; optional single flush of one stale buffer slot."""
    if flush > 0:
        cap.grab()
    for _ in range(max(1, int(retries))):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return True, frame
        time.sleep(0.01)
    return False, None

def probe_dual_stream(
    cap_l: cv2.VideoCapture,
    cap_r: cv2.VideoCapture,
    pairs: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Alternate reads on both cams before YOLO / background threads.
    Raises RuntimeError if simultaneous streaming fails.
    """
    print(f"  Dual-stream probe ({pairs} alternating reads)...")
    ok_l = ok_r = 0
    last_l = last_r = None
    t0 = time.time()
    for i in range(pairs):
        l_ok, frame_l = read_cam(cap_l, retries=3, flush=0)
        r_ok, frame_r = read_cam(cap_r, retries=3, flush=0)
        if l_ok and frame_l is not None:
            ok_l += 1
            last_l = frame_l
        if r_ok and frame_r is not None:
            ok_r += 1
            last_r = frame_r
        if i % 10 == 9:
            print(f"    pair {i + 1}/{pairs}: L={ok_l} R={ok_r} OK")
        time.sleep(0.005)
    dt = max(time.time() - t0, 1e-6)
    print(
        f"  Dual-stream result: L {ok_l}/{pairs}  R {ok_r}/{pairs}  "
        f"in {dt:.2f}s (~{pairs / dt:.1f} pair/s)"
    )
    if last_l is None or last_r is None or ok_l < pairs // 2 or ok_r < pairs // 2:
        raise RuntimeError(
            "Simultaneous OpenCV streaming failed (select timeout / USB stall). "
            "Both devices work alone with v4l2-ctl, so use MJPG 640x480, open BOTH "
            "devices before any background reader, and prefer USB2/3 ports on different hubs."
        )
    return last_l, last_r

class LatestFrameCam:
    """
    Gentle background reader — one read() per tick, no grab-storm.
    Aggressive grab()/retrieve loops cause V4L2 select() timeouts with dual UVC.
    """

    def __init__(self, cap: cv2.VideoCapture, name: str = "cam") -> None:
        self.cap = cap
        self.name = name
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ok_reads = 0
        self._fail_reads = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name=f"cam-latest-{name}", daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None and getattr(frame, "size", 0) > 0:
                with self._lock:
                    self._frame = frame
                    self._ok_reads += 1
                # ~25 Hz max — leaves USB bandwidth for the sibling camera.
                time.sleep(0.04)
            else:
                self._fail_reads += 1
                time.sleep(0.02)

    def read(self, retries: int = 5) -> tuple[bool, np.ndarray | None]:
        for _ in range(max(1, int(retries))):
            with self._lock:
                if self._frame is not None:
                    return True, self._frame.copy()
            time.sleep(0.02)
        return False, None

    def stats(self) -> str:
        return f"{self.name}: ok={self._ok_reads} fail={self._fail_reads}"

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.5)
        try:
            self.cap.release()
        except Exception:
            pass

def normalize_orient(rotate_deg: int, flip_ud: bool = False) -> tuple[int, bool]:
    """Canonical orient: rotate in {0, 90, 270}, flip_ud bool.

    Legacy plane.json with rotate_deg=180 becomes flip_ud=True.
    """
    rd = int(rotate_deg)
    fu = bool(flip_ud)
    if rd == 180:
        fu = True
        rd = int(cfg.SIDEWAYS_ROTATE_DEG) if cfg.CAMERAS_SIDEWAYS else 0
    elif rd not in (0, 90, 270):
        rd = int(cfg.SIDEWAYS_ROTATE_DEG) if cfg.CAMERAS_SIDEWAYS else 0
    return rd, fu

def orient_from_dict(
    data: dict,
    rotate_deg: int,
    flip_ud: bool,
    *,
    apply_rotate: bool = True,
    apply_flip: bool = True,
) -> tuple[int, bool]:
    """Merge saved plane.json orient into current values (CLI can skip either)."""
    rd, fu = rotate_deg, flip_ud
    if apply_rotate and "rotate_deg" in data:
        rd = int(data["rotate_deg"])
    if apply_flip and "flip_ud" in data:
        fu = bool(data["flip_ud"])
    elif apply_flip and apply_rotate and int(data.get("rotate_deg", rd)) == 180:
        fu = True
    return normalize_orient(rd, fu)

def orient_frame(
    frame: np.ndarray,
    rotate_deg: int,
    flip_ud: bool = False,
) -> np.ndarray:
    """Apply sideways rotate (0/90/270), then optional upside-down 180°."""
    rd, fu = normalize_orient(rotate_deg, flip_ud)
    out = frame
    if rd == 270:
        # Sideways helper: skip if the driver already delivers portrait.
        if not (cfg.CAMERAS_SIDEWAYS and out.shape[0] >= out.shape[1]):
            out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rd == 90:
        if not (cfg.CAMERAS_SIDEWAYS and out.shape[0] >= out.shape[1]):
            out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    if fu:
        out = cv2.rotate(out, cv2.ROTATE_180)
    return out

def rotate_label(deg: int, flip_ud: bool = False) -> str:
    rd, fu = normalize_orient(deg, flip_ud)
    if rd == 0:
        base = "rotate=off"
    elif rd == 270:
        base = "rotate=270 CCW"
    elif rd == 90:
        base = "rotate=90 CW"
    else:
        base = f"rotate={rd}"
    return f"{base} flip={'ON' if fu else 'off'}"

def draw_camera_hud(
    img: np.ndarray,
    which_cam: str,
    show_north: bool = True,
    flip_ud: bool = False,
) -> None:
    """L/R badge; optional N at TOP (sky) when aligning / checking orient."""
    col = COL_POST if which_cam == "L" else COL_BAR
    put_label(img, "L" if which_cam == "L" else "R", (8, 26), col, 0.75, 2)
    h, w = img.shape[:2]
    if not show_north:
        return
    cx = w // 2
    cv2.arrowedLine(img, (cx, 36), (cx, 10), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
    put_label(img, "N", (cx - 10, 58), (0, 255, 255), 0.7, 2)
    put_label(img, "up/sky", (cx - 28, 76), (180, 255, 255), 0.4, 1)
    put_label(
        img,
        f"flip={'ON' if flip_ud else 'off'}",
        (8, h - 14),
        (0, 0, 255) if h < w else (0, 200, 255),
        0.45,
        2,
    )

# ─── Goal plane geometry ─────────────────────────────────────────────────────

def apply_affine(
    uv: tuple[float, float],
    affine: np.ndarray | None,
) -> tuple[float, float]:
    if affine is None:
        return uv
    x, y = uv
    return (
        float(affine[0, 0] * x + affine[0, 1] * y + affine[0, 2]),
        float(affine[1, 0] * x + affine[1, 1] * y + affine[1, 2]),
    )

def project_safe(
    P: np.ndarray,
    xyz: np.ndarray,
    affine: np.ndarray | None = None,
) -> tuple[int, int] | None:
    uv = project_world(P, xyz)
    if uv is None:
        return None
    u, v = uv
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    # Reject points behind the camera (negative depth in camera frame ≈ hom[2] sign
    # already handled in project_world via division; still clamp absurd values)
    if abs(u) > 1e5 or abs(v) > 1e5:
        return None
    u, v = apply_affine((u, v), affine)
    if not (math.isfinite(u) and math.isfinite(v)) or abs(u) > 1e5 or abs(v) > 1e5:
        return None
    return int(round(u)), int(round(v))

def draw_line_world(
    img: np.ndarray,
    P: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    n_seg: int = 24,
    affine: np.ndarray | None = None,
) -> None:
    """Draw a 3D segment by sampling so perspective looks correct."""
    pts: list[tuple[int, int]] = []
    for i in range(n_seg + 1):
        t = i / n_seg
        xyz = a * (1.0 - t) + b * t
        uv = project_safe(P, xyz, affine)
        if uv is not None:
            pts.append(uv)
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)

def draw_goal_plane(
    img: np.ndarray,
    P: np.ndarray,
    baseline_m: float,
    extend_up_m: float,
    pos_l: np.ndarray,
    pos_r: np.ndarray,
    which_cam: str,
    grid_nx: int = 6,
    grid_ny: int = 8,
    y_vis_min: float = 0.0,
    affine: np.ndarray | None = None,
    compact: bool = False,
) -> None:
    """Draw posts, orange crossbar at Y=0, and grid. Optional 2D snap affine."""
    _ = (pos_l, pos_r, which_cam)  # kept for call-site compatibility
    half = baseline_m / 2.0
    y1 = max(0.4, float(extend_up_m))
    if compact:
        grid_nx, grid_ny = 3, 4

    # Orange = real crossbar (Y = 0). Yellow posts run from the bar upward.
    draw_line_world(
        img, P, np.array([-half, 0.0, 0.0]), np.array([half, 0.0, 0.0]),
        COL_BAR, 3 if not compact else 2, affine=affine,
    )
    draw_line_world(
        img, P, np.array([-half, 0.0, 0.0]), np.array([-half, y1, 0.0]),
        COL_POST, 3, affine=affine,
    )
    draw_line_world(
        img, P, np.array([half, 0.0, 0.0]), np.array([half, y1, 0.0]),
        COL_POST, 3, affine=affine,
    )
    if not compact:
        draw_line_world(
            img, P, np.array([-half, y1, 0.0]), np.array([half, y1, 0.0]),
            COL_GRID, 1, affine=affine,
        )

    for i in range(grid_nx + 1):
        x = -half + (baseline_m * i / grid_nx)
        a = np.array([x, 0.0, 0.0])
        b = np.array([x, y1, 0.0])
        draw_line_world(img, P, a, b, COL_GRID, 1, n_seg=12 if compact else 16, affine=affine)

    for j in range(1, grid_ny + 1):
        y = y1 * j / grid_ny
        a = np.array([-half, y, 0.0])
        b = np.array([half, y, 0.0])
        draw_line_world(img, P, a, b, COL_GRID, 1, n_seg=12 if compact else 16, affine=affine)

    if compact:
        return

    y_note = max(0.0, float(y_vis_min))
    uv = project_safe(P, np.array([0.0, y_note, 0.0]), affine)
    if uv is not None and y_note > 0.05:
        put_label(img, f"FOV from {y_note:.2f}m above bar", (uv[0] + 6, uv[1] + 16), COL_DEBUG, 0.4, 1)

def fit_height(img: np.ndarray, target_h: int) -> np.ndarray:
    if target_h <= 0 or img.shape[0] == target_h:
        return img
    scale = target_h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * scale)), target_h), interpolation=cv2.INTER_AREA)

def side_by_side_full(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, int]:
    """Stack both views at a common height. Uniform scale, no crop."""
    h = max(left.shape[0], right.shape[0])
    fl = fit_height(left, h)
    fr = fit_height(right, h)
    return np.hstack([fl, fr]), fl.shape[1]

def attach_status_bar(
    img: np.ndarray,
    lines: list[str],
    accent_color: tuple[int, int, int] | None = None,
    accent_line: str = "",
) -> tuple[np.ndarray, int]:
    """Append a status strip under the image so UI never covers the picture."""
    bar_h = 20 * max(len(lines), 1) + 12
    bar = np.zeros((bar_h, img.shape[1], 3), dtype=img.dtype)
    for i, line in enumerate(lines):
        col = accent_color if (accent_line and line == accent_line) else COL_STATUS
        cv2.putText(
            bar, line, (10, 18 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA,
        )
    return np.vstack([img, bar]), bar_h

def letterbox(img: np.ndarray, win_w: int, win_h: int) -> tuple[np.ndarray, int, int, int, int]:
    """
    Fit the whole image into the window with black bars. No crop, no stretch.
    Returns (canvas, x0, y0, content_w, content_h).
    """
    ih, iw = img.shape[:2]
    if win_w <= 0 or win_h <= 0 or iw <= 0 or ih <= 0:
        return img, 0, 0, iw, ih
    scale = min(win_w / iw, win_h / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    if (nw, nh) == (iw, ih):
        resized = img
    else:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.zeros((win_h, win_w, 3), dtype=img.dtype)
    x0 = (win_w - nw) // 2
    y0 = (win_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, x0, y0, nw, nh

# ─── Click state ─────────────────────────────────────────────────────────────

class ClickState:
    def __init__(self) -> None:
        self.far_l: list[tuple[int, int]] = []
        self.far_r: list[tuple[int, int]] = []

    def add(self, side: str, xy: tuple[int, int]) -> int:
        bucket = self.far_l if side == "L" else self.far_r
        bucket.append(xy)
        if len(bucket) > 2:
            del bucket[:-2]
        tag = "BOTTOM of post" if len(bucket) == 1 else "TOP of post"
        extra = (
            "  → now click the TOP of the same post"
            if len(bucket) == 1
            else "  → snapping yellow post"
        )
        print(f"  {side} {tag}: {xy}{extra}")
        return len(bucket)

    def clear(self) -> None:
        self.far_l.clear()
        self.far_r.clear()
        print("  Cleared all click marks")

    def draw_on(self, img_l: np.ndarray, img_r: np.ndarray) -> None:
        def _draw(img, pts):
            for i, (x, y) in enumerate(pts):
                name = ("BOT", "TOP")[i] if i < 2 else f"{i+1}"
                cv2.circle(img, (x, y), 7, COL_CLICK_FAR, -1, cv2.LINE_AA)
                put_label(img, name, (x + 8, y - 8), COL_CLICK_FAR, 0.5, 1)
            if len(pts) >= 2:
                cv2.line(img, pts[0], pts[1], COL_CLICK_FAR, 2, cv2.LINE_AA)
                cv2.arrowedLine(
                    img, pts[0], pts[1], COL_CLICK_FAR, 2, cv2.LINE_AA, tipLength=0.12,
                )

        _draw(img_l, self.far_l)
        _draw(img_r, self.far_r)

    def to_dict(self) -> dict:
        return {"far_left": self.far_l, "far_right": self.far_r}

    def load_dict(self, data: dict) -> None:
        self.far_l = [(int(p[0]), int(p[1])) for p in data.get("far_left", [])]
        self.far_r = [(int(p[0]), int(p[1])) for p in data.get("far_right", [])]

# ─── Ball (YOLO + triangulation vs the Z=0 plane) ────────────────────────────

def default_ball_model() -> str:
    custom = PROJECT_ROOT / "models" / "football_yolov8n.pt"
    return str(custom) if custom.exists() else COCO_MODEL

def load_ball_model(model_str: str) -> tuple[YOLO, list[int] | None]:
    path = Path(model_str)
    if not path.is_absolute():
        full = PROJECT_ROOT / path
        if full.exists():
            path = full
    is_coco = path.name == COCO_MODEL
    print(f"Loading {'COCO sports-ball' if is_coco else 'custom'} model: {path}")
    model = YOLO(str(path))
    try:
        model.fuse()
    except Exception:
        pass
    return model, ([SPORTS_BALL_CLASS] if is_coco else None)

def detect_both(
    model: YOLO,
    frame_l,
    frame_r,
    conf: float,
    classes: list[int] | None,
    imgsz: int = IMAGE_SIZE,
    device: str = YOLO_DEVICE,
):
    kw: dict = dict(
        conf=conf,
        imgsz=int(imgsz),
        verbose=False,
        device=device,
        max_det=1,
    )
    if classes:
        kw["classes"] = classes
    results = model.predict([frame_l, frame_r], **kw)

    def best(res):
        if res.boxes is None or len(res.boxes) == 0:
            return None
        boxes = res.boxes
        i = int(boxes.conf.argmax())
        xyxy = boxes.xyxy[i].cpu().numpy()
        c = float(boxes.conf[i])
        cx = 0.5 * (xyxy[0] + xyxy[2])
        cy = 0.5 * (xyxy[1] + xyxy[3])
        return cx, cy, c, xyxy

    return best(results[0]), best(results[1])

class BallDetectScheduler:
    """
    Motion-gated YOLO with adaptive stride (keeps YOLO rare).

    - Idle: skip entirely.
    - New motion / search (no 3D yet): first frame immediately, then SEARCH_STRIDE.
    - Near plane or fast: every NEAR_STRIDE frames (default 1 — needed for calls).
    - Coasting deep in field: every idle_stride frames; reuse last boxes.
    """

    def __init__(
        self,
        idle_stride: int = IDLE_YOLO_STRIDE,
        search_stride: int = SEARCH_STRIDE,
        near_stride: int = NEAR_STRIDE,
        stale_frames: int = STALE_DET_FRAMES,
    ) -> None:
        self.idle_stride = max(1, int(idle_stride))
        self.search_stride = max(1, int(search_stride))
        self.near_stride = max(1, int(near_stride))
        self.stale_frames = max(1, int(stale_frames))
        self.frame_i = 0
        self.since_yolo = 10**9
        self.last_xyz: np.ndarray | None = None
        self.last_xyz_t: float | None = None
        self.armed_until = 0
        self.last_mode = "idle"
        self._last_speed = 0.0

    def reset(self) -> None:
        self.frame_i = 0
        self.since_yolo = 10**9
        self.last_xyz = None
        self.last_xyz_t = None
        self.armed_until = 0
        self.last_mode = "idle"
        self._last_speed = 0.0

    def should_run(self, has_motion: bool, _now: float = 0.0) -> tuple[bool, str]:
        self.frame_i += 1
        self.since_yolo += 1

        was_idle = self.frame_i > self.armed_until
        if has_motion:
            self.armed_until = self.frame_i + ARM_AFTER_MOTION

        near = False
        fast = self._last_speed >= FAST_SPEED_M_S
        if self.last_xyz is not None:
            near = abs(float(self.last_xyz[2])) <= NEAR_PLANE_Z_M

        # Score zone — still frequent so GOAL/WIDE calls stay reliable.
        if near or fast:
            self.armed_until = max(self.armed_until, self.frame_i + ARM_AFTER_MOTION)
            self.last_mode = "near" if near else "fast"
            return self.since_yolo >= self.near_stride, self.last_mode

        # Have a track but deep / coasting — rare YOLO.
        if self.last_xyz is not None and self.frame_i <= self.armed_until:
            self.last_mode = "coast"
            return self.since_yolo >= self.idle_stride, self.last_mode

        # Motion wake or brief search with no 3D yet.
        if has_motion or self.frame_i <= self.armed_until:
            self.last_mode = "motion" if has_motion else "search"
            if was_idle and has_motion:
                return True, self.last_mode
            return self.since_yolo >= self.search_stride, self.last_mode

        self.last_mode = "idle"
        return False, self.last_mode

    def note_yolo(self, _dets: tuple, _now: float) -> None:
        self.since_yolo = 0

    def note_track(self, xyz: np.ndarray | None, now: float) -> None:
        if xyz is None:
            if self.since_yolo >= self.stale_frames:
                self.last_xyz = None
                self.last_xyz_t = None
                self._last_speed = 0.0
            return
        if self.last_xyz is not None and self.last_xyz_t is not None:
            dt = max(now - self.last_xyz_t, 1e-3)
            self._last_speed = float(np.linalg.norm(xyz - self.last_xyz) / dt)
        else:
            self._last_speed = 0.0
        self.last_xyz = np.asarray(xyz, dtype=np.float64).reshape(3).copy()
        self.last_xyz_t = now
        self.armed_until = max(self.armed_until, self.frame_i + ARM_AFTER_MOTION)

def estimate_ball_xyz(
    det_l,
    det_r,
    P1: np.ndarray,
    P2: np.ndarray,
    pos_l: np.ndarray,
    pos_r: np.ndarray,
) -> np.ndarray | None:
    if det_l is None or det_r is None:
        return None
    pt_l, pt_r, _c, _r = calib_centres_from_dets(det_l, det_r, pos_l, pos_r, P1, P2)
    xyz = triangulate_3d(P1, P2, pt_l, pt_r)
    if xyz is None:
        return None
    if reproj_err_px(P1, xyz, pt_l) > REPROJ_MAX_PX:
        return None
    if reproj_err_px(P2, xyz, pt_r) > REPROJ_MAX_PX:
        return None
    return xyz

def plane_side_label(z: float, eps: float = PLANE_CROSS_M) -> str:
    if z > eps:
        return "field"
    if z < -eps:
        return "through"
    return "on-plane"

def overlay_snap_to_clicks(
    P: np.ndarray,
    pts: list[tuple[int, int]],
    post_x: float,
    y0: float,
    y1: float,
) -> tuple[np.ndarray | None, float, str]:
    """
    Keep the default 3D plane. Slide/rotate the whole overlay in the image
    so the yellow far post lies on BOT→TOP. Orange bar and grid move with it.
    Returns (2x3 affine, mean pixel shift, note).
    """
    if len(pts) < 2:
        return None, -1.0, "need BOTTOM then TOP"
    bot = np.array(pts[0], dtype=np.float64)
    top = np.array(pts[1], dtype=np.float64)
    dst = top - bot
    if float(np.linalg.norm(dst)) < 4.0:
        return None, -1.0, "BOT and TOP are too close"

    uv_hi = project_world(P, np.array([post_x, y1, 0.0], dtype=np.float64))
    uv_lo = None
    for y in (y0, y0 + 0.25 * (y1 - y0), 0.5 * (y0 + y1)):
        cand = project_world(P, np.array([post_x, y, 0.0], dtype=np.float64))
        if cand is None:
            continue
        if not all(math.isfinite(v) for v in cand):
            continue
        if max(abs(cand[0]), abs(cand[1])) > 1e5:
            continue
        uv_lo = cand
        break
    if uv_lo is None or uv_hi is None:
        return None, -1.0, "far post did not project"
    if not all(math.isfinite(v) for v in (*uv_lo, *uv_hi)):
        return None, -1.0, "far post did not project"
    sa = np.array(uv_lo, dtype=np.float64)
    sb = np.array(uv_hi, dtype=np.float64)
    src = sb - sa
    if float(np.linalg.norm(src)) < 4.0:
        return None, -1.0, "far post projection is too short"
    if float(np.dot(src, dst)) <= 0.0:
        return None, -1.0, "click BOTTOM of the post first, then TOP"

    ang = math.atan2(dst[1], dst[0]) - math.atan2(src[1], src[0])
    c, s = math.cos(ang), math.sin(ang)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    mid = 0.5 * (sa + sb)
    sa_r = rot @ (sa - mid) + mid
    sb_r = rot @ (sb - mid) + mid

    def closest(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = float(np.dot(p - a, ab)) / denom
        return a + t * ab

    shift = 0.5 * (
        (bot - closest(sa_r, sb_r, bot)) + (top - closest(sa_r, sb_r, top))
    )
    # x' = rot @ (x - mid) + mid + shift
    tcol = mid - rot @ mid + shift
    affine = np.hstack([rot, tcol.reshape(2, 1)])
    moved = float(np.linalg.norm(shift))
    deg = math.degrees(ang)
    return affine, moved, f"shift {shift[0]:+.0f},{shift[1]:+.0f}px  rot {deg:+.1f}°"

def reproj_err_px(P: np.ndarray, xyz: np.ndarray, uv: tuple[float, float]) -> float:
    pred = project_world(P, xyz)
    if pred is None:
        return 1e9
    return math.hypot(pred[0] - uv[0], pred[1] - uv[1])

def inside_posts(x: float, baseline: float, radius: float = 0.0) -> bool:
    """True only if the whole ball (centre ± radius) sits between the posts."""
    return abs(x) + radius <= baseline / 2.0 + POST_MARGIN_M

def live_ball_call(
    x: float,
    z: float,
    baseline: float,
    radius: float = 0.0,
) -> str:
    """
    Score uses the 3D centre, expanded by ball radius.
    A 22 cm ball on a 1 m goal is IN only if |X| + 0.11 m ≤ 0.50 m.
    """
    in_posts = inside_posts(x, baseline, radius)
    if z > PLANE_CROSS_M:
        return "approaching" if in_posts else "wide_field"
    if in_posts:
        return "goal"
    return "wide"

def update_call_latch(
    call: str,
    z: float | None,
    now: float,
    state: dict,
) -> str:
    """Latch GOAL/WIDE briefly after a crossing, but never if the ball is gone or 1-cam."""
    if call in ("none", "one", "reject"):
        if call == "none":
            state["prev_z"] = None
            state["latch"] = None
        return call

    prev_z = state.get("prev_z")
    if (
        z is not None
        and prev_z is not None
        and prev_z > PLANE_CROSS_M
        and z <= PLANE_CROSS_M
        and call in ("goal", "wide")
    ):
        state["latch"] = call
        state["latch_until"] = now + CALL_LATCH_S
        print(f"  Plane crossed → {call.upper()}")
    state["prev_z"] = z

    # A later WIDE reading must not keep a stale GOAL flash
    if call == "wide":
        state["latch"] = "wide"
        return "wide"

    latch = state.get("latch")
    until = float(state.get("latch_until") or 0.0)
    if latch in ("goal", "wide") and now < until:
        return latch
    return call

def call_style(call: str) -> tuple[tuple[int, int, int], str, bool]:
    """colour, short label, pulse. Empty label = do not highlight."""
    if call == "approaching":
        return COL_CALL_BLUE, "IN FIELD", True
    if call == "wide_field":
        return COL_CALL_BLUE, "WIDE", True
    if call == "goal":
        return COL_CALL_GREEN, "GOAL", False
    if call == "wide":
        return COL_CALL_RED, "WIDE", False
    return (80, 80, 80), "", False

def draw_flash_border(img: np.ndarray, color: tuple[int, int, int], now: float, pulse: bool) -> None:
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return
    if pulse:
        thick = 6 + int(6 * (0.5 + 0.5 * math.sin(now * 14.0)))
    else:
        thick = 10
    thick = max(4, min(thick, min(h, w) // 8))
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), color, thick)

def paint_ball_box(img: np.ndarray, det, color: tuple[int, int, int], label: str) -> None:
    if det is None:
        return
    x1, y1, x2, y2 = map(int, det[3])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.circle(img, (int(det[0]), int(det[1])), 5, color, -1)
    if label:
        put_label(img, label, (x1, max(20, y1 - 8)), color, 0.48, 2)

def paint_ball_pair(
    disp_l: np.ndarray,
    disp_r: np.ndarray,
    det_l,
    det_r,
    P1: np.ndarray,
    P2: np.ndarray,
    pos_l: np.ndarray,
    pos_r: np.ndarray,
    baseline: float,
    now: float,
    latch_state: dict,
) -> tuple[str, str]:
    """Draw the ball AFTER the grid. No coloured frame unless 3D from BOTH cameras."""
    if det_l is None and det_r is None:
        update_call_latch("none", None, now, latch_state)
        return "", "none"

    if det_l is None or det_r is None:
        update_call_latch("one", None, now, latch_state)
        paint_ball_box(disp_l, det_l, COL_BALL, "")
        paint_ball_box(disp_r, det_r, COL_BALL, "")
        which = "L only" if det_r is None else "R only"
        half = baseline / 2.0
        return f"dbg  {which}  (need both for 3D)  posts ±{half:.2f}m", "one"

    pt_l, pt_r, _corr, _rough = calib_centres_from_dets(
        det_l, det_r, pos_l, pos_r, P1, P2,
    )
    pos3d = triangulate_3d(P1, P2, pt_l, pt_r)
    if pos3d is None:
        update_call_latch("reject", None, now, latch_state)
        paint_ball_box(disp_l, det_l, COL_BALL, "")
        paint_ball_box(disp_r, det_r, COL_BALL, "")
        return "dbg  both seen  3D failed", "reject"

    x, y, z = map(float, pos3d)
    err_l = reproj_err_px(P1, pos3d, pt_l)
    err_r = reproj_err_px(P2, pos3d, pt_r)
    if err_l > REPROJ_MAX_PX or err_r > REPROJ_MAX_PX:
        update_call_latch("reject", None, now, latch_state)
        paint_ball_box(disp_l, det_l, COL_BALL, "")
        paint_ball_box(disp_r, det_r, COL_BALL, "")
        return (
            f"dbg  3D mismatch  reproj L{err_l:.0f}px R{err_r:.0f}px  "
            f"(ignore call)"
        ), "reject"

    radius = cfg.BALL_RADIUS_M
    d_l = dist3(pos3d, pos_l)
    d_r = dist3(pos3d, pos_r)
    raw = live_ball_call(x, z, baseline, radius)
    call = update_call_latch(raw, z, now, latch_state)
    color, banner, pulse = call_style(call)
    # Per-camera stereo range to the ball (metres).
    label_l = f"{banner}  {d_l:.2f}m" if banner else f"{d_l:.2f}m"
    label_r = f"{banner}  {d_r:.2f}m" if banner else f"{d_r:.2f}m"
    if banner:
        paint_ball_box(disp_l, det_l, color, label_l)
        paint_ball_box(disp_r, det_r, color, label_r)
        draw_flash_border(disp_l, color, now, pulse)
        draw_flash_border(disp_r, color, now, pulse)
    else:
        paint_ball_box(disp_l, det_l, COL_BALL, label_l)
        paint_ball_box(disp_r, det_r, COL_BALL, label_r)

    half = baseline / 2.0
    in_posts = inside_posts(x, baseline, radius)
    x0, x1 = x - radius, x + radius
    line = (
        f"dbg  dL={d_l:.2f}m dR={d_r:.2f}m  "
        f"X={x:+.2f} ball[{x0:+.2f},{x1:+.2f}] posts±{half:.2f} "
        f"{'IN' if in_posts else 'OUT'}  "
        f"Y={y:+.2f}  Z={z:+.2f} ({plane_side_label(z)})  "
        f"reproj {err_l:.0f}/{err_r:.0f}px"
        + (f"  {banner}" if banner else "")
    )
    return line, call

def compact_ball_status(ball_line: str, ball_call: str, yolo_mode: str = "") -> str:
    """One short status line for the live HUD."""
    _, banner, _ = call_style(ball_call)
    dbit = ""
    if "dL=" in ball_line and "dR=" in ball_line:
        try:
            dl = ball_line.split("dL=")[1].split()[0]
            dr = ball_line.split("dR=")[1].split()[0]
            dbit = f"dL {dl}  dR {dr}"
        except (IndexError, ValueError):
            dbit = ""
    if banner:
        core = f"{banner}  {dbit}".strip() if dbit else banner
    elif "need both" in ball_line:
        core = "1-cam"
    elif "ignore" in ball_line or "mismatch" in ball_line:
        core = "reject"
    elif "3D failed" in ball_line:
        core = "no-3D"
    elif dbit:
        core = dbit
    elif ball_line.startswith("dbg"):
        zbit = ""
        if "Z=" in ball_line:
            try:
                zbit = ball_line.split("Z=")[1].split()[0]
                zbit = f"Z{zbit}"
            except (IndexError, ValueError):
                zbit = "track"
        core = zbit or "track"
    else:
        core = "—"
    mode = f" · {yolo_mode}" if yolo_mode else ""
    return f"{core}{mode}"

# ─── Save / load ─────────────────────────────────────────────────────────────

def save_plane(
    baseline_m: float,
    behind_m: float,
    h_angle: float,
    v_angle: float,
    extend_up_m: float,
    focal_px: float,
    img_w: int,
    img_h: int,
    clicks: ClickState,
    rotate_deg: int,
    h_left: float | None = None,
    h_right: float | None = None,
    v_left: float | None = None,
    v_right: float | None = None,
    roll_l: float = 0.0,
    roll_r: float = 0.0,
    snap_l: np.ndarray | None = None,
    snap_r: np.ndarray | None = None,
    flip_ud: bool = False,
) -> None:
    PLANE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline_m": baseline_m,
        "cam_behind_post_m": behind_m,
        "h_angle_deg": h_angle,
        "h_left_deg": h_left if h_left is not None else h_angle,
        "h_right_deg": h_right if h_right is not None else -abs(h_angle),
        "v_angle_deg": v_angle,
        "v_left_deg": v_left if v_left is not None else v_angle,
        "v_right_deg": v_right if v_right is not None else v_angle,
        "roll_left_deg": roll_l,
        "roll_right_deg": roll_r,
        "overlay_snap_left": None if snap_l is None else snap_l.tolist(),
        "overlay_snap_right": None if snap_r is None else snap_r.tolist(),
        "extend_up_m": extend_up_m,
        "focal_px": focal_px,
        "image_width": img_w,
        "image_height": img_h,
        "cameras_sideways": cfg.CAMERAS_SIDEWAYS,
        "rotate_deg": rotate_deg,
        "flip_ud": bool(flip_ud),
        "plane": {
            "origin": "midpoint of crossbar on the goal plane",
            "equation": "Z = 0 (goal plane / posts / crossbar)",
            "cameras_z": -behind_m,
            "left_post_x": -baseline_m / 2.0,
            "right_post_x": baseline_m / 2.0,
            "crossbar_y": 0.0,
            "note": "Y > 0 is above the bar; Z > 0 is into the field; cameras at Z = −behind",
        },
        "clicks": clicks.to_dict(),
        "camera": "B0332 + LN013 NOIR",
    }
    PLANE_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved → {PLANE_PATH}")

def load_plane() -> dict | None:
    if not PLANE_PATH.exists():
        return None
    try:
        return json.loads(PLANE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None

# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print(
        "stereo_plane_calib.py is a helper library for score_live.\n"
        "Run:  python capture/score_live.py",
        file=sys.stderr,
    )
    raise SystemExit(2)
