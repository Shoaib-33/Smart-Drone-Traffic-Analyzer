from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.services.job_store import job_store
from app.services.report_service import ReportService

logger = logging.getLogger(__name__)

TRACKER_CONFIG = str(Path(__file__).with_name("bytetrack_custom.yaml"))

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

FFMPEG_TIMEOUT_SECONDS = 300 


# ──────────────────────────────────────────────
# FFmpeg resolver
# ──────────────────────────────────────────────

def _find_ffmpeg() -> Optional[str]:
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and Path(env_path).is_file():
        return env_path

    which = shutil.which("ffmpeg")
    if which:
        return which

    if sys.platform == "win32":
        candidates = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            str(Path.home() / "ffmpeg" / "bin" / "ffmpeg.exe"),
        ]
        for c in candidates:
            if Path(c).is_file():
                return c

    return None


FFMPEG_BIN: Optional[str] = _find_ffmpeg()

if FFMPEG_BIN is None:
    logger.warning(
        "FFmpeg not found on startup. Video re-encoding will fail. "
        "Install FFmpeg or set the FFMPEG_PATH environment variable."
    )
else:
    logger.info("FFmpeg resolved: %s", FFMPEG_BIN)


def reencode_for_browser(src: Path, dst: Path) -> None:
    """Re-encode src to H.264 mp4 suitable for browser <video> playback.

    Raises:
        RuntimeError: If FFmpeg is not available.
        subprocess.CalledProcessError: If FFmpeg exits with a non-zero code.
        subprocess.TimeoutExpired: If FFmpeg hangs beyond FFMPEG_TIMEOUT_SECONDS.
    """
    if FFMPEG_BIN is None:
        raise RuntimeError(
            "FFmpeg not found. Install FFmpeg or set the FFMPEG_PATH environment variable."
        )

    logger.info("Re-encoding %s → %s", src, dst)
    subprocess.run(
        [
            FFMPEG_BIN,
            "-y",
            "-i", str(src),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(dst),
        ],
        check=True,
        timeout=FFMPEG_TIMEOUT_SECONDS,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,  # Capture stderr for error reporting
    )
    logger.info("Re-encode complete: %s", dst)


def _cleanup_files(*paths: Path) -> None:
    """Delete files if they exist. Logs but does not raise on failure."""
    for p in paths:
        try:
            if p.exists():
                p.unlink()
                logger.debug("Cleaned up: %s", p)
        except OSError as exc:
            logger.warning("Could not delete %s: %s", p, exc)


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray, scale: float = 1.5) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    h, w = frame.shape[:2]
    return cv2.resize(
        frame,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_CUBIC,
    )


def scale_boxes_down(boxes: np.ndarray, scale: float) -> np.ndarray:
    return (boxes / scale).astype(int)


# ──────────────────────────────────────────────
# Weighted vote (windowed)
# ──────────────────────────────────────────────

def _weighted_vote(votes: List[Tuple[str, float]], recent_n: int = 10) -> str:
    recent = votes[-recent_n:]
    weight_map: Dict[str, float] = defaultdict(float)
    for vtype, conf in recent:
        weight_map[vtype] += conf
    return max(weight_map, key=weight_map.__getitem__)


# ──────────────────────────────────────────────
# Line Counter
# ──────────────────────────────────────────────

class LineCounter:
    def __init__(self) -> None:
        self.line_y_ratio: float = 0.45
        self.min_hits: int = 1
        self.min_movement: int = 5
        self._frame_w: int = 1
        self._frame_h: int = 1

        self.recent_count_memory: List[Dict] = []
        self.recent_count_max_age_frames: int = 6

        self.track_hits: Dict[int, int] = defaultdict(int)
        self.track_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.counted_ids: Set[int] = set()

        self.vehicle_type_votes: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        self.vehicle_type_by_id: Dict[int, str] = {}

        self.first_seen: Dict[int, Tuple[int, float, Tuple[int, int]]] = {}
        self.last_seen: Dict[int, Tuple[int, float, Tuple[int, int]]] = {}
        self.events: List[Dict] = []

    @staticmethod
    def _ccw(A, B, C) -> bool:
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _crosses_line(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        line_y: int,
        width: int,
    ) -> bool:
        ls = (0, line_y)
        le = (width, line_y)
        return self._ccw(p1, ls, le) != self._ccw(p2, ls, le)

    def _cleanup_recent_memory(self, frame_id: int) -> None:
        self.recent_count_memory = [
            item
            for item in self.recent_count_memory
            if frame_id - item["frame_id"] <= self.recent_count_max_age_frames
        ]

    def _near_counted(self, cx: int, cy: int, frame_id: int, stable_type: str) -> bool:
        threshold_x = int(self._frame_w * 0.01)
        threshold_y = int(self._frame_h * 0.015)
        for item in self.recent_count_memory:
            if (
                frame_id - item["frame_id"] <= self.recent_count_max_age_frames
                and item["vehicle_type"] == stable_type
                and abs(cx - item["center"][0]) < threshold_x
                and abs(cy - item["center"][1]) < threshold_y
            ):
                return True
        return False

    def update(
        self,
        objs: List[Dict],
        frame_id: int,
        ts: float,
        w: int,
        h: int,
    ) -> None:
        self._frame_w = w
        self._frame_h = h
        self._cleanup_recent_memory(frame_id)

        line_y = int(h * self.line_y_ratio)
        line_band = max(10, int(h * 0.01))

        for obj in objs:
            tid = obj["track_id"]
            vtype = obj["vehicle_type"]
            conf = obj["confidence"]
            box = obj["box"]

            x1, y1, x2, y2 = box
            track_x = int((x1 + x2) / 2)
            track_y = y2
            track_pt = (track_x, track_y)

            self.track_hits[tid] += 1
            self.track_history[tid].append(track_pt)
            self.track_history[tid] = self.track_history[tid][-20:]

            self.vehicle_type_votes[tid].append((vtype, conf))
            stable_type = _weighted_vote(self.vehicle_type_votes[tid], recent_n=10)
            self.vehicle_type_by_id[tid] = stable_type

            if tid not in self.first_seen:
                self.first_seen[tid] = (frame_id, ts, track_pt)
            self.last_seen[tid] = (frame_id, ts, track_pt)

            if len(self.track_history[tid]) < 2:
                continue

            prev_pt = self.track_history[tid][-2]
            crossed = self._crosses_line(prev_pt, track_pt, line_y, w)
            has_moved = abs(track_y - self.first_seen[tid][2][1]) > self.min_movement

            first_y = self.first_seen[tid][2][1]
            first_seen_near_line = abs(first_y - line_y) <= line_band
            moved_beyond_line = track_y > line_y + line_band
            fallback_crossed = first_seen_near_line and moved_beyond_line

            should_count = (
                (crossed or fallback_crossed)
                and self.track_hits[tid] >= self.min_hits
                and has_moved
                and tid not in self.counted_ids
                and not self._near_counted(track_x, track_y, frame_id, stable_type)
            )

            if should_count:
                self.counted_ids.add(tid)
                self.events.append(
                    {
                        "vehicle_id": tid,
                        "vehicle_type": stable_type,
                        "crossed_line": "horizontal",
                        "counted_frame": frame_id,
                        "counted_timestamp_seconds": round(ts, 2),
                    }
                )
                self.recent_count_memory.append(
                    {
                        "vehicle_id": tid,
                        "vehicle_type": stable_type,
                        "center": (track_x, track_y),
                        "frame_id": frame_id,
                    }
                )

    def summary(self, duration: float) -> Dict:
        cnt = Counter(e["vehicle_type"] for e in self.events)
        return {
            "total_vehicle_count": len(self.events),
            "breakdown_by_type": {
                "car": cnt.get("car", 0),
                "bus": cnt.get("bus", 0),
                "truck": cnt.get("truck", 0),
                "motorcycle": cnt.get("motorcycle", 0),
            },
            "processing_duration_seconds": round(duration, 2),
        }

    def build_rows(self) -> List[Dict]:
        rows: List[Dict] = []
        for event in self.events:
            tid = event["vehicle_id"]
            first = self.first_seen.get(tid)
            last = self.last_seen.get(tid)
            rows.append(
                {
                    "vehicle_id": tid,
                    "vehicle_type": event["vehicle_type"],
                    "crossed_line": event["crossed_line"],
                    "first_detected_frame": first[0] if first else None,
                    "first_detected_timestamp_seconds": round(first[1], 2) if first else None,
                    "counted_frame": event["counted_frame"],
                    "counted_timestamp_seconds": event["counted_timestamp_seconds"],
                    "last_detected_frame": last[0] if last else None,
                    "last_detected_timestamp_seconds": round(last[1], 2) if last else None,
                }
            )
        return rows


# ──────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────

def detect(model: YOLO, frame: np.ndarray) -> List[Dict]:
    scale = 1.5
    f = preprocess_frame(frame, scale)

    results = model.track(
        source=f,
        persist=True,
        tracker=TRACKER_CONFIG,
        conf=0.15,
        iou=0.4,
        imgsz=960,
        classes=list(VEHICLE_CLASSES.keys()),
        augment=False,
        verbose=False,
    )

    if not results or results[0].boxes is None or results[0].boxes.id is None:
        return []

    boxes = scale_boxes_down(results[0].boxes.xyxy.cpu().numpy(), scale)
    cls = results[0].boxes.cls.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    objs: List[Dict] = []
    for b, c, i, conf in zip(boxes, cls, ids, confs):
        class_id = int(c)
        if class_id not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, b)
        objs.append(
            {
                "track_id": int(i),
                "vehicle_type": VEHICLE_CLASSES[class_id],
                "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                "box": (x1, y1, x2, y2),
                "confidence": round(float(conf), 3),
            }
        )
    return objs


# ──────────────────────────────────────────────
# Main Job Entry Point
# ──────────────────────────────────────────────

def process_video_job(job_id: str) -> None:
    report_service = ReportService()
    job = job_store.get_job(job_id)
    if not job:
        logger.error("Job %s not found in store — aborting.", job_id)
        return

    input_path = Path(job["input_path"])
    out_path = Path("outputs/processed_videos") / f"{job_id}.mp4"
    browser_path = out_path.with_name(f"{job_id}_final.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap: Optional[cv2.VideoCapture] = None
    writer: Optional[cv2.VideoWriter] = None

    try:
        # ── Pre-flight checks ──────────────────
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        if not Path(TRACKER_CONFIG).exists():
            raise FileNotFoundError(f"Tracker config not found: {TRACKER_CONFIG}")

        if FFMPEG_BIN is None:
            raise RuntimeError(
                "FFmpeg not found. Install FFmpeg or set the FFMPEG_PATH "
                "environment variable before starting the server."
            )

        # ── Model & video setup ────────────────
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Job %s — device: %s", job_id, device)
        model = YOLO("yolov8m.pt")
        model.to(device)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            logger.warning(
                "Job %s — invalid FPS (%s) in video metadata, falling back to 25.0. "
                "Timestamps in the report may be inaccurate.",
                job_id,
                fps,
            )
            fps = 25.0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        if w <= 0 or h <= 0:
            raise RuntimeError(
                f"Invalid video dimensions ({w}x{h}). The file may be corrupt."
            )

        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for: {out_path}")

        # ── Frame processing loop ──────────────
        counter = LineCounter()
        start = time.time()
        frame_id = 0
        SKIP = 2
        last_objs: List[Dict] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % SKIP == 0:
                last_objs = detect(model, frame)
                counter.update(last_objs, frame_id, frame_id / fps, w, h)

            line_y = int(h * counter.line_y_ratio)
            cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"Counted: {len(counter.counted_ids)}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )

            for o in last_objs:
                x1, y1, x2, y2 = o["box"]
                tid = o["track_id"]
                stable_type = counter.vehicle_type_by_id.get(tid, o["vehicle_type"])
                counted = tid in counter.counted_ids
                color = (0, 255, 0) if counted else (0, 0, 255)

                label = f"ID:{tid} {stable_type} {o['confidence']:.2f}"
                if counted:
                    label += " COUNTED"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                )

                history = counter.track_history.get(tid, [])
                for i in range(1, len(history)):
                    cv2.line(frame, history[i - 1], history[i], color, 1)

            writer.write(frame)
            frame_id += 1

            if frame_id % 10 == 0:
                progress = min(int((frame_id / total_frames) * 100), 99)
                job_store.update_job(job_id, progress=progress)

    except Exception as e:
        logger.exception("Job %s failed during frame processing: %s", job_id, e)
        job_store.update_job(job_id, status="failed", error=str(e), progress=0)
        _cleanup_files(out_path, browser_path)
        return

    finally:
        # Always release handles — even if an exception occurred mid-loop
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()

    # ── FFmpeg re-encode (separate try block for clear error messages) ──
    try:
        reencode_for_browser(out_path, browser_path)
    except subprocess.TimeoutExpired:
        err = f"FFmpeg timed out after {FFMPEG_TIMEOUT_SECONDS}s re-encoding the video."
        logger.error("Job %s — %s", job_id, err)
        job_store.update_job(job_id, status="failed", error=err, progress=0)
        _cleanup_files(out_path, browser_path)
        return
    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.decode(errors="replace") if e.stderr else "no stderr"
        err = f"FFmpeg re-encode failed (exit code {e.returncode}): {stderr_output}"
        logger.error("Job %s — %s", job_id, err)
        job_store.update_job(job_id, status="failed", error=err, progress=0)
        _cleanup_files(out_path, browser_path)
        return
    except Exception as e:
        logger.exception("Job %s — unexpected FFmpeg error: %s", job_id, e)
        job_store.update_job(job_id, status="failed", error=str(e), progress=0)
        _cleanup_files(out_path, browser_path)
        return
    finally:
        _cleanup_files(out_path)

    # ── Reports & job completion ───────────────
    try:
        duration = time.time() - start
        summary = counter.summary(duration)
        rows = counter.build_rows()
        report_paths = report_service.generate_reports(job_id, rows, summary)

        job_store.update_job(
            job_id,
            status="completed",
            progress=100,
            result={
                **summary,
                **report_paths,
                "processed_video_path": str(browser_path),
                "processed_video_url": f"/outputs/processed_videos/{browser_path.name}",
                "detections": rows,
            },
        )
        logger.info("Job %s completed. %d vehicles counted.", job_id, summary["total_vehicle_count"])

    except Exception as e:
        logger.exception("Job %s failed during report generation: %s", job_id, e)
        job_store.update_job(
            job_id,
            status="failed",
            error=f"Report generation failed: {e}",
            progress=0,
        )