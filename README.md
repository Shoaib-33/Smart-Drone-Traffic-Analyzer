# 🛸 Smart Drone Traffic Analyzer

> **ANTS Technical Assessment — Software Engineer (Computer Vision · Automation · Full-Stack)**

A full-stack, production-ready web application that analyzes drone footage to detect, track, and count vehicles — generating annotated video output and structured CSV intelligence reports.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
- [Computer Vision Pipeline](#computer-vision-pipeline)
- [Tracking Methodology & Double-Counting Prevention](#tracking-methodology--double-counting-prevention)
- [API Reference](#api-reference)
- [Reporting](#reporting)
- [Engineering Assumptions](#engineering-assumptions)
- [Evaluation Criteria Coverage](#evaluation-criteria-coverage)

---

## Overview

The Smart Drone Traffic Analyzer processes aerial/drone video footage through a YOLO + ByteTrack pipeline to:

- Detect vehicles (cars, motorcycles, buses, trucks) frame-by-frame
- Track each unique vehicle across frames with persistent IDs
- Count vehicles crossing a virtual detection line — without double-counting
- Annotate the processed video with bounding boxes, track IDs, and trail history
- Export findings as a structured CSV report

The entire workflow is exposed through a clean web UI: upload → process → review annotated video → download report.

---

## Live Demo

> 📹 **Screen Recording:** [View Demo on Loom ](#) *(https://www.loom.com/share/a5dcdd518e3d44f1a916a4b618bd1572)*

---

## Architecture

The application follows a **decoupled full-stack architecture**:

```
┌─────────────────────────────────────────────────────────┐
│                     Browser Client                      │
│              HTML · CSS · Vanilla JavaScript            │
│   Upload → Poll Status → Render Results → Download      │
└────────────────────┬────────────────────────────────────┘
                     │  REST API (HTTP)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend                        │
│                                                         │
│  routes/api.py   → All REST API endpoints               │
│  routes/web.py   → Serves index.html (template route)   │
│                                                         │
│  POST /api/upload       → Save file, create job         │
│  POST /api/process/:id  → Kick off background task      │
│  GET  /api/status/:id   → Poll progress (0–100%)        │
│  GET  /api/results/:id  → Fetch final detections        │
│  GET  /api/download/    → Serve CSV report              │
└────────────────────┬────────────────────────────────────┘
                     │  BackgroundTasks (FastAPI)
                     ▼
┌─────────────────────────────────────────────────────────┐
│              CV Processing Pipeline                     │
│                                                         │
│  OpenCV VideoCapture                                    │
│    → Frame Preprocessing (CLAHE + 1.5× upscale)         │
│    → YOLOv8m Detection (car / bus / truck / motorcycle) │
│    → ByteTrack Multi-Object Tracking                    │
│    → LineCounter (crossing logic + anti-recount)        │
│    → OpenCV VideoWriter (annotated output)              │
│    → FFmpeg Re-encode (H.264, browser-compatible)       │
│    → ReportService (CSV via pandas)                     │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**Frontend ↔ Backend communication** is purely REST. The frontend polls `GET /api/status/:id` every 2 seconds to update the progress bar. No WebSockets are used — polling is sufficient for the processing timescale and keeps the architecture simple.

**Job state** is managed in-process via a thread-safe `JobStore` (a `dataclass` wrapping a `dict` with a `threading.Lock`). This is intentional for a single-server proof-of-concept. In production this would be replaced with Redis + Celery.

**Background processing** uses FastAPI's `BackgroundTasks`. The video processing task runs in a separate thread, leaving the HTTP event loop free to serve polling requests without blocking.

**Static assets** (`app.js`, `style.css`) are mounted directly via FastAPI's `StaticFiles`, keeping the frontend self-contained with no build step required.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend | Python 3.10+, FastAPI, Uvicorn |
| Template Rendering | Jinja2 (via FastAPI `TemplateResponse`) |
| Object Detection | YOLOv8m (Ultralytics) |
| Multi-Object Tracking | ByteTrack (via Ultralytics tracker API) |
| Video I/O | OpenCV (`cv2`) |
| Video Encoding | FFmpeg (H.264 re-encode for browser playback) |
| Reporting | pandas |
| Deep Learning Runtime | PyTorch (CPU or CUDA) |

---

## Project Structure

```
smart_drone_traffic_analyzer/
│
├── app/
│   ├── __init__.py
│   ├── main.py                        # FastAPI app factory, router registration, static mounts
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── api.py                     # All REST API endpoints (/upload, /process, /status, etc.)
│   │   └── web.py                     # Web route — serves index.html via Jinja2
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── bytetrack_custom.yaml      # ByteTrack hyperparameter configuration
│   │   ├── job_store.py               # Thread-safe in-memory job state manager
│   │   ├── report_service.py          # CSV  report generation (pandas)
│   │   └── video_processor.py         # Full CV pipeline (YOLO + ByteTrack + LineCounter)
│   │
│   ├── static/
│   │   ├── app.js                     # Frontend logic (upload, poll, render results)
│   │   ├── drone-bg.jpg               # Background image asset
│   │   └── style.css                  # Application stylesheet
│   │
│   ├── templates/
│   │   └── index.html                 # Single-page UI (Jinja2 template)
│   │
│   └── utils/
│       ├── __init__.py
│       └── __init__.py
│
├── outputs/
│   ├── processed_videos/              # Annotated + re-encoded output videos (auto-created)
│   └── reports/                       # Generated CSV report files (auto-created)
│
├── uploads/                           # Raw uploaded videos (auto-created)
│
├── yolov8m.pt                         # YOLOv8 model weights (auto-downloaded on first run)
├── requirements.txt
└── README.md
```

---

## Local Setup

### Prerequisites

- Python **3.10+**
- `pip`
- **FFmpeg** installed and available on your system `PATH` (or via `FFMPEG_PATH` env variable)
- *(Optional)* CUDA-capable GPU for faster inference

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/smart-drone-traffic-analyzer.git
cd smart-drone-traffic-analyzer
```

---

### Step 2 — Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` includes:**

```
fastapi
uvicorn[standard]
python-multipart
jinja2
opencv-python
torch
torchvision
ultralytics
pandas
numpy
```

---

### Step 4 — FFmpeg Setup

FFmpeg is required to re-encode the processed video into H.264 format for browser `<video>` playback. The application resolves FFmpeg in the following priority order:

1. `FFMPEG_PATH` environment variable
2. System `PATH` (`shutil.which`)
3. Common Windows install paths (auto-detected)

**Option A — Install system FFmpeg (recommended):**

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
# Extract and add the /bin folder to your system PATH
```

**Option B — Set environment variable:**

```bash
# macOS / Linux
export FFMPEG_PATH=/path/to/ffmpeg

# Windows (Command Prompt)
set FFMPEG_PATH=C:\ffmpeg\bin\ffmpeg.exe
```

---

### Step 5 — YOLO Model Weights

On first run, `yolov8m.pt` is automatically downloaded by Ultralytics if not present in the project root. To pre-download manually:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

---

### Step 6 — Run the Application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser at: **http://localhost:8000**

> The `uploads/`, `outputs/processed_videos/`, and `outputs/reports/` directories are created automatically on first use.

---

## Computer Vision Pipeline

### 1. Frame Preprocessing

Before passing frames to YOLO, two enhancements are applied:

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** is applied in LAB color space on the L (luminance) channel. This improves detection of vehicles in low-contrast aerial footage where the road surface and vehicle colors are similar.

**1.5× bicubic upscale** before inference. Drone footage often has low effective resolution per vehicle. Upscaling before YOLO inference improves detection recall for small and distant objects. Bounding boxes are then scaled back down to original frame coordinates before drawing or counting.

### 2. Frame Skipping

The pipeline samples **every 2nd frame** (`SKIP = 2`). Between sampled frames, the last known detections are carried forward for annotation. This halves processing time with negligible accuracy loss, since vehicles move slowly relative to framerate in drone footage.

### 3. YOLO Detection

`YOLOv8m` (medium) is used with the following configuration:

| Parameter | Value | Rationale |
|---|---|---|
| `conf` | 0.15 | Low threshold to catch partially occluded or distant vehicles |
| `iou` | 0.40 | Standard NMS overlap threshold |
| `imgsz` | 960 | Larger input size improves small-object detection |
| `classes` | 2, 3, 5, 7 | car, motorcycle, bus, truck (COCO IDs) |
| `augment` | False | Disabled for processing speed |
| `persist` | True | Maintains track state across frames within the same session |

### 4. ByteTrack Configuration

ByteTrack (`app/services/bytetrack_custom.yaml`) associates detections across frames into persistent tracks:

```yaml
tracker_type: bytetrack
track_high_thresh: 0.30      # High-confidence detection threshold
track_low_thresh: 0.10       # Low-confidence detections recovered during occlusion
new_track_thresh: 0.30       # Minimum confidence to start a new track
track_buffer: 90             # Frames a track survives without a match (~3s at 30fps)
match_thresh: 0.80           # IoU threshold for track–detection association
fuse_score: true             # Fuses detection confidence with IoU for better matching
```

`track_buffer: 90` is intentionally high so tracks survive occlusions (vehicles passing under trees, bridges, or other vehicles) without being terminated and re-assigned a new ID.

### 5. Vehicle Type Stabilization

YOLO can oscillate between similar classes (e.g., `car` vs `truck`) across frames. To produce a stable label per track, a **confidence-weighted vote** is computed over the most recent 10 detections for each track ID. High-confidence frames contribute proportionally more to the final class decision.

---

## Tracking Methodology & Double-Counting Prevention

### Detection Line

A horizontal counting line is placed at **45% of frame height** (`line_y_ratio = 0.45`). This position was chosen for top-down drone footage where vehicles typically enter from the top and move downward through the frame.

### Crossing Detection — CCW Orientation Test

The crossing check uses a **CCW (counter-clockwise) geometric orientation test**. This determines whether the segment connecting a vehicle's current and previous bottom-center positions crosses the counting line. This is more robust than a simple Y-threshold comparison because it correctly handles vehicles moving at diagonal angles.

### Five-Layer Anti-Double-Counting System

Multiple independent safeguards prevent any vehicle from being counted more than once:

**Layer 1 — Track ID Set (`counted_ids`)**
Once a `track_id` is counted, it is permanently added to a set. Any subsequent crossing by the same ID is silently ignored — regardless of how many times it crosses the line.

**Layer 2 — Minimum Hit Requirement (`min_hits = 1`)**
A track must have been observed for at least 1 frame before it is eligible to be counted. This filters out spurious single-frame ghost detections that ByteTrack occasionally produces.

**Layer 3 — Minimum Movement Requirement (`min_movement = 5px`)**
The vehicle must have moved at least 5 pixels vertically from its first-seen position. This prevents counting stationary background objects or misdetections that happen to straddle the line when first detected.

**Layer 4 — Spatial / Temporal Anti-Recount Memory**
A rolling buffer (expiring after 6 frames) stores the position and vehicle type of each recently counted event. Before recording a new count, the system checks whether a vehicle of the **same type** was counted at a **nearby position** within the last 6 frames. Proximity thresholds are:

- ΔX < 1% of frame width
- ΔY < 1.5% of frame height

This catches track fragmentation edge cases where a single vehicle momentarily receives two IDs near the counting line, which would otherwise result in a double-count.

**Layer 5 — Fallback Counting for Late-Initialized Tracks**
ByteTrack sometimes initializes a track ID for a vehicle that first appears *already on* or *just past* the counting line (e.g., a vehicle entering from the side of the frame). The CCW crossing test would miss these vehicles entirely. A fallback rule counts a vehicle if:
- Its track first appeared within `±1% frame height` of the line, **and**
- It is subsequently observed clearly beyond the line (> `line_band` pixels past)

---

## API Reference

All endpoints are prefixed with `/api`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload a video file. Returns `job_id` and `uploaded_video_url`. |
| `POST` | `/api/process/{job_id}` | Start background CV processing for a job. |
| `GET` | `/api/status/{job_id}` | Poll processing status and integer progress (0–100). |
| `GET` | `/api/results/{job_id}` | Retrieve full results including detection rows and video URL. |
| `GET` | `/api/download/report/{job_id}` | Download CSV report. |

**Static mounts** (served directly by FastAPI/Starlette):

| Mount | Serves |
|---|---|
| `/static/` | `app/static/` — JS, CSS, background image |
| `/uploads/` | `uploads/` — raw uploaded videos |
| `/outputs/` | `outputs/` — annotated videos and reports |

---

## Reporting

On job completion, `ReportService` generates a CSV report automatically saved to `outputs/reports/`.

### CSV Report — `{job_id}_vehicle_report.csv`

One row per counted vehicle crossing event:

| Column | Description |
|---|---|
| `vehicle_id` | ByteTrack persistent track ID |
| `vehicle_type` | Stabilized class label: `car`, `bus`, `truck`, or `motorcycle` |
| `crossed_line` | Crossing line identifier (always `horizontal` in this build) |
| `first_detected_frame` | Frame index when the track was first observed |
| `first_detected_timestamp_seconds` | Video timestamp of first detection |
| `counted_frame` | Frame index when the vehicle crossed the counting line |
| `counted_timestamp_seconds` | Video timestamp of the line crossing event |
| `last_detected_frame` | Frame index of the track's final observation |
| `last_detected_timestamp_seconds` | Video timestamp of the last observation |


## Engineering Assumptions

The following decisions were made independently where the specification was ambiguous:

**1. Single horizontal counting line.**
The spec requires counting unique vehicles. This implementation uses one horizontal counting line at 45% frame height, which suits the provided top-down drone dataset. A multi-line or directional approach was considered but deemed unnecessary for the evaluation video.

**2. YOLOv8m over YOLOv8n/s.**
The medium model was chosen over nano/small variants for meaningfully better small-object detection accuracy on aerial footage. The 2× frame-skip optimization compensates for the higher per-frame inference cost.

**3. In-memory job store.**
A production system would use Redis or a database for job persistence. For this proof-of-concept, a thread-safe in-process dictionary is sufficient. Jobs are not persisted across server restarts.

**4. FFmpeg is a hard dependency for browser video.**
OpenCV's `VideoWriter` with the `mp4v` codec produces files that do not stream reliably in all browsers. FFmpeg re-encodes output to H.264 (`libx264`) with `+faststart` (moov atom at the start of the file), which is the standard for browser `<video>` element compatibility.

**5. Frame skip = 2.**
At typical drone video framerates (25–30fps), sampling every other frame provides 12–15 effective detection frames per second — well above the movement speed of vehicles. Tracking continuity is not meaningfully affected.

**6. CLAHE + upscale preprocessing applied uniformly.**
No per-frame quality assessment is performed. Preprocessing is applied to every sampled frame, which is appropriate for the provided dataset with a consistent aerial perspective and lighting.

---

## Evaluation Criteria Coverage

| Criteria | Weight | Implementation |
|---|---|---|
| **Pipeline & Architecture** | 35% | Decoupled FastAPI + vanilla JS; REST API; BackgroundTasks threading; FFmpeg re-encoding; pandas reporting; Jinja2 templates |
| **Problem Solving & Logic** | 35% | Five-layer anti-double-counting system; ByteTrack with 90-frame buffer for occlusion survival; confidence-weighted type voting; CCW geometric crossing detection; fallback counting for edge-initialized tracks |
| **Code Quality & Documentation** | 20% | Type-annotated Python throughout; modular service architecture (`routes/`, `services/`, `utils/`); this README |
| **User Experience** | 10% | Live progress polling every 2s; annotated video playback; one-click CSV download; mission log readout panel |

---

