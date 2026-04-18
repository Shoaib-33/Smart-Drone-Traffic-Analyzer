from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.services.job_store import job_store
from app.services.video_processor import process_video_job

router = APIRouter(tags=["Traffic Analyzer API"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a video file.")

    job_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}{suffix}"

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_store.create_job(job_id=job_id, filename=file.filename, input_path=str(save_path))

    return {
        "job_id": job_id,
        "filename": file.filename,
        "message": "Upload successful.",
        "uploaded_video_url": f"/uploads/{save_path.name}",
    }


@router.post("/process/{job_id}")
def start_processing(job_id: str, background_tasks: BackgroundTasks):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] == "processing":
        return {"message": "Job is already processing.", "job_id": job_id}

    if job["status"] == "completed":
        return {"message": "Job already completed.", "job_id": job_id}

    job_store.update_job(job_id, status="processing", progress=1, error=None)
    background_tasks.add_task(process_video_job, job_id)
    return {"message": "Processing started.", "job_id": job_id}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "error": job["error"],
    }


@router.get("/results/{job_id}")
def get_results(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job.get("result", {}),
    }


@router.get("/download/report/{job_id}")
def download_report(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet.")

    result = job.get("result", {})
    report_path = result.get("csv_report_path")

    if not report_path or not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Report not found.")

    return FileResponse(
        path=report_path,
        filename=Path(report_path).name,
        media_type="text/csv",
    )