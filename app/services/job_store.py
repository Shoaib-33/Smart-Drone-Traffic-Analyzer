from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Optional


@dataclass
class JobStore:
    jobs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    def create_job(self, job_id: str, filename: str, input_path: str) -> None:
        with self.lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "filename": filename,
                "input_path": input_path,
                "status": "uploaded",
                "progress": 0,
                "error": None,
                "result": None,
            }

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(kwargs)


job_store = JobStore()
