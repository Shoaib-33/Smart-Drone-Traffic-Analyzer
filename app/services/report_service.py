from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


class ReportService:
    def __init__(self, reports_dir: str = "outputs/reports") -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_reports(
        self,
        job_id: str,
        rows: List[Dict],
        summary: Dict,
    ) -> Dict[str, str]:
        detail_df = pd.DataFrame(rows)

        csv_path = self.reports_dir / f"{job_id}_vehicle_report.csv"
        detail_df.to_csv(csv_path, index=False)

        return {
            "csv_report_path": str(csv_path),
        }