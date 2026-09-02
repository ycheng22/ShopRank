import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()


@router.get("/ablation")
async def get_ablation_data() -> dict:
    """Returns the raw eval run JSONs for the UI to render ablation and cross-tabs."""
    results_dir = Path("evals/results")
    data = []
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            try:
                data.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, OSError):
                continue

    return {"runs": data}
