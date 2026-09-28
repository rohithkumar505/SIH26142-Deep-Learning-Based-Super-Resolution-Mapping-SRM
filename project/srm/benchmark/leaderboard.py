from srm.inference.engine import run_srm_pipeline
from srm.scenes import SCENE_BY_ID
from srm.config import SUPPORTED_MODELS


def run_leaderboard(scale: int = 4) -> dict:
    rows = []
    for sid in SCENE_BY_ID:
        for model in SUPPORTED_MODELS[:3]:
            r = run_srm_pipeline(sid, model, scale, 0.88, False, None)
            rows.append(
                {
                    "scene_id": sid,
                    "model": model,
                    "psnr_db": r["psnr_db"],
                    "ssim": r["ssim"],
                    "sam_degrees": r["sam_degrees"],
                }
            )
    rows.sort(key=lambda x: x["psnr_db"], reverse=True)
    return {"leaderboard": rows[:15], "total_runs": len(rows)}
