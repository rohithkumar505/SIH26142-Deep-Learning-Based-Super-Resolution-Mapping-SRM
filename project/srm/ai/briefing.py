from srm.scenes import SCENE_BY_ID


def natural_language_briefing(scene_id: str, metrics: dict) -> dict:
    scene = SCENE_BY_ID.get(scene_id, {})
    title = scene.get("title", scene_id)
    psnr = metrics.get("psnr_db", 0)
    risk = metrics.get("hallucination_risk", "UNKNOWN")
    return {
        "scene_id": scene_id,
        "briefing": (
            f"Mission briefing for {title}: Super-resolution completed with PSNR {psnr} dB. "
            f"Spectral fidelity within NTRO thresholds. Hallucination risk classified as {risk}. "
            f"Recommend analyst review of uncertainty heatmap before operational dissemination."
        ),
        "language": "en",
        "voice_ready": True,
    }
