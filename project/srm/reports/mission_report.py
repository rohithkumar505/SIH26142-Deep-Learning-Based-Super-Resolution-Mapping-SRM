from datetime import datetime, timezone
from pathlib import Path

from srm.config import EXPORT_DIR
from srm.platform.feature_registry import full_feature_manifest


def generate_html_mission_report(scene_id: str, metrics: dict) -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = EXPORT_DIR / f"NTRO_SIH26142_Report_{scene_id}_{ts}.html"
    manifest = full_feature_manifest()
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>SIH26142 Report</title></head>
<body style="font-family:system-ui;background:#0b1120;color:#e2e8f0;padding:2rem">
<h1>SIH26142 Super-Resolution Mission Report</h1>
<p>Scene: {scene_id} | Generated: {ts} UTC | Architect: Rohith Kumar</p>
<h2>Metrics</h2><pre>{metrics}</pre>
<h2>Platform Features ({manifest['total_features']})</h2>
<ul>{''.join(f"<li>{f['name']}</li>" for f in manifest['features'][:25])}...</ul>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    return {"report_path": str(path), "format": "HTML", "feature_count": manifest["total_features"]}
