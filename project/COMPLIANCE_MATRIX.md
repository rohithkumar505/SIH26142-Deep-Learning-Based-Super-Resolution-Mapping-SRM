# SIH26142 Compliance Matrix (NTRO Problem Statement)

| # | NTRO Expected Requirement | Status | Implementation |
|---|---------------------------|--------|----------------|
| 1 | 10m Sentinel-2 medium-res input | ✅ | `srm/scenes.py`, `GET /api/v1/copernicus/search` |
| 2 | Pre-processing (cloud mask, tiling, BOA) | ✅ | `srm/preprocessing/pipeline.py`, `POST /api/v1/srm/preprocess` |
| 3 | Generative / CNN / Transformer SR | ✅ | `srm/models/factory.py`, `GET /api/v1/models/zoo` |
| 4 | Output sharper than 4m GSD (4× from 10m → 2.5m) | ✅ | `srm/inference/engine.py` |
| 5 | Spectral & geospatial consistency | ✅ | SAM, NDVI, ERGAS in `srm/metrics/rs_metrics.py` |
| 6 | Uncertainty & hallucination risk | ✅ | `srm/uncertainty/estimator.py` |
| 7 | Validation vs HR reference | ✅ | `POST /api/v1/srm/validate`, `validate.py` |
| 8 | Training with paired datasets | ✅ | `srm/training/trainer.py`, `train.py`, `POST /api/v1/training/start` |
| 9 | Crop monitoring use case | ✅ | `GET /api/v1/applications/crop/{scene_id}` |
| 10 | Urban analysis use case | ✅ | `GET /api/v1/applications/urban/{scene_id}` |
| 11 | Disaster assessment use case | ✅ | `GET /api/v1/applications/disaster/{scene_id}` |
| 12 | Copernicus dataset integration | ✅/⚠️ | Live STAC + fallback (`srm/copernicus/stac.py`) |
| 13 | Production HR model weights (Planet 3m pairs) | ⚠️ Roadmap | Train on real tiles; checkpoint JSON demo in `checkpoints/` |

**Honest note:** Full nation-scale training on Copernicus + commercial HR pairs needs GPU cluster, TB storage, and licensed reference imagery. This repo provides the **complete framework** and **real metric pipeline**; swap synthetic patches for downloaded STAC assets to go production.
