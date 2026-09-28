# SIH26142 Enterprise Architecture (v5)

## Layers

1. **Ingestion** — Copernicus STAC, scene catalog, upload API  
2. **Pre-processing** — BOA, cloud mask, tiling (`srm/preprocessing`)  
3. **Inference** — Model zoo, ensemble, batch (`srm/models`, `srm/inference`, `srm/batch`)  
4. **Trust** — Metrics, uncertainty, XAI, audit chain (`srm/metrics`, `srm/uncertainty`, `srm/xai`, `srm/audit`)  
5. **Applications** — Crop, urban, disaster, water, forestry, defense  
6. **MLOps** — Training jobs, federated stub, edge ONNX manifest, K8s, CI  
7. **Experience** — `index.html` studio + `mission_control.html` + OpenAPI  

## API Surface

- `/api/v1/*` — Core SRM + compliance + applications  
- `/api/v2/*` — Enterprise extensions (50+ catalogued features)  

Lead: **Rohith Kumar** | Sponsor: **NTRO** | Problem: **SIH26142**
