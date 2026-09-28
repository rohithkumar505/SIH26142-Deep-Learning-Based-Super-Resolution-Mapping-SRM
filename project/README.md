# SIH26142 - Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries

Dedicated project codebase for **Smart India Hackathon 2026** problem statement `SIH26142`, sponsored by the **National Technical Research Organisation (NTRO)**.

**Lead Architect & Developer:** **Rohith Kumar** ([https://github.com/rohithkumar505](https://github.com/rohithkumar505))  
**Theme:** Space Technology  
**Dataset Reference:** [Copernicus Data Space Ecosystem (Sentinel-2)](https://browser.dataspace.copernicus.eu)

---

## 🚀 Execution Instructions

### Instant UI Browser View (Interactive Studio)
Serve via Python:
```bash
python -m http.server 8080
```
Visit: **[http://localhost:8080](http://localhost:8080)**

### Full FastAPI Backend
```bash
pip install -r requirements.txt
python app.py
```
- API Endpoint: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### Automated Pytest Suite
```bash
pytest test_app.py -v
```

### Docker Containerization
```bash
docker-compose up --build
```
