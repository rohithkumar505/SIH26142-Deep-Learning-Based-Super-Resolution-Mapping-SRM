# SIH26142 - Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries

[![Smart India Hackathon 2026](https://img.shields.io/badge/SIH-2026-blue.svg)](https://sih.gov.in)
[![Category](https://img.shields.io/badge/Category-Software-emerald.svg)](https://sih.gov.in)
[![Ministry / Org](https://img.shields.io/badge/Organization-National%20Technical%20Research%20Organisation%20(NTRO)-indigo.svg)](https://ntro.gov.in)
[![Theme](https://img.shields.io/badge/Theme-Space%20Technology-purple.svg)](https://sih.gov.in)
[![Developer](https://img.shields.io/badge/Developer-Rohith%20Kumar-sky.svg)](https://github.com/rohithkumar505)

---

## 🎯 Problem Statement Overview
- **Problem Statement ID:** `SIH26142`
- **Title:** Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries
- **Sponsoring Organization:** National Technical Research Organisation (NTRO)
- **Department:** National Technical Research Organisation (NTRO)
- **Category:** Software
- **Theme:** Space Technology
- **Lead Developer & System Architect:** **Rohith Kumar** ([github.com/rohithkumar505](https://github.com/rohithkumar505))
- **Primary Dataset:** [Copernicus Data Space Ecosystem (Sentinel-2 MSI)](https://browser.dataspace.copernicus.eu)

---

## 💡 Solution Architecture by Rohith Kumar

1. **Interactive Mission Control Dashboard (`project/index.html`):** Glassmorphic dark-mode web application featuring real-time before/after sub-pixel satellite comparison slider, multispectral band switching (RGB, CIR Infrared, NDVI, Uncertainty Heatmaps), and exportable audit logs.
2. **FastAPI Microservice Engine (`project/app.py`):** High-throughput Python REST backend with Pydantic v2 validation, Real-ESRGAN-RS and SwinIR-Satellite deep learning inference endpoints, SHA-256 tamper-evident audit logs, and OpenAPI Swagger documentation.
3. **Comprehensive Technical Whitepaper (`project/solution.md`):** Complete architectural breakdown, mathematical formulations, Spectral Angle Mapper (SAM) loss, PostGIS schemas, and deployment topologies.
4. **Automated Test Suite (`project/test_app.py`):** Built-in unit and integration tests verifying all REST endpoints.
5. **Turnkey Containerization (`project/Dockerfile` & `project/docker-compose.yml`):** Ready for one-command deployment on Docker / Kubernetes.

---

## 🚀 Quick Start Guide

### Option 1: Instant Browser Demo (Zero Setup)
Serve locally via Python:
```bash
cd project
python -m http.server 8080
```
Open **[http://localhost:8080](http://localhost:8080)** to access the command center.

### Option 2: Run Full Python FastAPI Microservice
```bash
cd project
pip install -r requirements.txt
python app.py
```
- API Server: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
- Interactive OpenAPI Docs: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

### Option 3: Run Automated Tests
```bash
cd project
pytest test_app.py -v
```

---

## 📂 Repository Structure
```plaintext
SIH26142 - Deep Learning Based Super Resolution Mapping (SRM)/
├── README.md                           # Main problem statement pitch & guide
├── problem_statement.json              # Official SIH 2026 metadata
└── project/
    ├── index.html                      # Interactive Satellite SRM Studio by Rohith Kumar
    ├── app.py                          # FastAPI REST API Microservice by Rohith Kumar
    ├── test_app.py                     # Pytest automated test suite
    ├── solution.md                     # Deep-dive 8-section technical whitepaper
    ├── requirements.txt                # Python backend dependencies
    ├── Dockerfile                      # Production Docker container definition
    ├── docker-compose.yml              # Multi-container orchestration
    └── README.md                       # Project execution manual
```

---

## 👨‍💻 Developer Credits
- **Rohith Kumar**
- GitHub: [https://github.com/rohithkumar505](https://github.com/rohithkumar505)
- Email: [dharmendrabrohithd@gmail.com](mailto:dharmendrabrohithd@gmail.com)
