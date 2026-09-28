-- PostGIS schema for SIH26142 SRM (run against docker-compose postgis service)
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS srm_tiles (
    id SERIAL PRIMARY KEY,
    scene_id VARCHAR(64) NOT NULL,
    geom GEOMETRY(Polygon, 4326),
    input_gsd_m REAL DEFAULT 10.0,
    output_gsd_m REAL,
    psnr_db REAL,
    ssim REAL,
    sam_degrees REAL,
    audit_hash VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_srm_tiles_scene ON srm_tiles(scene_id);
CREATE INDEX IF NOT EXISTS idx_srm_tiles_geom ON srm_tiles USING GIST(geom);
