"""
FastAPI backend for the Feature Selection Algorithm Recommender.
"""

import io
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from meta_features import compute_meta_features, MAX_SAMPLES
from predictor import predict

app = FastAPI(title="FS Algorithm Recommender")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_upload(upload: UploadFile) -> pd.DataFrame:
    content = upload.file.read()
    name = (upload.filename or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(content))
    raise HTTPException(
        status_code=415,
        detail="Unsupported file type. Please upload CSV, Excel, or Parquet.",
    )


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
def index():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.post("/api/columns")
async def get_columns(file: UploadFile = File(...)):
    """Return the column names of the uploaded file so the user can pick the target."""
    try:
        df = _read_upload(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    return JSONResponse({"columns": df.columns.tolist(), "shape": list(df.shape)})


@app.post("/api/recommend")
async def recommend(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    accuracy_weight: float = Form(0.6),
    compression_weight: float = Form(0.4),
    use_sampling: bool = Form(True),
):
    """
    Upload a dataset, specify the target column, and get algorithm rankings.
    """
    try:
        df = _read_upload(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    if target_col not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Column '{target_col}' not found in the uploaded file.",
        )

    numeric_cols = [
        c for c in df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(numeric_cols) == 0:
        raise HTTPException(
            status_code=422,
            detail="No numeric feature columns found (excluding target).",
        )

    if df[target_col].nunique() < 2:
        raise HTTPException(
            status_code=422,
            detail="Target column must have at least 2 distinct classes.",
        )

    try:
        mf = compute_meta_features(df, target_col, use_sampling=use_sampling)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error computing meta-features: {e}"
        )

    # Replace any NaN meta-features with 0 to avoid prediction failures
    mf_clean = {k: (v if np.isfinite(v) else 0.0) for k, v in mf.items()}

    try:
        result = predict(mf_clean, accuracy_weight, compression_weight)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating recommendations: {e}"
        )

    # Round meta-feature values for display
    mf_display = {k: round(v, 6) for k, v in mf.items()}

    n_original = int(df.shape[0])
    sampled = use_sampling and n_original > MAX_SAMPLES

    return JSONResponse(
        {
            "meta_features": mf_display,
            "dataset_info": {
                "n_samples": n_original,
                "n_samples_used": min(n_original, MAX_SAMPLES),
                "sampled": sampled,
                "n_features": int(len(numeric_cols)),
                "n_classes": int(df[target_col].nunique()),
                "target_col": target_col,
            },
            "rankings": result["rankings"],
            "weights": {
                "accuracy": round(accuracy_weight / (accuracy_weight + compression_weight), 4),
                "compression": round(compression_weight / (accuracy_weight + compression_weight), 4),
            },
        }
    )
