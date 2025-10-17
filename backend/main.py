# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
import os, tempfile, torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from .utils import load_audio_safe, save_stems_to_zip

app = FastAPI(title="VocalSplit API", version="0.1.0")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "htdemucs"
model = get_model(MODEL_NAME).to(DEVICE).eval()

@app.get("/")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_NAME}

MAX_BYTES = 200 * 1024 * 1024  # ~200MB cap

@app.post("/separate")
async def separate(file: UploadFile = File(...)):
    # allow formats soundfile can read (no mp3 here)
    if not file.filename.lower().endswith((".wav", ".flac", ".ogg", ".aiff", ".aif")):
        raise HTTPException(400, "Upload WAV/FLAC/OGG/AIFF only.")

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (limit ~200MB).")

    # save to temp file
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(data)

    try:
        waveform, sr = load_audio_safe(tmp_path)             # (C, T)
        with torch.no_grad():
            batch = waveform.unsqueeze(0).to(DEVICE)         # [1, C, T]
            stems = apply_model(model, batch, device=DEVICE)[0]  # [S, C, T]

        zip_bytes = save_stems_to_zip(stems, sr)
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename=\"{os.path.splitext(file.filename)[0]}_stems.zip\"'},
        )
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass