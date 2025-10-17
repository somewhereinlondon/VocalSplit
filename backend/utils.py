# backend/utils.py
import io
import zipfile
import torch
import soundfile as sf
from torchaudio.functional import resample as ta_resample

TARGET_SR = 44100

def load_audio_safe(path: str):
    data, sr = sf.read(path, always_2d=True)   # (T, C)
    wave = torch.from_numpy(data.T).float()    # (C, T)
    if sr != TARGET_SR:
        wave = ta_resample(wave, sr, TARGET_SR)
        sr = TARGET_SR
    return wave, sr

def save_stems_to_zip(stems: torch.Tensor, sr: int, names=None) -> bytes:
    if names is None:
        names = ["vocals", "drums", "bass", "other"]
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, stem in enumerate(stems):
            buf = io.BytesIO()
            # soundfile expects (T, C)
            sf.write(buf, stem.cpu().T.numpy(), sr, format="WAV")
            zf.writestr(f"{names[i]}.wav", buf.getvalue())
    mem.seek(0)
    return mem.getvalue()
    
