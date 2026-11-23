"""
Jareeda TTS server — بيشغّل VoiceTut-TTS و 3arab-TTS على جهاز فيه GPU، والتلفزيون بيطلب منه الصوت.

تشغيل (على جهاز/Colab فيه GPU):
    pip install fastapi uvicorn
    # VoiceTut:
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install git+https://github.com/k2-fsa/OmniVoice.git voicetut-tts
    # 3arab-TTS: git clone https://github.com/sherif1313/3arab-TTS  ثم  uv sync --extra cu128 جوه الفولدر
    export AB_DIR=/path/to/3arab-TTS
    python server.py            # بيفتح على بورت 8000

بعدها اعمل tunnel (مثلاً: cloudflared tunnel --url http://localhost:8000) وحط الرابط في js/config.js → JAREEDA_TTS.url
GET /tts?model=vt|ab&speaker=<اسم أو وصف>&text=<النص>   →  audio/wav
"""
import hashlib, os, subprocess, tempfile, threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

AB_DIR = os.environ.get("AB_DIR", "./3arab-TTS")
AB_CKPT = os.environ.get("AB_CKPT", "sherif1313/3arab-TTS-500M-v2-VoiceDesign")
VT_CKPT = os.environ.get("VT_CKPT", "mohammedaly22/VoiceTut-TTS")
CACHE = os.path.join(tempfile.gettempdir(), "jareeda_tts"); os.makedirs(CACHE, exist_ok=True)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
lock = threading.Lock()          # GPU واحد: طلب واحد في المرة
_vt = None

def vt_synth(text, speaker, out):
    global _vt
    if _vt is None:
        from voicetut_tts import VoiceTutTTS
        _vt = VoiceTutTTS.from_pretrained(VT_CKPT)
    _vt.synthesize(text, speaker=speaker, output=out)

def ab_synth(text, caption, out):
    # بيستخدم infer.py الرسمي (بيحمّل الموديل كل مرة → أبطأ). لو عايزه أسرع، حمّل الموديل مرة واحدة هنا من كود infer.py.
    cmd = ["uv", "run", "python", "infer.py", "--hf-checkpoint", AB_CKPT, "--text", text,
           "--caption", caption, "--no-ref", "--output-wav", out]
    subprocess.run(cmd, cwd=AB_DIR, check=True, timeout=300)

@app.get("/tts")
def tts(model: str, text: str, speaker: str = ""):
    if model not in ("vt", "ab") or not text.strip() or len(text) > 600:
        raise HTTPException(400, "bad request")
    out = os.path.join(CACHE, hashlib.sha1(f"{model}|{speaker}|{text}".encode()).hexdigest() + ".wav")
    if not os.path.exists(out):
        with lock:
            if not os.path.exists(out):
                tmp = out + ".part.wav"
                try:
                    (vt_synth if model == "vt" else ab_synth)(text, speaker or ("Mohamed" if model == "vt" else "نبره اخباريه"), tmp)
                    os.replace(tmp, out)
                except Exception as e:
                    raise HTTPException(500, str(e)[:200])
    return FileResponse(out, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
