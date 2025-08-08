import os, time
import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer

# ——————————————
# A) Dosya yollarını ayarla
script_dir = os.path.dirname(os.path.abspath(__file__))    # .../speech2text/src
root_dir   = os.path.dirname(script_dir)                    # .../speech2text
audio_path = os.path.join(root_dir, "data", "ornek.wav")
txt_path   = os.path.join(root_dir, "data", "ornek_tr.txt")

print("Audio:", audio_path)
print("Transcript:", txt_path)

# ——————————————
# B) Model ve cihaz
# Daha hızlı indirme için base-xlsr versiyonunu kullanıyoruz (~600 MB)
model_id = "facebook/wav2vec2-base-xlsr-53"
print("Downloading/using model:", model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ——————————————
# C) Processor ve model (internet ile çekilecek)
processor = Wav2Vec2Processor.from_pretrained(model_id)
model     = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()

# ——————————————
# D) Ses dosyasını oku ve sample rate kontrol et
speech, sr = sf.read(audio_path)
assert sr == 16000, f"16 kHz bekleniyor, dosyan {sr} Hz."
print(f"Loaded audio: {speech.shape[0]} samples @ {sr} Hz")

# ——————————————
# E) Tokenize & infer
inputs       = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
input_values = inputs.input_values.to(device)

start = time.perf_counter()
with torch.no_grad():
    logits = model(input_values).logits
pred_ids = torch.argmax(logits, dim=-1)
latency  = time.perf_counter() - start

# ——————————————
# F) Decode & WER
hypothesis = processor.batch_decode(pred_ids)[0].lower().strip()
truth      = open(txt_path, "r", encoding="utf-8").read().lower().strip()
error      = wer(truth, hypothesis)

# ——————————————
# G) Sonuçları yazdır
print("Hypothesis:", hypothesis)
print("Ground truth:", truth)
print(f"WER: {error:.3f}")
print(f"Latency: {latency:.2f}s")
