import json, wave, os
from vosk import Model, KaldiRecognizer
from jiwer import wer

# 1) Proje köküne veya script'e göre mutlak yolu bulun
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir   = os.path.dirname(script_dir)
audio_path = os.path.join(root_dir, "data", "ornek_16k.wav")      # veya ornek_16k.wav
txt_path   = os.path.join(root_dir, "data", "ornek_tr.txt")

# 2) Model yolunu seçin (TR veya EN)
model_path = os.path.join(root_dir, "models", "vosk-model-small-tr-0.3")
# model_path = os.path.join(root_dir, "models", "vosk-model-small-en-us-0.15")

# 3) WAV dosyasını açın
wf = wave.open(audio_path, "rb")
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
    raise RuntimeError("Lütfen 16 kHz mono, 16-bit WAV kullanın (ffmpeg ile dönüştürün).")

# 4) Recognizer’ı oluşturun
model = Model(model_path)
rec   = KaldiRecognizer(model, wf.getframerate())

# 5) Parça parça transkribe edin
full_hyp = ""
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        res = json.loads(rec.Result())
        full_hyp += " " + res.get("text", "")
# son kalan için
res = json.loads(rec.FinalResult())
full_hyp += " " + res.get("text", "")

hypothesis = full_hyp.strip()

# 6) WER hesapla
truth = open(txt_path, "r", encoding="utf-8").read().strip()
error = wer(truth, hypothesis)

# 7) Sonuçları yazdır
print("Hypothesis:", hypothesis)
print("Ground truth:", truth)
print(f"WER: {error:.3f}")
