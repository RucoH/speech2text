import os
import whisper
from jiwer import wer

# 1) Script klasörünü bul
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2) Ornek ses ve transkript dosyalarının mutlak yolları
audio_path = os.path.join(script_dir, "../data/ornek.wav")
txt_path   = os.path.join(script_dir, "../notebooks/ornek_tr.txt")

print("Audio path:", audio_path)
print("Transkript path:", txt_path)

# 3) Model yükle ve transkribe et
model = whisper.load_model("medium")
result = model.transcribe(audio_path, language="tr")

# 4) Sonuçları yazdır
print("Hypothesis:", result["text"])
print("WER:", wer(open(txt_path).read(), result["text"]))
