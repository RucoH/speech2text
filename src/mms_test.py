import os, time
import torch
import soundfile as sf
from transformers import pipeline
from jiwer import wer

# ——————————————
# A) Dosya yolları
root     = os.path.dirname(os.path.dirname(__file__))   # …/speech2text
audio    = os.path.join(root, "data", "ornek.wav")
trans_txt= os.path.join(root, "data", "ornek_tr.txt")

print("Audio file:", audio)
print("Transcript:", trans_txt)

# ——————————————
# B) Pipeline oluştur (MMS-1B-All)
print("Loading MMS-1B-All model… (~4.8 GB)")
asr = pipeline(
    "automatic-speech-recognition",
    model="facebook/mms-1b-all",
    device=0 if torch.cuda.is_available() else -1,  # -1=CPU, >=0=GPU id
)

# ——————————————
# C) Transkripsiyon ve ölçümler
start   = time.perf_counter()
output  = asr(audio, chunk_length_s=30)  # uzun kayıt için parça parça transkripte imkan
latency = time.perf_counter() - start

# Pipeline çıktısı sözlük biçiminde
hyp     = output["text"].strip().lower()
truth   = open(trans_txt, encoding="utf-8").read().strip().lower()
error   = wer(truth, hyp)

# ——————————————
# D) Sonuçları yazdır
print("Hypothesis:", hyp)
print("Ground truth:", truth)
print(f"WER: {error:.3f}")
print(f"Latency: {latency:.2f}s")
