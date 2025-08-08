#!/usr/bin/env python3
import os
import time
import torch
import soundfile as sf
from transformers import pipeline
from jiwer import wer

def main():
    # 1) Dosya yollarını belirle
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir   = os.path.dirname(script_dir)
    audio_path = os.path.join(root_dir, "data", "ornek.wav")
    txt_path   = os.path.join(root_dir, "data", "ornek_tr.txt")

    print(f"> Audio file:      {audio_path}")
    print(f"> Transcript file: {txt_path}")

    # 2) Model kimliği ve cihaz
    model_id = "facebook/wav2vec2-xls-r-300m"  # 300M multilingual model
    device   = 0 if torch.cuda.is_available() else -1  # 0=first GPU, -1=CPU
    print(f"> Model: {model_id} (device={device})")

    # 3) Pipeline’i doğru pozisyonel argümanla başlat
    asr = pipeline(
        "automatic-speech-recognition",  # pozisyonel task adı
        model=model_id,
        device=device,
        chunk_length_s=30
    )

    # 4) Ses dosyasını oku ve kontrol et
    speech, sr = sf.read(audio_path)
    assert sr == 16000, f"Model 16 kHz bekliyor, gelen {sr} Hz"
    print(f"> Loaded audio: {speech.shape[0]} samples @ {sr} Hz")

    # 5) İnference ve süre ölçümü
    start  = time.perf_counter()
    result = asr(audio_path)
    latency = time.perf_counter() - start

    # 6) Hypothesis ve WER hesapla
    hypothesis = result.get("text", "").strip().lower()
    truth      = open(txt_path, encoding="utf-8").read().strip().lower()
    error      = wer(truth, hypothesis)

    # 7) Sonuçları yazdır
    print("\n── Results ──")
    print(f"Hypothesis : {hypothesis}")
    print(f"Ground truth: {truth}")
    print(f"WER         : {error:.3f}")
    print(f"Latency     : {latency:.2f}s")

if __name__ == "__main__":
    main()
