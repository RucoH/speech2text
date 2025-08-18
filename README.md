# 🎙️ speech2text

## 📘 Project Description

**speech2text** is a flexible toolkit for **offline** and **real-time (streaming) speech-to-text (STT) conversion** with multi-model support. It offers:
- Local audio recording utilities
- Benchmarking for latency, memory usage, and accuracy (WER/CER)
- Easy integration of multiple STT models
- Multilingual transcription (focus on Turkish and English)

## ✨ Features

* 🎙️ Offline and streaming transcription
* 📝 Benchmarking: latency, memory, accuracy
* 🎚️ Compare multiple models side-by-side
* 🗂️ Local audio capture and management
* 🌍 Multilingual support (Turkish, English, more)

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/RucoH/speech2text.git
cd speech2text
```

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Install `ffmpeg` and ensure your PyTorch installation matches your hardware (GPU/CPU).

## 🚀 Usage

### 🎤 Record Audio from Microphone
```bash
python record_voice.py \
  --out data/demo.wav \
  --seconds 15 \
  --rate 16000
```

### 🔍 Transcribe Audio (example)
```bash
python src/fasterwhisper_test.py \
  --model large-v3 \
  --audio data/demo.wav \
  --language tr \
  --compute-type float16
```

## 🧠 Supported Models

* **Faster-Whisper** (CTranslate2, CPU/GPU)
* **OpenAI Whisper** (Large/Medium/Small)
* **SeamlessM4T-v2** (Meta multilingual)
* **Wav2Vec2** (e.g., `facebook/wav2vec2-large-xlsr-53`)
* **MMS-1B-ALL** (Meta multilingual speech)
* **Vosk** (lightweight, local)
* **Silero STT** (lightweight, local)
* ~~Deepspeech / Coqui-STT~~ (archived/no longer maintained)

## 🗂️ Project Structure
```
speech2text/
├── data/                     # Sample input/output audio and transcripts
├── model_checkpoints/        # (Optional) Local model weights/cache
│   └── models--facebook--wav2vec2-large-xlsr-53/
├── models/                   # Model wrappers/adapters (if any)
├── src/                      # Executable scripts and helper modules
├── record_voice.py           # Quick audio recording script
├── requirements.txt          # Dependencies
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## 🛠 Troubleshooting

* **CUDA not found / slow performance:** Verify driver and CUDA compatibility.
* **FFmpeg missing:** Check with `ffmpeg -version` and install if missing.
* **Long recordings cut off:** Increase segment length and enable `vad`.

## 🗺 Roadmap

- [ ] **Streaming ASR example:** Implement microphone → live transcript pipeline.
- [ ] **YouTube audio ingestion:** Download and transcribe directly from YouTube URLs.
- [ ] **Web interface:** Add Gradio or similar web UI for easy interaction.
- [ ] **VAD & diarization:** Integrate voice activity detection and speaker separation.
- [ ] **Benchmark dashboard:** Visualize latency, memory, and WER results.
- [ ] **Language expansion:** Add support for more languages and accents.
- [ ] **Deployment options:** Docker, Hugging Face Spaces, and cloud hosting.

## 📄 License

Distributed under the [MIT License](LICENSE).

## 👤 Author

* GitHub: [@RucoH](https://github.com/RucoH)
* Live Site: [https://rucoh.github.io/](https://rucoh.github.io/)
