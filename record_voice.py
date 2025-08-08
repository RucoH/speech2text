import sounddevice as sd
import soundfile as sf

# Ayarlar
DURATION = 45     # saniye cinsinden kayıt süresi
FS = 44100        # örnekleme hızı (Hz)
FILENAME = "data/ornek.wav"

# Kayıt
print(f"{DURATION} saniyelik kayıt başlıyor...")
recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
sd.wait()
print("Kayıt tamamlandı, dosya kaydediliyor...")

# Dosyaya yaz
sf.write(FILENAME, recording, FS)
print(f"Ses kaydı '{FILENAME}' olarak kaydedildi.")
