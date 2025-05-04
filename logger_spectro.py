import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import time
import datetime
import sounddevice as sd
import os

# === Configuration ===
duration = 60  # seconds per recording chunk
sample_rate = 44100  # Hz
channels = 1  # Mono
output_folder = "spectrogram_logs"
os.makedirs(output_folder, exist_ok=True)

# === Continuous Logging Loop ===
try:
    print("Starting overnight VLF spectrogram logger...")
    while True:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Recording chunk at {timestamp}...")
        
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
        sd.wait()

        # Save WAV file
        wav_filename = os.path.join(output_folder, f"recording_{timestamp}.wav")
        wav.write(wav_filename, sample_rate, recording)
        print(f"Saved WAV: {wav_filename}")

        # Generate Spectrogram
        plt.figure(figsize=(10, 4))
        plt.specgram(recording.flatten(), NFFT=1024, Fs=sample_rate, noverlap=512, cmap='inferno')
        plt.title(f"Spectrogram {timestamp}")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.ylim(0, 5000)  # Focus on VLF and lower range
        plt.colorbar(label='Intensity [dB]')
        spectro_filename = os.path.join(output_folder, f"spectrogram_{timestamp}.png")
        plt.savefig(spectro_filename)
        plt.close()
        print(f"Saved Spectrogram: {spectro_filename}")

except KeyboardInterrupt:
    print("Logger stopped by user. Exiting...")
