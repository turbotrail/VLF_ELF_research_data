import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import butter, filtfilt

# === Configuration ===
# Path to your .wav file
wav_file_path = 'third.wav'  # Change this to your WAV file name

# === Load WAV file ===
sample_rate, data = wav.read(wav_file_path)

# If stereo, take only one channel
if len(data.shape) == 2:
    data = data[:, 0]

# === Apply Band-pass Filter (e.g., 3000 Hz to 30000 Hz) to remove low-frequency harmonics ===
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Apply the filter
lowcut = 3000  # 3 kHz
highcut = 20000  # 20 kHz (within Nyquist limit for 44.1kHz sample rate)
data = bandpass_filter(data, lowcut, highcut, sample_rate)

# === Perform FFT on entire data ===
n = len(data)
frequencies = np.fft.rfftfreq(n, d=1/sample_rate)
fft_magnitude = np.abs(np.fft.rfft(data)) / n

# === Plot Spectrum (VLF Range) ===
plt.figure(figsize=(15, 6))
plt.plot(frequencies, fft_magnitude)
plt.title('Frequency Spectrum of VLF Antenna Recording (VLF Range 3kHz–30kHz)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.xlim(3000, 30000)  # Focus on VLF Range 3kHz–30kHz
plt.grid()
plt.show()

# === Optional: Plot Full Range ===
plt.figure(figsize=(15, 6))
plt.plot(frequencies, fft_magnitude)
plt.title('Full Frequency Spectrum (Full Data)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.xlim(0, sample_rate/2)  # Full range up to Nyquist frequency
plt.grid()
plt.show()

# === Create Spectrogram and Highlight Sferics ===
from scipy.signal import spectrogram

# Compute spectrogram
f_spec, t_spec, Sxx = spectrogram(data, fs=sample_rate, nperseg=2048, noverlap=1024)

# Remove low background noise
threshold = np.median(Sxx) + 3*np.std(Sxx)  # Dynamic threshold
Sxx_clean = np.where(Sxx > threshold, Sxx, 0)  # Suppress background

# Plot clean spectrogram with improved visibility
plt.figure(figsize=(15, 6), facecolor='white')
plt.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx_clean + 1e-10), shading='gouraud', cmap='plasma')
plt.title('Cleaned Spectrogram - Highlighting Sferics', fontsize=14)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Frequency [Hz]', fontsize=12)
plt.ylim(3000,10000)  # Focus on ELF/VLF range
plt.colorbar(label='Power [dB]')
plt.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.show()