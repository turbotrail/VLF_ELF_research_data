import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal

# Load your WAV file
sample_rate, data = wav.read('rain_long.wav')

# If stereo, select only one channel
if len(data.shape) == 2:
    data = data[:, 0]

# Normalize
if data.dtype == np.int16:
    data = data.astype(np.float32) / 32768.0

# Configure how many seconds of data to analyze
# duration_focus = 20  # seconds
# data = data[:int(sample_rate * duration_focus)]

# Perform STFT (full sample rate, full resolution)
# f, t, Zxx = signal.stft(data, fs=sample_rate, nperseg=4096, noverlap=2048)

# # Plot the STFT Spectrogram - Dual Panel (0-5kHz and 5-20kHz)
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), facecolor='white', sharex=True)

# # Top panel: 0–5kHz (Natural VLF focus)
# ax1.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='plasma')
# ax1.set_title('High-Resolution STFT Spectrogram (0-5kHz)', fontsize=14)
# ax1.set_ylabel('Frequency [Hz]', fontsize=12)
# ax1.set_ylim(0, 5000)
# ax1.grid(True, color='gray', linestyle='--', alpha=0.5)

# # Bottom panel: 5–20kHz (Transmitters, high-frequency components)
# ax2.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='plasma')
# ax2.set_title('High-Resolution STFT Spectrogram (5-20kHz)', fontsize=14)
# ax2.set_xlabel('Time [s]', fontsize=12)
# ax2.set_ylabel('Frequency [Hz]', fontsize=12)
# ax2.set_ylim(5000, 20000)
# ax2.grid(True, color='gray', linestyle='--', alpha=0.5)

# plt.colorbar(ax1.collections[0], ax=[ax1, ax2], label='Magnitude')
# plt.tight_layout()
# plt.show()

# === Detect Sudden Bursts ===
# Use the spectrogram Sxx directly
f_spec, t_spec, Sxx = signal.spectrogram(data, fs=sample_rate, nperseg=4096, noverlap=2048)

# Configurable threshold factor
threshold_sigma_factor = 4  # Change to 3, 4, 5 as needed

# Threshold: median + N * std deviation
threshold = np.median(Sxx) + threshold_sigma_factor * np.std(Sxx)

# Sum energy across frequency axis (collapse into time series)
energy_per_time = np.sum(Sxx, axis=0)

# Find time indices where energy exceeds threshold
burst_indices = np.where(energy_per_time > threshold)[0]
burst_times = t_spec[burst_indices]
burst_energies = energy_per_time[burst_indices]

# Log detected burst times and energies
print("\n=== Detected Sferic-like Bursts ===")
for bt, be in zip(burst_times, burst_energies):
    print(f"Burst at {bt:.2f} sec | Energy: {be:.2e}")

print(f"\nTotal bursts detected: {len(burst_times)} (using {threshold_sigma_factor}σ threshold)")