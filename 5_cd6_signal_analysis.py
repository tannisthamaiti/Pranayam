"""
=============================================================
CODE 5: CD6 Raw Signal Analysis
=============================================================
Reads cd6_write.csv  (28k+ rows, columns: time, data)
Produces:
  - cd6_raw_signal.png
  - cd6_spectrogram.png
  - cd6_fft.png
  - cd6_rolling_stats.png

The CD6 appears to be a continuous physiological sensor
(e.g., PPG, skin conductance, or similar wearable signal).
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import warnings
warnings.filterwarnings("ignore")

CSV_PATH = "NEEMA_Day1-20260408T100839Z-3-001\\NEEMA_Day1\\backup_2025-12-26_12-53-29_NEEMA_DAY1\\cd6_write.csv"


def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("time").reset_index(drop=True)
    # Estimate sampling rate
    dt = np.diff(df["time"].values)
    fs_est = 1.0 / np.median(dt)
    return df, fs_est


def plot_raw_signal(df):
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(df["time"] / 60, df["data"], linewidth=0.5, color="#2c3e50", alpha=0.8)
    ax.set_xlabel("Time (minutes)", fontsize=11)
    ax.set_ylabel("Signal Value", fontsize=11)
    ax.set_title("CD6 Raw Signal — Full Session", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cd6_raw_signal.png", dpi=150, bbox_inches="tight")
    print("Saved: cd6_raw_signal.png")
    plt.close()


def plot_spectrogram(df, fs):
    data = df["data"].values.astype(float)
    # Remove DC
    data = data - np.mean(data)

    nperseg = min(512, len(data) // 4)
    f, t, Sxx = scipy_signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=nperseg//2)

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.pcolormesh(t / 60, f, 10 * np.log10(Sxx + 1e-12),
                       shading="gouraud", cmap="plasma")
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    ax.set_xlabel("Time (minutes)", fontsize=11)
    ax.set_title("CD6 Signal Spectrogram", fontsize=13, fontweight="bold")
    ax.set_ylim(0, min(f.max(), fs / 2))
    plt.colorbar(im, ax=ax, label="Power (dB)")
    plt.tight_layout()
    plt.savefig("cd6_spectrogram.png", dpi=150, bbox_inches="tight")
    print("Saved: cd6_spectrogram.png")
    plt.close()


def plot_fft(df, fs):
    data = df["data"].values.astype(float)
    data = data - np.mean(data)

    N = len(data)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    fft_mag = np.abs(np.fft.rfft(data)) / N

    # Only show up to 5 Hz
    mask = freqs <= 5.0
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.semilogy(freqs[mask], fft_mag[mask], color="#e74c3c", linewidth=0.8)

    # Annotate top peaks
    peak_idx = np.argsort(fft_mag[mask])[-10:]
    for idx in peak_idx:
        if freqs[mask][idx] > 0.01:
            ax.axvline(freqs[mask][idx], color="#3498db", alpha=0.4, linewidth=1)
            ax.text(freqs[mask][idx], fft_mag[mask][idx] * 1.5,
                    f"{freqs[mask][idx]:.3f} Hz", fontsize=7, color="#3498db", rotation=45)

    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Magnitude", fontsize=11)
    ax.set_title("CD6 Signal — FFT Frequency Spectrum (0–5 Hz)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cd6_fft.png", dpi=150, bbox_inches="tight")
    print("Saved: cd6_fft.png")
    plt.close()


def plot_rolling_stats(df, fs):
    """Rolling mean, std, and energy over 60-second windows."""
    window = int(fs * 60)  # 60-second window
    data = df["data"]

    rolling_mean  = data.rolling(window, center=True, min_periods=1).mean()
    rolling_std   = data.rolling(window, center=True, min_periods=1).std()
    rolling_energy = (data**2).rolling(window, center=True, min_periods=1).mean()

    t = df["time"] / 60

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("CD6 Rolling Statistics (60-second window)", fontsize=13, fontweight="bold")

    axes[0].plot(t, data, alpha=0.2, color="grey", linewidth=0.5)
    axes[0].plot(t, rolling_mean, color="#2980b9", linewidth=2, label="Rolling Mean")
    axes[0].set_ylabel("Value"); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, rolling_std, color="#e74c3c", linewidth=1.5, label="Rolling Std Dev")
    axes[1].set_ylabel("Std Dev"); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, rolling_energy, color="#27ae60", linewidth=1.5, label="Rolling Energy")
    axes[2].set_ylabel("Energy (signal²)"); axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("Time (minutes)", fontsize=11)

    plt.tight_layout()
    plt.savefig("cd6_rolling_stats.png", dpi=150, bbox_inches="tight")
    print("Saved: cd6_rolling_stats.png")
    plt.close()


def main():
    df, fs = load_data()
    print(f"CD6 data: {len(df)} samples")
    print(f"Estimated sampling rate: {fs:.2f} Hz")
    print(f"Duration: {df['time'].max()/60:.1f} minutes")
    print(f"Signal range: {df['data'].min():.2f} to {df['data'].max():.2f}")

    plot_raw_signal(df)
    plot_spectrogram(df, fs)
    plot_fft(df, fs)
    plot_rolling_stats(df, fs)


if __name__ == "__main__":
    main()
