from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_communication(input_signal, output_signal, output_dir, sample_rate=200e6):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(len(input_signal), 2000)
    time_us = np.arange(count) / sample_rate * 1e6
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].plot(time_us, np.abs(input_signal[:count]), label="Entrada RFWebLab")
    axes[0].plot(time_us, np.abs(output_signal[:count]), label="Saída RFWebLab")
    axes[0].set_ylabel("Amplitude (V)")
    axes[0].set_title("RFWebLab: entrada e saída no tempo")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].scatter(input_signal.real[:count], output_signal.real[:count],
                    s=5, alpha=0.35)
    axes[1].set_xlabel("Entrada (parte real, V)")
    axes[1].set_ylabel("Saída (parte real, V)")
    axes[1].set_title("RFWebLab: resposta entrada-saída")
    axes[1].grid(alpha=0.3)
    figure.savefig(output_dir / "rfweblab_communication.png", dpi=180)
    plt.close(figure)
