"""Generate a waveform visualization showing why separation causes hallucination.

This creates a figure for REPORT.md / README.md that directly visualizes the
separation-tax phenomenon: at low overlap, oracle-separated tracks have long
silent regions where Whisper hallucinates (CER=24.25, CR=16.33 for pro_006).

Usage:
    .venv/bin/python scripts/docs/make_separation_tax_waveform.py
"""
from __future__ import annotations

import os
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CON_PATH = os.path.join(REPO, "resources", "snippets", "con_006.wav")
PRO_PATH = os.path.join(REPO, "resources", "snippets", "pro_006.wav")
OUT_PATH = os.path.join(REPO, "results", "figures", "report", "fig5_separation_tax_waveform.png")


def load_mono(path: str) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return sr, data.astype(np.float32)


def make_mix(con: np.ndarray, pro: np.ndarray, overlap_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic mix at the given overlap ratio.

    Returns (mixed, spk1_track, spk2_track) where spk1_track has silence
    where spk2 was talking and vice versa.
    """
    # Pad shorter signal to match
    max_len = max(len(con), len(pro))
    con_p = np.zeros(max_len, dtype=np.float32)
    pro_p = np.zeros(max_len, dtype=np.float32)
    con_p[: len(con)] = con
    pro_p[: len(pro)] = pro

    # At overlap_ratio r, pro starts at (1-r) * len(con) into the mix
    # So spk2 (pro) is silent for the first (1-r)*len(con) samples
    offset = int((1.0 - overlap_ratio) * len(con))
    mixed = np.zeros(max_len, dtype=np.float32)
    mixed[: len(con)] += con_p[: len(con)]
    pro_len_in_mix = min(len(pro), max_len - offset)
    if pro_len_in_mix > 0:
        mixed[offset : offset + pro_len_in_mix] += pro_p[:pro_len_in_mix]

    # Separated tracks (oracle): each has silence where the other speaker was
    spk1_track = np.zeros(max_len, dtype=np.float32)
    spk1_track[: len(con)] = con_p[: len(con)]
    spk2_track = np.zeros(max_len, dtype=np.float32)
    spk2_len = min(len(pro), max_len - offset)
    if spk2_len > 0:
        spk2_track[offset : offset + spk2_len] = pro_p[:spk2_len]

    return mixed, spk1_track, spk2_track


def main() -> None:
    sr, con = load_mono(CON_PATH)
    _, pro = load_mono(PRO_PATH)

    overlap_ratio = 0.05  # The catastrophic case from phase_curve.csv
    mixed, spk1, spk2 = make_mix(con, pro, overlap_ratio)

    t_mixed = np.arange(len(mixed)) / sr
    t_spk1 = np.arange(len(spk1)) / sr
    t_spk2 = np.arange(len(spk2)) / sr

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1, 1]})

    # Panel A: Mixed audio
    axes[0].plot(t_mixed, mixed / max(abs(mixed).max(), 1), color="#2196F3", linewidth=0.5)
    axes[0].set_ylabel("Amplitude", fontsize=10)
    axes[0].set_title("(A) Mixed audio — both speakers overlap at r=0.05\n"
                       "Whisper transcribes correctly: CER = 0.44", fontsize=11, loc="left")
    axes[0].set_xlim(0, t_mixed[-1])
    axes[0].set_ylim(-1.1, 1.1)

    # Panel B: Separated speaker 1 (con) — has speech + trailing silence
    axes[1].plot(t_spk1, spk1 / max(abs(spk1).max(), 1), color="#4CAF50", linewidth=0.5)
    axes[1].set_ylabel("Amplitude", fontsize=10)
    axes[1].set_title("(B) Oracle-separated Speaker 1 (con_006)\n"
                       "Speech ends ~2.1s, then SILENCE — Whisper transcribes OK: CER = 0.44",
                       fontsize=11, loc="left")
    # Highlight silent region
    sil_start = len(con) / sr
    axes[1].axvspan(sil_start, t_spk1[-1], alpha=0.15, color="red", label="silent region")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_xlim(0, t_spk1[-1])
    axes[1].set_ylim(-1.1, 1.1)

    # Panel C: Separated speaker 2 (pro) — has leading silence + speech
    axes[2].plot(t_spk2, spk2 / max(abs(spk2).max(), 1), color="#FF5722", linewidth=0.5)
    axes[2].set_ylabel("Amplitude", fontsize=10)
    axes[2].set_xlabel("Time (seconds)", fontsize=10)
    axes[2].set_title("(C) Oracle-separated Speaker 2 (pro_006) — CATASTROPHIC HALLUCINATION\n"
                       "Leading silence → Whisper enters token-id repetition loop: "
                       "CER = 24.25, CR = 16.33 (transcript 24× longer than reference!)",
                       fontsize=11, loc="left", color="#D32F2F")
    # Highlight silent region
    offset_t = int((1.0 - overlap_ratio) * len(con)) / sr
    axes[2].axvspan(0, offset_t, alpha=0.15, color="red", label="silent region (hallucination zone)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_xlim(0, t_spk2[-1])
    axes[2].set_ylim(-1.1, 1.1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PATH.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {OUT_PATH}")
    print(f"Wrote: {OUT_PATH.replace('.png', '.pdf')}")

    # Print the key numbers for verification
    print(f"\nVisualization summary:")
    print(f"  con_006 duration: {len(con)/sr:.2f}s")
    print(f"  pro_006 duration: {len(pro)/sr:.2f}s")
    print(f"  overlap ratio: {overlap_ratio}")
    print(f"  Speaker 2 leading silence: {offset_t:.2f}s")
    print(f"  From phase_curve.csv: CER_sep2=24.25, CR_sep2=16.33 (pair=5, r=0.05)")


if __name__ == "__main__":
    main()
