"""V4.3 Fig. 1 regeneration on the manuscript's own 64x61 front-end lattice.

Faithful re-implementation of ComplexSTFT + PhysicalTriMaskTeacher from
src/vimd_amc/models/spectral.py, applied to one immutable record extracted
from standards/cache_factor_headline_1024_v2/hard_interference.
"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
import torch.nn.functional as F

MODULATIONS = ("BPSK","PI/2-BPSK","QPSK","8PSK","16QAM","64QAM","256QAM","GMSK","CPFSK","4FSK")
PROFILES = ("TDL-A","TDL-B","TDL-C","TDL-D","TDL-E")
JAMMERS = ("tone","multitone","chirp","sweep","pulse","partial band","comb","cochannel","OFDM-like")
NFFT, HOP = 64, 16

d = np.load("teacher_record_v43.npz"); meta = json.load(open("teacher_record_v43.json"))
t = {n: torch.from_numpy(d[n]).unsqueeze(0).float() for n in ("x","clean","jammer","unexplained")}
win = torch.hann_window(NFFT, periodic=True)

def stft(v):
    return torch.stft(torch.complex(v[:,0], v[:,1]), n_fft=NFFT, hop_length=HOP,
                      win_length=NFFT, window=win, center=False, normalized=True,
                      onesided=False, return_complex=True)

with torch.no_grad():
    spectra = {n: stft(v) for n, v in t.items()}
    ps, pj, pu = (stft(t[n]).abs().square() for n in ("clean","jammer","unexplained"))
    total = ps + pj + pu
    eps = (1e-8 * total.mean(dim=(1,2), keepdim=True)).clamp_min(torch.finfo(total.dtype).tiny)
    den = total.clamp_min(eps)
    qs, qj, qu = ps/den, pj/den, pu/den
    empty = total <= eps
    masks = torch.stack((F.relu(qs-qj), F.relu(qj-qs), qu + 2.0*torch.minimum(qs,qj)), dim=1)
    if empty.any():
        masks[:,0] = masks[:,0].masked_fill(empty, 0.0)
        masks[:,1] = masks[:,1].masked_fill(empty, 0.0)
        masks[:,2] = masks[:,2].masked_fill(empty, 1.0)
    masks = masks / masks.sum(dim=1, keepdim=True).clamp_min(1e-8)
masks = masks.squeeze(0).numpy()
frames, bins = spectra["x"].shape[-1], spectra["x"].shape[-2]
assert (bins, frames) == (64, 61), (bins, frames)
print("mask sum check:", float(np.abs(masks.sum(axis=0) - 1).max()))

shifted = lambda v: np.fft.fftshift(v, axes=0)
def power_db(v):
    p = v.abs().square().squeeze(0).cpu().numpy()
    return shifted(10.0*np.log10(np.maximum(p/max(float(p.max()), np.finfo(np.float32).tiny), 1e-8)))

plt.rcParams.update({"font.family":"serif","font.size":8,"axes.titlesize":8,
                     "axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7})
fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.18), constrained_layout=True)
ticks = [0, 30, 60]
img = None
for ax,(n,title) in zip(axes[0], (("x","Received mixture"),("clean","Tracked target"),
                                  ("jammer","Tracked jammer"),("unexplained","Noise + receiver artifact"))):
    img = ax.imshow(power_db(spectra[n]), origin="lower", aspect="auto", cmap="magma",
                    vmin=-80.0, vmax=0.0, extent=(0, frames-1, -0.5, 0.5))
    ax.set_title(title); ax.set_xlabel("STFT frame"); ax.set_xticks(ticks)
    if ax is not axes[0, 0]:
        ax.set_yticklabels([])
fig.colorbar(img, ax=axes[0].tolist(), label="Relative power (dB)", shrink=0.82, pad=0.01)

mimg = None
for r,(ax,title) in enumerate(zip(axes[1,:3], ("Target-power\n"+r"dominant $M_s^\star$",
                                               "Jammer-power\n"+r"dominant $M_j^\star$",
                                               "Unexplained-or-\n"+r"power-ambiguous $M_o^\star$"))):
    mimg = ax.imshow(shifted(masks[r]), origin="lower", aspect="auto", cmap="viridis",
                     vmin=0.0, vmax=1.0, extent=(0, frames-1, -0.5, 0.5))
    ax.set_title(title, fontsize=7.5); ax.set_xlabel("STFT frame"); ax.set_xticks(ticks)
    if ax is not axes[1, 0]:
        ax.set_yticklabels([])
fig.colorbar(mimg, ax=axes[1,:3].tolist(), label="Teacher allocation", shrink=0.82, pad=0.01)

active = [JAMMERS[k] for k in np.flatnonzero(np.array(meta["jam_labels"]) > 0.5)]
s = axes[1,3]; s.axis("off")
s.text(0.02, 0.98, "\n".join((
    "Immutable cache record",
    f"split: {meta['split']}",
    f"source index: {meta['index']}, view: {meta['view']}",
    f"modulation: {MODULATIONS[meta['label']]}",
    f"jammer: {'+'.join(active) if active else 'none'}",
    f"SNR/SIR: {meta['snr_db']:.2f}/{meta['sir_db']:.2f} dB",
    f"target/jammer TDL: {PROFILES[meta['target_profile_index']]}/{PROFILES[meta['jammer_profile_index']]}",
    f"lattice: {bins}x{frames} (NFFT {NFFT}, hop {HOP})",
    "", "Admitted structured jammer.", "Teacher only; no learned",
    "prediction or performance", "evidence is shown.")),
    ha="left", va="top", linespacing=1.25)

fig.supylabel("Normalized frequency", fontsize=8)
fig.savefig("physical_teacher_example.pdf", bbox_inches="tight")
fig.savefig("physical_teacher_example.png", dpi=300, bbox_inches="tight")
print("rendered", bins, "x", frames, "| mod", MODULATIONS[meta['label']],
      "| jam", active, "| SNR/SIR", meta['snr_db'], meta['sir_db'])
