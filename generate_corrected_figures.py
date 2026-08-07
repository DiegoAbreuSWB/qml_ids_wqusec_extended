import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Consome a saida de reproduce_corrected_results.py e regenera as figuras
# fig_qml_perf.png e fig_noise_sweep.png usadas em paper/sbrc.tex.
HERE = Path(__file__).resolve().parent
JSON_PATH = HERE / "reproduce_output.json"
OUT_DIR = HERE / "paper" / "figures"

with open(JSON_PATH) as f:
    R = json.load(f)

plt.rcParams.update({"font.size": 11})
Q_C, C_C = "#4C72B0", "#DD8452"

# ── Figura: curva de aprendizado com IC (R=10) ──────────────────────────────
train_sizes = [30, 50, 80, 120, 180, 250]
qk_mean = [R["learning_curve"][str(n)]["qk"][0] * 100 for n in train_sizes]
qk_std  = [R["learning_curve"][str(n)]["qk"][1] * 100 for n in train_sizes]
rbf_mean = [R["learning_curve"][str(n)]["rbf"][0] * 100 for n in train_sizes]
rbf_std  = [R["learning_curve"][str(n)]["rbf"][1] * 100 for n in train_sizes]

fig, ax = plt.subplots(figsize=(7, 5))
qk_mean = np.array(qk_mean); qk_std = np.array(qk_std)
rbf_mean = np.array(rbf_mean); rbf_std = np.array(rbf_std)
ax.plot(train_sizes, qk_mean, "o-", color=Q_C, lw=2, label="QKSVM (media, R=10)")
ax.fill_between(train_sizes, qk_mean - qk_std, qk_mean + qk_std, color=Q_C, alpha=0.2, label="QKSVM +-1 desvio-padrao")
ax.plot(train_sizes, rbf_mean, "s--", color=C_C, lw=2, label="SVM-RBF (media, R=10)")
ax.fill_between(train_sizes, rbf_mean - rbf_std, rbf_mean + rbf_std, color=C_C, alpha=0.2, label="SVM-RBF +-1 desvio-padrao")
ax.set_xlabel("Tamanho do conjunto de treino (amostras)")
ax.set_ylabel("Acuracia no teste (%)")
ax.set_title("Curva de aprendizado com incerteza\nQKSVM vs. SVM-RBF (R=10 repeticoes, splits independentes)")
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(alpha=0.3)
ax.set_ylim(75, 100)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_qml_perf.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Salvo fig_qml_perf.png")

# ── Figura: varredura de dephasing com IC + marcadores de backend ──────────
T_GATE = 0.1
backend_deph_p = {
    "ibm_brisbane": 1 - np.exp(-T_GATE / 300),
    "ibm_kyoto": 1 - np.exp(-T_GATE / 150),
    "ibm_nairobi": 1 - np.exp(-T_GATE / 80),
    "noisy_extreme": 1 - np.exp(-T_GATE / 20),
}
ps = sorted(float(k) for k in R["dephasing_sweep"].keys())
means = np.array([R["dephasing_sweep"][str(p)][0] * 100 if str(p) in R["dephasing_sweep"]
                   else R["dephasing_sweep"][repr(p)][0] * 100 for p in ps])
# JSON keys are python float-str; reload directly by matching
means = []
stds = []
for k, v in sorted(R["dephasing_sweep"].items(), key=lambda kv: float(kv[0])):
    means.append(v[0] * 100); stds.append(v[1] * 100)
ps = sorted(float(k) for k in R["dephasing_sweep"].keys())
means = np.array(means); stds = np.array(stds)

ideal_mean = R["noise_ideal_baseline"]["bin"][0] * 100
ideal_std = R["noise_ideal_baseline"]["bin"][1] * 100

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(np.array(ps) * 100, means, "o-", color="#4C72B0", lw=2.5, label="Dephasing (media, R=8)")
ax.fill_between(np.array(ps) * 100, means - stds, means + stds, color="#4C72B0", alpha=0.2, label="+-1 desvio-padrao")
ax.axhline(ideal_mean, color="#55A868", lw=1.5, ls=":", label=f"Baseline sem ruido ({ideal_mean:.1f}% +- {ideal_std:.1f}%)")
ax.axhspan(ideal_mean - ideal_std, ideal_mean + ideal_std, color="#55A868", alpha=0.1)

colors = {"ibm_brisbane": "#4C72B0", "ibm_kyoto": "#8172B2", "ibm_nairobi": "#E8795B", "noisy_extreme": "#C44E52"}
for name, p in backend_deph_p.items():
    ax.axvline(p * 100, color=colors[name], lw=1.2, ls="--", alpha=0.8)
    ax.text(p * 100 + 0.05, 78, name, fontsize=7.5, color=colors[name], rotation=90, va="bottom")

p005_mean = R["dephasing_p005"]["mean"] * 100
p005_std = R["dephasing_p005"]["std"] * 100
ax.annotate(f"p=0,05: {p005_mean:.1f}% +- {p005_std:.1f}%\n(Delta={p005_mean-ideal_mean:+.1f} pp, R=8)",
            xy=(5, p005_mean), xytext=(3.0, 82),
            fontsize=8.5, arrowprops=dict(arrowstyle="->", color="black", lw=1))

ax.set_xlabel("Parametro de dephasing p (%)")
ax.set_ylabel("Acuracia binaria no teste (%)")
ax.set_title("Degradacao do QKSVM sob dephasing (T2), com incerteza\n(R=8 repeticoes, reamostragem completa por repeticao)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
ax.set_ylim(75, 100)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_noise_sweep.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Salvo fig_noise_sweep.png")
