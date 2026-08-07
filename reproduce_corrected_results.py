"""
Reexecucao corrigida do pipeline QML — Netslab-5G-ORAN-IDD
=============================================================
Objetivo: gerar numeros DEFENSAVEIS (com repeticoes / incerteza,
selecao de features documentada e baseline unica e consistente)
para corrigir o artigo, respondendo aos apontamentos do revisor.

Usa o MESMO simulador ZZFeatureMap (statevector exato, NumPy) e os
MESMOS hiperparametros do pipeline original (qml_pipeline.py /
qml_noise_pipeline.py), partindo do dataset ja reduzido
(oran_dataset_reduced.csv, 15 features, 120.000 linhas) que
acompanha o repositorio.
"""
import json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier
from scipy.optimize import minimize

# Este script reproduz a Tabela 3, Tabela 4, curva de aprendizado e varredura de
# dephasing do artigo (paper/sbrc.tex) com repeticoes/incerteza, a partir do
# dataset ja reduzido que acompanha o repositorio (oran_dataset_reduced.csv).
HERE = Path(__file__).resolve().parent
BASE = HERE
OUT_FIG = HERE / "paper" / "figures"
OUT_JSON = HERE / "reproduce_output.json"

RANDOM_STATE = 42
N_QUBITS = 6
DIM = 2 ** N_QUBITS

# ══════════════════════════════════════════════════════════════════
# Simulador (identico ao original)
# ══════════════════════════════════════════════════════════════════
I2 = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def Ry(t):
    a = t / 2
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=complex)

def Rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=complex)

def kron_n(*ops):
    r = ops[0]
    for op in ops[1:]:
        r = np.kron(r, op)
    return r

def cnot_matrix(ctrl, tgt, n):
    dim = 2 ** n
    U = np.eye(dim, dtype=complex)
    for b in range(dim):
        bits = list(format(b, f'0{n}b'))
        if bits[ctrl] == '1':
            bits[tgt] = '1' if bits[tgt] == '0' else '0'
            U[int(''.join(bits), 2), b] = 1
            U[b, b] = 0
    return U

H_ALL = kron_n(*[H] * N_QUBITS)
CNOTS = [cnot_matrix(i, i + 1, N_QUBITS) for i in range(N_QUBITS - 1)]

def zzfeaturemap(x, reps=2):
    s = np.zeros(DIM, dtype=complex); s[0] = 1.0
    for _ in range(reps):
        s = H_ALL @ s
        s = kron_n(*[Rz(2 * x[q]) for q in range(N_QUBITS)]) @ s
        for i in range(N_QUBITS - 1):
            s = CNOTS[i] @ s
            ops = [I2] * N_QUBITS
            ops[i + 1] = Rz(2 * (np.pi - x[i]) * (np.pi - x[i + 1]))
            s = kron_n(*ops) @ s
            s = CNOTS[i] @ s
    return s

def build_kernel_matrix(X1, X2, reps=2):
    phi1 = np.array([zzfeaturemap(x, reps) for x in X1])
    phi2 = np.array([zzfeaturemap(x, reps) for x in X2])
    return np.abs(phi1.conj() @ phi2.T) ** 2

def qik_kernel(X1, X2, gamma=1.0):
    from sklearn.metrics.pairwise import rbf_kernel
    def encode(X): return np.hstack([np.cos(X), np.sin(X), np.cos(2 * X), np.sin(2 * X)])
    return rbf_kernel(encode(X1), encode(X2), gamma=gamma)

def vqc_circuit(x, theta):
    s = zzfeaturemap(x, reps=1)
    n_layers = theta.shape[0]
    for l in range(n_layers):
        for q in range(N_QUBITS):
            ops_y = [I2] * N_QUBITS; ops_y[q] = Ry(theta[l, q, 0])
            s = kron_n(*ops_y) @ s
            ops_z = [I2] * N_QUBITS; ops_z[q] = Rz(theta[l, q, 1])
            s = kron_n(*ops_z) @ s
        for q in range(N_QUBITS):
            s = cnot_matrix(q, (q + 1) % N_QUBITS, N_QUBITS) @ s
    return s

def measure_z0(state):
    ops = [I2] * N_QUBITS; ops[0] = Z
    Z_op = kron_n(*ops)
    return np.real(state.conj() @ Z_op @ state)

def vqc_predict_proba(X, theta):
    return np.array([(1 - measure_z0(vqc_circuit(x, theta))) / 2 for x in X])

def vqc_loss(theta_flat, X, y, n_layers):
    theta = theta_flat.reshape(n_layers, N_QUBITS, 2)
    p = np.clip(vqc_predict_proba(X, theta), 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

print("=" * 70)
print("PARTE 0 — Dados reduzidos (15 features, 120.000 linhas)")
print("=" * 70)
df = pd.read_csv(f"{BASE}/oran_dataset_reduced.csv")
FEATURES_15 = [c for c in df.columns if c not in ("attack_category", "label", "label_bin")]
print(f"  15 features: {FEATURES_15}")
print(f"  Shape: {df.shape}")

# ══════════════════════════════════════════════════════════════════
# PARTE 1 — Selecao TRANSPARENTE das 6 features (RF importance, top-6 das 15)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PARTE 1 — Selecao das 6 features (Random Forest, top-6 das 15)")
print("=" * 70)

rf_sel = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
rf_sel.fit(df[FEATURES_15], df["label"])
imp = pd.Series(rf_sel.feature_importances_, index=FEATURES_15).sort_values(ascending=False)
print(imp.round(4).to_string())
TOP6 = imp.head(6).index.tolist()
print(f"\n  TOP6 selecionadas: {TOP6}")

# ══════════════════════════════════════════════════════════════════
# PARTE 2 — Amostra QML (250/classe = 1500), split 75/25
# ══════════════════════════════════════════════════════════════════
def make_qml_sample(seed, n_per_class=250):
    frames = []
    for cat in df["attack_category"].unique():
        sub = df[df["attack_category"] == cat].sample(n=n_per_class, random_state=seed)
        frames.append(sub)
    dfq = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed)
    le = LabelEncoder().fit(dfq["attack_category"])
    y_multi = le.transform(dfq["attack_category"])
    y_bin = dfq["label_bin"].values
    X_raw = dfq[TOP6].values.astype(float)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_raw)
    return X_scaled, y_multi, y_bin, le, dfq

# duplicidade de registros entre treino/teste (diagnostico para a Secao 3 / limitacao split)
print("\n  Diagnostico de duplicatas (registros com as 15 features identicas):")
dup_mask = df.duplicated(subset=FEATURES_15, keep=False)
print(f"    Registros duplicados (15 features): {dup_mask.sum()} de {len(df)} ({dup_mask.mean()*100:.2f}%)")

# ══════════════════════════════════════════════════════════════════
# PARTE 3 — Tabela 3 corrigida: R repeticoes, media +- desvio, baseline majoritaria, macro-F1
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PARTE 3 — Comparativo QML vs classico, COM repeticoes (R=10)")
print("=" * 70)

R_MAIN = 10
rows = {k: {"acc": [], "f1": [], "f1_macro": [], "auc": []} for k in
        ["QKSVM", "VQC", "QIK-SVM", "SVM-RBF", "RF", "LR", "Majoritaria_bin", "Majoritaria_multi"]}
class_balance_bin = []
vqc_converged = []

R_VQC = 5  # VQC e caro (COBYLA); repetido menos vezes, disclosure explicito no texto
for rep in range(R_MAIN):
    seed = RANDOM_STATE + rep
    X_scaled, y_multi, y_bin, le, dfq = make_qml_sample(seed)
    X_tr, X_te, ym_tr, ym_te, yb_tr, yb_te = train_test_split(
        X_scaled, y_multi, y_bin, test_size=0.25, random_state=seed, stratify=y_multi)

    class_balance_bin.append(yb_te.mean())  # proporcao de "ataque"=1 no teste

    # Majoritaria
    dm_b = DummyClassifier(strategy="most_frequent").fit(X_tr, yb_tr)
    pb = dm_b.predict(X_te)
    rows["Majoritaria_bin"]["acc"].append(accuracy_score(yb_te, pb))
    rows["Majoritaria_bin"]["f1_macro"].append(f1_score(yb_te, pb, average="macro"))
    dm_m = DummyClassifier(strategy="most_frequent").fit(X_tr, ym_tr)
    pm = dm_m.predict(X_te)
    rows["Majoritaria_multi"]["acc"].append(accuracy_score(ym_te, pm))
    rows["Majoritaria_multi"]["f1_macro"].append(f1_score(ym_te, pm, average="macro"))

    # QKSVM
    K_tr = build_kernel_matrix(X_tr, X_tr, reps=2)
    K_te = build_kernel_matrix(X_te, X_tr, reps=2)
    qksvm = SVC(kernel="precomputed", probability=True, C=1.0, random_state=RANDOM_STATE).fit(K_tr, yb_tr)
    qk_pred = qksvm.predict(K_te); qk_proba = qksvm.predict_proba(K_te)[:, 1]
    rows["QKSVM"]["acc"].append(accuracy_score(yb_te, qk_pred))
    rows["QKSVM"]["f1"].append(f1_score(yb_te, qk_pred))
    rows["QKSVM"]["f1_macro"].append(f1_score(yb_te, qk_pred, average="macro"))
    rows["QKSVM"]["auc"].append(roc_auc_score(yb_te, qk_proba))

    # SVM-RBF
    svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE).fit(X_tr, yb_tr)
    sp = svm.predict(X_te); spp = svm.predict_proba(X_te)[:, 1]
    rows["SVM-RBF"]["acc"].append(accuracy_score(yb_te, sp))
    rows["SVM-RBF"]["f1"].append(f1_score(yb_te, sp))
    rows["SVM-RBF"]["f1_macro"].append(f1_score(yb_te, sp, average="macro"))
    rows["SVM-RBF"]["auc"].append(roc_auc_score(yb_te, spp))

    # LR
    lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE).fit(X_tr, yb_tr)
    lp = lr.predict(X_te); lpp = lr.predict_proba(X_te)[:, 1]
    rows["LR"]["acc"].append(accuracy_score(yb_te, lp))
    rows["LR"]["f1"].append(f1_score(yb_te, lp))
    rows["LR"]["f1_macro"].append(f1_score(yb_te, lp, average="macro"))
    rows["LR"]["auc"].append(roc_auc_score(yb_te, lpp))

    # RF multiclasse
    rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1).fit(X_tr, ym_tr)
    rfp = rf.predict(X_te)
    rows["RF"]["acc"].append(accuracy_score(ym_te, rfp))
    rows["RF"]["f1_macro"].append(f1_score(ym_te, rfp, average="macro"))

    # QIK-SVM multiclasse
    Kq_tr = qik_kernel(X_tr, X_tr); Kq_te = qik_kernel(X_te, X_tr)
    qi = SVC(kernel="precomputed", probability=True, random_state=RANDOM_STATE).fit(Kq_tr, ym_tr)
    qip = qi.predict(Kq_te)
    rows["QIK-SVM"]["acc"].append(accuracy_score(ym_te, qip))
    rows["QIK-SVM"]["f1_macro"].append(f1_score(ym_te, qip, average="macro"))

    # VQC (menos repeticoes)
    if rep < R_VQC:
        N_LAYERS = 2; N_VQC = 100
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_tr), N_VQC, replace=False)
        X_vqc = X_tr[idx]; y_vqc = yb_tr[idx]
        theta_init = rng.uniform(0, 2 * np.pi, (N_LAYERS, N_QUBITS, 2))
        t0 = time.time()
        opt = minimize(vqc_loss, theta_init.flatten(), args=(X_vqc, y_vqc, N_LAYERS),
                        method="COBYLA", options={"maxiter": 100, "rhobeg": 0.3})
        theta_opt = opt.x.reshape(N_LAYERS, N_QUBITS, 2)
        vp = vqc_predict_proba(X_te, theta_opt)
        vqc_pred = (vp >= 0.5).astype(int)
        rows["VQC"]["acc"].append(accuracy_score(yb_te, vqc_pred))
        rows["VQC"]["f1"].append(f1_score(yb_te, vqc_pred))
        rows["VQC"]["f1_macro"].append(f1_score(yb_te, vqc_pred, average="macro"))
        rows["VQC"]["auc"].append(roc_auc_score(yb_te, vp))
        vqc_converged.append(bool(opt.success))
        print(f"  [rep {rep}] VQC treinado em {time.time()-t0:.1f}s | conv={opt.success}")

    print(f"  [rep {rep}] QKSVM Acc={rows['QKSVM']['acc'][-1]:.4f} | SVM-RBF Acc={rows['SVM-RBF']['acc'][-1]:.4f}")

def mstd(lst):
    a = np.array(lst)
    return float(a.mean()), float(a.std())

table3 = {}
for k, v in rows.items():
    table3[k] = {m: mstd(vals) for m, vals in v.items() if len(vals) > 0}

print("\n  RESUMO TABELA 3 (media +- desvio, R=%d exceto VQC R=%d):" % (R_MAIN, R_VQC))
for k, v in table3.items():
    print(f"    {k}: {v}")

print(f"\n  Balanceamento binario (proporcao de ataque no teste): "
      f"{np.mean(class_balance_bin)*100:.1f}% +- {np.std(class_balance_bin)*100:.1f}%")
print(f"  VQC convergiu em {sum(vqc_converged)}/{len(vqc_converged)} repeticoes")

# ══════════════════════════════════════════════════════════════════
# PARTE 4 — Curva de aprendizado COM repeticoes
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PARTE 4 — Curva de aprendizado (vantagem de amostragem), R=10")
print("=" * 70)

train_sizes = [30, 50, 80, 120, 180, 250]
R_CURVE = 10
curve = {n: {"qk": [], "rbf": []} for n in train_sizes}

for rep in range(R_CURVE):
    seed = RANDOM_STATE + 100 + rep
    X_scaled, y_multi, y_bin, le, dfq = make_qml_sample(seed)
    X_tr, X_te, ym_tr, ym_te, yb_tr, yb_te = train_test_split(
        X_scaled, y_multi, y_bin, test_size=0.25, random_state=seed, stratify=y_multi)
    rng = np.random.default_rng(seed)
    for n in train_sizes:
        idx = rng.choice(len(X_tr), min(n, len(X_tr)), replace=False)
        Xn, yn = X_tr[idx], yb_tr[idx]
        _K = build_kernel_matrix(Xn, Xn, reps=2); _Kte = build_kernel_matrix(X_te, Xn, reps=2)
        _m = SVC(kernel="precomputed").fit(_K, yn)
        curve[n]["qk"].append(accuracy_score(yb_te, _m.predict(_Kte)))
        _m2 = SVC(kernel="rbf").fit(Xn, yn)
        curve[n]["rbf"].append(accuracy_score(yb_te, _m2.predict(X_te)))
    print(f"  [rep {rep}] curva concluida")

curve_summary = {n: {"qk": mstd(v["qk"]), "rbf": mstd(v["rbf"])} for n, v in curve.items()}
print("\n  n_treino | QKSVM (media+-std) | SVM-RBF (media+-std)")
for n in train_sizes:
    q = curve_summary[n]["qk"]; r = curve_summary[n]["rbf"]
    print(f"  {n:8d} | {q[0]*100:.1f}+-{q[1]*100:.1f}%      | {r[0]*100:.1f}+-{r[1]*100:.1f}%")

# ══════════════════════════════════════════════════════════════════
# PARTE 5 — Ruido: baseline UNICA e consistente + dephasing com repeticoes
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PARTE 5 — Modelos de ruido: baseline consistente + dephasing com CI")
print("=" * 70)

N_GATES = 20
T_GATE = 0.1
IBM_BACKENDS = {
    "ibm_brisbane_2024": {"depol_p": 0.001, "dephasing_p": 1 - np.exp(-T_GATE/300),
                           "amp_damp_p": 1 - np.exp(-T_GATE/400), "n_shots": 4096},
    "ibm_kyoto_2023":    {"depol_p": 0.003, "dephasing_p": 1 - np.exp(-T_GATE/150),
                           "amp_damp_p": 1 - np.exp(-T_GATE/200), "n_shots": 4096},
    "ibm_nairobi_2022":  {"depol_p": 0.008, "dephasing_p": 1 - np.exp(-T_GATE/80),
                           "amp_damp_p": 1 - np.exp(-T_GATE/100), "n_shots": 1024},
    "ibm_noisy_extreme": {"depol_p": 0.02,  "dephasing_p": 1 - np.exp(-T_GATE/20),
                           "amp_damp_p": 1 - np.exp(-T_GATE/30),  "n_shots": 512},
}

def noise_depolarizing(K, p, n_gates=N_GATES):
    alpha = (1 - 4*p/3) ** n_gates
    Kn = alpha * K + (1 - alpha) * np.full_like(K, 1.0/DIM)
    np.fill_diagonal(Kn, 1.0)
    return np.clip(Kn, 0, 1)

def noise_dephasing(K, p, n_gates=N_GATES):
    decay = np.exp(-p * n_gates)
    Kn = decay * K.copy()
    np.fill_diagonal(Kn, 1.0)
    return np.clip(Kn, 0, 1)

def noise_amplitude_damping(K, gamma, n_gates=N_GATES):
    fid = (1 - gamma/2) ** n_gates
    Kn = fid * K + (1 - fid) * np.full_like(K, 1.0/DIM)
    np.fill_diagonal(Kn, 1.0)
    return np.clip(Kn, 0, 1)

def noise_shot(K, n_shots, rng):
    std = np.sqrt(np.clip(K*(1-K), 0, 1) / n_shots)
    Kn = np.clip(K + rng.normal(0, std), 0, 1)
    if Kn.shape[0] == Kn.shape[1]:
        Kn = (Kn + Kn.T) / 2
    np.fill_diagonal(Kn, 1.0)
    return Kn

def noise_combined(K, p, rng):
    Kn = noise_depolarizing(K, p["depol_p"])
    Kn = noise_dephasing(Kn, p["dephasing_p"])
    Kn = noise_amplitude_damping(Kn, p["amp_damp_p"])
    Kn = noise_shot(Kn, p["n_shots"], rng)
    return Kn

def run_svm(K_tr, K_te, y_tr, y_te, multi=False):
    svm = SVC(kernel="precomputed", C=1.0, random_state=RANDOM_STATE).fit(K_tr, y_tr)
    pred = svm.predict(K_te)
    avg = "macro" if multi else "binary"
    return accuracy_score(y_te, pred), f1_score(y_te, pred, average=avg)

# --- baseline UNICA sem ruido algum (nem shot noise), repetida R vezes p/ IC ---
R_NOISE = 8
NOISE_LEVELS = np.linspace(0, 0.05, 15)
ideal_acc_runs, ideal_acc_m_runs = [], []
dephasing_sweep_runs = {p: [] for p in NOISE_LEVELS}
backend_runs = {b: {"acc_b": [], "acc_m": []} for b in IBM_BACKENDS}
N_SAMPLES_NOISE = 100  # 100/classe x 6 classes = 600, igual ao qml_noise_pipeline.py original

for rep in range(R_NOISE):
    seed = RANDOM_STATE + 200 + rep
    frames = [df[df["attack_category"] == cat].sample(n=N_SAMPLES_NOISE, random_state=seed)
              for cat in df["attack_category"].unique()]
    dfq = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed)
    le = LabelEncoder().fit(dfq["attack_category"])
    y_multi = le.transform(dfq["attack_category"]); y_bin = dfq["label_bin"].values
    X_raw = dfq[TOP6].values.astype(float)
    X_scaled = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_raw)
    X_tr, X_te, ym_tr, ym_te, yb_tr, yb_te = train_test_split(
        X_scaled, y_multi, y_bin, test_size=0.25, random_state=seed, stratify=y_multi)

    K_tr_ideal = build_kernel_matrix(X_tr, X_tr, reps=2)
    K_te_ideal = build_kernel_matrix(X_te, X_tr, reps=2)

    a_ideal, _ = run_svm(K_tr_ideal, K_te_ideal, yb_tr, yb_te)
    a_ideal_m, _ = run_svm(K_tr_ideal, K_te_ideal, ym_tr, ym_te, multi=True)
    ideal_acc_runs.append(a_ideal); ideal_acc_m_runs.append(a_ideal_m)

    for p in NOISE_LEVELS:
        Kn_tr = noise_dephasing(K_tr_ideal, p); Kn_te = noise_dephasing(K_te_ideal, p)
        a, _ = run_svm(Kn_tr, Kn_te, yb_tr, yb_te)
        dephasing_sweep_runs[p].append(a)

    rng_b = np.random.default_rng(seed)
    for bname, bp in IBM_BACKENDS.items():
        Kn_tr = noise_combined(K_tr_ideal, bp, rng_b)
        Kn_te = noise_combined(K_te_ideal, bp, rng_b)
        a_b, _ = run_svm(Kn_tr, Kn_te, yb_tr, yb_te)
        a_m, _ = run_svm(Kn_tr, Kn_te, ym_tr, ym_te, multi=True)
        backend_runs[bname]["acc_b"].append(a_b)
        backend_runs[bname]["acc_m"].append(a_m)

    print(f"  [rep {rep}] baseline ideal bin={a_ideal:.4f} multi={a_ideal_m:.4f}")

ideal_mean, ideal_std = mstd(ideal_acc_runs)
ideal_m_mean, ideal_m_std = mstd(ideal_acc_m_runs)
print(f"\n  BASELINE UNICA (sem ruido, R={R_NOISE}): bin={ideal_mean*100:.2f}+-{ideal_std*100:.2f}% "
      f"| multi={ideal_m_mean*100:.2f}+-{ideal_m_std*100:.2f}%")

dephasing_summary = {float(p): mstd(v) for p, v in dephasing_sweep_runs.items()}
p005 = min(dephasing_summary.keys(), key=lambda x: abs(x - 0.05))
print(f"  Dephasing p={p005:.4f}: acc={dephasing_summary[p005][0]*100:.2f}% "
      f"+- {dephasing_summary[p005][1]*100:.2f}% (R={R_NOISE}) | Delta vs ideal = "
      f"{(dephasing_summary[p005][0]-ideal_mean)*100:+.2f} pp")

backend_summary = {}
for bname, v in backend_runs.items():
    ab_m, ab_s = mstd(v["acc_b"]); am_m, am_s = mstd(v["acc_m"])
    backend_summary[bname] = {
        "acc_b": (ab_m, ab_s), "acc_m": (am_m, am_s),
        "delta_b_pp": (ab_m - ideal_mean) * 100,
        "delta_m_pp": (am_m - ideal_m_mean) * 100,
    }
    print(f"  {bname}: bin={ab_m*100:.2f}+-{ab_s*100:.2f}% (D={backend_summary[bname]['delta_b_pp']:+.2f}pp) | "
          f"multi={am_m*100:.2f}+-{am_s*100:.2f}% (D={backend_summary[bname]['delta_m_pp']:+.2f}pp)")

# ══════════════════════════════════════════════════════════════════
# Salvar tudo em JSON para consumo no texto do artigo
# ══════════════════════════════════════════════════════════════════
out = {
    "features_15": FEATURES_15,
    "top6_importance": imp.round(4).to_dict(),
    "top6": TOP6,
    "duplicates_15feat_pct": float(dup_mask.mean() * 100),
    "duplicates_15feat_n": int(dup_mask.sum()),
    "table3": table3,
    "class_balance_bin_pct": [float(x*100) for x in class_balance_bin],
    "vqc_converged": vqc_converged,
    "r_main": R_MAIN, "r_vqc": R_VQC, "r_curve": R_CURVE, "r_noise": R_NOISE,
    "learning_curve": {str(n): curve_summary[n] for n in train_sizes},
    "learning_curve_raw": {str(n): curve[n] for n in train_sizes},
    "noise_ideal_baseline": {"bin": [ideal_mean, ideal_std], "multi": [ideal_m_mean, ideal_m_std]},
    "dephasing_sweep": {str(k): v for k, v in dephasing_summary.items()},
    "dephasing_p005": {"p": p005, "mean": dephasing_summary[p005][0], "std": dephasing_summary[p005][1]},
    "backend_summary": backend_summary,
}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSalvo: {OUT_JSON}")
print("\nCONCLUIDO.")
