"""
Pipeline QML — Netslab-5G-ORAN-IDD
====================================
Técnicas implementadas from-scratch (NumPy/SciPy, statevector exato):
  1. Simulador de circuitos quânticos
  2. ZZFeatureMap + Quantum Kernel SVM  (QKSVM)
  3. Variational Quantum Classifier     (VQC)
  4. Quantum-Inspired Kernel SVM        (QIK-SVM)
  5. Comparação com SVM-RBF, LR, RF
  6. Curva de aprendizado, ROC, análise do espaço quântico
"""
import warnings, os, time
warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.decomposition import PCA
from matplotlib.patches import Patch

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# A — PRIMITIVAS DO SIMULADOR QUÂNTICO
# ══════════════════════════════════════════════════════════════════════════════
N_QUBITS = 6
DIM      = 2**N_QUBITS   # 64

I2 = np.eye(2, dtype=complex)
H  = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
Z  = np.array([[1,0],[0,-1]], dtype=complex)

def Rx(t): a=t/2; return np.array([[np.cos(a), -1j*np.sin(a)],[-1j*np.sin(a), np.cos(a)]], dtype=complex)
def Ry(t): a=t/2; return np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]], dtype=complex)
def Rz(t): return np.array([[np.exp(-1j*t/2),0],[0,np.exp(1j*t/2)]], dtype=complex)

def kron_n(*ops):
    r = ops[0]
    for op in ops[1:]: r = np.kron(r, op)
    return r

def cnot_matrix(ctrl, tgt, n):
    dim = 2**n; U = np.eye(dim, dtype=complex)
    for b in range(dim):
        bits = list(format(b, f'0{n}b'))
        if bits[ctrl] == '1':
            bits[tgt] = '1' if bits[tgt]=='0' else '0'
            U[int(''.join(bits),2), b] = 1; U[b,b] = 0
    return U

# Pré-computar gates independentes de x (feito uma vez)
H_ALL   = kron_n(*[H]*N_QUBITS)
CNOTS   = [cnot_matrix(i, i+1, N_QUBITS) for i in range(N_QUBITS-1)]

print("=" * 65)
print("PARTE A — Simulador inicializado")
print(f"  N_QUBITS={N_QUBITS} | Hilbert dim={DIM}")
print(f"  Gates pré-computados: H_ALL (64×64), {len(CNOTS)} CNOTs")

# ══════════════════════════════════════════════════════════════════════════════
# B — ZZFEATUREMAP E QUANTUM KERNEL (vetorizados)
# ══════════════════════════════════════════════════════════════════════════════
def zzfeaturemap(x, reps=2):
    """ZZFeatureMap: H → Rz(2xi) → CNOT·Rz(ZZ)·CNOT para cada par vizinho."""
    s = np.zeros(DIM, dtype=complex); s[0] = 1.0
    for _ in range(reps):
        s = H_ALL @ s
        s = kron_n(*[Rz(2*x[q]) for q in range(N_QUBITS)]) @ s
        for i in range(N_QUBITS-1):
            s = CNOTS[i] @ s
            ops = [I2]*N_QUBITS; ops[i+1] = Rz(2*(np.pi-x[i])*(np.pi-x[i+1]))
            s = kron_n(*ops) @ s
            s = CNOTS[i] @ s
    return s

def build_kernel_matrix(X1, X2, reps=2):
    """K[i,j] = |⟨φ(X1[i])|φ(X2[j])⟩|² — vetorizado via batch dot product."""
    phi1 = np.array([zzfeaturemap(x, reps) for x in X1])  # (n1, 64)
    phi2 = np.array([zzfeaturemap(x, reps) for x in X2])  # (n2, 64)
    return np.abs(phi1.conj() @ phi2.T)**2

# ══════════════════════════════════════════════════════════════════════════════
# C — VQC (Variational Quantum Classifier)
# ══════════════════════════════════════════════════════════════════════════════
def vqc_circuit(x, theta):
    """Ansatz: ZZFeatureMap(reps=1) + camadas RyRz + CNOT ring."""
    s = zzfeaturemap(x, reps=1)
    n_layers = theta.shape[0]
    for l in range(n_layers):
        for q in range(N_QUBITS):
            ops_y = [I2]*N_QUBITS; ops_y[q] = Ry(theta[l,q,0])
            s = kron_n(*ops_y) @ s
            ops_z = [I2]*N_QUBITS; ops_z[q] = Rz(theta[l,q,1])
            s = kron_n(*ops_z) @ s
        # CNOT ring
        for q in range(N_QUBITS):
            s = cnot_matrix(q, (q+1)%N_QUBITS, N_QUBITS) @ s
    return s

def measure_z0(state):
    """⟨Z⟩ no qubit 0."""
    ops = [I2]*N_QUBITS; ops[0] = Z
    Z_op = kron_n(*ops)
    return np.real(state.conj() @ Z_op @ state)

def vqc_predict_proba(X, theta):
    return np.array([(1 - measure_z0(vqc_circuit(x, theta)))/2 for x in X])

def vqc_loss(theta_flat, X, y):
    theta = theta_flat.reshape(N_LAYERS, N_QUBITS, 2)
    p = np.clip(vqc_predict_proba(X, theta), 1e-7, 1-1e-7)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

# ══════════════════════════════════════════════════════════════════════════════
# D — QUANTUM-INSPIRED KERNEL (escalável)
# ══════════════════════════════════════════════════════════════════════════════
def qik_kernel(X1, X2, gamma=1.0):
    """Fourier quantum encoding: φ(x) = [cos(x), sin(x), cos(2x), sin(2x)]."""
    from sklearn.metrics.pairwise import rbf_kernel
    def encode(X): return np.hstack([np.cos(X), np.sin(X), np.cos(2*X), np.sin(2*X)])
    return rbf_kernel(encode(X1), encode(X2), gamma=gamma)

# ══════════════════════════════════════════════════════════════════════════════
# E — DADOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PARTE E — Dados")
df = pd.read_csv("/mnt/user-data/outputs/oran_dataset_reduced.csv")

N_SAMPLES = 250   # por classe — viável com kernel vetorizado
TOP6 = ["dst_port","conn_state_enc","log_dst_bytes",
        "duration","byte_ratio","bytes_per_pkt_dst"]

frames = []
for cat in df["attack_category"].unique():
    sub = df[df["attack_category"]==cat].sample(n=N_SAMPLES, random_state=RANDOM_STATE)
    frames.append(sub)
df_qml = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)

X_raw  = df_qml[TOP6].values.astype(float)
le     = LabelEncoder().fit(df_qml["attack_category"])
y_multi = le.transform(df_qml["attack_category"])
y_bin   = df_qml["label_bin"].values
class_names = le.classes_

scaler  = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X_raw)

X_tr, X_te, ym_tr, ym_te, yb_tr, yb_te = train_test_split(
    X_scaled, y_multi, y_bin,
    test_size=0.25, random_state=RANDOM_STATE, stratify=y_multi)

print(f"  Total: {len(df_qml)} | Treino: {len(X_tr)} | Teste: {len(X_te)}")
print(f"  Qubits: {N_QUBITS} | Hilbert: {DIM} dimensões")

# ══════════════════════════════════════════════════════════════════════════════
# F — TREINAMENTO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PARTE F — Treinamento")
results = {}

# F1 — QKSVM binário
print(f"\n  [1/5] QKSVM — construindo kernel {len(X_tr)}×{len(X_tr)}...")
t0 = time.time()
K_tr = build_kernel_matrix(X_tr, X_tr, reps=2)
K_te = build_kernel_matrix(X_te, X_tr, reps=2)
t_k  = time.time()-t0
qksvm = SVC(kernel="precomputed", probability=True, C=1.0, random_state=RANDOM_STATE)
qksvm.fit(K_tr, yb_tr)
qk_pred  = qksvm.predict(K_te)
qk_proba = qksvm.predict_proba(K_te)[:,1]
results["QKSVM"] = dict(acc=accuracy_score(yb_te,qk_pred), f1=f1_score(yb_te,qk_pred),
    auc=roc_auc_score(yb_te,qk_proba), time=t_k, type="quantum", task="bin",
    pred=qk_pred, proba=qk_proba)
print(f"        Kernel em {t_k:.1f}s | Acc={results['QKSVM']['acc']:.4f} "
      f"| F1={results['QKSVM']['f1']:.4f} | AUC={results['QKSVM']['auc']:.4f}")

# F2 — QIK-SVM multiclasse
print("\n  [2/5] QIK-SVM multiclasse (Quantum-Inspired Kernel)...")
t0 = time.time()
Kq_tr = qik_kernel(X_tr, X_tr); Kq_te = qik_kernel(X_te, X_tr)
t_qik = time.time()-t0
qiksvm = SVC(kernel="precomputed", probability=True, random_state=RANDOM_STATE)
qiksvm.fit(Kq_tr, ym_tr)
qi_pred = qiksvm.predict(Kq_te)
qi_proba_all = qiksvm.predict_proba(Kq_te)
results["QIK-SVM"] = dict(acc=accuracy_score(ym_te,qi_pred),
    f1=f1_score(ym_te,qi_pred,average="macro"), auc=None,
    time=t_qik, type="quantum-inspired", task="multi", pred=qi_pred)
print(f"        {t_qik:.2f}s | Acc={results['QIK-SVM']['acc']:.4f} | F1={results['QIK-SVM']['f1']:.4f}")

# F3 — VQC binário
print("\n  [3/5] VQC (Variational Quantum Classifier)...")
N_LAYERS = 2
N_VQC    = 100   # subconjunto para VQC
idx = np.random.choice(len(X_tr), N_VQC, replace=False)
X_vqc = X_tr[idx]; y_vqc = yb_tr[idx]
theta_init = np.random.uniform(0, 2*np.pi, (N_LAYERS, N_QUBITS, 2))
loss_history = []
def vqc_loss_tracked(tf, X, y):
    l = vqc_loss(tf, X, y)
    loss_history.append(l)
    if len(loss_history) % 20 == 0: print(f"        iter {len(loss_history)} | loss={l:.4f}")
    return l
t0 = time.time()
opt = minimize(vqc_loss_tracked, theta_init.flatten(), args=(X_vqc, y_vqc),
               method="COBYLA", options={"maxiter":100,"rhobeg":0.3})
t_vqc = time.time()-t0
theta_opt = opt.x.reshape(N_LAYERS, N_QUBITS, 2)
vp = vqc_predict_proba(X_te, theta_opt)
vqc_pred = (vp >= 0.5).astype(int)
results["VQC"] = dict(acc=accuracy_score(yb_te,vqc_pred), f1=f1_score(yb_te,vqc_pred),
    auc=roc_auc_score(yb_te,vp), time=t_vqc, type="quantum", task="bin",
    pred=vqc_pred, proba=vp, loss_history=loss_history, converged=opt.success)
print(f"        {t_vqc:.1f}s | Acc={results['VQC']['acc']:.4f} "
      f"| F1={results['VQC']['f1']:.4f} | AUC={results['VQC']['auc']:.4f} | Conv={opt.success}")

# F4 — SVM-RBF
print("\n  [4/5] SVM-RBF clássico...")
t0=time.time()
svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE).fit(X_tr, yb_tr)
sp=svm.predict(X_te); spp=svm.predict_proba(X_te)[:,1]; t_svm=time.time()-t0
results["SVM-RBF"] = dict(acc=accuracy_score(yb_te,sp), f1=f1_score(yb_te,sp),
    auc=roc_auc_score(yb_te,spp), time=t_svm, type="classico", task="bin", pred=sp, proba=spp)
print(f"        {t_svm:.3f}s | Acc={results['SVM-RBF']['acc']:.4f} | F1={results['SVM-RBF']['f1']:.4f}")

# F5 — RF e LR
print("\n  [5/5] RF + LR clássicos...")
t0=time.time()
rf=RandomForestClassifier(50,max_depth=8,random_state=RANDOM_STATE).fit(X_tr,ym_tr)
rfp=rf.predict(X_te); t_rf=time.time()-t0
results["RF"] = dict(acc=accuracy_score(ym_te,rfp), f1=f1_score(ym_te,rfp,average="macro"),
    auc=None, time=t_rf, type="classico", task="multi", pred=rfp)
t0=time.time()
lr=LogisticRegression(max_iter=500,random_state=RANDOM_STATE).fit(X_tr,yb_tr)
lrp=lr.predict(X_te); lrpp=lr.predict_proba(X_te)[:,1]; t_lr=time.time()-t0
results["LR"] = dict(acc=accuracy_score(yb_te,lrp), f1=f1_score(yb_te,lrp),
    auc=roc_auc_score(yb_te,lrpp), time=t_lr, type="classico", task="bin", pred=lrp, proba=lrpp)
print(f"        RF: Acc={results['RF']['acc']:.4f} | LR: Acc={results['LR']['acc']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# G — CURVA DE APRENDIZADO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PARTE G — Curva de aprendizado")
train_sizes=[30,50,80,120,180,250]
qk_scores, rbf_scores, qk_times, rbf_times = [],[],[],[]
for n in train_sizes:
    idx=np.random.choice(len(X_tr), min(n,len(X_tr)), replace=False)
    Xn,yn=X_tr[idx],yb_tr[idx]
    t0=time.time()
    _K=build_kernel_matrix(Xn,Xn,reps=2); _Kte=build_kernel_matrix(X_te,Xn,reps=2)
    _m=SVC(kernel="precomputed").fit(_K,yn)
    qk_scores.append(accuracy_score(yb_te,_m.predict(_Kte))); qk_times.append(time.time()-t0)
    t0=time.time()
    _m2=SVC(kernel="rbf").fit(Xn,yn)
    rbf_scores.append(accuracy_score(yb_te,_m2.predict(X_te))); rbf_times.append(time.time()-t0)
    print(f"  n={n:3d} | QKSVM={qk_scores[-1]:.3f} ({qk_times[-1]:.1f}s) | RBF={rbf_scores[-1]:.3f} ({rbf_times[-1]:.3f}s)")

# ══════════════════════════════════════════════════════════════════════════════
# H — PCA DO ESPAÇO QUÂNTICO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PARTE H — PCA do espaço quântico")
N_VIS=20
quantum_states, labels_vis = [], []
for cls in sorted(df_qml["attack_category"].unique()):
    sub=df_qml[df_qml["attack_category"]==cls].head(N_VIS)
    for _,row in sub.iterrows():
        x=scaler.transform([row[TOP6].values.astype(float)])[0]
        quantum_states.append(np.abs(zzfeaturemap(x,reps=2))**2)
        labels_vis.append(cls)
quantum_states=np.array(quantum_states)
pca=PCA(n_components=2)
q2d=pca.fit_transform(quantum_states)
print(f"  Variância explicada (2D): {pca.explained_variance_ratio_.sum():.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# I — FIGURAS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PARTE I — Gerando figuras")
sns.set_style("whitegrid")
PAL    = sns.color_palette("muted",8)
Q_C    = "#4C72B0"
C_C    = "#DD8452"
QI_C   = "#55A868"
V_C    = "#8172B2"
type_color = {"quantum":Q_C,"quantum-inspired":QI_C,"classico":C_C}

# ── Fig 1: Circuito esquemático + distribuição kernel + PCA ──────────────────
fig1 = plt.figure(figsize=(18,10))
fig1.suptitle("QML para IDS em O-RAN — Arquitetura e Espaço Quântico",
              fontsize=14, fontweight="bold")
gs1 = gridspec.GridSpec(2,3, figure=fig1, hspace=0.45, wspace=0.35)
ax_c  = fig1.add_subplot(gs1[0,:2])
ax_hb = fig1.add_subplot(gs1[0,2])
ax_km = fig1.add_subplot(gs1[1,0])
ax_pca= fig1.add_subplot(gs1[1,1])
ax_ov = fig1.add_subplot(gs1[1,2])

# Circuito esquemático
ax_c.set_xlim(0,10); ax_c.set_ylim(-0.6,N_QUBITS-0.4); ax_c.axis("off")
ax_c.set_title(f"ZZFeatureMap (reps=2) + Ansatz VQC — {N_QUBITS} qubits", fontsize=10)
blues = plt.cm.Blues(np.linspace(0.45,0.85,N_QUBITS))
stages = [
    (0.15, 0.55, "H",       "#4C72B0", "white"),
    (0.85, 0.9,  "Rz(2xᵢ)","#E8795B", "white"),
    (2.5,  1.3,  "ZZᵢ,ᵢ₊₁","#2a6099", "white"),
    (4.3,  1.0,  "Ry/Rz(θ)","#55A868", "white"),
    (5.8,  0.9,  "CNOT⊕",  "#C44E52", "white"),
    (7.0,  0.7,  "⟨Z⟩",    "#9C755F", "white"),
]
for q in range(N_QUBITS):
    ax_c.axhline(y=q, color="gray", lw=0.8, alpha=0.3)
    ax_c.text(-0.1, q, f"q{q}|0⟩", ha="right", va="center", fontsize=8)
for (xpos, width, label, fc, tc) in stages:
    for q in range(N_QUBITS):
        r=plt.Rectangle((xpos,q-0.21),width,0.42,facecolor=fc,edgecolor="k",lw=0.7)
        ax_c.add_patch(r)
        ax_c.text(xpos+width/2,q,label,ha="center",va="center",fontsize=7,color=tc)
    ax_c.text(xpos+width/2,-0.55,label,ha="center",fontsize=7.5,color="gray",style="italic")
# ZZ ligações
for q in range(N_QUBITS-1):
    ax_c.plot([2.2,2.2],[q,q+1],"o-",color=Q_C,lw=2,ms=6)

# Hilbert dim
nq_r=np.arange(1,13)
ax_hb.semilogy(nq_r, 2**nq_r, "o-", color=Q_C, lw=2)
ax_hb.axvline(N_QUBITS, color="red", ls="--", alpha=0.7, label=f"{N_QUBITS} qubits (este trabalho)")
ax_hb.set_xlabel("Nº de qubits"); ax_hb.set_ylabel("Dim. espaço de Hilbert")
ax_hb.set_title("Crescimento exponencial\ndo espaço de Hilbert"); ax_hb.legend(fontsize=8)

# Kernel matrix
n_show = min(80, len(K_tr))
im=ax_km.imshow(K_tr[:n_show,:n_show], cmap="viridis", aspect="auto")
plt.colorbar(im, ax=ax_km)
ax_km.set_title(f"Quantum Kernel Matrix\nK[i,j]=|⟨φ(xᵢ)|φ(xⱼ)⟩|²  ({n_show}×{n_show})")
ax_km.set_xlabel("Amostras de treino"); ax_km.set_ylabel("Amostras de treino")

# PCA
uniq = sorted(set(labels_vis))
for i, cls in enumerate(uniq):
    m=[l==cls for l in labels_vis]; pts=q2d[m]
    ax_pca.scatter(pts[:,0],pts[:,1],label=cls,alpha=0.75,color=PAL[i],s=40,edgecolors="white",lw=0.3)
ax_pca.set_title(f"Espaço quântico 2D (PCA)\n{pca.explained_variance_ratio_.sum():.1%} variância")
ax_pca.legend(fontsize=7); ax_pca.set_xlabel("PC1"); ax_pca.set_ylabel("PC2")

# Overview resultados
names_ov = list(results.keys())
accs_ov  = [results[m]["acc"]*100 for m in names_ov]
cols_ov  = [type_color[results[m]["type"]] for m in names_ov]
bars=ax_ov.bar(names_ov, accs_ov, color=cols_ov, edgecolor="black", lw=0.6)
ax_ov.set_ylim(0,115); ax_ov.set_ylabel("Accuracy (%)"); ax_ov.set_title("Comparativo geral")
ax_ov.tick_params(axis="x", rotation=25, labelsize=8)
for b,a in zip(bars,accs_ov):
    ax_ov.text(b.get_x()+b.get_width()/2, a+1.2, f"{a:.1f}%", ha="center", va="bottom", fontsize=8)
leg=[Patch(facecolor=Q_C,label="Quântico"),Patch(facecolor=QI_C,label="Quantum-inspired"),
     Patch(facecolor=C_C,label="Clássico")]
ax_ov.legend(handles=leg, fontsize=8)

fig1.savefig(os.path.join(OUTPUT_DIR,"qml_fig1_arquitetura.png"),dpi=120,bbox_inches="tight")
plt.close(fig1); print("  Salvo: qml_fig1_arquitetura.png")

# ── Fig 2: Curva de aprendizado + ROC + Loss VQC + F1 vs Tempo ───────────────
fig2, axes2 = plt.subplots(2,2, figsize=(16,12))
fig2.suptitle("QML — Análise de Desempenho Detalhada", fontsize=14, fontweight="bold")

# Curva de aprendizado
ax=axes2[0,0]
ax.plot(train_sizes,[s*100 for s in qk_scores],"o-",color=Q_C,lw=2,label="QKSVM")
ax.plot(train_sizes,[s*100 for s in rbf_scores],"s--",color=C_C,lw=2,label="SVM-RBF")
ax.fill_between(train_sizes,[s*100 for s in qk_scores],[s*100 for s in rbf_scores],alpha=0.1,color=Q_C)
ax.set_xlabel("Tamanho do conjunto de treino"); ax.set_ylabel("Accuracy no teste (%)")
ax.set_title("Curva de Aprendizado\n(QKSVM vs SVM-RBF)"); ax.legend(); ax.grid(alpha=0.3)

# ROC
ax=axes2[0,1]
roc_models=[("QKSVM",Q_C,"-"),("VQC",V_C,"-."),("SVM-RBF",C_C,"--"),("LR",PAL[3],":")]
for nm,col,ls in roc_models:
    if results[nm].get("proba") is not None:
        fpr,tpr,_=roc_curve(yb_te,results[nm]["proba"])
        ax.plot(fpr,tpr,color=col,lw=2,ls=ls,label=f"{nm} (AUC={results[nm]['auc']:.3f})")
ax.plot([0,1],[0,1],"k--",alpha=0.3); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("Curva ROC — Benigno vs Ataque"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Loss VQC
ax=axes2[1,0]
lh=results["VQC"]["loss_history"]
ax.plot(range(len(lh)),lh,color=V_C,lw=2)
ax.axhline(lh[-1],color="gray",ls="--",alpha=0.6,label=f"Loss final={lh[-1]:.4f}")
ax.set_xlabel("Iteração COBYLA"); ax.set_ylabel("Binary Cross-Entropy")
ax.set_title(f"Convergência do VQC\n({N_LAYERS} camadas, {N_VQC} amostras, {N_QUBITS} qubits)")
ax.legend(); ax.grid(alpha=0.3)

# F1 vs Tempo
ax=axes2[1,1]
for nm in results:
    r=results[nm]; col=type_color[r["type"]]
    ax.scatter(np.log1p(r["time"]), r["f1"]*100, color=col, s=130, edgecolors="black", lw=0.7, zorder=3)
    ax.annotate(nm,(np.log1p(r["time"]),r["f1"]*100+0.4),ha="center",fontsize=8)
ax.set_xlabel("log(1 + Tempo) [s]"); ax.set_ylabel("F1-score (%)")
ax.set_title("Trade-off: Qualidade vs Custo Computacional"); ax.grid(alpha=0.3)
leg2=[Patch(facecolor=Q_C,label="Quântico"),Patch(facecolor=QI_C,label="Quantum-inspired"),Patch(facecolor=C_C,label="Clássico")]
ax.legend(handles=leg2, fontsize=9)

plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR,"qml_fig2_desempenho.png"),dpi=120,bbox_inches="tight")
plt.close(fig2); print("  Salvo: qml_fig2_desempenho.png")

# ── Fig 3: Kernel distributions + confusion matrices ─────────────────────────
fig3, axes3 = plt.subplots(1,3, figsize=(18,6))
fig3.suptitle("QML — Análise do Kernel Quântico e Matrizes de Confusão", fontsize=14, fontweight="bold")

# Distribuição do kernel por par de classes
n_s=min(len(K_tr),120); K_s=K_tr[:n_s,:n_s]; lb_s=yb_tr[:n_s]
same,diff=[],[]
for i in range(n_s):
    for j in range(i+1,n_s):
        (same if lb_s[i]==lb_s[j] else diff).append(K_s[i,j])
axes3[0].hist(same,bins=35,alpha=0.65,color=Q_C,label="Mesma classe",density=True)
axes3[0].hist(diff,bins=35,alpha=0.65,color=C_C,label="Classes distintas",density=True)
axes3[0].set_title("Distribuição K(xᵢ,xⱼ)\npor par de classes (benigno vs ataque)")
axes3[0].set_xlabel("Fidelidade quântica |⟨φ(xᵢ)|φ(xⱼ)⟩|²")
axes3[0].set_ylabel("Densidade"); axes3[0].legend()

# Confusion matrices
for ax,nm,pred in [(axes3[1],"QKSVM",qk_pred),(axes3[2],"SVM-RBF",results["SVM-RBF"]["pred"])]:
    cm=confusion_matrix(yb_te,pred)
    cm_p=cm/cm.sum(axis=1,keepdims=True)*100
    sns.heatmap(cm_p,ax=ax,annot=True,fmt=".1f",cmap="Blues",
                xticklabels=["Benigno","Ataque"],yticklabels=["Benigno","Ataque"],
                annot_kws={"size":13})
    ax.set_title(f"Matriz de Confusão (%) — {nm}"); ax.set_xlabel("Predito"); ax.set_ylabel("Real")

plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR,"qml_fig3_kernel_cm.png"),dpi=120,bbox_inches="tight")
plt.close(fig3); print("  Salvo: qml_fig3_kernel_cm.png")

# ── Fig 4: QIK-SVM multiclasse ───────────────────────────────────────────────
fig4, axes4 = plt.subplots(1,2, figsize=(14,6))
fig4.suptitle("QIK-SVM Multiclasse (6 classes de ataque)", fontsize=14, fontweight="bold")

cm_qi=confusion_matrix(ym_te,qi_pred)
cm_qi_p=cm_qi/cm_qi.sum(axis=1,keepdims=True)*100
sns.heatmap(cm_qi_p,ax=axes4[0],annot=True,fmt=".1f",cmap="Blues",
            xticklabels=class_names,yticklabels=class_names,annot_kws={"size":9})
axes4[0].set_title("QIK-SVM — Confusion Matrix (%)"); axes4[0].set_xlabel("Predito")
axes4[0].set_ylabel("Real"); axes4[0].tick_params(axis="x",rotation=30,labelsize=8)

# F1 por classe: QIK vs RF
from sklearn.metrics import f1_score as f1_per
f1_qi=f1_score(ym_te,qi_pred,average=None); f1_rf=f1_score(ym_te,rfp,average=None)
x_cls=np.arange(len(class_names))
axes4[1].bar(x_cls-0.2,f1_qi*100,0.4,label="QIK-SVM",color=QI_C,edgecolor="k",lw=0.5)
axes4[1].bar(x_cls+0.2,f1_rf*100,0.4,label="RF (clássico)",color=C_C,edgecolor="k",lw=0.5)
axes4[1].set_xticks(x_cls); axes4[1].set_xticklabels(class_names,rotation=25,fontsize=9)
axes4[1].set_ylabel("F1-score (%)"); axes4[1].set_title("F1 por classe: QIK-SVM vs RF")
axes4[1].legend(); axes4[1].set_ylim(0,115); axes4[1].grid(alpha=0.3,axis="y")

plt.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR,"qml_fig4_multiclasse.png"),dpi=120,bbox_inches="tight")
plt.close(fig4); print("  Salvo: qml_fig4_multiclasse.png")

# ══════════════════════════════════════════════════════════════════════════════
# J — RELATÓRIO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("RELATÓRIO FINAL")
print("="*65)

sep="="*60
rpt=[
"RELATÓRIO — Pipeline QML | Netslab-5G-ORAN-IDD", sep,"",
"CONFIGURAÇÃO DO EXPERIMENTO",
f"  Qubits:       {N_QUBITS}  |  Hilbert dim: {DIM}",
f"  Feature map:  ZZFeatureMap (reps=2) — interações ZZ entre qubits vizinhos",
f"  Ansatz VQC:   {N_LAYERS} camadas RyRz + CNOT ring | otimizador: COBYLA",
f"  Amostras:     {len(df_qml)} ({N_SAMPLES}/classe) | Treino={len(X_tr)} | Teste={len(X_te)}",
f"  Features:     {', '.join(TOP6)}",
f"  Normalização: MinMaxScaler → [0, π]","",
"RESULTADOS — MODELOS QUÂNTICOS",
f"  QKSVM (Quantum Kernel SVM)",
f"    K(xi,xj) = |⟨φ(xi)|φ(xj)⟩|²  |  Kernel matrix em {t_k:.1f}s",
f"    Acc={results['QKSVM']['acc']:.4f} | F1={results['QKSVM']['f1']:.4f} | AUC={results['QKSVM']['auc']:.4f}",
f"    Complexidade: O(N²·2^n) = O(N²·{DIM})", "",
f"  VQC (Variational Quantum Classifier)",
f"    Parâmetros: {N_LAYERS}×{N_QUBITS}×2 = {N_LAYERS*N_QUBITS*2} | Treino: {N_VQC} amostras | {t_vqc:.1f}s",
f"    Acc={results['VQC']['acc']:.4f} | F1={results['VQC']['f1']:.4f} | AUC={results['VQC']['auc']:.4f}",
f"    Convergência: {results['VQC']['converged']} | Loss final: {results['VQC']['loss_history'][-1]:.4f}","",
f"  QIK-SVM (Quantum-Inspired Kernel) — multiclasse",
f"    Fourier encoding: φ(x)=[cos(x),sin(x),cos(2x),sin(2x)]  |  {t_qik:.3f}s",
f"    Acc={results['QIK-SVM']['acc']:.4f} | F1-macro={results['QIK-SVM']['f1']:.4f}","",
"RESULTADOS — MODELOS CLÁSSICOS (mesma amostra)",
f"  SVM-RBF:  Acc={results['SVM-RBF']['acc']:.4f} | F1={results['SVM-RBF']['f1']:.4f} | AUC={results['SVM-RBF']['auc']:.4f} | {results['SVM-RBF']['time']:.3f}s",
f"  RF:       Acc={results['RF']['acc']:.4f} | F1-macro={results['RF']['f1']:.4f} | {results['RF']['time']:.3f}s",
f"  LR:       Acc={results['LR']['acc']:.4f} | F1={results['LR']['f1']:.4f} | AUC={results['LR']['auc']:.4f} | {results['LR']['time']:.3f}s","",
"ANÁLISE: QUANDO QML FARIA SENTIDO EM O-RAN?",
"  * Com hardware quântico real, o QKSVM elimina o custo O(2^n) da simulação.",
"  * O QIK-SVM é hoje o mais viável: kernel expressivo + custo polinomial.",
"  * O VQC é análogo a uma rede neural rasa quântica — precisa de mais qubits",
"    e mais dados para superar SVMs clássicos na era NISQ.",
"  * Para near-RT RIC (<10ms): apenas QIK-SVM é diretamente aplicável.",
"  * Vantagem potencial do QML: eficiência em dados de alta dimensão com",
"    estrutura quântica (ex: canais de propagação em mmWave/5G).","",
"CURVA DE APRENDIZADO (QKSVM vs SVM-RBF)",
"  n_treino | QKSVM   | SVM-RBF | ΔQKSVM-RBF"]
for n,qa,ra in zip(train_sizes,qk_scores,rbf_scores):
    delta=qa-ra
    rpt.append(f"  {n:8d} | {qa:.4f}  | {ra:.4f}  | {'+' if delta>=0 else ''}{delta:.4f}")
rpt+=["","ARQUIVOS GERADOS",
"  qml_fig1_arquitetura.png  — circuito, Hilbert, kernel matrix, PCA, overview",
"  qml_fig2_desempenho.png   — curva de aprendizado, ROC, loss VQC, F1 vs tempo",
"  qml_fig3_kernel_cm.png    — distribuição do kernel, confusion matrices",
"  qml_fig4_multiclasse.png  — QIK-SVM multiclasse, F1 por classe vs RF",
"  qml_pipeline.py           — código completo reproduzível"]

report_text="\n".join(rpt)
print(report_text)
with open(os.path.join(OUTPUT_DIR,"relatorio_qml.txt"),"w") as f:
    f.write(report_text+"\n")
print("\nPipeline QML concluído.")
