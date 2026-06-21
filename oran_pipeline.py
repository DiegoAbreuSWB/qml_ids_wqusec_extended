"""
Pipeline de Análise — Netslab-5G-ORAN-IDD (Network Dataset)
============================================================
Etapas:
  1. Carregamento com amostragem estratificada
  2. EDA — distribuições, correlações, nulos
  3. Limpeza e engenharia de features
  4. Seleção de features (importância + correlação)
  5. Geração do dataset reduzido
  6. Avaliação de modelos de ML (binário e multiclasse)
  7. Relatório final em texto
"""

import warnings
warnings.filterwarnings("ignore")

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample

INPUT_FILE  = "/mnt/user-data/uploads/Network_Dataset.csv"
OUTPUT_DIR  = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_PER_CLASS = 20_000   # registros por classe para manter balanceamento razoável
RANDOM_STATE     = 42
N_TOP_FEATURES   = 15

# ──────────────────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO COM AMOSTRAGEM ESTRATIFICADA
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("ETAPA 1 — Carregamento com amostragem estratificada")
print("=" * 65)

t0 = time.time()
full = pd.read_csv(INPUT_FILE, usecols=["attack_category"])
class_counts = full["attack_category"].value_counts()
classes = class_counts.index.tolist()

# Lê o arquivo em chunks e coleta amostras por classe
chunk_size = 200_000
buffers = {c: [] for c in classes}

for chunk in pd.read_csv(INPUT_FILE, chunksize=chunk_size):
    for c in classes:
        sub = chunk[chunk["attack_category"] == c]
        buffers[c].append(sub)

frames = []
for c in classes:
    combined = pd.concat(buffers[c], ignore_index=True)
    n = min(SAMPLE_PER_CLASS, len(combined))
    frames.append(combined.sample(n=n, random_state=RANDOM_STATE))

df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"  Dataset total: {len(full):,} linhas | Amostra de trabalho: {len(df):,} linhas")
print(f"  Tempo de leitura: {time.time()-t0:.1f}s\n")
print("  Distribuição da amostra:")
print(df["attack_category"].value_counts().to_string())
print()

del full, buffers, frames

# ──────────────────────────────────────────────────────────────────────────────
# 2. EDA
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("ETAPA 2 — EDA")
print("=" * 65)

# Nulos
null_summary = df.isnull().sum()
null_pct     = (null_summary / len(df) * 100).round(2)
null_df      = pd.DataFrame({"nulos": null_summary, "%": null_pct})
null_df      = null_df[null_df["nulos"] > 0]
print("\n  Colunas com valores nulos:")
print("  Nenhuma." if null_df.empty else null_df.to_string())

NUM_COLS = ["duration", "src_bytes", "dst_bytes", "missed_bytes",
            "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
            "http_trans_depth", "files_total_bytes"]
CAT_COLS = ["proto", "service", "conn_state"]

print("\n  Stats numéricas:")
print(df[NUM_COLS].describe().T[["mean","std","min","50%","max"]].round(2).to_string())

print("\n  Cardinalidade das colunas categóricas:")
for c in CAT_COLS + ["history"]:
    print(f"    {c}: {df[c].nunique()} valores únicos")

# ──────────────────────────────────────────────────────────────────────────────
# 3. LIMPEZA E ENGENHARIA DE FEATURES
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ETAPA 3 — Limpeza e engenharia de features")
print("=" * 65)

# 3a. Remover colunas de identificação / leak / alta cardinalidade
DROP_COLS = ["uid", "src_ip", "dst_ip", "history",  # identificadores ou quasi-identificadores
             "attack_type",                           # granularidade maior da label — evitar leak
             "traffic_type"]                          # binário derivado de attack_category
df.drop(columns=DROP_COLS, inplace=True, errors="ignore")
print(f"  Colunas removidas: {DROP_COLS}")

# 3b. Log-transform em features com distribuição muito assimétrica
LOG_COLS = ["src_bytes","dst_bytes","src_ip_bytes","dst_ip_bytes","missed_bytes","files_total_bytes"]
for c in LOG_COLS:
    df[f"log_{c}"] = np.log1p(df[c])
print(f"  Log-transform aplicado em: {LOG_COLS}")

# 3c. Feature derivada: razão bytes enviados / recebidos
df["byte_ratio"] = np.where(
    df["dst_bytes"] + df["src_bytes"] > 0,
    df["src_bytes"] / (df["src_bytes"] + df["dst_bytes"] + 1e-9),
    0.0
)
# Razão pacotes
df["pkt_ratio"] = np.where(
    df["dst_pkts"] + df["src_pkts"] > 0,
    df["src_pkts"] / (df["src_pkts"] + df["dst_pkts"] + 1e-9),
    0.0
)
# Bytes por pacote
df["bytes_per_pkt_src"] = np.where(df["src_pkts"] > 0, df["src_bytes"] / df["src_pkts"], 0)
df["bytes_per_pkt_dst"] = np.where(df["dst_pkts"] > 0, df["dst_bytes"] / df["dst_pkts"], 0)
print("  Features derivadas criadas: byte_ratio, pkt_ratio, bytes_per_pkt_src, bytes_per_pkt_dst")

# 3d. Encoding de categóricas
le = LabelEncoder()
for c in CAT_COLS:
    df[c + "_enc"] = le.fit_transform(df[c].astype(str))
print(f"  Label encoding aplicado em: {CAT_COLS}")

# 3e. Encoding da label alvo
label_enc = LabelEncoder()
df["label"] = label_enc.fit_transform(df["attack_category"])
df["label_bin"] = (df["attack_category"] != "benign").astype(int)
print(f"  Classes mapeadas: { {k: v for k, v in zip(label_enc.classes_, range(len(label_enc.classes_)))} }")

print(f"\n  Shape após limpeza: {df.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. SELEÇÃO DE FEATURES
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ETAPA 4 — Seleção de features")
print("=" * 65)

# Features candidatas (todas as numéricas + encodings categóricos, sem a label)
EXCLUDE = ["attack_category", "label", "label_bin",
           "proto", "service", "conn_state",   # versões originais (já encodadas)
           "ip_proto"]                          # redundante com proto_enc
candidate_cols = [c for c in df.columns if c not in EXCLUDE]

X_feat = df[candidate_cols].select_dtypes(include=[np.number])
y_feat = df["label"]

# Random Forest para importância de features
rf_feat = RandomForestClassifier(n_estimators=100, max_depth=10,
                                  random_state=RANDOM_STATE, n_jobs=-1)
rf_feat.fit(X_feat, y_feat)
importances = pd.Series(rf_feat.feature_importances_, index=X_feat.columns).sort_values(ascending=False)

print("\n  Top 20 features por importância (Random Forest):")
print(importances.head(20).round(4).to_string())

TOP_FEATURES = importances.head(N_TOP_FEATURES).index.tolist()
print(f"\n  Features selecionadas ({N_TOP_FEATURES}): {TOP_FEATURES}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. DATASET REDUZIDO
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ETAPA 5 — Geração do dataset reduzido")
print("=" * 65)

reduced_cols = TOP_FEATURES + ["attack_category", "label", "label_bin"]
df_reduced = df[reduced_cols].copy()
out_reduced = os.path.join(OUTPUT_DIR, "oran_dataset_reduced.csv")
df_reduced.to_csv(out_reduced, index=False)
print(f"  Salvo: {out_reduced}")
print(f"  Shape: {df_reduced.shape} | Colunas: {TOP_FEATURES}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. AVALIAÇÃO DE MODELOS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ETAPA 6 — Avaliação de modelos de ML")
print("=" * 65)

X = df_reduced[TOP_FEATURES]
y_multi = df_reduced["label"]
y_bin   = df_reduced["label_bin"]

X_train, X_test, ym_train, ym_test, yb_train, yb_test = train_test_split(
    X, y_multi, y_bin, test_size=0.2, random_state=RANDOM_STATE, stratify=y_multi)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

MODELS = {
    "Decision Tree":       (DecisionTreeClassifier(max_depth=15, random_state=RANDOM_STATE), False),
    "Random Forest":       (RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1), False),
    "Gradient Boosting":   (GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE), False),
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
    "KNN (k=5)":           (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), True),
}

results = []
print()
for name, (model, use_scaled) in MODELS.items():
    Xtr = X_train_sc if use_scaled else X_train
    Xte = X_test_sc  if use_scaled else X_test

    # Multiclasse
    t1 = time.time()
    model.fit(Xtr, ym_train)
    ym_pred = model.predict(Xte)
    elapsed = time.time() - t1

    acc  = accuracy_score(ym_test, ym_pred)
    f1m  = f1_score(ym_test, ym_pred, average="macro")
    f1w  = f1_score(ym_test, ym_pred, average="weighted")

    # Binário (re-treina no binário)
    model2 = type(model)(**model.get_params())
    model2.fit(Xtr, yb_train)
    yb_pred = model2.predict(Xte)
    acc_b = accuracy_score(yb_test, yb_pred)
    f1_b  = f1_score(yb_test, yb_pred)

    results.append({
        "Modelo": name,
        "Acc (multi)": round(acc * 100, 2),
        "F1-macro (multi)": round(f1m * 100, 2),
        "F1-w (multi)":  round(f1w * 100, 2),
        "Acc (bin)":  round(acc_b * 100, 2),
        "F1 (bin)":   round(f1_b * 100, 2),
        "Tempo (s)":  round(elapsed, 1),
        "_model":     model,
        "_ym_pred":   ym_pred,
    })
    print(f"  [{name}] Acc={acc:.4f} | F1-macro={f1m:.4f} | F1-bin={f1_b:.4f} | {elapsed:.1f}s")

results_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
print("\n  Tabela resumo:")
print(results_df.set_index("Modelo").to_string())

# Melhor modelo para confusion matrix
best_idx = results_df["F1-macro (multi)"].idxmax()
best_name = results_df.loc[best_idx, "Modelo"]
best_pred = results[best_idx]["_ym_pred"]
best_model = results[best_idx]["_model"]

print(f"\n  Melhor modelo: {best_name}")
print(f"\n  Classification Report ({best_name}):")
print(classification_report(ym_test, best_pred,
                             target_names=label_enc.classes_))

# ──────────────────────────────────────────────────────────────────────────────
# 7. FIGURAS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ETAPA 7 — Gerando figuras")
print("=" * 65)

sns.set_style("whitegrid")
PALETTE = sns.color_palette("muted", 6)

# ── Figura 1: EDA ────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(16, 9))
fig1.suptitle("EDA — Netslab-5G-ORAN-IDD (Network)", fontsize=14, fontweight="bold")

# 1a. Distribuição de classes
cat_counts = df["attack_category"].value_counts()
axes[0,0].barh(cat_counts.index, cat_counts.values, color=PALETTE)
axes[0,0].set_title("Distribuição das classes")
axes[0,0].set_xlabel("Amostras")
for i, v in enumerate(cat_counts.values):
    axes[0,0].text(v + 100, i, f"{v:,}", va="center", fontsize=8)

# 1b. Distribuição de attack_type
type_counts = df["attack_type"].value_counts() if "attack_type" in df.columns else pd.Series()

# Como removemos attack_type do df, recarregar do original para plot
_tmp = pd.read_csv(INPUT_FILE, usecols=["attack_type"], nrows=len(df)*3)
_type_c = _tmp["attack_type"].value_counts().head(10)
axes[0,1].barh(_type_c.index, _type_c.values, color=PALETTE[1])
axes[0,1].set_title("Top-10 tipos de ataque (dataset completo)")
axes[0,1].set_xlabel("Amostras")

# 1c. Distribuição de proto
proto_c = df["proto"].value_counts()
axes[0,2].pie(proto_c.values, labels=proto_c.index, autopct="%1.1f%%", colors=PALETTE)
axes[0,2].set_title("Protocolo")

# 1d. Log duration por classe
log_dur = np.log1p(df["duration"])
for i, cat in enumerate(cat_counts.index):
    mask = df["attack_category"] == cat
    axes[1,0].hist(log_dur[mask], bins=40, alpha=0.5, label=cat, color=PALETTE[i])
axes[1,0].set_title("Distribuição de log(duration) por classe")
axes[1,0].set_xlabel("log(duration + 1)")
axes[1,0].legend(fontsize=7)

# 1e. Mapa de calor de correlação (top features)
corr = df[TOP_FEATURES[:10]].corr()
mask_tri = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=axes[1,1], cmap="coolwarm", annot=True, fmt=".1f",
            annot_kws={"size": 7}, mask=mask_tri, linewidths=0.3)
axes[1,1].set_title("Correlação — top 10 features")
axes[1,1].tick_params(axis="x", rotation=45, labelsize=7)
axes[1,1].tick_params(axis="y", labelsize=7)

# 1f. Importância de features
top_imp = importances.head(15)
axes[1,2].barh(top_imp.index[::-1], top_imp.values[::-1], color=PALETTE[2])
axes[1,2].set_title("Importância das features (RF)")
axes[1,2].set_xlabel("Importância")
axes[1,2].tick_params(labelsize=8)

plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, "fig1_eda.png"), dpi=120, bbox_inches="tight")
plt.close(fig1)
print("  Salvo: fig1_eda.png")

# ── Figura 2: Resultados ML ──────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle("Resultados dos Modelos de ML", fontsize=14, fontweight="bold")

model_names = results_df["Modelo"].tolist()
x = np.arange(len(model_names))
w = 0.25

# 2a. Comparativo de métricas multiclasse
axes2[0].bar(x - w, results_df["Acc (multi)"], w, label="Accuracy", color=PALETTE[0])
axes2[0].bar(x,     results_df["F1-macro (multi)"], w, label="F1-macro", color=PALETTE[1])
axes2[0].bar(x + w, results_df["F1-w (multi)"], w, label="F1-weighted", color=PALETTE[2])
axes2[0].set_xticks(x)
axes2[0].set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
axes2[0].set_ylabel("Score (%)")
axes2[0].set_title("Multiclasse (6 classes)")
axes2[0].legend(fontsize=8)
axes2[0].set_ylim(0, 110)
for bar in axes2[0].patches:
    h = bar.get_height()
    if h > 1:
        axes2[0].text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=6)

# 2b. Comparativo binário
axes2[1].bar(x - w/2, results_df["Acc (bin)"], w, label="Accuracy", color=PALETTE[3])
axes2[1].bar(x + w/2, results_df["F1 (bin)"],  w, label="F1-score",  color=PALETTE[4])
axes2[1].set_xticks(x)
axes2[1].set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
axes2[1].set_ylabel("Score (%)")
axes2[1].set_title("Binário (benigno vs. ataque)")
axes2[1].legend(fontsize=8)
axes2[1].set_ylim(0, 110)
for bar in axes2[1].patches:
    h = bar.get_height()
    if h > 1:
        axes2[1].text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=6)

# 2c. Confusion matrix do melhor modelo
use_scaled = MODELS[best_name][1]
Xte_cm = X_test_sc if use_scaled else X_test
cm = confusion_matrix(ym_test, best_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, ax=axes2[2], annot=True, fmt=".1f", cmap="Blues",
            xticklabels=label_enc.classes_, yticklabels=label_enc.classes_,
            annot_kws={"size": 9})
axes2[2].set_title(f"Matriz de Confusão — {best_name} (%)")
axes2[2].set_xlabel("Predito")
axes2[2].set_ylabel("Real")
axes2[2].tick_params(axis="x", rotation=30, labelsize=8)
axes2[2].tick_params(axis="y", rotation=0, labelsize=8)

plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "fig2_resultados_ml.png"), dpi=120, bbox_inches="tight")
plt.close(fig2)
print("  Salvo: fig2_resultados_ml.png")

# ── Figura 3: Feature analysis por classe ────────────────────────────────────
fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9))
fig3.suptitle("Top features por classe de ataque", fontsize=14, fontweight="bold")

PLOT_FEATURES = ["log_src_bytes", "log_dst_bytes", "duration",
                 "byte_ratio", "src_pkts", "pkt_ratio"]
PLOT_LABELS   = ["log(src_bytes)", "log(dst_bytes)", "duration",
                 "byte_ratio", "src_pkts", "pkt_ratio"]

for ax, feat, lbl in zip(axes3.flat, PLOT_FEATURES, PLOT_LABELS):
    for i, cat in enumerate(cat_counts.index):
        vals = df.loc[df["attack_category"] == cat, feat]
        # Clip para melhor visualização
        clip_val = vals.quantile(0.99)
        vals = vals.clip(upper=clip_val)
        ax.hist(vals, bins=50, alpha=0.5, label=cat, color=PALETTE[i], density=True)
    ax.set_title(lbl)
    ax.set_xlabel(lbl)
    ax.set_ylabel("Densidade")
    ax.legend(fontsize=6)

plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "fig3_features_por_classe.png"), dpi=120, bbox_inches="tight")
plt.close(fig3)
print("  Salvo: fig3_features_por_classe.png")

# ──────────────────────────────────────────────────────────────────────────────
# RELATÓRIO FINAL
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RELATÓRIO FINAL")
print("=" * 65)

report_lines = [
    "RELATÓRIO — Pipeline 5G-ORAN-IDD (Network Dataset)",
    "=" * 60,
    "",
    "1. DATASET",
    f"   Total de registros: 1.723.817",
    f"   Amostra de trabalho: {len(df):,} (estratificada, {SAMPLE_PER_CLASS:,}/classe)",
    f"   Features originais: 26 | Após engenharia: {len(candidate_cols)}",
    f"   Features selecionadas (top-{N_TOP_FEATURES}): {', '.join(TOP_FEATURES)}",
    "",
    "2. CLASSES",
]
for cat, cnt in class_counts.items() if hasattr(class_counts, 'items') else cat_counts.items():
    report_lines.append(f"   {cat:<15} {cnt:>8,} registros")

report_lines += [
    "",
    "3. RESULTADOS DOS MODELOS",
    results_df.set_index("Modelo").to_string(),
    "",
    f"4. MELHOR MODELO: {best_name}",
    f"   F1-macro (multi): {results_df.loc[best_idx, 'F1-macro (multi)']}%",
    f"   Accuracy (multi): {results_df.loc[best_idx, 'Acc (multi)']}%",
    "",
    "5. ARQUIVOS GERADOS",
    f"   oran_dataset_reduced.csv   — dataset reduzido ({len(df_reduced):,} x {len(TOP_FEATURES)+3})",
    "   fig1_eda.png               — análise exploratória",
    "   fig2_resultados_ml.png     — comparativo de modelos + confusion matrix",
    "   fig3_features_por_classe.png — distribuição de features por classe",
]

report_text = "\n".join(report_lines)
print(report_text)

report_path = os.path.join(OUTPUT_DIR, "relatorio_pipeline.txt")
with open(report_path, "w") as f:
    f.write(report_text + "\n")
print(f"\n  Salvo: {report_path}")
print("\nPipeline concluído.")
