# -*- coding: utf-8 -*-

"""
María Cañas Francia
TFG: Redes Bayesianas para Diagnóstico Clínico bajo Incertidumbre
Dataset: UCI Cleveland Heart Disease (303 pacientes, 13 vars + target)
"""

# Importamos librerías:
import pandas as pd #tablas tipo matriz
import numpy as np #álg. numérica y arrays
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt #graficos
import matplotlib.patches as mpatches
import seaborn as sns #visualizaciones est.
import networkx as nx #grafos
import json, copy, time, warnings #json: guardar aristas
import logging as _pgmpy_log
import ssl as _ssl

# Silenciar logger de pgmpy globalmente (evita mensajes INFO durante VE/Gibbs)
import logging as _pgmpy_log
_pgmpy_log.getLogger('pgmpy').setLevel(_pgmpy_log.ERROR)
warnings.filterwarnings('ignore')  #ocultamos mensajes irrelevantes

from collections import Counter
from itertools import product as iproduct
from scipy import stats
from scipy.stats import (shapiro, normaltest, skew, kurtosis,
                          kruskal, f_oneway, levene, chi2_contingency,
                          mannwhitneyu, norm as sp_norm)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import (StratifiedKFold, train_test_split,
                                     cross_val_predict)
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score, log_loss,
                              confusion_matrix, cohen_kappa_score,
                              balanced_accuracy_score, roc_curve, auc)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier # XGBoost (mejor que GradientBoosting sklearn)
from ucimlrepo import fetch_ucirepo # dataset UCI

from pgmpy.models import DiscreteBayesianNetwork
#en pgmpy >= 1.0 los scores de estructura se movieron a pgmpy.structure_score
try:
    from pgmpy.structure_score import BIC as pgBIC, BDeu as pgBDeu
except ImportError:
    #fallback para versiones antiguas (< 1.0)
    from pgmpy.estimators import BicScore as pgBIC, BDeuScore as pgBDeu
from pgmpy.estimators import (HillClimbSearch,
                               PC,
                               MaximumLikelihoodEstimator, BayesianEstimator)
# pgmpy >= 1.1: los estimadores de parametros se movieron a parameter_estimator
try:
    from pgmpy.parameter_estimator import DiscreteMLE, DiscreteBayesianEstimator
    _USE_NEW_ESTIMATORS = True
except ImportError:
    _USE_NEW_ESTIMATORS = False   # versión < 1.1, usamos los antiguos
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.sampling import BayesianModelSampling

from imblearn.over_sampling import SMOTENC

# bnlearn para independencias condicionales y comparación de modelos
try:
    import bnlearn as bn
    BNLEARN_OK = True
except ImportError:
    BNLEARN_OK = False
    print("AVISO: bnlearn no disponible -> independencias condicionales se harán manualmente")



TARGET = 'target'

print("=" * 70)
print("TFG Redes Bayesianas :UCI Cleveland Heart Disease")
print("=" * 70)


# =============================================================================
# BLOQUE 1: Carga y calidad del dato
# =============================================================================

print("\n" + "=" * 70)
print("BLOQUE 1: CALIDAD DEL DATO")
print("=" * 70)

heart = fetch_ucirepo(id=45) #dataset
df_raw = pd.concat([heart.data.features, heart.data.targets], axis=1)
df_raw = df_raw.replace('?', np.nan).apply(pd.to_numeric, errors='coerce') 

# 1.1 — variable target: MANTENEMOS las 5 categorías originales de num (0,1,2,3,4)
df_raw[TARGET] = df_raw['num'].astype(int)
df_raw = df_raw.drop(columns=['num'])
print(f"Tamaño dataset: {df_raw.shape}")
vc = df_raw[TARGET].value_counts().sort_index()
print("Distribución target (grado estenosis coronaria):")
for k, v in vc.items():
    bar = '█' * int(v / len(df_raw) * 40)
    print(f"  Clase {k}: {v:>3} ({v/len(df_raw)*100:.1f}%)  {bar}")

# 1.2 — Duplicados
n_dup = df_raw.duplicated().sum()
print(f"\nFilas duplicadas: {n_dup}")
if n_dup:
    df_raw = df_raw.drop_duplicates()

# 1.3 — Valores imposibles (rangos fisiológicos)
RANGOS = {
    'age':      (18,  100),
    'trestbps': (60,  250),
    'chol':     (100, 600),
    'thalach':  (50,  220),
    'oldpeak':  (-0.1, 10),
}
total_imp = 0
for col, (mn, mx) in RANGOS.items():  #valores fuera de rango? -> missing
    mask = df_raw[col].notna() & ((df_raw[col] < mn) | (df_raw[col] > mx))
    n = mask.sum()
    total_imp += n
    df_raw.loc[mask, col] = np.nan
print(f"Valores imposibles convertidos a NaN: {total_imp}")

# 1.4 — análisis de Missing Values + MCAR
miss = df_raw.isnull().sum() #nº missings por var
miss_pct = miss / len(df_raw) * 100
miss_tab = pd.DataFrame({'N': miss, '%': miss_pct.round(2)})
miss_tab = miss_tab[miss_tab['N'] > 0].sort_values('N', ascending=False)

if miss_tab.empty:
    print("No hay variables con valores faltantes.")
else: #mapa de calor de faltantes y MCAR
    print("\nResumen de missings:")
    print(miss_tab.to_string())
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    sns.heatmap(df_raw[miss_tab.index].isnull().astype(int).T,
                cbar=False, cmap='Reds', ax=axes[0])
    axes[0].set_title('Patrón de missings')
    axes[1].bar(miss_tab.index, miss_tab['%'], color='#e05c4b', alpha=0.8)
    axes[1].set_ylabel('% faltante')
    axes[1].set_title('% missings por variable')
    axes[1].tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig('uci_missings.png', dpi=120, bbox_inches='tight')
    plt.close()
    
    # TEST MCAR: necesitamos comprobar que
    # la probabilidad de que un valor falte NO depende de ninguna variable observada.
    vars_con_miss = list(miss_tab.index)
    # Condición 1: P(target | Rⱼ=1) = P(target | Rⱼ=0)
    print("\nCondición 1 -> Mann-Whitney U: ¿difiere TARGET entre missing/no-missing?")
    for col in vars_con_miss:
        g_miss = df_raw.loc[df_raw[col].isnull(),  TARGET].dropna() #target cuando esa variable es missing
        g_nomiss = df_raw.loc[df_raw[col].notna(), TARGET].dropna() #target cuando no es missing
        if len(g_miss) >= 3 and len(g_nomiss) >= 3:
            stat, pval = mannwhitneyu(g_miss, g_nomiss, alternative='two-sided') # test estadistico
            mec = "MAR/MNAR" if pval < 0.05 else "MCAR compatible"
            print(f"  {col:<10}  n_miss={len(g_miss):>3}  p={pval:.4f}  ->{mec}")
        else:
            print(f"  {col:<10}  n_miss={len(g_miss):>3}  -> muestra insuficiente")
    # Condición 2: P(Xₖ | Rⱼ=1) = P(Xₖ | Rⱼ=0) para k ≠ j
    print("\nCondición 2 -> Mann-Whitney U: ¿difieren OTRAS VARIABLES entre grupo missing / no-missing?")
    mcar_cond2_ok = True
    for col_miss in vars_con_miss:
        mask_miss = df_raw[col_miss].isnull() # true si esa fila tiene missing en esa variable
        otras = [c for c in df_raw.columns if c != col_miss and c != TARGET]
        for col_otra in otras:
            g1 = df_raw.loc[mask_miss,  col_otra].dropna() # valores de otra variable cuando hay missing
            g2 = df_raw.loc[~mask_miss, col_otra].dropna() # cuando no hay missing
            if len(g1) >= 3 and len(g2) >= 3:
                _, pval = mannwhitneyu(g1, g2, alternative='two-sided')
                if pval < 0.05:
                    print(f"  R({col_miss}) ~ {col_otra}  p={pval:.4f}  ->posible MAR")
                    mcar_cond2_ok = False
    if mcar_cond2_ok:
        print("Ninguna variable muestra asociación significativa con los missings.")
    
    # Condición 3: Test de Little (implementación manual)
    def little_mcar_test(data):
        """
        Implementación manual de test de Little para MCAR.
        H0: los datos son MCAR
        """
        data = data.copy()
        #solo usamos columnas numéricas con al menos 1 missing
        all_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        n, k = data.shape[0], len(all_cols)
        #estimamos globalmente μ y Σ con todos los datos disponibles (EM simplificado: medias y cov completos)
        mu = data[all_cols].mean()
        sigma = data[all_cols].cov()
        try:
            sigma_inv = np.linalg.pinv(sigma.values)
        except Exception:
            return None, None, None
        #identificamos patrones de missing
        patterns = data[all_cols].isnull().drop_duplicates()
        d2_total = 0.0
        gl_total = 0
        for _, pattern in patterns.iterrows():
            obs_cols = [c for c in all_cols if not pattern[c]]# columnas observadas en este patrón
            if not obs_cols:
                continue
            mask = (data[all_cols].isnull() == pattern).all(axis=1) #filas que siguen este patrón
            subdf = data.loc[mask, obs_cols]
            nj = mask.sum()
            if nj == 0:
                continue
            yj = subdf.mean() #media global
            diff = (yj - mu[obs_cols]).values #diferencia con media global
            idx_c = [all_cols.index(c) for c in obs_cols] #indice de covarianza
            S_sub = sigma_inv[np.ix_(idx_c, idx_c)]
            d2_total += nj * diff @ S_sub @ diff #contribución al estadístico (forma cuadrática: chi-cuadrado)
            gl_total += len(obs_cols)
        gl = gl_total - k
        if gl <= 0:
            return d2_total, None, None
        pval = 1 - stats.chi2.cdf(d2_total, df=gl)
        return d2_total, gl, pval

    print("\nCondición 3 -> Test de Little (MCAR global):")
    d2, gl, pval_little = little_mcar_test(df_raw.select_dtypes(include=[np.number]))
    if pval_little is not None:
        res_l = "No se rechaza MCAR" if pval_little > 0.05 else "Se rechaza MCAR"
        print(f"  d²={d2:.3f}  gl={gl}  p={pval_little:.4f}  ->{res_l}")
        if pval_little > 0.05:
            print(" -> MCAR confirmado: imputamos por moda (variables categóricas ordinales, n_miss ≤ 4) y por mediana (continuas)")
        else:
            print(" -> MCAR no compatibile: MAR/MNAR")
    else:
        print("Test de Little no aplicable (grados de libertad insuficientes).")

df = df_raw.copy()


# =============================================================================
# BLOQUE 2: EDA 
# =============================================================================

print("\n" + "=" * 70)
print("BLOQUE 2: EDA")
print("=" * 70)

# 2.0 — definición de variables
VARS_CONTINUAS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
VARS_CATEGORICAS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

colores  = ['#4e79a7', '#76b7b2', '#f28e2b', '#e05c4b', '#59a14f']
labels_target = {0:'Grado 0 (sano)', 1:'Grado 1 (leve)',
                 2:'Grado 2 (mod.)', 3:'Grado 3 (grave)', 4:'Grado 4 (muy grave)'}

# 2.1 - Distribuciones marginales: P(Xj)
print("\n--- 2.1 Distribuciones marginales ---")
print("\n Variables continuas: estadísticos descriptivos.")
print(df[VARS_CONTINUAS].describe().round(2).to_string())
print("\nVariables categóricas: frecuencias y moda.")
for col in VARS_CATEGORICAS:
    vc2 = df[col].value_counts().sort_index()
    cats = "  ".join([f"{int(k)}:{v}" for k, v in vc2.items()])
    print(f"  {col:<10}: {cats}")
print("\nVariable target 'num': distribución.")
vc_t = df[TARGET].value_counts().sort_index()
for cat, cnt in vc_t.items():
    bar = '█' * int(cnt/len(df)*50)
    print(f"  Grado {cat}: {cnt} ({cnt/len(df)*100:.1f}%):  {bar}")

# 2.2 — Distribuciones condicionadas: P(Xj ∣Y=1), P(Xj​∣Y=0) -> histogramas por clase
print("\n--- 2.2 Distribuciones condicionadas ---")
feats  = [c for c in df.columns if c != TARGET] #vars predictoras (features)
n_cols = 4
n_rows = (len(feats) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
axes = axes.flatten()
for i, col in enumerate(feats):
    if col in VARS_CONTINUAS:
        # Histograma superpuesto por clase
        for clase in sorted(df[TARGET].unique()):
            datos = df[df[TARGET] == clase][col].dropna()
            axes[i].hist(datos, bins=15, alpha=0.55,
                         color=colores[clase], edgecolor='white',
                         label=labels_target[clase])
    else: #bar plot con posiciones enteras
        cats_col = sorted(df[col].dropna().unique().astype(int))
        n_clases = len(sorted(df[TARGET].unique()))
        width    = 0.8 / n_clases
        for j, clase in enumerate(sorted(df[TARGET].unique())):
            datos = df[df[TARGET] == clase][col].dropna()
            vc3 = datos.value_counts().sort_index()
            x_pos = np.array(cats_col, dtype=float)
            y_vals = np.array([vc3.get(c, 0) for c in cats_col], dtype=float)
            offset = (j - n_clases / 2 + 0.5) * width
            axes[i].bar(x_pos + offset, y_vals, width=width * 0.9,
                        alpha=0.8, color=colores[clase], label=labels_target[clase])
        axes[i].set_xticks(cats_col)
        axes[i].set_xticklabels([str(c) for c in cats_col], fontsize=7)

    axes[i].set_title(col, fontsize=7)
    axes[i].legend(fontsize=4, ncol=2)
    axes[i].tick_params(labelsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('UCI Cleveland: Distribuciones condicionadas P(Xⱼ | grado estenosis)',
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig('uci_distribuciones.png', dpi=120, bbox_inches='tight')
plt.close()

# segun las distribuciones vistas en las graficas, definimos las variables continuas como:
VARS_DISCRETAS_CLINICAS = ['age', 'oldpeak']   # se discretizarán en B3A
VARS_CONTINUAS_GAUSSIANAS = ['trestbps', 'chol', 'thalach']

# 2.3 - tests de normalidad:
# 2.3.1 — test de normalidad GLOBAL
print("\n--- 2.3.1 Tests de normalidad GLOBAL ---")
normalidad_global = {}
label_da = "D'A p-val"
print(f"  {'Variable':<12} {'Skewness':>10} {'Kurtosis':>10} {'SW p-val':>10} {label_da:>10} {'Aprox normal?':>14}")
print("  " + "─" * 72)

for col in VARS_CONTINUAS:
    datos = df[col].dropna()
    sk_v = skew(datos) #asimetria
    ku_v = kurtosis(datos) #curtosis
    _, p_sw = shapiro(datos) if len(datos) <= 5000 else (None, np.nan)
    _, p_da = normaltest(datos)
    # Criterio diferenciado según el tipo de variable
    if col in VARS_CONTINUAS_GAUSSIANAS:
        # Umbral permisivo (0.01) para continuas gaussianas
        es_norm = (np.isnan(p_sw) or p_sw > 0.01) and p_da > 0.01 and abs(sk_v) < 1.0
    else:
        # Umbral estricto (0.05) para age y oldpeak
        es_norm = (np.isnan(p_sw) or p_sw > 0.05) and p_da > 0.05

    normalidad_global[col] = es_norm
    sw_str = f"{p_sw:.4f}" if not np.isnan(p_sw) else "N/A"
    marca = "sí aprox" if es_norm else "no normal"
    print(f"  {col:<12} {sk_v:>10.3f} {ku_v:>10.3f} {sw_str:>10} "
          f"{p_da:>10.4f} {marca:>14}")

# Justificación: N~300 hace que SW sea hipersensible -> criterio visual
# del tutor confirma distribuciones campaniformes para estas 3 variables.

# 2.3.2 — Diferencia entre clases
print("\n--- 2.3.2 Tests de diferencia entre clases ---")
clases_unicas = sorted(df[TARGET].unique()) #lista ordenada de los valores únicos del target (0,1,2,3,4)
grupos_por_clase = {col: [df[df[TARGET] == c][col].dropna().values
                           for c in clases_unicas]
                    for col in VARS_CONTINUAS}

print(f"  {'Variable':<12} {'Tipo test':>14} {'Estadístico':>13} "
      f"{'p-valor':>10} {'Efecto η²':>11} {'Sig.':>6}")
print("  " + "─" * 73)
# vars continuas: tests de normalidad POR CLASES (ANOVA o Kruskal-Wallis)
for col in VARS_CONTINUAS:
    grupos = [g for g in grupos_por_clase[col] if len(g) >= 3]
    if len(grupos) < 2:
        continue
    if col in VARS_CONTINUAS_GAUSSIANAS:
        #normalidad por clase (Shapiro)
        norm_por_clase = []
        for g in grupos:
            if len(g) >= 8:
                _, p_s = shapiro(g)
                norm_por_clase.append(p_s > 0.01)
            else:
                norm_por_clase.append(False)
        todos_norm = all(norm_por_clase)
        #Levene (homocedasticidad)
        _, p_lev = levene(*grupos)
        homoc = p_lev > 0.05
        if todos_norm and homoc:
            stat, p_val = f_oneway(*grupos)
            tipo_test = "ANOVA"
        else:
            stat, p_val = kruskal(*grupos)
            tipo_test = "Kruskal-W"
            razon = []
            if not todos_norm:
                razon.append("no normal")
            if not homoc:
                razon.append(f"Levene p={p_lev:.3f}")
            tipo_test += f" ({'/'.join(razon)})"
    else:
        # age y oldpeak: directamente no paramétrico
        stat, p_val = kruskal(*grupos)
        tipo_test = "Kruskal-W"
    #Eta² (tamaño del efecto)
    n_tot = sum(len(g) for g in grupos)
    k_g = len(grupos)
    eta2 = max(0.0, (stat - k_g + 1) / (n_tot - k_g)) if n_tot > k_g else 0.0
    sig = "Sí" if p_val < 0.05 else "no"
    print(f"  {col:<12} {tipo_test:>14} {stat:>13.3f} {p_val:>10.4f} "
          f"{eta2:>11.3f} {sig:>6}")

#vars categóricas: tests de diferencias (Chi² + Cramér's V)
print(f"\n  {'Variable':<12} {'Test':>12} {'Chi²':>10} {'p-valor':>10} "
      f"{'Cramér V':>10} {'Sig.':>6}")
print("  " + "─" * 65)
# H₀: la variable categórica y target son independientes
for col in VARS_CATEGORICAS:
    ct = pd.crosstab(df[col], df[TARGET]) #tabla de contingencia
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, p_chi, _, _ = chi2_contingency(ct)
        n   = ct.values.sum()
        r, c_n = ct.shape
        crv = np.sqrt(chi2 / (n * (min(r, c_n) - 1)))
        sig = "Sí" if p_chi < 0.05 else "no"
        print(f"  {col:<12} {'Chi²':>12} {chi2:>10.3f} {p_chi:>10.4f} "
              f"{crv:>10.3f} {sig:>6}")

# 2.4 — Outliers: detección GLOBAL + boxplot POR CLASE 
print("\n--- 2.4 Outliers ---")

# 2.4.1 - detección GLOBAL (IQR + Z-score sobre todos los datos) 
print("  Outliers global:")
fig, axes = plt.subplots(1, len(VARS_CONTINUAS), figsize=(16, 4))
for i, col in enumerate(VARS_CONTINUAS):
    datos = df[col].dropna()
    Q1, Q3 = datos.quantile(0.25), datos.quantile(0.75)
    IQR = Q3 - Q1
    n_iqr  = ((datos < Q1 - 1.5 * IQR) | (datos > Q3 + 1.5 * IQR)).sum()
    n_zsco = (np.abs((datos - datos.mean()) / datos.std()) > 3).sum()
    axes[i].boxplot(datos, patch_artist=True,
                    boxprops=dict(facecolor='#4e79a7', alpha=0.6))
    axes[i].set_title(f'{col}\nIQR:{n_iqr} Z:{n_zsco}', fontsize=8)
    axes[i].tick_params(labelsize=7)
    print(f"  {col:<12}: IQR outliers={n_iqr:>3}  Z-score={n_zsco:>3}")
plt.suptitle('UCI: Boxplot GLOBAL', fontsize=8)
plt.tight_layout()
plt.savefig('uci_outliers_global.png', dpi=120, bbox_inches='tight')
plt.close()

# 2.4.2 - boxplot POR CLASE 
print("\n Outliers por clase:")
fig, axes = plt.subplots(1, len(VARS_CONTINUAS), figsize=(16, 5))
for i, col in enumerate(VARS_CONTINUAS):
    data_by_class = [df[df[TARGET] == c][col].dropna().values
                     for c in clases_unicas]
    bp = axes[i].boxplot(data_by_class, patch_artist=True,
                          labels=[f'C{c}' for c in clases_unicas])
    for patch, color in zip(bp['boxes'], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    # IQR por clase
    resumen_clases = []
    for c_i, c in enumerate(clases_unicas):
        d = df[df[TARGET] == c][col].dropna()
        if len(d) > 0:
            q1, q3 = d.quantile(0.25), d.quantile(0.75)
            iqr = q3 - q1
            n_out = ((d < q1 - 1.5 * iqr) | (d > q3 + 1.5 * iqr)).sum()
            resumen_clases.append(f"C{c}:{n_out}")
    axes[i].set_title(f'{col}\n{" ".join(resumen_clases)}', fontsize=8)
    axes[i].tick_params(labelsize=7)
plt.suptitle('UCI: Boxplots POR CLASE', fontsize=8)
plt.tight_layout()
plt.savefig('uci_outliers_clase.png', dpi=120, bbox_inches='tight')
plt.close()

#2.5 — MULTICOLINEALIDAD (vars continuas)
print("\n--- 2.5 Multicolinealidad ---")

LR_sk = LinearRegression  # alias usado en bloque 5.3

# Imputamos temporalmente para el cálculo (no admite NaN)
df_mc = df[VARS_CONTINUAS_GAUSSIANAS].copy()
for col in df_mc.columns:
    df_mc[col] = df_mc[col].fillna(df_mc[col].median())

# 2.5.1 — Correlaciones de Pearson entre continuas
print("\n  Correlaciones de Pearson entre vars continuas de la RB híbrida:")
corr_pearson = df_mc.corr(method='pearson').round(3)
print(corr_pearson.to_string())

# 2.5.2 — VIF
print("\n  VIF (Variance Inflation Factor):")
print(f"  {'Variable':<12} {'VIF':>8}  {'Interpretación':>20}")
print("  " + "─" * 45)
for col in VARS_CONTINUAS_GAUSSIANAS:
    otras = [c for c in VARS_CONTINUAS_GAUSSIANAS if c != col]
    X_vif = df_mc[otras].values
    y_vif = df_mc[col].values
    r2 = LinearRegression().fit(X_vif, y_vif).score(X_vif, y_vif)
    vif = 1 / (1 - r2) if r2 < 1.0 else np.inf
    if vif < 5:
        interp = "Sin multicolinealidad"
    elif vif < 10:
        interp = "Moderada"
    else:
        interp = "SEVERA"
    print(f"  {col:<12} {vif:>8.3f}  {interp:>20}")

# 2.5.3 — Heatmap de Pearson 
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr_pearson, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax, annot_kws={'size': 10})
ax.set_title('Multicolinealidad: Pearson entre vars. continuas RB híbrida', fontsize=10)
plt.tight_layout()
plt.savefig('uci_multicolinealidad.png', dpi=120, bbox_inches='tight')
plt.close()

# 2.6 — Matriz de correlaciones: Spearman (continuas/ordinales) + Cramér's V (categóricas)
print("\n--- 2.6 Matrices de correlación ---")

# Spearman: correlación por rangos (todas vars)
corr_spear = df[feats + [TARGET]].corr(method='spearman')
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
mask = np.triu(np.ones_like(corr_spear, dtype=bool))
sns.heatmap(corr_spear, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5, ax=axes[0],
            annot_kws={'size': 7})
axes[0].set_title('Correlación de Spearman (todas las variables)', fontsize=8)

# Cramér's V: asociación entre categóricas (V = 0 ->independencia)
all_vars_cat = VARS_CATEGORICAS 
v_matrix = pd.DataFrame(np.zeros((len(all_vars_cat), len(all_vars_cat))),
                          index=all_vars_cat, columns=all_vars_cat)
for i_v, c1 in enumerate(all_vars_cat):
    for j_v, c2 in enumerate(all_vars_cat):
        if i_v == j_v:
            v_matrix.loc[c1, c2] = 1.0 # variable consigo misma = asociación perfecta
        elif i_v < j_v:
            ct = pd.crosstab(df[c1].dropna(), df[c2].dropna())
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                chi2_v, _, _, _ = chi2_contingency(ct)
                n_v   = ct.values.sum()
                r_v, c_v = ct.shape
                v_val = np.sqrt(chi2_v / (n_v * (min(r_v, c_v) - 1)))
                v_matrix.loc[c1, c2] = v_val
                v_matrix.loc[c2, c1] = v_val # la matriz es simétrica
mask_v = np.triu(np.ones_like(v_matrix, dtype=bool))
sns.heatmap(v_matrix, mask=mask_v, annot=True, fmt='.2f',
            cmap='YlOrRd', vmin=0, vmax=1, linewidths=0.5, ax=axes[1],
            annot_kws={'size': 8})
axes[1].set_title("Cramér's V (vars. categóricas + target)", fontsize=8)
plt.suptitle('UCI Cleveland: Matrices de correlación/asociación', fontsize=13)
plt.tight_layout()
plt.savefig('uci_correlaciones.png', dpi=120, bbox_inches='tight')
plt.close()

# 2.7 — Información mutua vs |Spearman|
print("\n--- 2.7 Información Mutua vs |Spearman| ---")
# Imputamos temporalmente para MI (no admite NaN)
df_tmp = df.copy()
for col in df_tmp.columns:
    if col in VARS_CONTINUAS:
        df_tmp[col] = df_tmp[col].fillna(df_tmp[col].median())
    else:
        df_tmp[col] = df_tmp[col].fillna(df_tmp[col].mode()[0])

discrete_mask = [col in VARS_CATEGORICAS 
                 for col in df_tmp.drop(columns=[TARGET]).columns]
mi_scores_full = mutual_info_classif(
    df_tmp.drop(columns=[TARGET]),
    df_tmp[TARGET],
    discrete_features=discrete_mask,
    random_state=42)
mi_df_full = pd.DataFrame({
    'var': df_tmp.drop(columns=[TARGET]).columns,
    'MI':  mi_scores_full,
    #comparamos con |Spearman| para detectar relaciones no lineales
    'Spearman': [abs(corr_spear.loc[c, TARGET])
                 for c in df_tmp.drop(columns=[TARGET]).columns]
}).sort_values('MI', ascending=False)
print(mi_df_full.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(mi_df_full))
#MI en azul, |Spearman| en rojo
bars1 = ax.bar(x_pos - 0.2, mi_df_full['MI'], 0.38, 
               label='Inf. Mutua', color='#4e79a7', alpha=0.85)
bars2 = ax.bar(x_pos + 0.2, mi_df_full['Spearman'], 0.38, 
               label='|Spearman|', color='#e05c4b', alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(mi_df_full['var'], rotation=30, ha='right', fontsize=7)
ax.set_ylabel('Valor'); ax.legend(fontsize=10)
ax.set_title('UCI — Información Mutua vs |Spearman| con target', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('uci_mi_spearman.png', dpi=120)
plt.close()

# 2.8 — Pairplot continuas por clase
print("\n--- 2.8 Pairplots vars continuas por clase: ---")
df_pair = df[VARS_CONTINUAS + [TARGET]].copy()
df_pair[TARGET] = df_pair[TARGET].astype(str)
g = sns.pairplot(df_pair, hue=TARGET,
                 palette={'0':'#4e79a7','1':'#76b7b2','2':'#f28e2b',
                          '3':'#e05c4b','4':'#59a14f'},
                 plot_kws={'alpha':0.5, 'edgecolor':'none'},
                 diag_kind='kde', height=2.2)
g.figure.suptitle('UCI Cleveland: Pairplot continuas por grado', fontsize=10, y=1.02)
plt.savefig('uci_pairplot.png', dpi=120, bbox_inches='tight')
plt.close()


# =============================================================================
# BLOQUE 3A: DISCRETIZACIÓN (RED BAYESIANA HÍBRIDA)
# =============================================================================

print("\n" + "=" * 70)
print("BLOQUE 3A: DISCRETIZACIÓN Y DATASET HÍBRIDO")
print("=" * 70)

# Tenemos vars discretas y continuas -> red HÍBRIDA (CLG: Conditional Linear Gaussian).

# 3A.1 — Comparativa de estrategias de discretización 
def entropia_cond(serie, target_s):
    """
    Entropía condicional H(Y | X_disc).
    Menor valor: la discretización captura mejor la relación con el target.
    H(Y|X) = Σₓ P(X=x) · H(Y|X=x)
    H(Y|X=x) = −Σᵧ P(Y=y|X=x) log₂ P(Y=y|X=x)
    input:
        serie: variable discretizada X
        target: variable Y
    output:
        h: entropía total
    """
    df_t = pd.DataFrame({'x': serie, 'y': target_s}).dropna() #(x,y)
    h = 0.0
    for v in df_t['x'].unique():
        mask_v = df_t['x'] == v
        p_x   = mask_v.mean()
        py_x  = df_t.loc[mask_v, 'y'].value_counts(normalize=True)
        h    += p_x * (-np.sum(py_x * np.log2(py_x + 1e-12)))
    return h

print("--- 3A.1 Comparativa de discretización (ejemplo con variable chol) ---")
chol_ew   = pd.cut(df['chol'], bins=3, labels=[0, 1, 2])
chol_ef3  = pd.qcut(df['chol'], q=3, labels=[0, 1, 2], duplicates='drop')
chol_clin = pd.cut(df['chol'], bins=[0, 200, 240, 600], labels=[0, 1, 2])

for nom, s in [('Equal-width k=3', chol_ew),
               ('Equal-freq  q=3', chol_ef3),
               ('Clínica AHA',     chol_clin)]:
    print(f"  {nom:<26}  H = {entropia_cond(s, df[TARGET]):.4f} bits")

# Comparar con percentiles reales del dataset
p33 = df['chol'].quantile(0.333)
p67 = df['chol'].quantile(0.667)
print(f"""
  Percentiles reales (qcut q=3): P33={p33:.0f}  P67={p67:.0f} mg/dL
  Cortes clínicos AHA:           200  y  240  mg/dL

  Se aprecia similitud:
    P33≈{p33:.0f} vs 200 AHA: diferencia = {abs(p33-200):.0f} mg/dL
    P67≈{p67:.0f} vs 240 AHA: diferencia = {abs(p67-240):.0f} mg/dL

  Elegimos los cortes clínicos ya que se aproximan bastante a los cuantiles naturales del dataset ({abs(p33-200):.0f} y {abs(p67-240):.0f} mg/dL de diferencia).
  y tienen mayor interpretabilidad médica directa validada.
""")

# Misma comparativa para age y oldpeak
print("  Comparativa para age:")
age_eq3 = pd.cut(df['age'], bins=3, labels=[0, 1, 2])
age_ef4 = pd.qcut(df['age'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
age_cl  = pd.cut(df['age'], bins=[0, 40, 55, 70, 120], labels=[0, 1, 2, 3])
for nom, s in [('Equal-width k=3', age_eq3),
               ('Equal-freq  q=4', age_ef4),
               ('Clínica (<40/40-55/55-70/>70)', age_cl)]:
    print(f"    {nom:<38}  H = {entropia_cond(s, df[TARGET]):.4f} bits")
for q_i, q_v in [(0.25, 'P25'), (0.5, 'P50'), (0.75, 'P75')]:
    print(f"    {q_v} age real = {df['age'].quantile(q_i):.0f} años  "
          f"(cortes clínicos: 40 / 55 / 70)")

print("\n  Comparativa para oldpeak:")
op_eq4 = pd.cut(df['oldpeak'], bins=4, labels=[0, 1, 2, 3], duplicates='drop')
# oldpeak tiene muchos ceros -> qcut genera bins duplicados que se eliminan,
# dejando menos de 4 intervalos. Detectamos cuántos quedan y ajustamos labels.
_op_bins = df['oldpeak'].dropna().quantile([0, 0.25, 0.5, 0.75, 1.0])
_op_bins_uniq = _op_bins.unique()
_n_op = len(_op_bins_uniq) - 1
op_ef4 = pd.qcut(df['oldpeak'], q=4, labels=list(range(_n_op)), duplicates='drop')
op_cl  = pd.cut(df['oldpeak'], bins=[-0.1, 0, 1, 2, 10], labels=[0, 1, 2, 3])
for nom, s in [('Equal-width k=4', op_eq4),
               ('Equal-freq  q=4', op_ef4),
               ('Clínica (0/1/2/>2)', op_cl)]:
    print(f"    {nom:<38}  H = {entropia_cond(s, df[TARGET]):.4f} bits")
for q_i, q_v in [(0.25, 'P25'), (0.5, 'P50'), (0.75, 'P75')]:
    print(f"    {q_v} oldpeak real = {df['oldpeak'].quantile(q_i):.0f} mm  "
          f"(cortes clínicos: 0/1/2/>2)")    

# 3A.2 — Diccionarios de discretización clínica (solo age y oldpeak)
print("\n--- 3A.2 Discretización clínica: age y oldpeak ---")

UCI_DISC = {
    'age':      ([0,40,55,70,120], [0,1,2,3]), # <40|40-55|55-70|>70
    'oldpeak':  ([-0.1,0,1,2,10], [0,1,2,3]), # ninguna|leve|mod|severa
}
UCI_ETIQ = {
    'age':     {0: '<40', 1: '40-55', 2: '55-70', 3: '>70'},
    'oldpeak': {0: 'ninguna', 1: 'leve', 2: 'moderada', 3: 'severa'},
}

def discretizar_uci(df_in):
    """
    Aplica discretización clínica SOLO a age y oldpeak, convirtiendo estas variables 
    continuas en discretas mediante intervalos clínicamente definidos.
    """
    df_d = df_in.copy()
    for col, (bins, labels) in UCI_DISC.items():
        if col in df_d.columns:
            df_d[col] = pd.cut(df_in[col], bins=bins, labels=labels)
            df_d[col] = pd.to_numeric(df_d[col], errors='coerce') #categorias -> numeros
    return df_d

df_disc = discretizar_uci(df)

print("  Verificación discretización:")
for col, etiq in UCI_ETIQ.items():
    cnt   = df_disc[col].value_counts().sort_index()
    parts = [f"{etiq[k]}:{cnt.get(k, 0)}" for k in etiq]
    print(f"    {col:<10} " + "  ".join(parts))

# 3A.3 — Imputación final
## Dataset para la RB discreta (solo vars discretas)

VARS_DISCRETAS_RB = ['age', 'sex', 'cp', 'fbs', 'restecg', 'exang',
                      'slope', 'ca', 'thal', 'oldpeak', TARGET]
VARS_CONTINUAS_RB = ['trestbps', 'chol', 'thalach']

# df_clean: todas las vars en escala original, missings imputados 
df_clean = df.copy()
for col in df_clean.columns:
    if col in VARS_CONTINUAS_RB:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())  # mediana
    else:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0]) # moda

# df_norm: igual que df_clean pero z-score sobre vars numéricas
feats_all = [c for c in df_clean.columns if c != TARGET]
num_cols   = [c for c in feats_all if df_clean[c].nunique() > 4]
df_norm    = df_clean.copy()
df_norm[num_cols] = StandardScaler().fit_transform(df_clean[num_cols])

# df_hybrid: age y oldpeak discretizadas, 11 vars discretas (int) imputadas por moda y 2 vars continuas (float) imputados por mediana 
df_hybrid = df_disc.copy()  # df_disc = discretizar_uci(df): age/oldpeak ya discretizadas
for col in VARS_DISCRETAS_RB:
    # imputa por moda solo las discretas (ca y thal son las únicas con missings)
    df_hybrid[col] = df_hybrid[col].fillna(df_hybrid[col].mode()[0]).astype(int)
for col in VARS_CONTINUAS_RB:
    # toma las continuas de df_clean: ya imputadas por mediana, escala original
    df_hybrid[col] = df_clean[col].astype(float)

print(f"  df_clean  {df_clean.shape}  (age/oldpeak continuos, todo imputado)")
print(f"  df_norm   {df_norm.shape}  (z-score)")
print(f"  df_hybrid {df_hybrid.shape}  (11 disc int + 3 cont float)")
 

# =============================================================================
# BLOQUE 3B: BALANCEO DE CLASES
# =============================================================================

print("\n" + "=" * 70)
print("BLOQUE 3B: BALANCEO DE CLASES")
print("=" * 70)
# Se usa el dataset original sin modificar para no modificar P(target)

vc_target = df_hybrid[TARGET].value_counts().sort_index()
print("  Distribución original (se mantiene sin modificar):")
for k, v in vc_target.items():
    ratio = v / len(df_hybrid)
    bar = '█' * int(ratio * 40)
    print(f"    Clase {k}: {v:>3} ({ratio*100:.1f}%):  {bar}")

ratio_desbalanceo = vc_target.max() / vc_target.min()
print(f"  Ratio de desbalanceo: {ratio_desbalanceo:.1f}x "
      f"({'SEVERO' if ratio_desbalanceo > 3 else 'moderado'})")


# =============================================================================
# BLOQUE 3C: DIVISIÓN TRAIN / TEST (80/20)
# =============================================================================
print("\n" + "=" * 70)
print("BLOQUE 3C: DIVISIÓN TRAIN/TEST 80/20")
print("=" * 70)

# Con N=303 pacientes usamos 80% train / 20% test (estratificado)

X_all_hyb = df_hybrid.drop(columns=[TARGET])   # 13 features: 11 disc + 3 cont
y_all         = df_hybrid[TARGET]
X_all_full    = df_clean.drop(columns=[TARGET])    # 13 features: age/oldpeak continuos
X_all_norm    = df_norm.drop(columns=[TARGET])     # 13 features normalizados

# División train/test 80/20 estratificada (SIEMPRE sobre datos originales)
# Usamos X_all_hibrido para mantener los índices originales del df_hybrid
X_train_hyb, X_test_hyb, y_train, y_test = train_test_split(
    X_all_hyb, y_all, test_size=0.20, random_state=42, stratify=y_all)

# Todos los datasets tienen exactamente los mismos pacientes en train y test
X_train_full  = X_all_full.loc[X_train_hyb.index]
X_train_norm  = X_all_norm.loc[X_train_hyb.index]
X_test_full   = X_all_full.loc[X_test_hyb.index]
X_test_norm   = X_all_norm.loc[X_test_hyb.index]

print(f"  Train: {len(y_train):>3} pacientes ({len(y_train)/len(y_all)*100:.0f}%)")
print(f"  Test:  {len(y_test):>3} pacientes ({len(y_test)/len(y_all)*100:.0f}%)")

print("\n  Distribución de clases (verificación estratificación):")
for nombre, y_s in [('Train', y_train), ('Test', y_test)]:
    dist = y_s.value_counts(normalize=True).sort_index()
    print(f"  {nombre}: " + "  ".join([f"C{k}:{v*100:.0f}%" for k, v in dist.items()]))

# Construcción del dataset de entrenamiento final:
df_train_hibrido = pd.concat(
    [X_train_hyb.reset_index(drop=True),
     y_train.reset_index(drop=True)], axis=1)

for col in VARS_DISCRETAS_RB:
    if col in df_train_hibrido.columns:
        df_train_hibrido[col] = df_train_hibrido[col].astype(int)
for col in VARS_CONTINUAS_RB:
    if col in df_train_hibrido.columns:
        df_train_hibrido[col] = df_train_hibrido[col].astype(float)

# Dataset de entrenamiento final: original sin modificar:
# Extraemos X e y del dataset final
X_train_res = df_train_hibrido.drop(columns=[TARGET])
y_train_res = df_train_hibrido[TARGET]
# Dataset solo con vars discretas (para pgmpy/algoritmos que no admiten continuas)
df_train_disc_only = df_train_hibrido[VARS_DISCRETAS_RB].copy().astype(int)

# Cross-Validation: 5 folds estratificados sobre el train final
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"\n  CV: {SKF.n_splits}-fold estratificado sobre train final")
print(f"  Total muestras train: {len(y_train_res)}")
print(f"  Total muestras test:  {len(y_test)} (no tocadas)")

# Resumen final
print("\n  Distribución final en train_res:")
vc_final = y_train_res.value_counts().sort_index()
for k, v in vc_final.items():
    ratio = v / len(y_train_res)
    bar = '█' * int(ratio * 40)
    print(f"    Clase {k}: {v:>3} ({ratio*100:.1f}%):  {bar}")
 
    
## Para el BLOQUE 4: Discretización de vars continuas para aprendizaje de estructura -> df_train_disc_all (todas vars discretizadas)
DISC_CONTINUAS = {
    'chol':     ([0, 200, 240, 600],   [0, 1, 2]),   # AHA: bajo/límite/alto
    'trestbps': ([0, 120, 140, 250],   [0, 1, 2]),   # AHA: normal/elevada/HTA
    'thalach':  ([0, 100, 140, 220],   [0, 1, 2]),   # baja/normal/alta
}

# df_disc_all: 14 vars TODAS discretizadas (int)
df_disc_all = df_hybrid.copy()
for col, (bins, labels) in DISC_CONTINUAS.items():
    df_disc_all[col] = pd.cut(df_clean[col], bins=bins,    # df_clean: ya imputado
                               labels=labels, include_lowest=True)
    df_disc_all[col] = pd.to_numeric(df_disc_all[col],
                                      errors='coerce').astype(int)
    # no necesita fillna: df_clean no tiene missings en estas variables

df_train_disc_all = df_disc_all.loc[X_train_hyb.index].reset_index(drop=True)
for col in df_train_disc_all.columns:
    df_train_disc_all[col] = df_train_disc_all[col].astype(int)

print(f"\n  df_train_disc_all (todas vars discretizadas): "
      f"{df_train_disc_all.shape}")
print(f"  Columnas: {list(df_train_disc_all.columns)}")

 # Verificación
for col in ['trestbps', 'chol', 'thalach']:
    vc = df_train_disc_all[col].value_counts().sort_index()
    etiq = {0: 'bajo', 1: 'normal', 2: 'alto'}
    partes = [f"{etiq[k]}:{v}" for k, v in vc.items()]
    print(f"  {col:<12}: {'  '.join(partes)}")


# =============================================================================
# BLOQUE 4: APRENDIZAJE DE ESTRUCTURA DE LA RB
# =============================================================================

print("\n" + "=" * 70)
print("BLOQUE 4: ESTRUCTURA DE LA RED BAYESIANA")
print("=" * 70)
 
"""
  El aprendizaje de estructura requiere que TODAS las variables sean
  discretas (restricción de pgmpy). Se usa df_train_disc_all (14 vars
  todas discretizadas). En el Bloque 5, los nodos continuos originales
  (trestbps, chol, thalach) se reparametrizarán con distribuciones
  gaussianas lineales (CLG) usando df_train_hibrido.
"""
 
try:
    import pybnesian as pbn
    PYBNESIAN_OK = True
    print("  pybnesian disponible")
except ImportError:
    PYBNESIAN_OK = False
    print("  AVISO: pybnesian no instalado. Solo red discreta pgmpy.")
 
# 0 - Funciones de análisis estructural
 
def get_parents_map(dag_edges):
    """
    Construye un diccionario con los padres directos de un nodo en el DAG.
    dag_edges: lista de arcos dirigidos (u,v)
    Devuelve:
        dict: diccionario {nodo: cto de padres}
    """
    pm = {}
    for u, v in dag_edges:
        pm.setdefault(v, set()).add(u)
        pm.setdefault(u, set())
    return pm
 
def get_children_map(dag_edges):
    """
    Construye un diccionario con los hijos directos de un nodo en el DAG.
    dag_edges: lista de arcos dirigidos (u,v)
    Devuelve:
        dict: diccionario {nodo: cto de padres}
    """
    cm = {}
    for u, v in dag_edges:
        cm.setdefault(u, set()).add(v)
        cm.setdefault(v, set())
    return cm
 
    
def get_v_structures(model):
    """
    Devuelve lista (X,Y,Z) de v-structures (X->Z<-Y) donde X e Y no son adyacentes
    """
    dag_edges = list(model.edges()) #arcos
    edge_set  = set(dag_edges) | {(v, u) for u, v in dag_edges} #cto adyacencia: 2 nodos conectados? sin direccion
    pm = get_parents_map(dag_edges)  #padres de cada nodo
    vs = []
    for z in model.nodes(): #examinamos todos los pares posibles de padres (X,Y) de Z
        pads = list(pm.get(z, set()))
        for i in range(len(pads)):
            for j in range(i + 1, len(pads)):
                x, y = pads[i], pads[j]
                if (x, y) not in edge_set and (y, x) not in edge_set: # si X,Y padres no adyacentes -> v-structure
                    vs.append((x, z, y))
    return vs
 
def esqueleto(model):
    """
    Obtiene el esqueleto del DAG eliminando la orientación de los arcos
    Devuelve:
        set: conjunto de pares de nodos no dirigidos
    """
    return {frozenset(e) for e in model.edges()}
 
def son_markov_equivalentes(m1, m2):
    """
    Comprueba si dos RRBB cumplen la Equivalencia Markov: mismo esqueleto Y mismas v-structures.
    Devuelve:
        tuple (bool, str): indica si existe equivalencia y la razón.
    """
    esq1, esq2 = esqueleto(m1), esqueleto(m2)
    if esq1 != esq2:
        return False, (f"esqueletos distintos: "
                       f"|solo M1|={len(esq1-esq2)}, |solo M2|={len(esq2-esq1)}")
    # si los esqueletos coinciden, calculamos las vstructures y comparamos
    vs1 = {frozenset([x, z, y]) for x, z, y in get_v_structures(m1)}
    vs2 = {frozenset([x, z, y]) for x, z, y in get_v_structures(m2)}
    if vs1 == vs2:
        return True, "Markov-equivalentes (mismo esqueleto + mismos h-to-h patterns)"
    return False, (f"mismo esqueleto, v-structs distintos: "
                   f"+{len(vs1-vs2)} en M1, +{len(vs2-vs1)} en M2")
 
def shd(edges1, edges2):
    """
    Calcula la Structural Hamming Distance (SHD) entre dos DAGs. 
    Mide el número mínimo de operaciones necesarias para transformar una estructura en otra (inserciones, eliminaciones y inversiones)
    """
    s1 = set(map(tuple, edges1)) #convertimos las listas de arcos en conjuntos
    s2 = set(map(tuple, edges2))
    dist = 0
    procesadas = set()
    for e in s1: #recorremos arcos del dag1
        par = frozenset(e)
        if par in procesadas:
            continue
        procesadas.add(par)
        rev = (e[1], e[0]) #arco invertido
        if e in s2: #mismo arco
            pass
        elif rev in s2: #arco invertido 
            dist += 1 # inversión
        else: #arco eliminado
            dist += 1 # eliminación
    for e in s2: #recorremos arcos nuevos en dag2
        par = frozenset(e)
        if par in procesadas:
            continue
        procesadas.add(par)
        if (e[1], e[0]) not in s1: #añadimos
            dist += 1 # inserción
    return dist
 
def markov_blanket_completo(model, nodo):
    """
    Obtiene la manta de Markov de un nodo: MB(X) = Pa(X) ∪ Ch(X) ∪ CoP(X)
    Este conjunto contiene toda la información necesaria para predecir el nodo, haciéndolo independiente del resto de vars de la red
    CoP(X) = {Pa(Y) \\ {X} : Y ∈ Ch(X)}  
    """
    if nodo not in model.nodes(): #existe el nodo?
        return set(), set(), set()
    dag_edges = list(model.edges()) #arcos del dag
    parents_map  = get_parents_map(dag_edges)
    children_map = get_children_map(dag_edges)
    padres = parents_map.get(nodo, set())
    hijos  = children_map.get(nodo, set())
    copadres = set()
    for h in hijos:
        for p in parents_map.get(h, set()):
            if p != nodo:
                copadres.add(p)
    return padres, hijos, copadres
 
def mb(model, nodo):
    """
    MB(nodo) como lista ordenada.
    """
    p, h, cp = markov_blanket_completo(model, nodo)
    return sorted(p | h | cp)
 
# métricas de puntuación para evaluar estructuras de RRBB:

def bic_formal(model, data):
    """
    Calcula el Bayesian Information Criterion (BIC) de una red: aproximación frecuentista
    Devuelve BIC final, Log-verosimilitud, penalización y nº de parámetros
    """
    data = data.copy()
    for c in data.columns: #convertimos vars a enteros 
        data[c] = pd.to_numeric(data[c], errors='coerce').fillna(0).astype(int)
    N = len(data); log_L = 0.0; dim_S = 0 #inicializamos
    for nodo in model.nodes(): #recorremos cada nodo: BIC global es la suma de contribuciones locales de cada nodo
        if nodo not in data.columns:
            continue
        r_i  = data[nodo].nunique() #cardinalidad
        pads = [u for u, v in model.edges() #padres
                if v == nodo and u in data.columns]
        if not pads: # A - nodo SIN padres
            dim_S += r_i - 1 #suma dist.multinomial = 1
            for k in data[nodo].unique():
                N_ijk = (data[nodo] == k).sum() #nº obs del estado k
                if N_ijk > 0:
                    log_L += N_ijk * np.log(N_ijk / N)
        else: # B - nodo CON padres
            grupos = data.groupby(pads) #agrupamos las obs por configuracion de padres
            q_i = grupos.ngroups
            dim_S += q_i * (r_i - 1) #complejidad del modelo
            for _, g in grupos:
                N_ij = len(g)
                if N_ij == 0:
                    continue
                for k in data[nodo].unique():
                    N_ijk = (g[nodo] == k).sum()
                    if N_ijk > 0:
                        log_L += N_ijk * np.log(N_ijk / N_ij)
    pen = dim_S * 0.5 * np.log(N)
    return log_L - pen, log_L, pen, dim_S
 
def bic_score(model, data):
    return bic_formal(model, data)[0]
 
def bdeu_score(model, data, ess=5):
    """
    Calcula la puntuación BDeu (Bayesian Dirichlet equivalent uniform): score bayesiano.
    Utiliza una distribución Dirichlet uniforme como prior
    Intenta pgmpy, si falla usa implementación manual de Heckerman 
    """
    from scipy.special import gammaln
    di = data.copy()
    for c in di.columns:
        di[c] = pd.to_numeric(di[c], errors='coerce').fillna(0).astype(int)
    try:
        scorer = pgBDeu(di, equivalent_sample_size=ess) #pgmpy
        score_val = sum(scorer.local_score(n, list(model.get_parents(n)))
                        for n in model.nodes() if n in di.columns)
        if np.isfinite(score_val):
            return float(score_val)
    except Exception:
        pass
    # Fallback manual basada en logΓ(x)
    score_total = 0.0
    for nodo in model.nodes():
        if nodo not in di.columns:
            continue
        vals_nodo = di[nodo].dropna().unique()
        r_i = len(vals_nodo) #cardinalidades
        if r_i == 0:
            continue
        pads = [u for u, v in model.edges()
                if v == nodo and u in di.columns]
        if not pads: # Caso SIN padres
            alpha = ess / r_i
            N_ij  = len(di)
            # αij​=ri​ * α
            term  = gammaln(r_i * alpha) - gammaln(r_i * alpha + N_ij)
            for k_val in vals_nodo:
                N_ijk = int((di[nodo] == k_val).sum())
                term += gammaln(alpha + N_ijk) - gammaln(alpha)
            score_total += term
        else: # Caso CON padres: se agrupan las obs según los padres
            grupos_iter = di.groupby(pads)
            q_i = grupos_iter.ngroups
            if q_i == 0:
                continue
            alpha = ess / (q_i * r_i) # BDeu reparte uniformemente el conocimiento previo ESS
            for _, grupo in grupos_iter:
                N_ij = len(grupo)
                term = gammaln(r_i * alpha) - gammaln(r_i * alpha + N_ij)
                for k_val in vals_nodo:
                    N_ijk = int((grupo[nodo] == k_val).sum())
                    term += gammaln(alpha + N_ijk) - gammaln(alpha)
                score_total += term #suma la contribución de cada nodo y de cada configuracion de padres
    return float(score_total)
 
def bic_local(nodo, padres, data):
    """
    BIC local de un único nodo dado sus padres.
    Misma fórmula que bic_formal() pero restringida a P(Xi|Pa(Xi))
    Muchas operaciones de aprendizaje estructural comparan únicamente cambios locales -> Usado para desempate de dirección en la red común.
    """
    d = data.copy()
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0).astype(int)
    N   = len(d)
    r_i = d[nodo].nunique()
    if r_i == 0:
        return -np.inf
    if not padres:
        dim = r_i - 1
        ll  = sum(n * np.log(n / N)
                  for k in d[nodo].unique()
                  if (n := int((d[nodo] == k).sum())) > 0)
        return ll - dim * 0.5 * np.log(N)
    grupos = d.groupby(padres)
    q_i   = grupos.ngroups
    dim   = q_i * (r_i - 1)
    ll    = 0.0
    for _, g in grupos:
        N_ij = len(g)
        if N_ij == 0:
            continue
        for k in d[nodo].unique():
            N_ijk = int((g[nodo] == k).sum())
            if N_ijk > 0:
                ll += N_ijk * np.log(N_ijk / N_ij)
    return ll - dim * 0.5 * np.log(N)
 
def bayes_ball_d_sep(dag_edges, X_set, Y_set, Z_set):
    """
    Algoritmo Bayes-Ball: comprueba si dos ctos de vars X e Y están d-separados por un cto Z.
    Devuelve True si X ⊥⊥ Y | Z en el DAG.
    """
    parents = {}; children = {}; all_nodes = set() #contruccion DAG
    for u, v in dag_edges:
        all_nodes.update([u, v])
        children.setdefault(u, set()).add(v)
        parents.setdefault(v,  set()).add(u)
    for n in all_nodes:
        children.setdefault(n, set()); parents.setdefault(n, set())
 
    def ancestros(zs):
        # Cálculo de ancestros de los nodos observados, ya que en B-BALL un colisionador se activa cuando
        # el colisionador está observado o alguno de sus descendientes esta observado
        anc = set(zs); cola = list(zs)
        while cola:
            n = cola.pop()
            for p in parents.get(n, set()):
                if p not in anc:
                    anc.add(p); cola.append(p)
        return anc
 
    anc_Z = ancestros(Z_set); Z_set = set(Z_set)
    visitados = set(); cola = []
    # inicializacion de la busqueda
    for x in X_set: #la pelota comienza en los nodos de X
        if x not in Z_set: cola.append((x, 'up')) #up: la bola llega desde abajo
        if x in anc_Z:     cola.append((x, 'up'))
    reachable = set()
    while cola:
        nodo, d = cola.pop()
        if (nodo, d) in visitados: continue
        visitados.add((nodo, d))
        if nodo not in Z_set:
            reachable.add(nodo)
        if d == 'up':
            if nodo not in Z_set:
                # no observado: fluye en ambas direcciones
                for p in parents.get(nodo, set()):  cola.append((p, 'up'))
                for c in children.get(nodo, set()): cola.append((c, 'down'))
            else:
                # observado llegado desde abajo: solo hacia padres
                for p in parents.get(nodo, set()):  cola.append((p, 'up'))
        elif d == 'down':
            if nodo not in Z_set:
                # no observado llegado desde arriba: hacia hijos
                for c in children.get(nodo, set()): cola.append((c, 'down'))
            # colisionador activo: observado O descendiente de observado
            if nodo in Z_set or nodo in anc_Z:
                for p in parents.get(nodo, set()):  cola.append((p, 'up'))
    return Y_set.isdisjoint(reachable) #comprueba si algun nodo de Y fue alcanzado -> sí: existe camino activo y no hay d-separacion. No: todos los caminos están bloqueados, hay d-separacion
 
def clasificar_estructuras_locales(model):
    """
    Clasifica tripletas en tail-to-tail, head-to-tail, head-to-head.
    """
    dag_edges = list(model.edges())
    cm = get_children_map(dag_edges); pm = get_parents_map(dag_edges)
    edge_set  = set(dag_edges) | {(v, u) for u, v in dag_edges}
    est = {'tail-to-tail': [], 'head-to-tail': [], 'head-to-head': []}
    for c in model.nodes():
        pads_c = pm.get(c, set()); hijos_c = cm.get(c, set())
        hl = list(hijos_c)
        # tail-to-tail: c tiene ≥2 hijos no adyacentes entre sí
        for i in range(len(hl)):
            for j in range(i+1, len(hl)):
                a, b = hl[i], hl[j]
                if (a,b) not in edge_set and (b,a) not in edge_set:
                    est['tail-to-tail'].append((a, c, b))
        # head-to-tail: padre ->c ->hijo
        for p in pads_c:
            for h in hijos_c:
                est['head-to-tail'].append((p, c, h))
        # head-to-head: v-structure
        pl = list(pads_c)
        for i in range(len(pl)):
            for j in range(i+1, len(pl)):
                x, y = pl[i], pl[j]
                if (x,y) not in edge_set and (y,x) not in edge_set:
                    est['head-to-head'].append((x, c, y))
    return est
 
def viz_grafo_completo(model, titulo, ax, vars_cont=None, target=TARGET):
    """
    Visualiza TODOS los nodos del grafo.
    Colores: rojo=target, azul=padre directo de target,
             naranja=continua originalmente discretizada para este bloque, verde=resto.
    """
    if vars_cont is None:
        vars_cont = set()
    G = nx.DiGraph(model.edges()) #convierte RB en grafo
    for n in model.nodes():
        if n not in G.nodes:
            G.add_node(n)
    def color(n):
        if n == target:           return '#e05c4b'
        if n in vars_cont:        return '#f28e2b'
        if G.has_edge(n, target): return '#4e79a7'
        return '#59a14f'
    cmap = [color(n) for n in G.nodes()]
    pos  = nx.spring_layout(G, seed=42, k=2.2)
    nx.draw_networkx(G, pos=pos, ax=ax, node_color=cmap, node_size=2000,
                 font_size=10, font_color='white', arrows=True,
                 arrowsize=15, edge_color='#444', width=1.1)
    ax.set_title(f'{titulo}\n({len(G.nodes)} nodos, {len(G.edges())} aristas)',
             fontsize=11, pad=5)
    ax.axis('off')
 
    
# 4.1 — Red Experta
print("\n--- 4.1 Red Experta ---")
EXPERT_EDGES = [
    ('age',   TARGET),    
    ('sex',   TARGET),     
    ('cp',    TARGET),   
    ('fbs',   TARGET),     
    (TARGET,  'exang'),    
    (TARGET,  'oldpeak'), 
    (TARGET,  'slope'),  
    (TARGET,  'ca'),       
    (TARGET,  'thal'),    
    (TARGET,  'restecg'),  
    (TARGET,  'thalach'), 
    (TARGET,  'chol'),    
    ('age',   'trestbps'),
]
 
model_expert = DiscreteBayesianNetwork(EXPERT_EDGES)
print(f"  Aristas: {len(EXPERT_EDGES)}")
 
p_exp, h_exp, cp_exp = markov_blanket_completo(model_expert, TARGET)
print(f"  MB(target): padres={sorted(p_exp)}, hijos={sorted(h_exp)}, "
      f"copadres={sorted(cp_exp)}")
print(f"  MB(target) = {sorted(p_exp | h_exp | cp_exp)}")

# 4.2 — Aprendizaje automático de estructura
print("\n--- 4.2 Algoritmos de aprendizaje de estructura ---") 
df_str = df_train_disc_all.copy()
for _col in df_str.columns:
    df_str[_col] = df_str[_col].astype('category')
 
print(f"  Dataset estructura: {df_str.shape} — {list(df_str.columns)}")
 
# Restricción CLG: en una RB híbrida, un nodo discreto NO puede tener padre continuo.
# Como aquí trabajamos con todas las vars discretizadas, en la práctica prohibimos 
# toda arista que salga de una variable continua.
from pgmpy.estimators import ExpertKnowledge
CLG_FORBIDDEN = [(u, v)
                 for u in VARS_CONTINUAS_RB
                 for v in list(df_str.columns)
                 if v != u]
print(f"  Restricción CLG: {len(CLG_FORBIDDEN)} aristas prohibidas "
      f"(ninguna variable continua puede ser padre en este contexto)")

struct_models = {}
 
# A) PC α=0.15
print("\n[A] PC-Algorithm (χ², α=0.15):")
try:
    dag_pc = PC(data=df_str).estimate(
        ci_test='chi_square', significance_level=0.15,
        return_type='dag', show_progress=False)
    edges_pc_raw = list(dag_pc.edges())
    # Filtrar aristas que violen CLG (cont->disc)
    edges_pc = [(u, v) for u, v in edges_pc_raw
                if u not in VARS_CONTINUAS_RB]
    n_filt = len(edges_pc_raw) - len(edges_pc)
    if n_filt > 0:
        print(f"    Filtradas {n_filt} aristas CLG: "
              f"{[(u,v) for u,v in edges_pc_raw if u in VARS_CONTINUAS_RB]}")
    struct_models['PC'] = DiscreteBayesianNetwork(edges_pc) if edges_pc else None
    print(f"    Aristas: {len(edges_pc)}")
except Exception as e:
    struct_models['PC'] = None; print(f"    Falló: {e}")
 
# B) HC-BIC
print("\n[B] Hill-Climbing + BIC:")
try:
    ek_clg = ExpertKnowledge(forbidden_edges=CLG_FORBIDDEN)
    dag_bic = HillClimbSearch(df_str).estimate(
        scoring_method='bic-d',
        expert_knowledge=ek_clg,
        max_indegree=5,
        max_iter=1000, show_progress=False)
    struct_models['HC-BIC'] = DiscreteBayesianNetwork(list(dag_bic.edges()))
    print(f"    Aristas: {len(list(dag_bic.edges()))}")
except Exception as e:
    struct_models['HC-BIC'] = None; print(f"    Falló: {e}")
    
# C) HC-BDeu (ESS=5)
print("\n[C] Hill-Climbing + BDeu (ESS=5):")
try:
    ek_clg_bdeu = ExpertKnowledge(forbidden_edges=CLG_FORBIDDEN)
    dag_bdeu = HillClimbSearch(df_str).estimate(
        scoring_method='bdeu',
        expert_knowledge=ek_clg_bdeu,
        max_indegree=5,
        max_iter=1000, show_progress=False)
    struct_models['HC-BDeu'] = DiscreteBayesianNetwork(list(dag_bdeu.edges()))
    print(f"    Aristas: {len(list(dag_bdeu.edges()))}")
except Exception as e:
    struct_models['HC-BDeu'] = None; print(f"    Falló: {e}")
 
# D) MMHC (Max-Min Hill-Climbing)
print("\n[D] MMHC (Max-Min Hill-Climbing):")
try:
    # Fase 1: esqueleto por PC
    pc_skel, _ = PC(data=df_str).estimate(
        ci_test='chi_square', significance_level=0.15,
        return_type='skeleton', show_progress=False)
    skel_edges = set(frozenset(e) for e in pc_skel.edges())
    all_vars   = list(df_str.columns)
    print(f"    Fase 1 (PC skeleton): {len(skel_edges)} pares candidatos")
 
    # Fase 2: forbidden = pares fuera del esqueleto + restricción CLG
    forbidden = []
    for i, u in enumerate(all_vars):
        for v in all_vars[i+1:]:
            if frozenset([u, v]) not in skel_edges:
                forbidden.append((u, v))
                forbidden.append((v, u))
    # Añadir restricción CLG
    for arista in CLG_FORBIDDEN:
        if arista not in forbidden:
            forbidden.append(arista)
 
    ek = ExpertKnowledge(forbidden_edges=forbidden)
    dag_mmhc = HillClimbSearch(df_str).estimate(
        scoring_method='bic-d',
        expert_knowledge=ek,
        max_indegree=5,
        max_iter=1000,
        show_progress=False)
    struct_models['MMHC'] = DiscreteBayesianNetwork(list(dag_mmhc.edges()))
    print(f"    Fase 2 (HC-BIC restringido + CLG): {len(list(dag_mmhc.edges()))} aristas")
except Exception as e:
    print(f"    MMHC falló ({e}). HC-Tabu como fallback:")
    try:
        ek_tabu = ExpertKnowledge(forbidden_edges=CLG_FORBIDDEN)
        dag_tabu = HillClimbSearch(df_str).estimate(
            scoring_method='bic-d',
            expert_knowledge=ek_tabu,
            tabu_length=10,
            max_indegree=5, max_iter=1000, show_progress=False)
        struct_models['MMHC'] = DiscreteBayesianNetwork(list(dag_tabu.edges()))
        print(f"    Aristas (HC-Tabu+CLG): {len(list(dag_tabu.edges()))}")
    except Exception as e2:
        struct_models['MMHC'] = None; print(f"    Falló: {e2}")
       
        
# 4.3 — Comparación: SHD, BIC, BDeu, v-structures, MB(target)
print("\n--- 4.3 Comparación de estructuras ---")
 
df_bic_str = df_train_disc_all.copy()
 
todos_modelos = {'Experta': model_expert,
                 **{k: v for k, v in struct_models.items() if v is not None}}
modelos_validos = {k: v for k, v in todos_modelos.items() if v is not None}
 
print(f"\n  {'Modelo':<12} {'Aristas':>7} {'SHD_exp':>9} {'BIC':>12} "
      f"{'BDeu':>12} {'V-str':>7}  MB(target)")
print("  " + "─" * 95)
 
bic_scores_dict = {}
for nom, m in modelos_validos.items():
    shd_v  = shd(list(m.edges()), EXPERT_EDGES) if nom != 'Experta' else 0
    bic_v  = bic_score(m, df_bic_str)
    bdeu_v = bdeu_score(m, df_bic_str)
    mb_v   = mb(m, TARGET)
    vs_v   = get_v_structures(m)
    bic_scores_dict[nom] = bic_v
    print(f"  {nom:<12} {len(list(m.edges())):>7} {shd_v:>9} "
          f"{bic_v:>12.1f} {bdeu_v:>12.1f} {len(vs_v):>7}  {mb_v}")
 
# BIC detallado
print(f"\n  BIC detallado (N_train={len(df_bic_str)}):")
print(f"  {'Modelo':<12} {'log L':>14} {'dim(S)':>8} {'penaliz.':>12} {'BIC':>14}")
print("  " + "─" * 72)
for nom, m in modelos_validos.items():
    try:
        bv, ll, pen, dim = bic_formal(m, df_bic_str)
        print(f"  {nom:<12} {ll:>14.1f} {dim:>8} {pen:>12.1f} {bv:>14.1f}")
    except Exception as e:
        print(f"  {nom:<12}  Error: {e}")
 
# ΔBIC respecto al mejor
mejor_nom = max(bic_scores_dict, key=bic_scores_dict.get)
mejor_bic = bic_scores_dict[mejor_nom]
print(f"\n  ΔBIC respecto al mejor ({mejor_nom}):")
for nom, bv in bic_scores_dict.items():
    delta = mejor_bic - bv
    ev = ("mejor" if delta == 0 else "débil" if delta < 2
          else "positiva" if delta < 6 else "fuerte" if delta < 10
          else "MUY FUERTE")
    print(f"    {nom:<12}  ΔBIC={delta:>8.1f}  ->evidencia {ev}")
 
# SHD pairwise
print(f"\n  SHD pairwise:")
nvs = list(modelos_validos.keys())
print(f"  {'':>12}", end="")
for n2 in nvs: print(f"  {n2:>12}", end="")
print()
for n1 in nvs:
    print(f"  {n1:>12}", end="")
    for n2 in nvs:
        d = shd(list(modelos_validos[n1].edges()), list(modelos_validos[n2].edges()))
        print(f"  {d:>12}", end="")
    print()
 

# 4.4 — Equivalencia Markov y elección del modelo principal
print("\n--- 4.4 Equivalencia Markov ---")
 
print(f"\n  V-structures por modelo:")
for nom, m in modelos_validos.items():
    vs = get_v_structures(m)
    vs_str = [f"({x}->{z}<-{y})" for x, z, y in vs[:4]]
    resto  = f" +{len(vs)-4} más" if len(vs) > 4 else ""
    print(f"  {nom:<12}  {len(vs):>2} h2h: {', '.join(vs_str)}{resto}")
 
print(f"\n  Equivalencia Markov pairwise:")
ml = [(n, m) for n, m in modelos_validos.items()]
print(f"  {'M1':<12} {'M2':<12} {'Equiv?':>8}  Razón")
print("  " + "─" * 72)
for i in range(len(ml)):
    for j in range(i+1, len(ml)):
        n1, m1 = ml[i]; n2, m2 = ml[j]
        try:
            eq, razon = son_markov_equivalentes(m1, m2)
            print(f"  {n1:<12} {n2:<12} {'SÍ ' if eq else 'NO ':>8}  {razon}")
        except Exception as e:
            print(f"  {n1:<12} {n2:<12} {'Error':>8}  {e}")
 
# Elección del modelo principal automático
MODELO_PRINCIPAL_NOM = None
for cand in ['MMHC', 'HC-BIC', 'HC-BDeu', 'PC']:
    if modelos_validos.get(cand) is not None:
        MODELO_PRINCIPAL_NOM = cand; break
 
print(f"\n  *** MODELO PRINCIPAL AUTOMÁTICO: {MODELO_PRINCIPAL_NOM} ***")
m_princ     = modelos_validos[MODELO_PRINCIPAL_NOM]
dag_e_princ = list(m_princ.edges())
 

# 4.5 — Red común (consenso de las 3 RB automáticas)
print("\n--- 4.5 Red común (aristas robustas — RB automáticas) ---")

RB_AUTOMATICAS = ['PC', 'HC-BIC', 'HC-BDeu', 'MMHC']
modelos_auto = {k: v for k, v in modelos_validos.items()
                if k in RB_AUTOMATICAS and v is not None}
n_cons_rc = len(modelos_auto)
modelos_consenso_rc = modelos_auto.copy()
 
print(f"  Modelos usados: {list(modelos_consenso_rc.keys())} ({n_cons_rc})")

# Paso 1: contar aristas sin dirección
cnt_rc = Counter()
for m in modelos_consenso_rc.values():
    for e in m.edges():
        cnt_rc[frozenset(e)] += 1
 
umbral_rob_rc = 3   # ≥3 de 4 modelos

# Paso 2: dirección de cada arista robusta
edges_rc_final = []
print(f"\n  Aristas robustas:")
for par, cnt in cnt_rc.items():
    if cnt < umbral_rob_rc:
        continue
    pl = list(par)
    a, b = pl[0], pl[1]
    ab = (a, b); ba = (b, a)
    votos_ab = sum(1 for m in modelos_consenso_rc.values()
                   if ab in list(m.edges()))
    votos_ba = sum(1 for m in modelos_consenso_rc.values()
                   if ba in list(m.edges()))
    if votos_ab > votos_ba:
        dir_elegida = ab; fuente = f"mayoría ({votos_ab} vs {votos_ba})"
    elif votos_ba > votos_ab:
        dir_elegida = ba; fuente = f"mayoría ({votos_ba} vs {votos_ab})"
    else:
        # Empate: primero se consulta la red experta.
        # Si la arista aparece en ella, se respeta esa dirección.
        # Si no está en la red experta, se usa BIC local.
        expert_set = set(map(tuple, EXPERT_EDGES))
        if ab in expert_set:
            dir_elegida = ab; fuente = "empate -> red experta"
        elif ba in expert_set:
            dir_elegida = ba; fuente = "empate -> red experta"
        else:
            try: #elegimos el bic con el que el padre explique mejor a los hijos
                bic_ab = bic_local(b, [a], df_bic_str)
                bic_ba = bic_local(a, [b], df_bic_str)
                if bic_ab >= bic_ba:
                    dir_elegida = ab
                    fuente = f"empate -> BIC_local({b}|{a})={bic_ab:.1f}"
                else:
                    dir_elegida = ba
                    fuente = f"empate -> BIC_local({a}|{b})={bic_ba:.1f}"
            except Exception:
                dir_elegida = ab; fuente = "empate -> fallback"
    edges_rc_final.append(dir_elegida)
    print(f"    {dir_elegida[0]:>12} ->{dir_elegida[1]:<12} "
          f"({cnt}/{n_cons_rc})  [{fuente}]")
 
print(f"\n  Total aristas robustas: {len(edges_rc_final)}")
 
# Paso 3: construir la red común
red_comun = None
try:
    red_comun = DiscreteBayesianNetwork(edges_rc_final)
    modelos_validos['Red_Comun'] = red_comun
    nodos_presentes = set(red_comun.nodes())
    ALL_VARS_DATASET = list(df_train_disc_all.columns)
    faltantes_final = [v for v in ALL_VARS_DATASET if v not in nodos_presentes]
    if faltantes_final:
        print(f"  [Aviso] Nodos ausentes en red común: {faltantes_final}")
    else:
        print(f"\n  Red común: {len(edges_rc_final)} aristas, "
              f"{len(nodos_presentes)} nodos (14/14) Bien")
    p_rc, h_rc, cp_rc = markov_blanket_completo(red_comun, TARGET)
    print(f"  MB(target) red común = {sorted(p_rc | h_rc | cp_rc)}")
except Exception as e_rc:
    print(f"  [Error] No se pudo construir red común: {e_rc}")

# he probado a construir la red comun con 
# 3 aristas en comun (BIC=-2090) y con 2 aristas en comun (BIC=-2600)
# por lo que he decidido escoger la de 3 aristas en común ya que aunque es más
# conservadora, el BIC es mejor.

# 4.6 — Análisis comparativo: Red Experta vs Red Común
print("\n--- 4.6 Comparativa: Red Experta vs Red Común ---")
 
if red_comun is not None:
    # Scores
    bic_exp  = bic_score(model_expert, df_bic_str)
    bic_rc   = bic_score(red_comun,   df_bic_str)
    bdeu_exp = bdeu_score(model_expert, df_bic_str)
    bdeu_rc  = bdeu_score(red_comun,   df_bic_str)
    shd_er   = shd(EXPERT_EDGES, list(red_comun.edges()))
 
    print(f"\n  {'Métrica':<25} {'Experta':>12} {'Común':>12}")
    print("  " + "─" * 52)
    print(f"  {'N° aristas':<25} {len(EXPERT_EDGES):>12} "
          f"{len(list(red_comun.edges())):>12}")
    print(f"  {'SHD entre sí':<25} {shd_er:>12}")
    print(f"  {'BIC':<25} {bic_exp:>12.1f} {bic_rc:>12.1f}")
    print(f"  {'BDeu':<25} {bdeu_exp:>12.1f} {bdeu_rc:>12.1f}")
 
    mb_exp = mb(model_expert, TARGET)
    mb_rc  = mb(red_comun,   TARGET)
    print(f"\n  MB(target) Experta [{len(mb_exp)} vars]: {mb_exp}")
    print(f"  MB(target) Común   [{len(mb_rc)} vars]:  {mb_rc}")
    solo_exp = set(mb_exp) - set(mb_rc)
    solo_rc  = set(mb_rc)  - set(mb_exp)
    if solo_exp: print(f"    Solo en MB Experta: {sorted(solo_exp)}")
    if solo_rc:  print(f"    Solo en MB Común:   {sorted(solo_rc)}")
 
    # Nodos presentes en cada red
    ALL_VARS = list(df_train_disc_all.columns)
    nodos_exp = set(model_expert.nodes())
    nodos_rc  = set(red_comun.nodes())
    print(f"\n  Nodos en Experta: {len(nodos_exp)}/14  "
          f"Ausentes: {sorted(set(ALL_VARS) - nodos_exp) or 'ninguno'}")
    print(f"  Nodos en Común:  {len(nodos_rc)}/14  "
          f"Ausentes: {sorted(set(ALL_VARS) - nodos_rc) or 'ninguno'}")
 
    # Aristas compartidas
    e_exp_set = set(map(tuple, EXPERT_EDGES))
    e_rc_set  = set(map(tuple, list(red_comun.edges())))
    print(f"\n  Aristas comunes:   {len(e_exp_set & e_rc_set)}")
    print(f"  Solo en Experta:   {sorted(e_exp_set - e_rc_set)}")
    print(f"  Solo en Común:     {sorted(e_rc_set - e_exp_set)}")
else:
    print("  [Aviso] Red común no disponible.")
 

# 4.7 — Completar nodos faltantes y re-análisis
print("\n--- 4.7 Completar nodos faltantes ---")
"""
  Si alguna de las dos redes no contiene los 14 nodos del dataset,
  se conectan los nodos faltantes con la variable ya presente en la
  red con la que tienen mayor correlación de Spearman (calculada sobre
  df_train_disc_all). La experta sí tiene los 14, por lo que solo completaremos
  la común. La dirección de la arista se determina consultando
  la red experta; si el par no aparece en ella, se usa BIC local.
  Tras añadir los nodos faltantes se repite el análisis.
  
  Son las que usaremos en el RESTO de bloques, ya que son necesarios todos los 
  nodos para la inferencia.
"""
 
corr_sp = df_train_disc_all.copy().astype(float).corr(method='spearman')
expert_set_dir = set(map(tuple, EXPERT_EDGES))
ALL_VARS = list(df_train_disc_all.columns)
 
def completar_red(model, nombre):
    """
    Añade a la red los nodos ausentes conectándolos con la variable ya presente
    más correlacionada (Spearman). La dirección del arco se obtiene a partir de
    la red experta o, en su defecto, comparando el BIC local de ambas orientaciones.
    """
    edges_actuales = list(model.edges())
    nodos_actuales = set(model.nodes())
    faltantes = [v for v in ALL_VARS if v not in nodos_actuales]
    if not faltantes:
        print(f"  {nombre}: todos los nodos presentes")
        return model
    print(f"  {nombre}: nodos faltantes = {faltantes}")
    nuevas = []
    for vf in faltantes: #restricción CLG
        candidatos = [(v, abs(corr_sp.loc[vf, v]))
                      for v in nodos_actuales
                      if v in corr_sp.columns and v != vf
                      and not (v in VARS_CONTINUAS_RB
                               and vf not in VARS_CONTINUAS_RB)]
        if not candidatos:
            continue
        mejor_var, mejor_corr = max(candidatos, key=lambda x: x[1])
        # Dirección: red experta, y sino, BIC local
        if (vf, mejor_var) in expert_set_dir:
            nueva = (vf, mejor_var); fuente = "red experta"
        elif (mejor_var, vf) in expert_set_dir:
            nueva = (mejor_var, vf); fuente = "red experta"
        else:
            try:
                bic_a = bic_local(vf,       [mejor_var], df_bic_str)
                bic_b = bic_local(mejor_var, [vf],        df_bic_str)
                if mejor_var in VARS_CONTINUAS_RB and vf not in VARS_CONTINUAS_RB:
                    nueva  = (vf, mejor_var)
                    fuente = "CLG: disc->cont forzado"
                elif bic_a >= bic_b:
                    nueva  = (mejor_var, vf) 
                    fuente = f"BIC local ({bic_a:.1f} vs {bic_b:.1f})"
                else:
                    nueva = (vf, mejor_var)
                    fuente = f"BIC local ({bic_b:.1f} vs {bic_a:.1f})"
            except Exception:
                nueva = (mejor_var, vf); fuente = "fallback"
        nuevas.append(nueva)
        nodos_actuales.add(vf)
        print(f"    [{vf}] -> {nueva[0]} -> {nueva[1]}  "
              f"(Spearman={mejor_corr:.3f}, {fuente})")
 
    modelo_completo = DiscreteBayesianNetwork(edges_actuales + nuevas)
    return modelo_completo
 
red_comun_full = completar_red(red_comun, 'Común') if red_comun else None
 

# 4.8 - Análisis estructural de RRBB completas   
print("\n--- 4.8 Análisis estructural: Red Experta vs Red Común (completas) ---")
 
for nombre_red, modelo_red in [('Red experta', model_expert),
                                ('Red común',   red_comun_full)]:
    if modelo_red is None:
        continue
    dag_red     = list(modelo_red.edges())
    dag_red_set = set(dag_red) | {(v, u) for u, v in dag_red}
    nodos_red   = sorted(modelo_red.nodes())
    print(f"\n  ── {nombre_red} ──────────────────────────────────────────")
 
    # Re-análisis: métricas y MB tras completar
    bic_v  = bic_score(modelo_red, df_bic_str)
    bdeu_v = bdeu_score(modelo_red, df_bic_str)
    mb_v   = mb(modelo_red, TARGET)
    print(f"  Nodos: {len(nodos_red)}/14  "
          f"Aristas: {len(dag_red)}  "
          f"BIC: {bic_v:.1f}  BDeu: {bdeu_v:.1f}")
    print(f"  MB(target) [{len(mb_v)} vars]: {mb_v}")
 
    # Estructuras locales
    try:
        est = clasificar_estructuras_locales(modelo_red)
        sim_tipos = {
            'tail-to-tail': 'A<-C -> B',
            'head-to-tail': 'A->C->B',
            'head-to-head': 'A->C<-B (colisionador)',
        }
        for tipo, lista in est.items():
            print(f"\n  {tipo.upper()}  ({len(lista)})  [{sim_tipos[tipo]}]:")
            for a, c, b in lista[:3]:
                interp = (f"({a}⊥⊥{b}) a priori, explaining away dado {c}"
                          if tipo == 'head-to-head' else f"({a}⊥⊥{b}) | {c}")
                print(f"    {a} · {c} · {b}  -> {interp}")
            if len(lista) > 3:
                print(f"    ... y {len(lista)-3} más")
    except Exception as e:
        print(f"  [Aviso] estructuras locales: {e}")
 
    # MB de todos los nodos
    print(f"\n  MB: todos los nodos:")
    print(f"  {'Nodo':<14} {'MB'}")
    print("  " + "─" * 70)
    for nodo_i in nodos_red:
        print(f"  {nodo_i:<14} {mb(modelo_red, nodo_i)}")
 
    # D-separación: todos los pares no adyacentes del grafo
    # condicionando en ∅ y en {target}
    print(f"\n  D-separación (Bayes-Ball):")
    print(f"  {'Xi':<12} {'Xj':<12} {'dado':<18} {'d-sep':>5}  relación")
    print("  " + "─" * 55)
 
    pares_no_adj = [
        (u, v) for i, u in enumerate(nodos_red)
        for v in nodos_red[i+1:]
        if frozenset([u, v]) not in {frozenset(e) for e in dag_red_set}
    ]
    print(f"  [{len(pares_no_adj)} pares no adyacentes]")
    for xi, xj in pares_no_adj:
        for cond in [set(), {TARGET}]:
            cc  = {c for c in cond if c in modelo_red.nodes()}
            sep = bayes_ball_d_sep(dag_red, {xi}, {xj}, cc)
            cs  = '{' + ','.join(sorted(cc)) + '}' if cc else '∅'
            res = 'Sí' if sep else 'No'
            rel = '⊥⊥' if sep else '~⊥⊥'
            print(f"  {xi:<12} {xj:<12} {cs:<18} {res:>5}  {rel}")

# 4.9 — Visualización de RRBB
print("\n--- 4.9 Visualización ---")

# Experta vs Común vs Común completa
fig, axs = plt.subplots(1, 3, figsize=(22, 7))
viz_grafo_completo(model_expert, 'UCI: Red Experta',
                   axs[0], vars_cont=set(VARS_CONTINUAS_RB))
if red_comun is not None:
    viz_grafo_completo(red_comun, 'UCI: Red Común (aristas robustas)',
                       axs[1], vars_cont=set(VARS_CONTINUAS_RB))
else:
    axs[1].text(0.5, 0.5, 'Red común no disponible',
                ha='center', va='center', transform=axs[1].transAxes)
    axs[1].axis('off')
if red_comun_full is not None:
    viz_grafo_completo(red_comun_full, 'UCI: Red Común (completa, 14 nodos)',
                       axs[2], vars_cont=set(VARS_CONTINUAS_RB))
else:
    axs[2].text(0.5, 0.5, 'Red común completa no disponible',
                ha='center', va='center', transform=axs[2].transAxes)
    axs[2].axis('off')

fig.legend(handles=[
    mpatches.Patch(color='#e05c4b', label='Target'),
    mpatches.Patch(color='#4e79a7', label='Padre directo de target'),
    mpatches.Patch(color='#59a14f', label='Efecto/otro nodo'),
    mpatches.Patch(color='#f28e2b', label='Orig. continua (disc. en B4)'),
], loc='lower center', ncol=4, fontsize=10)
plt.suptitle('UCI Cleveland: Red Experta vs Red Común', fontsize=13)
plt.tight_layout()
plt.savefig('uci_experta_vs_comun.png', dpi=120, bbox_inches='tight')
plt.close()


# Experta vs Común completa
fig, axs = plt.subplots(1, 2, figsize=(18, 9))  # más grande
viz_grafo_completo(model_expert, 'UCI: Red Experta',
                   axs[0], vars_cont=set(VARS_CONTINUAS_RB))
if red_comun_full is not None:
    viz_grafo_completo(red_comun_full,
                       'UCI: Red Común (completa, 14 nodos)',
                       axs[1],
                       vars_cont=set(VARS_CONTINUAS_RB))
else:
    axs[1].text(0.5, 0.5, 'Red común completa no disponible',
                ha='center', va='center',
                transform=axs[1].transAxes)
    axs[1].axis('off')
fig.legend(handles=[
    mpatches.Patch(color='#e05c4b', label='Target'),
    mpatches.Patch(color='#4e79a7', label='Padre directo de target'),
    mpatches.Patch(color='#59a14f', label='Efecto/otro nodo'),
    mpatches.Patch(color='#f28e2b', label='Orig. continua (disc. en B4)'),
], loc='lower center', ncol=4, fontsize=13)
plt.suptitle('UCI Cleveland: Red Experta vs Red Común Completa', fontsize=15)
plt.tight_layout()
plt.savefig('uci_experta_vs_comun_completa.png',
            dpi=150, bbox_inches='tight')
plt.close()

# Modelos automáticos
n_auto = len(struct_models)
nc = 2; nr = (n_auto + nc - 1) // nc
fig2, axs2 = plt.subplots(nr, nc, figsize=(18, 9 * nr))
axs2 = np.array(axs2).flatten()
for idx, (nom, m) in enumerate(struct_models.items()):
    if m is None:
        axs2[idx].axis('off'); continue
    viz_grafo_completo(m, f'UCI — {nom}', axs2[idx],
                       vars_cont=set(VARS_CONTINUAS_RB))
for j in range(idx+1, len(axs2)):
    axs2[j].axis('off')
fig2.legend(handles=[
    mpatches.Patch(color='#e05c4b', label='Target'),
    mpatches.Patch(color='#4e79a7', label='Padre directo de target'),
    mpatches.Patch(color='#59a14f', label='Efecto/otro nodo'),
    mpatches.Patch(color='#f28e2b', label='Orig. continua (disc. en B4)'),
], loc='lower center', ncol=4, fontsize=13)
plt.suptitle('UCI Cleveland: RB automáticas', fontsize=15)
plt.tight_layout()
plt.savefig('uci_automaticas.png', dpi=150, bbox_inches='tight')
plt.close()


# =============================================================================
# BLOQUE 5: APRENDIZAJE DE PARÁMETROS (RB HÍBRIDA CLG)
# =============================================================================

print("\n" + "=" * 70)
print("BLOQUE 5: APRENDIZAJE DE PARÁMETROS")
print("=" * 70)
"""
  Vamos a construir dos modelos híbridos CLG completos: modelo_expert y modelo_comun.
  Estos unifican las CPTs discretas y los parámetros CLG.   
"""

from sklearn.linear_model import LinearRegression as LR_sk
import json as _json

nodos_disc = set(VARS_DISCRETAS_RB) # 11 vars discretas
nodos_cont = set(VARS_CONTINUAS_RB) # 3 vars continuas: trestbps, chol, thalach


# 5.0 — Clase RedBayesianaHibridaCLG
print("\n--- 5.0 Clase RedBayesianaHibridaCLG ---")

class RedBayesianaHibridaCLG:
    """
    Clase que unifica en un único objeto la parametrización completa de una RB híbrida CLG:
        Las CPTs de los nodos discretos (θᵢⱼₖ = P(Xᵢ=k | Pa=j)  [BayesianEstimator ESS=5]) 
        y de los nodos continuos (P(Xc | Pa_d=k) = N(μₖ + βₖᵀz, σ²ₖ)  [OLS por grupos]).
        Se tiene en cuenta la restricción CLG: un nodo discreto NO puede tener padres continuos,
        ya que nuestra red solo tiene padres discretos.

    Atributos de la clase:
        nombre: str
        disc: DiscreteBayesianNetwork  (pgmpy, CPTs ESS=5)
        clg: dict {col_c: {estado_padre: (μ, [β], σ²)}}
        edges_all: list [(u,v), ...]  con todas las aristas del DAG
        nodes_disc: set de nodos discretos
        nodes_cont: set de nodos continuos

    """

    def __init__(self, nombre, model_disc, clg_params, edges_all,
                 nodes_disc, nodes_cont):
        self.nombre = nombre
        self.disc = model_disc
        self.clg = clg_params
        self.edges_all = list(edges_all)
        self.nodes_disc = set(nodes_disc)
        self.nodes_cont = set(nodes_cont)
        #inicializa padres e hijos
        self._parents  = {}
        self._children = {}
        all_nodes = set()
        for u, v in self.edges_all:
            all_nodes.update([u, v])
            self._parents.setdefault(v, []).append(u)
            self._children.setdefault(u, []).append(v)
        for n in all_nodes: #todos los nodos tienen entrada en ambos diccionarios
            self._parents.setdefault(n, [])
            self._children.setdefault(n, [])
        self._nodes = sorted(all_nodes)

    ## Topología del modelo:
    def nodes(self): # todos los nodos
        return self._nodes
    
    def edges(self): # todas las aristas
        return self.edges_all
    
    def parents(self, nodo): #padres de un nodo
        return self._parents.get(nodo, [])
    
    def children(self, nodo): # hijos de un nodo
        return self._children.get(nodo, [])
    
    def is_disc(self, nodo): # True si nodo discreto
        return nodo in self.nodes_disc
    
    def is_cont(self, nodo): # True si nodo continuo
        return nodo in self.nodes_cont
    
    def topological_order(self):
        """
        Orden topológico del DAG.
        Calcula el grado de entrada (cuántos podres tiene) de cada nodo.
        El algoritmo va "eliminando" nodos del grafo: al procesar un nodo, elimina 
        conceptualmente todos sus arcos salientes.
        """
        in_degree = {n: len(self._parents[n]) for n in self._nodes} # grado de entrada
        queue = [n for n in self._nodes if in_degree[n] == 0] #nodos sin padre -> nodos raíz
        order = []
        while queue:
            queue.sort() # desempate determinista por nombre alfabético
            n = queue.pop(0) #extrae el primero (menor nombre)
            order.append(n) #lo añadimos
            for c in self._children[n]:
                in_degree[c] -= 1 #eliminamos el arco n->c
                if in_degree[c] == 0: #si c ya no tiene padres pendientes
                    queue.append(c) # procesamos c
        return order

    ## Acceso a parámetros:
    def get_cpd_disc(self, nodo):
        """
        LLama a un método pgmpy para calcular la CPD del nodo discreto.
        Devuelve el objeto Tabular CPD con P(target=k|Pa=j).
        """
        if not self.is_disc(nodo):
            raise ValueError(f"{nodo} no es un nodo discreto.")
        return self.disc.get_cpds(nodo) 
    
    def get_clg(self, nodo, estado_padre):
        """
        Parámetros CLG del nodo continuo dado el estado del padre discreto.
        Devuelve (μ, β, σ²)
        """
        if not self.is_cont(nodo):
            raise ValueError(f"{nodo} no es un nodo continuo.")
        params = self.clg.get(nodo, {}) #dict de estimar_clg_manual() -> tuplas (int, ), int o string
        if estado_padre in params: 
            return params[estado_padre]
        try: #Intentar como int si viene como str
            return params[int(estado_padre)]
        except (ValueError, KeyError):
            pass
        #fallback: primer estado disponible
        if params:
            return next(iter(params.values()))
        raise KeyError(f"No hay parámetros CLG para {nodo} estado={estado_padre}")

    ## Evaluación de probabilidades: 
    def p_disc(self, nodo, k, pa_vals=None):
        """
        Calcula P(nodo=k | padres=pa_vals), siendo pa_vals: dict {nombre_padre: valor} para los padres discretos.
        Si el nodo no tiene padres, pa_vals puede ser None o {}.
        Intenta primero la CPT de pgmpy (model_disc) y, si el nodo no está en model_disc (nodo discreto con
        solo una arista que sea disc->cont), calcula la probabilidad directamente por conteo desde    
        df_train_disc_only con suavizado ESS=5.        
        """
        if pa_vals is None:
            pa_vals = {}
        try: # Intentamos CPT pgmpy
            cpd = self.get_cpd_disc(nodo)
            return float(cpd.get_value(**{nodo: k, **pa_vals}))
        except Exception:
            pass
        try: # Fallback: conteo directo desde df_train_disc_only (ESS=5)
            df = df_train_disc_only
            r  = int(df[nodo].nunique())
            if pa_vals:
                mask = np.ones(len(df), dtype=bool)
                for p, v in pa_vals.items():
                    if p in df.columns:
                        mask &= (df[p] == int(v))
                df_sub = df[mask]
            else:
                df_sub = df
            N_ij  = len(df_sub)
            N_ijk = (df_sub[nodo] == k).sum()
            return float((N_ijk + 5.0/r) / (N_ij + 5.0)) if N_ij > 0 \
                   else 1.0/r
        except Exception:
            r = _card(self, nodo)
            return 1.0/r if r > 0 else 0.0
    
    def p_cont(self, nodo, x, pa_disc=None):
        """
        Calcula la densidad gaussiana f(x | padres_disc=pa_disc), siendo
        pa_disc: valor del padre discreto (int, tuple, 'marginal').
        Devuelve la densidad N(μₖ + βₖᵀz, σ²ₖ) evaluada en x.
        """
        mu, beta, sig2 = self.get_clg(nodo, pa_disc if pa_disc is not None
                                       else 'marginal')
        sigma = np.sqrt(max(sig2, 1e-9)) #evita division por cero (si sig2 muy pequeño)
        return float(np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                     / (sigma * np.sqrt(2 * np.pi))) #densidad gaussiana f(x)

    ## Ancestral sampling: (adicional)
    def sample(self, n, seed=42):
        """
        Genera n muestras del modelo por forward sampling (muestreo ancestral).
        Recorre los nodos en orden topológico:
          - Nodo discreto : muestrea de la CPT P(Xᵢ | Pa=j)
          - Nodo continuo : muestrea de N(μₖ + βₖᵀz, σ²ₖ)
        Devuelve un DataFrame con columnas = nodos.
        """
        rng = np.random.default_rng(seed) #generador de nº aleatorios
        order = self.topological_order() # garantiza que cnd se muestrea un nodo, todos sus padres ya tienen valor
        # Inicializamos vectores con tipo correcto (tamaño n)
        data_disc = {n: np.zeros(n, dtype=int)   for n in order
                     if self.is_disc(n)}
        data_cont = {n: np.zeros(n, dtype=float) for n in order
                     if self.is_cont(n)}
        for i in range(n):
            vals = {}
            for nodo in order: #orden topologico
                pa = self._parents[nodo] #padres
                pa_disc_nodo = [p for p in pa if self.is_disc(p)]
                pa_cont_nodo = [p for p in pa if self.is_cont(p)]
                if self.is_disc(nodo): # A- Nodo discreto
                    pa_vals = {p: int(vals[p]) for p in pa_disc_nodo} #construye la evidencia
                    cpd  = self.get_cpd_disc(nodo) #CPT
                    #Obtenemos distribución condicional:
                    try:
                        if pa_vals: # si hay padres
                            probs = np.array([
                                float(cpd.get_value(**{nodo: k, **pa_vals}))
                                for k in range(cpd.variable_card)
                            ])
                        else: # si no hay padres -> distribución marginal
                            probs = cpd.values.flatten().astype(float)
                        probs = np.clip(probs, 0, None) #evita probs negativas
                        probs /= probs.sum() #normaliza
                        k_sample = int(rng.choice(len(probs), p=probs)) #muestrea estado
                    except Exception:
                        k_sample = 0
                    vals[nodo] = k_sample
                    data_disc[nodo][i] = k_sample #almacenamos en la fila correspondiente del dataset final
                else: # B- Nodo continuo
                    if pa_disc_nodo: # si tiene padres discretos
                        estado = (int(vals[pa_disc_nodo[0]])
                                  if len(pa_disc_nodo) == 1
                                  else tuple(int(vals[p])
                                             for p in pa_disc_nodo))
                    else:
                        estado = 'marginal'
                    mu, beta, sig2 = self.get_clg(nodo, estado) #parametros CLG
                    sigma  = np.sqrt(max(sig2, 1e-9))
                    # media condicional (padres continuos)
                    mu_tot = mu
                    for j, p_cont in enumerate(pa_cont_nodo):
                        mu_tot += (beta[j] * vals[p_cont]
                                   if j < len(beta) else 0.0)
                    x_sample = float(rng.normal(mu_tot, sigma)) #muestrea X∼N(μtot​,σ2)
                    vals[nodo] = x_sample
                    data_cont[nodo][i] = x_sample
        # Construimos DataFrame completo
        df_out = pd.DataFrame()
        for nodo in order:
            if self.is_disc(nodo):
                df_out[nodo] = data_disc[nodo]
            else:
                df_out[nodo] = data_cont[nodo]
        return df_out

    ## Resumen de la red:
    def summary(self):
        print(f"\n  {'═'*60}")
        print(f"  Red Bayesiana Híbrida CLG — {self.nombre}")
        print(f"  {'═'*60}")
        print(f"  Nodos totales : {len(self._nodes)}  "
              f"(disc={len(self.nodes_disc)}, cont={len(self.nodes_cont)})")
        print(f"  Aristas totales: {len(self.edges_all)}")
        print(f"  Orden topológico: {self.topological_order()}")
        print(f"\n  Nodos discretos ({len(self.nodes_disc)}):")
        for n in sorted(self.nodes_disc):
            if n in set(self.disc.nodes()):
                cpd  = self.disc.get_cpds(n)
                vals = cpd.values.flatten()
                pa   = self._parents[n]
                print(f"    {n:<12} Pa={pa}  "
                      f"P_min={vals.min():.4f}  P_max={vals.max():.4f}  "
                      f"celdas={len(vals)}")
        print(f"\n  Nodos continuos ({len(self.nodes_cont)}):")
        for n in sorted(self.nodes_cont):
            if n in self.clg:
                pa = self._parents[n]
                print(f"    {n:<12} Pa={pa}")
                for estado, (mu, beta, sig2) in self.clg[n].items():
                    b_str = (f"  β={[round(b,3) for b in beta]}"
                             if beta else "")
                    print(f"      estado={estado}: "
                          f"μ={mu:.4f}  σ²={sig2:.4f}{b_str}")
        print(f"  {'═'*60}")
    
    def __repr__(self):
        return (f"RedBayesianaHibridaCLG('{self.nombre}', "
                f"{len(self._nodes)} nodos, "
                f"{len(self.edges_all)} aristas)")


print("Clase RedBayesianaHibridaCLG definida")

# 5.1 — Comparativa de estimadores
print("\n--- 5.1 Comparativa de estimadores ---")

# 5.1a: Discretos: tabla comparativa sobre target marginal
r_t  = int(df_train_disc_only[TARGET].nunique())
N_tr = len(df_train_disc_only)
print(f"\n  [5.1a] Estimadores discretos: TARGET marginal  (N={N_tr}, r={r_t})")
print(f"  {'k':>4} {'Nₖ':>8} {'MLE':>10} {'Laplace':>12} "
      f"{'Jeffreys':>12} {'ESS=5':>12}")
print("  " + "─" * 62)
for k in sorted(df_train_disc_only[TARGET].unique()):
    N_k = int((df_train_disc_only[TARGET] == k).sum())
    print(f"  {k:>4} {N_k:>8} "
          f"{N_k/N_tr:>10.4f} "
          f"{(N_k+1)/(N_tr+r_t):>12.4f} "
          f"{(N_k+0.5)/(N_tr+r_t/2):>12.4f} "
          f"{(N_k+5/r_t)/(N_tr+5):>12.4f}")
    
#¿cómo cambia la probabilidad estimada cuando modifico la fuerza prior?
print(f"\n  Sensibilidad P_min de target al ESS (N={N_tr}, r={r_t}):")
print(f"  {'ESS':>6} {'P_min':>12} {'P_max':>10}  Suma=1?  Interpretación")
print("  " + "─" * 60)
for ess_v, desc in [(0, 'MLE puro'),
                    (1, 'Laplace'),
                    (2, 'ESS=2'),
                    (5, 'ESS=5'),
                    (10, 'ESS=10'),
                    (20, 'ESS=20'),
                    (50, 'Fuertemente informativo')]:
    Ns = [int((df_train_disc_only[TARGET]==k).sum())
          for k in sorted(df_train_disc_only[TARGET].unique())]
    if ess_v == 0:
        ps = [n/N_tr for n in Ns]
    else:
        ps = [(n + ess_v/r_t)/(N_tr + ess_v) for n in Ns]
    print(f"  {ess_v:>6} {min(ps):>12.6f} {max(ps):>10.4f}  "
          f"{'Bien':>6}   {desc}")
print("\nSe elige ESS=5 ya que proporciona suavizado moderado y evita probabilidad nula.")

# 5.1b: Continuos: demostración MLE ≡ OLS
print(f"\n  [5.1b] Continuos: demostración MLE ≡ OLS con ejemplo: (thalach | target)")
print(f"  {'target':>8} {'N':>5} {'μ̂ MLE':>12} {'μ̂ OLS':>12} "
      f"{'σ̂² MLE':>12} {'σ̂² OLS':>12} {'≡?':>4}")
print("  " + "─" * 74)
for k, grp in df_train_hibrido.groupby(TARGET)['thalach']: #agrupamos por target
    x = grp.dropna().values
    if len(x) < 3: continue
    # MLE directo:
    mu_mle = float(x.mean())
    s2_mle = float(np.mean((x - mu_mle)**2)) 
    # OLS (regresión sin predictores -> intercept = media muestral)
    reg     = LR_sk().fit(np.zeros((len(x),1)), x)
    mu_ols  = float(reg.intercept_)
    s2_ols  = float(np.mean((x - mu_ols)**2))
    ok = (np.isclose(mu_mle, mu_ols, atol=1e-6) and
          np.isclose(s2_mle, s2_ols, atol=1e-6))
    print(f"  {k:>8} {len(x):>5} {mu_mle:>12.4f} {mu_ols:>12.4f} "
          f"{s2_mle:>12.4f} {s2_ols:>12.4f} {'Bien' if ok else 'Mal':>4}")
print("  -> Se confirma: MLE gaussiano ≡ OLS  ")
print("     Usamos OLS para estimar μₖ, βₖ, σ²ₖ en 5.3")

# 5.2 — Estimación CPTs discretas (BayesianEstimator ESS=5)
print("\n--- 5.2 Estimación CPTs nodos discretos (ESS=5) ---")
"""
  Dataset: df_train_disc_only (11 vars discretas, int).
  Estimador: BayesianEstimator con prior Dirichlet ESS=5.
      log L(θ:D) = Σᵢ log Lᵢ(θᵢ:D)
      Cada nodo se estima independientemente dado su familia.

"""

def estimar_cpts_ess5(model, nombre, ess=5):
    """
    Esta función estima las CPTs de los nodos discretos con BayesianEstimator ESS=5.
    Devuelve una DiscreteBayesianNetwork pgmpy con las CPTs ajustadas.
    
    Incluye TODOS los nodos discretos del modelo, no solo los
    que participan en aristas disc->disc. Los nodos discretos que solo tienen
    aristas disc->cont (ej. fbs->trestbps) se añaden como nodos raíz con
    CPT marginal. Si no se hiciera esto, get_cpd_disc fallaría para esos
    nodos durante la inferencia.
    """
    edges_disc = [(u, v) for u, v in model.edges()
                  if u in nodos_disc and v in nodos_disc] # Aristas disc->disc 
    cols_disc  = [c for c in df_train_disc_only.columns
                  if c in model.nodes() and c in nodos_disc] #todos los nodos discretos
    df_d = df_train_disc_only[cols_disc].copy().astype(int) 
    m = DiscreteBayesianNetwork(edges_disc) # DAG discreto (solo estructura)
    # Añadimos los nodos discretos que no aparecen en ninguna arista disc->disc
    nodos_en_grafo = set(m.nodes())
    for nodo in cols_disc:
        if nodo not in nodos_en_grafo:
            m.add_node(nodo)
    try: #pgmpy P(Xi​∣Pa(Xi​)) utiliza frecuencias observadas
        if _USE_NEW_ESTIMATORS: #version nueva pgmpy
            m.fit(df_d, estimator=DiscreteBayesianEstimator(
                equivalent_sample_size=ess)) 
        else: #version vieja pgmpy
            m.fit(df_d, estimator=BayesianEstimator(
                m, prior_type='dirichlet', pseudo_counts=ess))
        print(f"  {nombre}: ESS={ess} Bien "
              f"({len(edges_disc)} aristas disc, {len(cols_disc)} nodos)")
        return m
    except Exception as e:
        print(f"  {nombre}: Error {e}")
        return None
model_expert_disc = estimar_cpts_ess5(model_expert, 'Experta')
model_comun_disc = estimar_cpts_ess5(red_comun_full, 'Común')
# Mostramos CPTs
print("\n  CPTs asignadas (ESS=5):")
for nombre, md in [('Experta', model_expert_disc),
                   ('Común',   model_comun_disc)]:
    if md is None: continue
    print(f"\n  ── {nombre} ──")
    for nodo in sorted(md.nodes()):
        cpd = md.get_cpds(nodo) #CPT ya estimada
        vals = cpd.values.flatten()
        pa = [u for u, v in md.edges() if v == nodo]
        suma_ok = np.allclose(cpd.values.sum(axis=0), 1.0, atol=1e-6)
        print(f"    {nodo:<12} Pa={pa}  "
              f"P_min={vals.min():.4f}  P_max={vals.max():.4f}  "
              f"celdas={len(vals):>5}  P≈0={(vals<1e-9).sum()}  "
              f"Σ=1:{'Bien' if suma_ok else 'Mal'}")


# 5.3 — Estimación parámetros CLG continuos (OLS por grupos = MLE gaussiano)
print("\n--- 5.3 Estimación parámetros CLG continuos (OLS por grupos) ---")
"""
  Estimamos para cada nodo continuo Xc con padres discretos Pa_d y continuos Pa_c 
  (en nuestras redes solo hay discretos) y por cada configuración de Pa_d = k.
  Dataset: df_train_hibrido 
  Nota: ddof=0 para coherencia con MLE 
"""

def estimar_clg_manual(model, nombre):
    """
    Estima manualmente los parámetros CLG para los nodos continuos de una red según sus padres.
    Devuelve dict {col_c: {estado_padre: (μ, [β], σ²)}}.
    """
    clg = {}
    print(f"\n  ── {nombre} ──")
    for col_c in VARS_CONTINUAS_RB:
        if col_c not in model.nodes(): continue #comprobar que la var existe en la red
        padres_c = [u for u, v in model.edges() if v == col_c] #padres
        padres_disc_c = [p for p in padres_c if p in nodos_disc]
        padres_cont_c = [p for p in padres_c if p in nodos_cont]
        clg[col_c] = {}
        cols_ok = [c for c in [col_c] + padres_disc_c + padres_cont_c
                   if c in df_train_hibrido.columns] #cols necesarias
        df_sub = df_train_hibrido[cols_ok].dropna()
        print(f"\n    {col_c}  Pa_disc={padres_disc_c}  Pa_cont={padres_cont_c}")
        # A) Caso sin padres: estimamos una única gaussiana
        if not padres_c:
            # Sin padres: distribución marginal
            x = df_sub[col_c].values
            mu_m = float(x.mean())
            sig2_m = max(float(np.mean((x - mu_m)**2)), 1e-6)
            clg[col_c]['marginal'] = (mu_m, [], sig2_m)
            print(f"      Marginal ->μ={mu_m:.4f}  σ²={sig2_m:.4f}  N={len(x)}")
        # B) Caso cpn padres discretos: una gaussiana distinta para cada estado del padre
        elif padres_disc_c:
            print(f"      {'Estado Pa_d':<30} {'μ̂ₖ':>10} {'σ̂²ₖ':>10} "
                  f"{'Nₖ':>6}  β̂ₖ")
            print(f"      {'─'*72}")
            for estado, grupo in df_sub.groupby(padres_disc_c): #agrupamos por estados
                xk = grupo[col_c].values #obs
                Nk = len(xk) #nº muestras
                if Nk < 3: #si hay pocos datos, fallback
                    mu_k = float(xk.mean()) if Nk > 0 else 0.0
                    clg[col_c][estado] = (mu_k, [], 1.0)
                    print(f"      {str(estado):<30} {mu_k:>10.4f} "
                          f"{'N/A':>10} {Nk:>6}  (Nₖ<3)")
                    continue
                if padres_cont_c:
                    Zk = grupo[padres_cont_c].values
                    reg = LR_sk().fit(Zk, xk)
                    mu_k = float(reg.intercept_)
                    betas = list(reg.coef_.astype(float))
                    resid = xk - reg.predict(Zk)
                else: #nuestro caso: solo padres discretos
                    mu_k  = float(xk.mean())
                    betas = []
                    resid = xk - mu_k
                # σ²ₖ con ddof=0 (MLE)
                sig2_k = max(float(np.mean(resid**2)), 1e-6)
                clg[col_c][estado] = (mu_k, betas, sig2_k)
                est_str = (f"{padres_disc_c[0]}={estado}"
                           if isinstance(estado, (int, np.integer))
                           else ", ".join(f"{p}={v}"
                                for p, v in zip(padres_disc_c, estado)))
                b_str = (f"β={[round(b,4) for b in betas]}"
                         if betas else "—")
                print(f"      {est_str:<30} {mu_k:>10.4f} {sig2_k:>10.4f} "
                      f"{Nk:>6}  {b_str}")
        else: # C) Solo padres continuos: OLS global
            Xc = df_sub[col_c].values
            Zc = df_sub[padres_cont_c].values
            if len(Xc) >= 3:
                reg = LR_sk().fit(Zc, Xc)
                mu_k = float(reg.intercept_)
                betas = list(reg.coef_.astype(float))
                sig2_k = max(float(np.mean((Xc - reg.predict(Zc))**2)), 1e-6)
            else:
                mu_k, betas, sig2_k = float(Xc.mean()), [], 1.0
            clg[col_c]['all'] = (mu_k, betas, sig2_k)
            print(f"      OLS global -> μ={mu_k:.4f}  σ²={sig2_k:.4f}  "
                  f"β={[round(b,4) for b in betas]}")
    return clg
clg_expert_params = estimar_clg_manual(model_expert,   'Red Experta')
clg_comun_params  = estimar_clg_manual(red_comun_full, 'Red Común')


# 5.4 — Construcción de los modelos híbridos finales
print("\n--- 5.4 Construcción de los modelos híbridos finales ---")
"""
  Unificamos CPTs discretas y parámetros CLG en un único
  objeto RedBayesianaHibridaCLG por red.
"""

modelo_expert = RedBayesianaHibridaCLG(
    nombre     = 'Red Experta',
    model_disc = model_expert_disc,
    clg_params = clg_expert_params,
    edges_all  = list(model_expert.edges()),
    nodes_disc = nodos_disc,
    nodes_cont = nodos_cont
)
modelo_comun = RedBayesianaHibridaCLG(
    nombre     = 'Red Común',
    model_disc = model_comun_disc,
    clg_params = clg_comun_params,
    edges_all  = list(red_comun_full.edges()),
    nodes_disc = nodos_disc,
    nodes_cont = nodos_cont
)
modelo_expert.summary()
modelo_comun.summary()


# 5.5 — Verificación: P>0, Σ=1, σ²>0
print("\n--- 5.5 Verificación de parámetros ---")
for m in [modelo_expert, modelo_comun]:
    print(f"\n  {m.nombre}:")
    # Discretos
    zeros, sumas_ok = 0, True
    for n in m.disc.nodes():
        cpd = m.disc.get_cpds(n)
        vals = cpd.values
        zeros += (vals < 1e-12).sum() #probs casi nulas
        if not np.allclose(vals.sum(axis=0), 1.0, atol=1e-6):
            sumas_ok = False
            print(f"    CUIDADO {n}: Σ ≠ 1")
    print(f"    Discretos: celdas P≈0 = {zeros}  "
          f"Σ=1 todos: {'Bien' if sumas_ok else 'Mal'}")
    # Continuos
    clg_ok = True
    for col_c, params in m.clg.items():
        for estado, (mu, beta, sig2) in params.items():
            if sig2 <= 0:
                print(f"    CUIDADO CLG {col_c} estado={estado}: σ²={sig2}≤0")
                clg_ok = False
    print(f"    Continuos: σ²>0 todos: {'Bien' if clg_ok else 'Mal'}")


# 5.6 — Coherencia P(target=k) vs frecuencia relativa
print("\n--- 5.6 Coherencia P(target=k) vs frecuencia relativa ---")
"""
  Con ESS=5 y N=242, la diferencia teórica es:
    Δmax ≈ ESS / (2·N) = 5 / (2·242) ≈ 0.010
  Verificamos que las diferencias observadas están dentro de ese rango.
"""

freq_rel = df_train_disc_only[TARGET].value_counts(
    normalize=True).sort_index() #frecuencias reales del dataset

def p_marginal_target(m, k):
    """
    Como target tiene padres discretos, paraa calcular su probabilidad marginal 
    P(target=k) pondemos por frecuencia empírica de los padres.
    """
    md = m.disc # parte discreta
    if TARGET not in md.nodes(): return float('nan')
    cpd = md.get_cpds(TARGET) #CPT de target
    pa  = [u for u, v in md.edges() if v == TARGET] #padres
    if not pa: # Caso 1: target SIN padres -> CPT marginal
        vals = cpd.values.flatten()
        return float(vals[k]) if k < len(vals) else float('nan')
    try: # Caso 2: target CON padres
        pesos = (df_train_disc_only.groupby(pa).size()
                 / len(df_train_disc_only))  #cuánto aparece cada conf de padres: P(Pa=j) empiricamente
        p_k = 0.0
        for config, peso in pesos.items(): #recorre cada configuracion de padres
            if not isinstance(config, tuple):
                config = (config,)
            idx = {TARGET: k, **dict(zip(pa, config))} #indice
            p_k += peso * float(cpd.get_value(**idx)) #prob en la CPT ponderada: P(target=k)=j∑​P(target=k∣Pa=j)P(Pa=j) (Tma Prob Total)
        return p_k #P​(target=k) según la RB aprendida
    except Exception:
        return float('nan')
#Tabla comparativa
print(f"\n  {'k':>4} {'Freq.rel':>12} {'P Experta':>12} {'P Común':>12} "
      f"{'Δ Exp':>10} {'Δ Com':>10}")
print("  " + "─" * 64)
for k in sorted(df_train_disc_only[TARGET].unique()):
    fr = float(freq_rel.get(k, 0.0)) #frecuencia real
    p_e = p_marginal_target(modelo_expert, k)
    p_c = p_marginal_target(modelo_comun,  k)
    print(f"  {k:>4} {fr:>12.4f} {p_e:>12.4f} {p_c:>12.4f} "
          f"{abs(fr-p_e):>10.4f} {abs(fr-p_c):>10.4f}")
print(f"\n  Δmax teórico ≈ ESS/(2N) = 5/(2·{N_tr}) ≈ {5/(2*N_tr):.4f}")


# 5.7 — Visualización distribuciones CLG
print("\n--- 5.7 Visualización distribuciones CLG ---")
"""
  Histograma de vars continuas por grupo de padres discrtos, con Curva gaussiana N(μ̂ₖ, σ̂²ₖ) 
  superpuesta con el MISMO color. Y test de Shapiro-Wilk por grupo para ver si la 
  gaussiana es una aproximación razonable (limitación del modelo CLG)
"""
COLORES_CLG = ['#4e79a7', '#f28e2b', '#e05c4b', '#76b7b2', '#59a14f',
               '#b07aa1', '#ff9da7', '#9c755f']
fig, axes = plt.subplots(2, len(VARS_CONTINUAS_RB),
                          figsize=(5.5 * len(VARS_CONTINUAS_RB), 9))
if len(VARS_CONTINUAS_RB) == 1:
    axes = axes.reshape(2, 1)

print(f"\n  Test de normalidad Shapiro-Wilk por grupo (p>0.05 -> gaussiana razonable):")
print(f"  {'Red':<10} {'Variable':<12} {'Grupo':<20} {'N':>5} "
      f"{'SW p-val':>10}  Gaussiana?")
print("  " + "─" * 65)

for row_i, m in enumerate([modelo_expert, modelo_comun]):
    for col_i, col_c in enumerate(VARS_CONTINUAS_RB):
        ax = axes[row_i, col_i]
        pa_disc_c = [p for p in m.parents(col_c) if p in nodos_disc]

        if pa_disc_c and col_c in m.clg:
            grp_col = pa_disc_c[0]
            grupos  = list(df_train_hibrido.groupby(grp_col)[col_c])
            # Rango global para el eje x (todos los grupos juntos)
            todos_x = df_train_hibrido[col_c].dropna().values
            x_min, x_max = todos_x.min(), todos_x.max()
            xr_global = np.linspace(x_min, x_max, 200)
            for g_idx, (est_g, dat_g) in enumerate(grupos):
                x     = dat_g.dropna().values
                color = COLORES_CLG[g_idx % len(COLORES_CLG)]
                label = f"{grp_col}={int(est_g)} (N={len(x)})"
                if len(x) < 5:
                    continue
                # Histograma con color del grupo
                ax.hist(x, bins=12, alpha=0.40, color=color,
                        label=label, edgecolor='white', density=True)
                # Gaussiana solo si N≥10 (estimación fiable)
                if len(x) >= 10:
                    try:
                        mu_k, _, sig2_k = m.get_clg(col_c, int(est_g))
                        sigma = np.sqrt(max(sig2_k, 1e-6))
                        gauss = (np.exp(-0.5*((xr_global - mu_k)/sigma)**2)
                                 / (sigma * np.sqrt(2*np.pi)))
                        ax.plot(xr_global, gauss, color=color, lw=1.8,
                                linestyle='-')
                    except Exception:
                        pass
                # Test Shapiro-Wilk (solo si 8 ≤ N ≤ 5000)
                if 8 <= len(x) <= 5000:
                    _, p_sw = stats.shapiro(x)
                    razonable = "Sí" if p_sw > 0.05 else "No (colas pesadas)"
                    grupo_str = f"{grp_col}={int(est_g)}"
                    print(f"  {m.nombre:<10} {col_c:<12} {grupo_str:<20} "
                          f"{len(x):>5} {p_sw:>10.4f}  {razonable}")
            ax.set_title(f"{m.nombre}: {col_c} | {grp_col}", fontsize=13)
            ax.legend(fontsize=10, loc='upper right')
            ax.set_xlim(x_min - (x_max - x_min)*0.05,
                        x_max + (x_max - x_min)*0.05)
        else:
            # Sin padres o marginal
            x = df_train_hibrido[col_c].dropna().values
            ax.hist(x, bins=15, color=COLORES_CLG[0],
                    alpha=0.8, edgecolor='white', density=True,
                    label=f"N={len(x)}")
            if col_c in m.clg and 'marginal' in m.clg[col_c]:
                mu_m, _, sig2_m = m.clg[col_c]['marginal']
                xr = np.linspace(x.min(), x.max(), 200)
                sigma = np.sqrt(max(sig2_m, 1e-6))
                ax.plot(xr,
                        np.exp(-0.5*((xr - mu_m)/sigma)**2)
                        / (sigma*np.sqrt(2*np.pi)),
                        color=COLORES_CLG[0], lw=1.8)
                if 8 <= len(x) <= 5000:
                    _, p_sw = stats.shapiro(x)
                    razonable = "Sí" if p_sw > 0.05 else "No"
                    print(f"  {m.nombre:<10} {col_c:<12} {'marginal':<20} "
                          f"{len(x):>5} {p_sw:>10.4f}  {razonable}")
            ax.set_title(f"{m.nombre}: {col_c} (marginal)", fontsize=8)
            ax.legend(fontsize=6)

        ax.set_xlabel(col_c, fontsize=13)
        ax.set_ylabel("Densidad", fontsize=10)
        ax.tick_params(labelsize=6)

plt.suptitle(
    "CLG: P(Xc | Pa_d=k) histograma real + N(μ̂ₖ, σ̂²ₖ) ajustada\n",
    fontsize=13)
plt.tight_layout()
plt.savefig('uci_clg_distribuciones.png', dpi=120, bbox_inches='tight')
plt.close()

# 5.9 — Visualización del DAG con parámetros anotados en aristas/nodos
print("\n--- 5.9 Visualización DAG con parámetros ---")
"""
  Para cada red pintamos el grado del DAG (como Bloque 4) y anotamos sus parámetros
 """

def viz_dag_con_params(m, titulo, ax):
    """
    Dibuja el DAG completo con parámetros anotados en las aristas:
    disc->disc: etiqueta con P_min–P_max del hijo dado ese padre
    disc->cont: tabla con los parámetros de las distribuciones gaussianas condicionadas
        para cada categoría del padre discreto.
    """
    G = nx.DiGraph(m.edges_all)
    for n in m.nodes():
        if n not in G.nodes:
            G.add_node(n)
    
    def color_nodo(n): #mismos colores que bloque4
        if n == TARGET:           return '#e05c4b'
        if m.is_cont(n):          return '#f28e2b'
        if n in [u for u,v in m.edges_all if v == TARGET]: return '#4e79a7'
        return '#59a14f'
    
    colores = [color_nodo(n) for n in G.nodes()]
    pos = nx.spring_layout(G, seed=42, k=2.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colores,
                           node_size=900, alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6,
                            font_color='white', font_weight='bold')
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=14,
                           edge_color='#555', width=1.2,
                           connectionstyle='arc3,rad=0.05')
    # Anotamos aristas discretas con parámetros:
    edge_labels = {}
    for u, v in m.edges_all:
        if m.is_disc(u) and m.is_disc(v):
            # disc->disc: P_min–P_max de la CPT del hijo
            try:
                cpd  = m.get_cpd_disc(v)
                vals = cpd.values.flatten()
                edge_labels[(u,v)] = f"P∈[{vals.min():.2f},{vals.max():.2f}]"
            except Exception:
                edge_labels[(u,v)] = ""

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                  ax=ax, font_size=5,
                                  bbox=dict(boxstyle='round,pad=0.2',
                                            fc='white', alpha=0.7, ec='none'),
                                  label_pos=0.35)
    # Anotamos CPDs CLG junto al nodo continuo
    for n in m.nodes_cont:
        if n in m.clg and n in pos:
            px, py = pos[n]
            padres = [p for p in m.parents(n) if p in nodos_disc]
            padre = padres[0] if len(padres) else "marginal"
            texto = [f"{n}"]
            lines = [f"{n}:"]
            for estado, (mu, _, sig2) in m.clg[n].items():
                est_str = str(estado).replace('(','').replace(',)','').replace(')','')
                lines.append(f" {padre}={est_str}: μ={mu:.1f} σ={np.sqrt(sig2):.1f}")
            txt = "\n".join(lines)
            ax.annotate(txt, xy=(px, py), xytext=(px+0.15, py+0.18),
                        fontsize=4.5, color='#333',
                        bbox=dict(boxstyle='round,pad=0.3', fc='#fff8e7',
                                  alpha=0.85, ec='#f28e2b', lw=0.8))

    ax.set_title(f"{titulo}\n({len(m.nodes())} nodos, "
                 f"{len(m.edges_all)} aristas)",
                 fontsize=8, pad=6)
    ax.axis('off')


fig5, axs5 = plt.subplots(1, 2, figsize=(20, 9))
viz_dag_con_params(modelo_expert, 'Red Experta: DAG + parámetros CLG', axs5[0])
viz_dag_con_params(modelo_comun,  'Red Común: DAG + parámetros CLG', axs5[1])

fig5.legend(handles=[
    mpatches.Patch(color='#e05c4b', label='Target'),
    mpatches.Patch(color='#4e79a7', label='Padre disc. de target'),
    mpatches.Patch(color='#f28e2b', label='Nodo continuo (CLG)'),
    mpatches.Patch(color='#59a14f', label='Nodo discreto'),
], loc='lower center', ncol=4, fontsize=7)
plt.suptitle('UCI Cleveland: DAG híbrido CLG con parámetros anotados\n'
             'Aristas disc->disc: rango P_min–P_max',
             fontsize=10)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('uci_dag_parametros.png', dpi=130, bbox_inches='tight')
plt.close()

# Guardar parámetros CLG en JSON
for nf, m in [('expert', modelo_expert), ('comun', modelo_comun)]:
    ser = {}
    for col_c, params in m.clg.items():
        ser[col_c] = {}
        for est, (mu, beta, sig2) in params.items():
            ser[col_c][str(est)] = {
                'mu': float(mu),
                'beta': [float(b) for b in beta],
                'sigma2': float(sig2)
            }
    with open(f'uci_clg_params_{nf}.json', 'w') as f:
        _json.dump(ser, f, indent=2)
    
    
# =============================================================================
# BLOQUE 6: INFERENCIA EXACTA
# =============================================================================

print("\n" + "=" * 70)
print("BLOQUE 6: INFERENCIA EXACTA: VE Y BP MANUALES (RED HÍBRIDA CLG)")
print("=" * 70)
"""
  Tratamiento de evidencia continua:
    - Si Xc = xc está observada, f(xc | Pa_d=k) = N(xc; μₖ, σ²ₖ)  
    - Si Xc NO está observada (marginalización): ∫ N(xc; μₖ, σ²ₖ) dxc = 1 
 
  Resultado: P(target | evidencia) exacta sobre la parte discreta,
  + la densidad P(Xc = xc | evidencia discreta) para evidencia continua.
"""
 
 
# 6.0- Funciones auxiliares 
def likelihood_continuo(m, nodo_cont, xc, estado_padre):
    """
    Implementa la verosimilitud de observar xc en el nodo continuo dado el estado del padre discreto.
    f(xc | Pa_d=estado_padre) = P(Xc​=xc​∣Pad​=k)  =  N(xc; μₖ, σ²ₖ)
    """
    try:
        mu, _, sig2 = m.get_clg(nodo_cont, estado_padre)
        return float(sp_norm.pdf(xc, loc=mu, scale=np.sqrt(max(sig2, 1e-9))))
    except Exception:
        return 1.0
 

# 6.1 — Variable Elimination (VE) manual híbrido
print("\n--- 6.1 Variable Elimination manual (red híbrida CLG) ---")
"""
  VE para P(target | E_disc, E_cont):
"""
 
def _card(m, nodo):
    """Número de estados únicos de un nodo discreto."""
    if nodo in df_train_disc_only.columns:
        return int(df_train_disc_only[nodo].nunique())
    try:
        return int(m.get_cpd_disc(nodo).variable_card)
    except Exception:
        return 2

def _estados(m, nodo):
    """
    Lista de estados posibles de un nodo discreto usando los valores
    REALES del dataset. Por ejemplo: cp={1,2,3,4} y thal={3,6,7}
    no están remapeados a 0-based, y las CPTs de pgmpy se estiman
    con esos valores reales.
    """
    if nodo in df_train_disc_only.columns:
        return sorted(df_train_disc_only[nodo].dropna().astype(int).unique().tolist())
    try:
        return list(range(m.get_cpd_disc(nodo).variable_card))
    except Exception:
        return [0, 1]

def ve_hibrido(m, query, evidencia_disc=None, evidencia_cont=None):
    """
    Variable Elimination manual sobre RedBayesianaHibridaCLG.
    Factores iniciales:
      · disc->disc : φ(Xi, Pa_disc) = P(Xi | Pa_disc)  desde m.p_disc()
      · disc->cont : φ(Pa_disc) = N(xc; μₖ, σ²ₖ) si Xc observado
                    no añade factor i Xc no observado (integral=1)
    """
    if evidencia_disc is None:
        evidencia_disc = {}
    if evidencia_cont is None:
        evidencia_cont = {}
    orden_topo = m.topological_order()
    
    # Paso 1: construir factores 
    factores = [] # cada CPT se convierte en un factor
    for nodo in orden_topo:
        if not m.is_disc(nodo):
            continue
        pa_disc = [u for u in m.parents(nodo) if m.is_disc(u)]
        r_nodo = _card(m, nodo)  #cardinalidad
        vars_f = tuple(pa_disc + [nodo])#vars del factor (depende de ellas)
        # Construcción de la tabla
        tabla = {}
        estados_nodo = _estados(m, nodo)
        if pa_disc: # Sí padres discretos
            for cfg_pa in iproduct(*[_estados(m, p) for p in pa_disc]): #producto cartesiano: todas las combinaciones posibles de estados de los padres
                pa_vals = dict(zip(pa_disc, cfg_pa))
                for k in estados_nodo:
                    tabla[tuple(list(cfg_pa) + [k])] = float(
                        m.p_disc(nodo, k, pa_vals))
        else: # No padres
            for k in estados_nodo:
                tabla[(k,)] = float(m.p_disc(nodo, k, {})) # el factor es su dist.marginal
        factores.append((vars_f, tabla))

    # Verosimilitud de nodos continuos observados -> factor sobre Pa_disc
    for nodo_c, xc in evidencia_cont.items(): 
        if nodo_c not in m.nodes_cont:
            continue
        pa_disc_c = [u for u in m.parents(nodo_c) if m.is_disc(u)]
        if not pa_disc_c:
            continue
        vars_f = tuple(pa_disc_c)
        tabla  = {}
        for cfg_pa in iproduct(*[_estados(m, p) for p in pa_disc_c]):
            tabla[cfg_pa] = likelihood_continuo(m, nodo_c, xc, cfg_pa)
        factores.append((vars_f, tabla))

    # Paso 2: reducir evidencia discreta 
    for nodo_ev, val_ev in evidencia_disc.items(): #recorremos todos los factores que contienen a la variable observada
        val_ev  = int(val_ev)
        nuevos  = [] #factores ya reducidos
        for vars_f, tabla in factores: #¿el factor contiene la evidencia?
            if nodo_ev not in vars_f: # No -> añadimos
                nuevos.append((vars_f, tabla))
                continue
            idx = vars_f.index(nodo_ev) # Posicion de la var ev
            # Filtramos solo las entradas donde nodo_ev == val_ev
            n_vars = tuple(v for v in vars_f if v != nodo_ev)  #eliminamos ev porque ya la sabemos
            n_tabla = {}
            for asig, val in tabla.items(): #recorremos filas de nueva tabla
                if asig[idx] == val_ev:  #comprobamos evidencia
                    n_asig = tuple(asig[i] for i in range(len(asig)) if i != idx) #"eliminamos" var obs
                    n_tabla[n_asig] = val
            nuevos.append((n_vars, n_tabla))
        factores = nuevos

    # Paso 3: eliminar variables según orden topológico inverso
    # Eliminamos todas las discretas excepto query (TARGET), en orden inverso topológico
    orden_elim = [n for n in reversed(orden_topo)
                  if m.is_disc(n) and n != query
                  and n not in evidencia_disc]
    for var_elim in orden_elim:
        con_var = [(vf, tb) for vf, tb in factores if var_elim in vf] # factores que contienen var_elim
        sin_var = [(vf, tb) for vf, tb in factores if var_elim not in vf] # factores que NO contienen var_elim
        if not con_var:
            continue
        # Unimos variables en los factores afectados
        vars_union = []
        for vf, _ in con_var:
            for v in vf:
                if v not in vars_union:
                    vars_union.append(v)
        vars_res = [v for v in vars_union if v != var_elim] # vars que quedarán tras eliminar
        estados_elim = _estados(m, var_elim)
        nueva_tabla = {}
        for cfg_res in iproduct(*[_estados(m, v) for v in vars_res]):
            cfg_dict = dict(zip(vars_res, cfg_res))
            suma = 0.0
            for k_elim in estados_elim:
                cfg_full = {**cfg_dict, var_elim: k_elim}
                prod = 1.0 # multiplicar todos los factores afectados
                for vf, tb in con_var:
                    asig = tuple(cfg_full[v] for v in vf)
                    prod *= tb.get(asig, 0.0)
                suma += prod #sumatorio
            nueva_tabla[cfg_res] = suma
        # sustituimos factores viejos por los nuevos creados
        sin_var.append((tuple(vars_res), nueva_tabla))
        factores = sin_var

    # Paso 4: multiplicar factores restantes y normalizar
    # Tras eliminar las vars, solo quedan factores sobre query (y quizás constantes)
    estados_query = _estados(m, query)
    resultado = np.ones(len(estados_query))
    for vars_f, tabla in factores:
        if query in vars_f: # A) Factor contiene la query
            idx_q = vars_f.index(query) #posicion
            for i, k in enumerate(estados_query):
                val_k = sum(val for asig, val in tabla.items() #fila compatible
                            if asig[idx_q] == k) #comprueba
                resultado[i] *= val_k
        elif tabla:
            resultado *= next(iter(tabla.values()))
    # Normalizamos:
    z = resultado.sum()
    return resultado / z if z > 1e-300 else np.ones(len(estados_query)) / len(estados_query)
 
 
# 6.2 — Belief Propagation (BP) manual híbrido
print("\n--- 6.2 Belief Propagation manual (red híbrida CLG) ---")
"""
  BP: paso de mensajes en dos fases (upward + downward).
  Para nodos continuos observados: λ_c(Pa_d=k) = N(xc; μₖ, σ²ₖ)
  que se propaga como un mensaje upward desde el nodo continuo hacia su padre discreto.
"""
 
def bp_hibrido(m, query, evidencia_disc=None, evidencia_cont=None):
    """
    Belief Propagation manual sobre RedBayesianaHibridaCLG.
    upward (λ) + downward (π)
        -> belief(X) ∝ π(X)λ(X)
    Evidencia discreta  -> λ puntual.
    Evidencia continua  ->  N(xc;μₖ,σ²ₖ) multiplicado en λ del padre.
    """
    if evidencia_disc is None:
        evidencia_disc = {}
    if evidencia_cont is None:
        evidencia_cont = {}
    
    # Paso 1: inicializo
    orden_topo = m.topological_order()
    nodos_disc_m = [n for n in orden_topo if m.is_disc(n)]
    cards = {n: _card(m, n) for n in nodos_disc_m}

    # Paso 2: Mensajes λ individuales de cada hijo hacia cada padre:
    lam_child = {} # lam_child[(hijo, padre)](k_padre) = λ_{hijo->padre}(k_padre)
    lam_self = {n: np.ones(cards[n]) for n in nodos_disc_m} #inicializo con 1s

    # Paso 3: Evidencia discreta -> λ puntual (indezado por posicion)
    for nodo_ev, val_ev in evidencia_disc.items():
        if nodo_ev in lam_self:
            estados_ev = _estados(m, nodo_ev)
            if int(val_ev) in estados_ev:
                idx = estados_ev.index(int(val_ev))
                lam_self[nodo_ev] = np.zeros(cards[nodo_ev])
                lam_self[nodo_ev][idx] = 1.0

    # Paso 4: Evidencia continua -> multiplica λ del padre discreto
    for nodo_c, xc in evidencia_cont.items():
        if nodo_c not in m.nodes_cont:
            continue
        pa_disc_c = [u for u in m.parents(nodo_c) if u in cards]
        if not pa_disc_c:
            continue
        pa_nodo = pa_disc_c[0]
        lk = np.array([likelihood_continuo(m, nodo_c, xc, (k,))
                       for k in _estados(m, pa_nodo)])
        lam_self[pa_nodo] *= lk #multiplicamos

    # Paso 5: Upward pass: calcular λ_{hijo->padre} 
    for nodo in reversed(nodos_disc_m): #orden topologico inverso
        pa_disc = [p for p in m.parents(nodo) if p in cards]
        if not pa_disc:
            continue
        estados_v = _estados(m, nodo)
        lam_nodo = lam_self[nodo].copy() #y luego añado mensajes hijo -> nodo
        for hijo in [c for c in m.children(nodo) if c in cards]:
            if (hijo, nodo) in lam_child:
                lam_nodo *= lam_child[(hijo, nodo)] # toda la evidencia que llega desde abajo
        # Calculamos mensaje hacia cada padre
        for pa in pa_disc:
            estados_pa = _estados(m, pa)
            msg = np.zeros(len(estados_pa))
            otros = [p for p in pa_disc if p != pa]
            for i_pa, k_pa in enumerate(estados_pa):
                for i_v, j in enumerate(estados_v):
                    if otros: # con otros padres (configuraciones) x∑​Pa∖U∑​P(x∣Pa)λ(x)
                        estados_otros = [_estados(m, p) for p in otros]
                        n_otros = np.prod([len(e) for e in estados_otros])
                        for cfg_o in iproduct(*estados_otros):
                            pa_vals = {pa: k_pa, **dict(zip(otros, cfg_o))}
                            msg[i_pa] += (m.p_disc(nodo, j, pa_vals)
                                          * lam_nodo[i_v] / n_otros)
                    else: #sin otros padres:λX->U​(u)=x∑​P(x∣u)λ(x)
                        msg[i_pa] += m.p_disc(nodo, j, {pa: k_pa}) * lam_nodo[i_v]
            lam_child[(nodo, pa)] = msg
    # λ total de cada nodo = lam_self · producto de mensajes de sus hijos
    lam_total = {}
    for nodo in nodos_disc_m:
        lt = lam_self[nodo].copy()
        for hijo in [c for c in m.children(nodo) if c in cards]:
            if (hijo, nodo) in lam_child:
                lt *= lam_child[(hijo, nodo)]
        lam_total[nodo] = lt

    # Paso 6: Downward pass
    pi_m = {n: np.ones(cards[n]) for n in nodos_disc_m} 
    for nodo in nodos_disc_m: #recorremos de arriba a bajo
        pa_disc = [p for p in m.parents(nodo) if p in cards] #padres
        if not pa_disc: # si no tiene padres no hay nada que propagar
            continue
        estados_v = _estados(m, nodo)
        pi_v = np.zeros(len(estados_v)) # π del nodo
        for cfg_pa in iproduct(*[_estados(m, p) for p in pa_disc]): #configuraciones de los padres
            pa_vals = dict(zip(pa_disc, cfg_pa))
            peso = 1.0 #pesos de la configuracion
            for p, kp in pa_vals.items():  # analizamos cada padre
                estados_p = _estados(m, p)
                i_kp = estados_p.index(kp)
                lt_excl = lam_self[p].copy() #λself​(padre) SIN este hijo
                for hijo in [c for c in m.children(p) if c in cards]:
                    if hijo != nodo and (hijo, p) in lam_child:
                        lt_excl *= lam_child[(hijo, p)] #mutiplico mensajes de otros hijos
                bel_excl = pi_m[p] * lt_excl #creencia parcial del padre
                z_p = bel_excl.sum() #normalizar
                peso *= (bel_excl[i_kp] / z_p) if z_p > 1e-300 else 0.0 #construye una probabilidad para la configuración de padres
            for i_v, j in enumerate(estados_v):
                pi_v[i_v] += m.p_disc(nodo, j, pa_vals) * peso #tma prob total: π(X)=Pa∑​P(X∣Pa)P(Pa)
        pi_m[nodo] = pi_v
        
    # Paso 7: Creencia final β(k) ∝ π(k) · λ_total(k) 
    b_query = (pi_m.get(query, np.ones(cards.get(query, 5))) *
               lam_total.get(query, np.ones(cards.get(query, 5))))
    z = b_query.sum() # normalizamos
    return b_query / z if z > 1e-300 else np.ones(len(b_query)) / len(b_query)
 
 # 6.3 — Perfiles clínicos: VE y BP sobre ambas redes
print("\n--- 6.3 Inferencia por perfiles clínicos ---")

def filtrar_evidencia_disc(ev_d, m):
    """
    Filtra la evidencia discreta: solo nodos presentes en el modelo,
    distintos de TARGET, y con valores dentro de los estados reales
    del dataset (_estados(m, nodo) devuelve los valores reales).
    Se usan los valores ORIGINALES del dataset UCI,
    no índices 0-based. 
    """
    ev_f = {}
    for nodo, val in ev_d.items():
        if nodo == TARGET or nodo not in m.nodes():
            continue
        if not m.is_disc(nodo):
            continue
        val_int = int(val)
        estados = _estados(m, nodo)
        if val_int in estados:
            ev_f[nodo] = val_int
        else:
            print(f"    CUIDADO {nodo}={val} no es un estado válido {estados}, descartado")
    return ev_f


# Perfiles con valores ORIGINALES del dataset UCI (cp y thal no remapeados)
# cp:   1=angina típica, 2=dolor atípico, 3=no anginoso, 4=asintomático
# thal: 3=normal, 6=defecto fijo, 7=defecto reversible
# age, oldpeak: discretizados 0-based en bloque 3A

PERFILES = [
    ('Alto riesgo: angina típica H58',
     {'sex': 1, 'age': 2, 'cp': 1, 'exang': 1, 'oldpeak': 2},
     {}),
    ('Bajo riesgo: asintomática M42',
     {'sex': 0, 'age': 1, 'cp': 4, 'exang': 0},
     {}),
    ('Solo pruebas objetivas (ca=2,thal=7)',
     {'ca': 2, 'thal': 7},
     {}),
    ('Con thalach continuo observado',
     {'sex': 1, 'age': 2, 'cp': 1},
     {'thalach': 120.0}),
    ('Con thalach alto (sano)',
     {'sex': 0, 'age': 1},
     {'thalach': 175.0}),
    ('Sin evidencia (prior)',
     {},
     {}),
]

SECUENCIA = [
    ('1. Sin evidencia (prior)',         {}, {}),
    ('2. + Demográficos (H, 55-70a)',    {'sex': 1, 'age': 2}, {}),
    ('3. + Angina típica (cp=1)',        {'sex': 1, 'age': 2, 'cp': 1}, {}),
    ('4. + Angina de esfuerzo',         {'sex': 1, 'age': 2, 'cp': 1,
                                          'exang': 1}, {}),
    ('5. + FC máx baja (thalach=115)',  {'sex': 1, 'age': 2, 'cp': 1,
                                          'exang': 1},
                                         {'thalach': 115.0}),
    ('6. + Dep. ST severa (oldpeak=2)', {'sex': 1, 'age': 2, 'cp': 1,
                                          'exang': 1, 'oldpeak': 2},
                                         {'thalach': 115.0}),
    ('7. + 2 vasos calcif. (ca=2)',     {'sex': 1, 'age': 2, 'cp': 1,
                                          'exang': 1, 'oldpeak': 2, 'ca': 2},
                                         {'thalach': 115.0}),
]


r_target = _card(modelo_expert, TARGET)  # nº estados de target (5: 0,1,2,3,4)

for nombre_m, m in [('Experta', modelo_expert), ('Común', modelo_comun)]:
    print(f"\n  ── {nombre_m} ──────────────────────────────────────────")
    print(f"  {'Perfil':<42} {'Método':>6}  {'MAP':>3}  "
          f"{'k0':>7}{'k1':>7}{'k2':>7}{'k3':>7}{'k4':>7}")
    print("  " + "─" * 90)
    for desc, ev_d, ev_c in PERFILES:  # cada paciente
        # Filtrar: nodos presentes, estados válidos, sin target
        ev_d_f = filtrar_evidencia_disc(ev_d, m)
        ev_c_f = {k: v for k, v in ev_c.items() if k in m.nodes_cont}
        for metodo, fn in [('VE', ve_hibrido), ('BP', bp_hibrido)]:  # inferencia exacta
            try:
                dist = fn(m, TARGET, ev_d_f, ev_c_f)  # P(TARGET|evidencia), 5 clases
                dist = np.array(dist, dtype=float)
                if len(dist) < r_target:  # aseguramos longitud correcta
                    dist = np.pad(dist, (0, r_target - len(dist)))
                dist = dist[:r_target]
                z = dist.sum()
                dist = dist / z if z > 1e-300 else np.ones(r_target) / r_target  # dist. válida
                # MAP: estado (grado de estenosis 0-4) más probable
                mapa = int(np.argmax(dist))  # arg max_k P(X=k|e)
                ks = "".join([f"{v:7.4f}" for v in dist])
                label = desc if metodo == 'VE' else ''
                print(f"  {label:<42} {metodo:>6}  {mapa:>3}  {ks}")
            except Exception as e:
                print(f"  {desc:<42} {metodo:>6}  Error: {e}")

# Verificación: VE vs BP
print("\n--- Verificación: VE ≡ BP ---")
"""
  Umbral: ‖VE-BP‖₁ < 1e-4 se considera coincidencia numérica exacta,
  comparando las 5 componentes de la distribución completa P(TARGET=0..4|e).
"""

print(f"  {'Perfil':<40} {'Red':<10} {'‖VE-BP‖₁':>10}  {'≡?':>4}")
print("  " + "─" * 68)

for nombre_m, m in [('Experta', modelo_expert), ('Común', modelo_comun)]:
    for desc, ev_d, ev_c in PERFILES:
        ev_d_f = filtrar_evidencia_disc(ev_d, m)
        ev_c_f = {k: v for k, v in ev_c.items() if k in m.nodes_cont}
        try:
            d_ve = np.array(ve_hibrido(m, TARGET, ev_d_f, ev_c_f), dtype=float)
            d_bp = np.array(bp_hibrido(m, TARGET, ev_d_f, ev_c_f), dtype=float)
            d_ve = d_ve / d_ve.sum() if d_ve.sum() > 0 else d_ve
            d_bp = d_bp / d_bp.sum() if d_bp.sum() > 0 else d_bp
            diff = float(np.abs(d_ve - d_bp).sum())
            ok = 'Bien' if diff < 1e-4 else 'CUIDADO'
            print(f"  {desc:<40} {nombre_m:<10} {diff:>10.2e}  {ok:>4}")
        except Exception as e:
            print(f"  {desc:<40} {nombre_m:<10} Error: {e}")

# 6.4 — Actualización secuencial de evidencia
print("\n--- 6.4 Actualización secuencial de P(target=k) ---")
"""
  Simulación de una consulta médica: se añade evidencia progresivamente
  y se observa cómo actualiza P(target=k | evidencia) para k=0,1,2,3,4.
  Con valores originales.
"""

print(f"\n  {'Paso':<42} {'Red':<10} {'MAP':>3}  "
      f"{'k0':>7}{'k1':>7}{'k2':>7}{'k3':>7}{'k4':>7}")
print("  " + "─" * 95)

for desc, ev_d, ev_c in SECUENCIA:
    for nombre_m, m in [('Experta', modelo_expert), ('Común', modelo_comun)]:
        ev_d_f = filtrar_evidencia_disc(ev_d, m)
        ev_c_f = {k: v for k, v in ev_c.items() if k in m.nodes_cont}
        try:
            dist = np.array(ve_hibrido(m, TARGET, ev_d_f, ev_c_f), dtype=float)
            dist = dist / dist.sum() if dist.sum() > 0 else dist
            mapa = int(np.argmax(dist))
            ks = "".join([f"{v:7.4f}" for v in dist])
            label = desc if nombre_m == 'Experta' else ''
            print(f"  {label:<42} {nombre_m:<10} {mapa:>3}  {ks}")
        except Exception as e:
            print(f"  {desc:<42} {nombre_m:<10} Error: {e}")                


# =============================================================================
# BLOQUE 7: INFERENCIA APROXIMADA 
# =============================================================================

print("\n" + "=" * 70)
print("BLOQUE 7: INFERENCIA APROXIMADA (Gibbs Sampling y LW) SOBRE LA RED BAYESIANA HÍBRIDA CLG")
print("=" * 70)
"""
  Implementación manual de ambos métodos de inferencia aproximada sobre
  RedBayesianaHibridaCLG, usando directamente modelo_expert y modelo_comun.
"""

# Referencia exacta VE (bloque 6)
# Usamos ejemplos del bloque 6.4 como casos de pruebas:
# Ejemplo A: solo evidencia discreta
EV_DISC_B7 = {'sex': 1, 'age': 2, 'cp': 1, 'exang': 1, 'oldpeak': 2}
EV_CONT_B7 = {}   
# Ejemplo B: evidencia discreta y continua
EV_DISC_B7_CONT = {'sex': 1, 'age': 2, 'cp': 1, 'exang': 1, 'oldpeak': 2}
EV_CONT_B7_CONT = {'thalach': 115.0}

print("\n  Referencia VE exacta ejemplos 6.4:")
print("  Ejemplo A: solo evidencia discreta (cp=1, exang=1, oldpeak=2)")
print("  Ejemplo B: + evidencia continua  (thalach=115.0)")


r_target_b7 = len(_estados(modelo_expert, TARGET))
resultados_b7 = {}
for nombre_m, m in [('Experta', modelo_expert), ('Común', modelo_comun)]:
    # Ejemplo A
    ev_d = filtrar_evidencia_disc(EV_DISC_B7, m)
    ev_c = {}
    t0 = time.time()
    dist_ve = np.array(ve_hibrido(m, TARGET, ev_d, ev_c), dtype=float)
    t_ve = (time.time() - t0) * 1000
    dist_ve /= dist_ve.sum()
    estados_tgt = _estados(m, TARGET)
    print(f"  {nombre_m:<10} A: " +
          "  ".join([f"k{k}:{dist_ve[i]:.3f}"
                     for i, k in enumerate(estados_tgt)]) +
          f"  MAP={int(np.argmax(dist_ve))}  [{t_ve:.1f} ms]")
    resultados_b7[f'VE_{nombre_m}'] = {'dist': dist_ve, 'ms': t_ve}
    # Ejemplo B 
    ev_d2 = filtrar_evidencia_disc(EV_DISC_B7_CONT, m)
    ev_c2 = {k: v for k, v in EV_CONT_B7_CONT.items() if k in m.nodes_cont}
    t0 = time.time()
    dist_ve2 = np.array(ve_hibrido(m, TARGET, ev_d2, ev_c2), dtype=float)
    t_ve2 = (time.time() - t0) * 1000
    dist_ve2 /= dist_ve2.sum()
    print(f"  {nombre_m:<10} B: " +
          "  ".join([f"k{k}:{dist_ve2[i]:.3f}"
                     for i, k in enumerate(estados_tgt)]) +
          f"  MAP={int(np.argmax(dist_ve2))}  [{t_ve2:.1f} ms]")
    resultados_b7[f'VE_{nombre_m}_cont'] = {'dist': dist_ve2, 'ms': t_ve2}


# 7.1 - Funciones auxiliares de muestreo 
def _norm_pdf_b7(x, mu, sigma):
    """
    Calcula la densidad de una distribución normal N(x; mu, sigma) sin scipy."""
    z = (float(x) - float(mu)) / max(float(sigma), 1e-9)
    return float(np.exp(-0.5 * z * z) / (sigma * np.sqrt(2 * np.pi)))

def prior_sample_clg(m, rng):
    """
    Genera una muestra del prior P(X) en orden topológico.
    """
    estado = {}
    for nodo in m.topological_order(): # recorre la red en orden topologico (padres se generan antes que hijos)
        if m.is_disc(nodo): # discreto
            pa_disc = [p for p in m.parents(nodo) if m.is_disc(p)]
            pa_vals = {p: estado[p] for p in pa_disc if p in estado} # estado de padres discretos
            estados_n = _estados(m, nodo) #estados posibles del nodo
            probs = np.array([float(m.p_disc(nodo, k, pa_vals))
                              for k in estados_n]) # P(nodo=k∣padres).
            probs = np.clip(probs, 1e-15, None) # evita probs nulas
            probs /= probs.sum() #normaliza
            estado[nodo] = int(rng.choice(estados_n, p=probs)) #muestrea un estado siguiendo esa dist.
        else:  # continuo
            pa_disc = [p for p in m.parents(nodo) if m.is_disc(p)]
            cfg = tuple(estado.get(p, 0) for p in pa_disc) # configuracion actual de los padres discretps
            estado_pa = cfg[0] if len(cfg) == 1 else (cfg if cfg else None) #adaptamos formato
            try:
                mu, _, sig2 = m.get_clg(nodo, estado_pa)
            except Exception:
                params = m.clg.get(nodo, {})
                mu, _, sig2 = next(iter(params.values())) if params else (0, [], 1)
            estado[nodo] = float(rng.normal(float(mu), float(np.sqrt(max(sig2, 1e-6))))) #muestra X∼N(μ,σ^2)
    return estado # devuelve una muestra completa del prior

def lk_disc_b7(m, nodo, val, estado):
    """
    Calcula P(nodo=val | Pa_disc del estado actual) según la CPT."""
    pa_disc = [p for p in m.parents(nodo) if m.is_disc(p)]
    pa_vals = {p: estado[p] for p in pa_disc if p in estado}
    return max(float(m.p_disc(nodo, val, pa_vals)), 1e-15)

def lk_cont_b7(m, nodo, val, estado):
    """
    Calcula la verosimilitud de una observación continua N(val; μₖ, σ²ₖ) dado 
    el estado del padre discreto.
    """
    pa_disc = [p for p in m.parents(nodo) if m.is_disc(p)]
    cfg = tuple(estado.get(p, 0) for p in pa_disc)
    estado_pa = cfg[0] if len(cfg) == 1 else (cfg if cfg else None)
    try:
        mu, _, sig2 = m.get_clg(nodo, estado_pa)
    except Exception:
        params = m.clg.get(nodo, {})
        mu, _, sig2 = next(iter(params.values())) if params else (0, [], 1)
    return _norm_pdf_b7(float(val), float(mu), float(np.sqrt(max(sig2, 1e-6))))


# 7.2 - GIBBS SAMPLING HÍBRIDO CLG
print("\n--- 7.2 - Gibbs Sampling Híbrido CLG ---")

def gibbs_cadena_clg(m, ev_disc, ev_cont, n_samples, burn_in, thin, seed):
    """
    Calcula una cadena de Gibbs sobre RedBayesianaHibridaCLG, muestrea
    diferenciando caso discreto y caso continuo.
    """
    rng = np.random.default_rng(seed)
    obs = set(ev_disc) | set(ev_cont) #evidencia obs
    libres = [n for n in m.topological_order() if n not in obs] #vars que Gibbs va a actualizar

    # Inicialización: prior sample para variables libres
    estado = prior_sample_clg(m, rng) # genera un estado inicial completo
    for v, val in ev_disc.items(): #sobrescribimos evidencia -> consistente
        estado[v] = int(val)
    for v, val in ev_cont.items():
        estado[v] = float(val)
    muestras_target = []
    total = burn_in + n_samples * thin #nº total de iteraciones
    # bucle principal
    for it in range(total):
        for nodo in libres:
            if m.is_disc(nodo): # Caso discreto
                estados_n = _estados(m, nodo) #estados posibles
                log_p = np.zeros(len(estados_n)) #vector log-probabilidades
                for i, k in enumerate(estados_n): #probamos cada estado
                    estado[nodo] = k #fijamos temporalmente
                    log_p[i] = np.log(lk_disc_b7(m, nodo, k, estado)) # Factor propio P(Xd=k | Pa_d)
                    for hijo in m.children(nodo): # factores de hijos incluir siempre (sea o no obs)
                        if m.is_disc(hijo): #hijo discreto
                            # Usamos valor observado si está en evidencia, si no el actual
                            v_hijo = ev_disc.get(hijo, estado.get(hijo))
                            if v_hijo is not None:
                                log_p[i] += np.log(lk_disc_b7(m, hijo, int(v_hijo), estado))
                        else:  # hijo continuo
                            v_hijo = ev_cont.get(hijo, estado.get(hijo))
                            if v_hijo is not None:
                                log_p[i] += np.log(lk_cont_b7(m, hijo, float(v_hijo), estado))
                # Muestrear
                log_p -= log_p.max() #evita underflow de P(X=k∣resto)
                probs = np.exp(log_p) #volvemos a probs
                probs /= probs.sum() #normalizamos
                estado[nodo] = int(rng.choice(estados_n, p=probs)) #Muestreo Gibbs: escoge aleatorio

            else:  # Caso continuo
                pa_disc = [p for p in m.parents(nodo) if m.is_disc(p)] #padres discretos
                cfg = tuple(estado.get(p, 0) for p in pa_disc) #configuracion
                est_pa = cfg[0] if len(cfg) == 1 else (cfg if cfg else None)
                try:
                    mu, _, sig2 = m.get_clg(nodo, est_pa) #obtenemos CLG
                except Exception:
                    params = m.clg.get(nodo, {})
                    mu, _, sig2 = next(iter(params.values())) if params else (0, [], 1)
                estado[nodo] = float(rng.normal(float(mu),
                                                float(np.sqrt(max(sig2, 1e-6))))) #muestrea: Xc​∼N(μ,σ2)
        if it >= burn_in and (it - burn_in) % thin == 0: #ignoramos 1ºs iteraciones y guardamos 1 de cada 5
            muestras_target.append(estado.get(TARGET, 0)) #guardamos target
    return np.array(muestras_target)

#Tengo varias cadenas de muestras de la posterior.
#¿Han convergido esas cadenas a la misma dist.posterior?:
def gelman_rubin_b7(cadenas):
    """
    Calcula el estadístico de convergencia R̂ de Gelman-Rubin 
    a partir de varias cadenas (con muestras del target) de Gibbs.
    """
    M = len(cadenas) #nº cadenas
    N = min(len(c) for c in cadenas) #long. comun
    if N < 10 or M < 2: #comprobación básica
        return float('nan')
    chains = np.array([(c[:N] == 0).astype(float) for c in cadenas]) # binarizamos target -> ψ=I(TARGET=0)
    psi_j = chains.mean(axis=1) #media de cada cadena fila a fila
    psi = psi_j.mean() #media global
    B = N / (M - 1) * np.sum((psi_j - psi) ** 2) #varianza entre cadenas: cuánto difieren las medias de las cadenas
    W = np.mean([chains[j].var(ddof=1) for j in range(M)]) #varianza dentro de cadenas
    if W < 1e-15:
        return float('nan')
    var_hat = (N - 1) / N * W + B / N #estimación de la varianza posterior
    return float(np.sqrt(var_hat / W))

#Ejemplo:
N_CHAINS = 4
N_SAMPLES = 5000
BURN_IN = 1000
THIN = 20
tabla_b7 = []   # para la tabla comparativa final
for nombre_m, m in [('Experta', modelo_expert), ('Común', modelo_comun)]:
    ev_d = filtrar_evidencia_disc(EV_DISC_B7, m)
    ev_c = {k: v for k, v in EV_CONT_B7.items() if k in m.nodes_cont}
    dist_ve_ref = resultados_b7[f'VE_{nombre_m}']['dist']
    estados_tgt = _estados(m, TARGET)
    print(f"\n  ── {nombre_m} ({N_CHAINS} cadenas × N={N_SAMPLES}, "
          f"burn_in={BURN_IN}, thin={THIN}) ──")
    cadenas = []
    t0_g = time.time()
    for ch in range(N_CHAINS):
        try:
            c = gibbs_cadena_clg(m, ev_d, ev_c,
                                  N_SAMPLES, BURN_IN, THIN, seed=42 + ch * 1000)
            cadenas.append(c)
            print(f"    Cadena {ch+1}: {len(c)} muestras Bien")
        except Exception as e_ch:
            print(f"    Cadena {ch+1}: error — {e_ch}")
    t_g = (time.time() - t0_g) * 1000
    if cadenas:
        todas = np.concatenate(cadenas)
        dist_gibbs = np.array([(todas == k).mean() for k in estados_tgt])
        mapa_gibbs = int(np.argmax(dist_gibbs))
        error_l1 = float(np.abs(dist_gibbs - dist_ve_ref).sum())
        r_hat = gelman_rubin_b7(cadenas)
        conv = f"R̂={r_hat:.4f} ->{'CONVERGIDO' if r_hat < 1.1 else 'MAL no convergido'}" \
                  if not np.isnan(r_hat) else "R̂=N/A"
        dist_str = "  ".join([f"k{k}:{dist_gibbs[i]:.3f}" for i, k in enumerate(estados_tgt)])
        print(f"  Gibbs {nombre_m}: MAP={mapa_gibbs}  ‖Gibbs-VE‖₁={error_l1:.4f}  [{t_g:.0f} ms]")
        print(f"  Distribución: [{dist_str}]")
        print(f"  {conv}")
        resultados_b7[f'Gibbs_{nombre_m}'] = {
            'dist': dist_gibbs, 'cadenas': cadenas, 'r_hat': r_hat, 't_ms': t_g}
    else:
        print(f"  Gibbs {nombre_m}: fallido.")


# 7.3 - LIKELIHOOD WEIGHTING HÍBRIDO CLG
print("\n--- 7.3 - Likelihood Weighting Híbrido CLG ---")
"""
  w(s) = ∏_{Xd∈E_disc} P(Xd=ed | Pa_d(s))   ·   ∏_{Xc∈E_cont} N(ec; μₖ, σ²ₖ)
  Las variables no observadas se muestrean del prior en orden topológico.
  Ref: K&F §12.2.3 (Alg. 12.2).
"""

def lw_clg(m, ev_disc, ev_cont, n, seed=42):
    """
    Likelihood Weighting sobre RedBayesianaHibridaCLG.
    Devuelve la distribución posterior aproximada P(target=k | E) para todos los estados de target.
    """
    rng = np.random.default_rng(seed)
    estados_tgt = _estados(m, TARGET)
    pesos  = np.zeros(n)
    muestras_tgt = np.zeros(n, dtype=int)
    for i in range(n): #generamos una muestra completa
        estado = {}
        w = 1.0 #peso inicial
        for nodo in m.topological_order():
            if nodo in ev_disc: # Caso 1: evidencia discreta
                # Fijar y acumular peso discreto
                estado[nodo] = int(ev_disc[nodo])
                w *= lk_disc_b7(m, nodo, int(ev_disc[nodo]), estado)
            elif nodo in ev_cont: # Caso 2: evidencia continua
                # Fijar y acumular peso gaussiano
                estado[nodo] = float(ev_cont[nodo])
                w *= lk_cont_b7(m, nodo, float(ev_cont[nodo]), estado)
            elif m.is_disc(nodo): # Caso 3: disc. NO observada -> muestreamos
                pa_disc = [p for p in m.parents(nodo) if m.is_disc(p)] #padres discretos
                pa_vals = {p: estado[p] for p in pa_disc if p in estado}
                ests = _estados(m, nodo) # estados posibles
                probs = np.array([float(m.p_disc(nodo, k, pa_vals)) for k in ests]) #CPT
                probs = np.clip(probs, 1e-15, None); probs /= probs.sum() #normalizacion
                estado[nodo] = int(rng.choice(ests, p=probs)) #muestreo ancestral
            else: # Caso 4: cont. NO observada -> muestreamos
                pa_disc = [p for p in m.parents(nodo) if m.is_disc(p)] #padres discretos
                cfg = tuple(estado.get(p, 0) for p in pa_disc)
                est_pa = cfg[0] if len(cfg) == 1 else (cfg if cfg else None)
                try:
                    mu, _, sig2 = m.get_clg(nodo, est_pa)
                except Exception:
                    params = m.clg.get(nodo, {})
                    mu, _, sig2 = next(iter(params.values())) if params else (0, [], 1)
                estado[nodo] = float(rng.normal(float(mu),
                                                float(np.sqrt(max(sig2, 1e-6))))) #muestreo ancestral continuo
        pesos[i] = max(w, 1e-300)
        muestras_tgt[i] = estado.get(TARGET, estados_tgt[0])
    suma_w = pesos.sum() #estimamos la posterior
    if suma_w < 1e-290:
        return np.ones(len(estados_tgt)) / len(estados_tgt)
    return np.array([float(np.sum(pesos * (muestras_tgt == k))) / suma_w
                     for k in estados_tgt]) #aproximacion
#ejemplo
NS_LW = [100, 500, 1000, 3000, 5000, 10000]
for nombre_m, m in [('Experta', modelo_expert), ('Común', modelo_comun)]:
    ev_d = filtrar_evidencia_disc(EV_DISC_B7, m)
    ev_c = {k: v for k, v in EV_CONT_B7.items() if k in m.nodes_cont}
    dist_ve_ref = resultados_b7[f'VE_{nombre_m}']['dist']
    estados_tgt = _estados(m, TARGET)
    print(f"\n  ── {nombre_m} ──")
    print(f"  {'N':>7}  {'MAP':>4}  {'‖LW-VE‖₁':>10}  {'ms':>6}")
    print("  " + "─" * 30)
    p_lw_list = []
    for n_s in NS_LW:
        t0 = time.time()
        try:
            dist_lw = lw_clg(m, ev_d, ev_c, n=n_s, seed=42)
            mapa_lw = int(np.argmax(dist_lw))
            error_l1 = float(np.abs(dist_lw - dist_ve_ref).sum())
        except Exception as e_lw:
            print(f"  [LW error N={n_s}]: {e_lw}")
            dist_lw = None
            mapa_lw = -1
            error_l1 = float('nan')
        t_ms = (time.time() - t0) * 1000
        p_lw_list.append(float(dist_lw[0]) if dist_lw is not None else float('nan'))
        print(f"  {n_s:>7}  {mapa_lw:>4}  {error_l1:>10.5f}  {t_ms:>6.0f}")
    resultados_b7[f'LW_{nombre_m}'] = {'p_lw_list': p_lw_list, 'ns': NS_LW}


# 7.4 - Gráficas
colores_k = ['#4e79a7', '#76b7b2', '#f28e2b', '#e05c4b', '#59a14f']
fig7, axes7 = plt.subplots(3, 2, figsize=(18, 17))

def acf_manual(x, max_lag=50):
    """
    ACF (Autocorrelation Function) manual hasta max_lag.
    Se analiza la calidad de la cadena de Gibbs.
    """
    x = np.array(x, dtype=float); x -= x.mean()
    var = np.var(x)
    if var < 1e-15:
        return np.zeros(max_lag + 1)
    return np.array([np.mean(x[:len(x)-k] * x[k:]) / var
                     for k in range(max_lag + 1)])

for col_i, (nombre_m, m) in enumerate([('Experta', modelo_expert),
                                        ('Común',   modelo_comun)]):
    dist_ve = resultados_b7[f'VE_{nombre_m}']['dist']
    lw_data = resultados_b7.get(f'LW_{nombre_m}', {})
    gib_data_m = resultados_b7.get(f'Gibbs_{nombre_m}', {})
    estados_m = _estados(m, TARGET)
    ev_d_m = filtrar_evidencia_disc(EV_DISC_B7, m)

    # Convergencia LW vs VE: usamos P(k=0) como métrica de convergencia
    ax_lw = axes7[0, col_i]
    p_k0_ve = float(dist_ve[0])
    valid = [(n, p) for n, p in zip(lw_data.get('ns', []),
                                     lw_data.get('p_lw_list', []))
             if not np.isnan(p)]
    if valid:
        ns_v, ps_v = zip(*valid)
        ax_lw.semilogx(ns_v, ps_v, 'o-', color='#4e79a7', lw=2,
                       markersize=6, label='LW')
    ax_lw.axhline(p_k0_ve, color='#e05c4b', linestyle='--', lw=2,
                  label=f'VE exacto = {p_k0_ve:.4f}')
    if 'dist' in gib_data_m:
        p_k0_g = float(gib_data_m['dist'][0])
        r_g = gib_data_m.get('r_hat', float('nan'))
        ax_lw.axhline(p_k0_g, color='#f28e2b', linestyle=':', lw=2,
                      label=f'Gibbs = {p_k0_g:.4f}  (R̂={r_g:.3f})')
    ax_lw.set_xlabel('N muestras LW (log)')
    ax_lw.set_ylabel('P(k=0 | ev)')
    ax_lw.set_title(f'Red {nombre_m}: convergencia LW vs VE', fontsize=15)
    ax_lw.legend(fontsize=13); ax_lw.grid(alpha=0.3)

    # ACF con y sin burn-in
    ax_acf = axes7[1, col_i]
    if 'cadenas' in gib_data_m:
        c_stat = gibbs_cadena_clg(m, ev_d_m, {},
                                   n_samples=500, burn_in=BURN_IN,
                                   thin=1, seed=42)
        c_burn = gibbs_cadena_clg(m, ev_d_m, {},
                                   n_samples=BURN_IN, burn_in=0,
                                   thin=1, seed=42)
        lags  = np.arange(51)
        acf_b = acf_manual(c_burn, 50)
        acf_s = acf_manual(c_stat, 50)
        ax_acf.bar(lags - 0.2, acf_b, width=0.35, color='#e05c4b',
                   alpha=0.7, label='Con burn-in (no estacionario)')
        ax_acf.bar(lags + 0.2, acf_s, width=0.35, color='#4e79a7',
                   alpha=0.7, label='Sin burn-in (estacionario)')
        ax_acf.axhline(0, color='black', lw=0.8)
        ic = 1.96 / np.sqrt(len(c_stat))
        ax_acf.axhline( ic, color='gray', linestyle='--', lw=1, label='IC 95%')
        ax_acf.axhline(-ic, color='gray', linestyle='--', lw=1)
    ax_acf.set_xlabel('Lag')
    ax_acf.set_ylabel('Autocorrelación')
    ax_acf.set_title(f'Red {nombre_m}: ACF target: cadena 1\n(con y sin burn-in)', fontsize=15)
    ax_acf.legend(fontsize=13); ax_acf.grid(alpha=0.3)

    # Evolución P(target=k)
    ax_evo = axes7[2, col_i]
    if 'cadenas' in gib_data_m:
        c_bi   = gibbs_cadena_clg(m, ev_d_m, {},
                                   n_samples=BURN_IN, burn_in=0,
                                   thin=1, seed=42)
        c_post = gibbs_cadena_clg(m, ev_d_m, {},
                                   n_samples=N_SAMPLES, burn_in=BURN_IN,
                                   thin=1, seed=42)
        c_tot  = np.concatenate([c_bi, c_post])
        iters  = np.arange(1, len(c_tot) + 1)
        for i_k, k in enumerate(estados_m):
            freq_k = np.cumsum(c_tot == k) / iters
            ax_evo.plot(iters, freq_k,
                        color=colores_k[i_k % len(colores_k)],
                        lw=1.2, label=f'P(target={k})')
            ax_evo.axhline(dist_ve[i_k],
                           color=colores_k[i_k % len(colores_k)],
                           linestyle=':', lw=0.8, alpha=0.6)
        ax_evo.axvline(BURN_IN, color='black', linestyle='--', lw=1.5,
                       label=f'Fin burn-in (it={BURN_IN})')
        ax_evo.axvspan(0, BURN_IN, alpha=0.08, color='red', label='Burn-in')
    ax_evo.set_xlabel('Iteración total (burn-in + muestras)')
    ax_evo.set_ylabel('Frecuencia acumulada')
    ax_evo.set_title(f'Red {nombre_m}: evolución P(target=k)\n'
                     f'(punteado = VE exacto)', fontsize=15)
    ax_evo.legend(fontsize=13, ncol=2); ax_evo.grid(alpha=0.3)

plt.suptitle('Bloque 7: Inferencia aproximada: Gibbs CLG + LW\n'
             'Evidencia: cp=1 (angina típica), exang=1, oldpeak=2  |  '
             'Red Experta (izq.) vs Red Común (dcha.)',
             fontsize=15)
plt.tight_layout()
plt.savefig('uci_b7_convergencia.png', dpi=120, bbox_inches='tight')
plt.close()


# =============================================================================
# BLOQUE 8: COMPARACIÓN RB HÍBRIDA vs MODELOS ML 
# =============================================================================

print("\n" + "=" * 70)
print("BLOQUE 8: COMPARACIÓN RB HÍBRIDA vs MODELOS ML")
print("=" * 70)
 
# Función auxiliar:
def discretizar_fila_rb(fila_dict):
    """
    Convierte las vars continuas age y oldpeak a sus estados discretos
    usando los mismos cortes clínicos del bloque 3A.
    Necesario para los ejemplos donde se dan los datos crudos.
    """
    out = dict(fila_dict)
    if 'age' in out and not pd.isna(out['age']):
        v = float(out['age'])
        out['age'] = 0 if v < 40 else (1 if v < 55 else (2 if v < 70 else 3))
    if 'oldpeak' in out and not pd.isna(out['oldpeak']):
        v = float(out['oldpeak'])
        out['oldpeak'] = 0 if v <= 0 else (1 if v <= 1 else (2 if v <= 2 else 3))
    return out

# 8.1 Entrenar modelos ML 
print("\n--- 8.1 Entrenamiento modelos ML ---")
modelos_ml = {
    'Random Forest':  RandomForestClassifier(n_estimators=200, random_state=42,
                                              class_weight='balanced'),
    'XGBoost':         XGBClassifier(n_estimators=200, random_state=42,
                                     eval_metric='mlogloss', verbosity=0),
    'SVM (RBF)':      SVC(kernel='rbf', probability=True, random_state=42,
                           class_weight='balanced'),
    'Log. Regression': LogisticRegression(max_iter=1000, random_state=42,
                                           class_weight='balanced',
                                           multi_class='ovr'),
}
 
clases = sorted(y_train.unique())
n_clases = len(clases)
resultados_ml = {}
for nombre, clf in modelos_ml.items():
    clf.fit(X_train_norm, y_train)
    y_pred = clf.predict(X_test_norm)
    y_prob = clf.predict_proba(X_test_norm)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    y_bin  = label_binarize(y_test, classes=clases)
    try:
        auc_val = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
    except Exception:
        auc_val = float('nan')
    resultados_ml[nombre] = {'acc': acc, 'f1': f1, 'auc': auc_val,
                              'clf': clf, 'y_pred': y_pred, 'y_prob': y_prob}
    print(f"  {nombre:<20}: Acc={acc:.4f}  F1={f1:.4f}  AUC={auc_val:.4f}")

# 8.2 RB híbrida en test set 
print("\n--- 8.2 RB híbrida (VE) en test set ---")
"""
  Para cada paciente del test set la RB recibe solo las variables observadas.
"""
y_pred_rb_exp = []
y_pred_rb_com = []
y_prob_rb_exp = []
y_prob_rb_com = []
 
X_test_rb = X_test_full.copy()
 
for idx in range(len(X_test_rb)):
    fila = X_test_rb.iloc[idx]
    # Variables discretas crudas
    ev_d_raw_cont = {col: fila[col]
                     for col in VARS_DISCRETAS_RB
                     if col in fila.index and not pd.isna(fila[col]) and col != TARGET}
    # Discretizamos age y oldpeak con los cortes clínicos del B3
    ev_d_raw = discretizar_fila_rb(ev_d_raw_cont)
    ev_d_raw = {k: int(v) for k, v in ev_d_raw.items()}
    ev_c_raw = {col: float(fila[col])
                for col in VARS_CONTINUAS_RB
                if col in fila.index and not pd.isna(fila[col])}
    for m, pred_list, prob_list in [
        (modelo_expert, y_pred_rb_exp, y_prob_rb_exp),
        (modelo_comun,  y_pred_rb_com, y_prob_rb_com),
    ]:
        ev_d = filtrar_evidencia_disc(ev_d_raw, m)
        ev_c = {k: v for k, v in ev_c_raw.items() if k in m.nodes_cont}
        try:
            dist = np.array(ve_hibrido(m, TARGET, ev_d, ev_c), dtype=float)
            dist /= dist.sum()
        except Exception:
            dist = np.ones(len(_estados(m, TARGET))) / len(_estados(m, TARGET))
        estados_m = _estados(m, TARGET)
        pred_list.append(estados_m[int(np.argmax(dist))])
        prob_list.append(dist.tolist())
y_test_arr = y_test.values
 
for nombre_rb, y_pred_rb, y_prob_rb in [
    ('RB Experta (EV)', np.array(y_pred_rb_exp), y_prob_rb_exp),
    ('RB Común  (EV)', np.array(y_pred_rb_com), y_prob_rb_com),
]:
    acc_rb = accuracy_score(y_test_arr, y_pred_rb)
    f1_rb = f1_score(y_test_arr, y_pred_rb, average='macro', zero_division=0)
    y_bin = label_binarize(y_test_arr, classes=clases)
    try:
        auc_rb = roc_auc_score(y_bin, np.array(y_prob_rb),
                               average='macro', multi_class='ovr')
    except Exception:
        auc_rb = float('nan')
    resultados_ml[nombre_rb] = {'acc': acc_rb, 'f1': f1_rb, 'auc': auc_rb,
                                 'y_pred': y_pred_rb, 'y_prob': np.array(y_prob_rb)}
    print(f"  {nombre_rb:<22}: Acc={acc_rb:.4f}  F1={f1_rb:.4f}  AUC={auc_rb:.4f}")

# 8.3 Tabla comparativa 
print("\n--- 8.3 Tabla comparativa, curvas ROC y matrices de confusión ---")
print(f"  {'Modelo':<25} {'Acc':>8} {'F1-macro':>10} {'AUC-OvR':>10}")
print("  " + "─" * 56)
for nombre, res in resultados_ml.items():
    marker = " #" if nombre.startswith('RB') else ""
    print(f"  {nombre:<25} {res['acc']:>8.4f} {res['f1']:>10.4f} "
          f"{res['auc']:>10.4f}{marker}")
 
# Curvas ROC: una figura por clase (OvR)
y_bin_test = label_binarize(y_test_arr, classes=clases)
n_plot     = min(n_clases, 5)
colores_mod = {
    'Random Forest': '#4e79a7',
    'XGBoost': '#f28e2b',
    'SVM (RBF)': '#e05c4b',
    'Log. Regression':'#59a14f',
    'RB Experta (VE)':'#b07aa1',
    'RB Común  (VE)': '#76b7b2',
}
estilos_mod = {k: ('-' if not k.startswith('RB') else '--')
               for k in colores_mod}

# Layout: 3 arriba, 2 abajo
from matplotlib.gridspec import GridSpec

fig_roc = plt.figure(figsize=(18, 10))
gs = GridSpec(4, 6, figure=fig_roc)

axes_roc = [
    fig_roc.add_subplot(gs[:2, :2]),   # fila sup izq
    fig_roc.add_subplot(gs[:2, 2:4]),  # fila sup centro
    fig_roc.add_subplot(gs[:2, 4:]),   # fila sup der
    fig_roc.add_subplot(gs[2:, 1:3]),  # fila inf izq-centro
    fig_roc.add_subplot(gs[2:, 3:5]),  # fila inf der-centro
]

for i_c, clase in enumerate(clases[:n_plot]):
    ax_r = axes_roc[i_c]
    for nombre, res in resultados_ml.items():
        y_prob_m = res.get('y_prob')
        if y_prob_m is None:
            continue
        y_prob_arr = np.array(y_prob_m)
        if y_prob_arr.ndim == 1 or y_prob_arr.shape[1] <= i_c:
            continue
        fpr, tpr, _ = roc_curve(y_bin_test[:, i_c], y_prob_arr[:, i_c])
        roc_auc_c   = auc(fpr, tpr)
        ax_r.plot(fpr, tpr,
                  color=colores_mod.get(nombre, 'gray'),
                  linestyle=estilos_mod.get(nombre, '-'),
                  lw=2, label=f"{nombre} ({roc_auc_c:.2f})")
    ax_r.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax_r.set_xlabel('FPR', fontsize=12)
    ax_r.set_ylabel('TPR', fontsize=12)
    ax_r.set_title(f'ROC: target={clase} (OvR)', fontsize=13)
    ax_r.legend(fontsize=12)
    ax_r.tick_params(labelsize=11)
    ax_r.grid(alpha=0.3)

plt.suptitle('Curvas ROC por clase (One-vs-Rest): ML vs RB híbrida', fontsize=15)
plt.tight_layout()
plt.savefig('uci_roc_curves.png', dpi=120, bbox_inches='tight')
plt.close()

# Matrices de confusión
from sklearn.metrics import ConfusionMatrixDisplay
 
fig_cm, axes_cm = plt.subplots(1, 2, figsize=(12, 5))
 
for ax_cm, (nombre_cm, y_pred_cm) in zip(
        axes_cm,
        [('RB Experta (VE)', np.array(y_pred_rb_exp)),
         ('Random Forest',   resultados_ml['Random Forest']['y_pred'])]):
    cm = confusion_matrix(y_test_arr, y_pred_cm, labels=clases)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=clases)
    disp.plot(ax=ax_cm, colorbar=False, cmap='Blues')
    acc_cm = accuracy_score(y_test_arr, y_pred_cm)
    f1_cm  = f1_score(y_test_arr, y_pred_cm, average='macro', zero_division=0)
    ax_cm.set_title(f'{nombre_cm}\nAcc={acc_cm:.3f}  F1-macro={f1_cm:.3f}')
 
plt.suptitle('Matrices de confusión: test set (N=61)\n'
             'Filas=real, Columnas=predicho', fontsize=11)
plt.tight_layout()
plt.savefig('uci_confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.show()
plt.close()

 
# =============================================================================
# BLOQUE 9: FEATURE IMPORTANCE  Y SENSIBILIDAD MARGINAL RB
# =============================================================================
 
# 9.1 Permutation Importance 
print("\n--- 9.1 Permutation Importance (RF) ---")

perm_results = {}
clf = resultados_ml['Random Forest']['clf']
perm = permutation_importance(clf, X_test_norm, y_test,
                               n_repeats=30, random_state=42,
                               scoring='roc_auc_ovr_weighted')
pi_mean = pd.Series(perm.importances_mean, index=X_test_norm.columns)
pi_std = pd.Series(perm.importances_std,  index=X_test_norm.columns)
perm_results['Random Forest'] = {'mean': pi_mean, 'std': pi_std,
                                  'raw': perm.importances}
pi_sorted = pi_mean.sort_values(ascending=False)
print(f"\n  Random Forest (AUC-OvR ponderado):")
print(f"  {'Variable':<14} {'ΔAUC-OvR':>12}  {'±std':>8}")
print("  " + "─" * 38)
for var in pi_sorted.index:
    barra = '█' * max(0, int(pi_sorted[var] * 60))
    print(f"  {var:<14} {pi_sorted[var]:>12.4f}  "
          f"±{pi_std[var]:.4f}  {barra}")
 
# 9.2 Sensibilidad marginal RB: discretas + continuas 
print("\n--- 9.2 Sensibilidad marginal RB (variables discretas y continuas) ---")

# Calculamos para ambas redes y guardamos en dict por red
sens_por_red = {}
for nombre_m_s, m_s in [('Experta', modelo_expert), ('Común', modelo_comun)]:
    sensibilidades = {}
    detalle_sens_m = {}

    # Variables discretas
    for nodo in sorted(nodos_disc):
        if nodo == TARGET:
            continue
        estados_n = _estados(m_s, nodo)
        if len(estados_n) < 2:
            continue
        p_vals = []
        for k in estados_n:
            ev_f = filtrar_evidencia_disc({nodo: k}, m_s)
            try:
                dist = np.array(ve_hibrido(m_s, TARGET, ev_f), dtype=float)
                dist /= dist.sum()
                p_vals.append((str(k), float(1 - dist[0])))
            except Exception:
                pass
        if len(p_vals) >= 2:
            p_min = min(p_vals, key=lambda x: x[1])
            p_max = max(p_vals, key=lambda x: x[1])
            sensibilidades[nodo] = p_max[1] - p_min[1]
            detalle_sens_m[nodo] = {'tipo': 'disc', 'min': p_min, 'max': p_max}

    # Variables continuas: barrer [P5, P95]
    RANGOS_CONT = {
        'thalach':  (df_train_hibrido['thalach'].quantile(0.05),
                     df_train_hibrido['thalach'].quantile(0.95)),
        'chol':     (df_train_hibrido['chol'].quantile(0.05),
                     df_train_hibrido['chol'].quantile(0.95)),
        'trestbps': (df_train_hibrido['trestbps'].quantile(0.05),
                     df_train_hibrido['trestbps'].quantile(0.95)),
    }
    N_GRID = 20
    for nodo_c, (vmin, vmax) in RANGOS_CONT.items():
        if nodo_c not in m_s.nodes_cont:
            continue
        grid   = np.linspace(vmin, vmax, N_GRID)
        p_vals = []
        for t in grid:
            try:
                dist = np.array(ve_hibrido(m_s, TARGET, {},
                                            {nodo_c: float(t)}), dtype=float)
                dist /= dist.sum()
                p_vals.append((f"{t:.1f}", float(1 - dist[0])))
            except Exception:
                pass
        if len(p_vals) >= 2:
            p_min = min(p_vals, key=lambda x: x[1])
            p_max = max(p_vals, key=lambda x: x[1])
            sensibilidades[nodo_c] = p_max[1] - p_min[1]
            detalle_sens_m[nodo_c] = {'tipo': 'cont', 'min': p_min, 'max': p_max,
                                       'rango': (vmin, vmax)}

    sens_por_red[nombre_m_s] = {
        'sorted': sorted(sensibilidades.items(), key=lambda x: x[1], reverse=True),
        'detalle': detalle_sens_m
    }

# Mostrar ambas redes lado a lado
for nombre_m_s in ['Experta', 'Común']:
    print(f"\n  ── Red {nombre_m_s} ──")
    sens_sorted_m = sens_por_red[nombre_m_s]['sorted']
    detalle_m     = sens_por_red[nombre_m_s]['detalle']
    print(f"  {'Variable':<14} {'Tipo':<8} {'ΔP(target>0)':>12}  "
      f"{'Val más sano':>14}  {'Val más enfermo':>16}")
    print("  " + "─" * 72)
    for nodo, delta in sens_sorted_m:
        d     = detalle_m[nodo]
        tipo  = d['tipo']
        p_min = d['min']; p_max = d['max']
        barra = '█' * int(delta * 45)
        if tipo == 'disc':
            print(f"  {nodo:<14} {'disc':<8} {delta:>10.4f}  "
                  f"k={p_min[0]} -> {p_min[1]:.3f}  "
                  f"k={p_max[0]} -> {p_max[1]:.3f}  {barra}")
        else:
            print(f"  {nodo:<14} {'cont':<8} {delta:>10.4f}  "
                  f"t={p_min[0]}→{p_min[1]:.3f}  "
                  f"t={p_max[0]}→{p_max[1]:.3f}  {barra}")

# 9.3 SHAP multiclase 
try:
    import shap
    print("\n--- 9.3 SHAP values (RF, multiclase) ---")
 
    rf_clf = resultados_ml['Random Forest']['clf']
    explainer = shap.TreeExplainer(rf_clf)
    shap_values = explainer.shap_values(X_test_norm)
    # Normalizar a array 3D (N, F, C)
    if isinstance(shap_values, list):
        shap_3d = np.stack(shap_values, axis=2) # versiones antiguas
    else:
        shap_3d = shap_values # versiones nuevas
 
    shap_imp = pd.Series(
        np.abs(shap_3d).mean(axis=(0, 2)),
        index=X_test_norm.columns
    ).sort_values(ascending=False)
 
    print(f"  {'Variable':<14} {'|SHAP| medio':>14}")
    print("  " + "─" * 30)
    for var, val in shap_imp.items():
        barra = '█' * int(val / shap_imp.max() * 30)
        print(f"  {var:<14} {val:>14.4f}  {barra}")
 
    shap_cls1 = shap_3d[:, :, 1] # (N, F) para target=1
 
    # Figura SHAP: importancia global + beeswarm (target=1)
    plt.close('all')
    fig_shap, axes_shap = plt.subplots(1, 2, figsize=(20, 8))

    shap_imp_sorted = shap_imp.sort_values()
    axes_shap[0].barh(shap_imp_sorted.index, shap_imp_sorted.values,
                      color='#4e79a7', alpha=0.85)
    axes_shap[0].set_xlabel('Mean |SHAP value|', fontsize=13)
    axes_shap[0].set_title('SHAP: importancia global\n'
                            '(media |SHAP| sobre todas las clases)', fontsize=13)
    axes_shap[0].tick_params(labelsize=12)
    axes_shap[0].grid(alpha=0.3, axis='x')
 
    plt.sca(axes_shap[1])
    try:
        expl = shap.Explanation(
            values=shap_cls1,
            base_values=explainer.expected_value[1]
                        if hasattr(explainer.expected_value, '__len__')
                        else float(explainer.expected_value),
            data=X_test_norm.values,
            feature_names=list(X_test_norm.columns)
        )
        shap.plots.beeswarm(expl, show=False, max_display=13,
                            order=shap.Explanation.abs.mean(0))
        axes_shap[1].set_title('SHAP beeswarm: target=1 vs resto\n'
                               '(rojo=aumenta riesgo, azul=reduce)', fontsize=13)
    except Exception as e_bee:
        shap.summary_plot(shap_cls1, X_test_norm,
                          show=False, plot_type='dot')
        axes_shap[1].set_title(f'SHAP dot: target=1  [{e_bee}]', fontsize=13)
 
    plt.suptitle('Valores SHAP: Random Forest (multiclase OvR)',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig('uci_shap.png', dpi=120, bbox_inches='tight')
    plt.show()
    plt.close('all')

except ImportError:
    print("\n  [shap no instalado: omitido]")
    shap_imp = None
except Exception as e_shap:
    print(f"\n  SHAP error: {e_shap}")
    shap_imp = None
 
# 9.4 Gráfica comparativa: RF permutation importance vs RB sensibilidad 
print("\n--- 9.4 Gráfica: RF permutation importance vs RB sensibilidad ---")
 
fig9, axes9 = plt.subplots(1, 3, figsize=(22, 7))
 
# RF: filtrar importancias positivas + barras de error
pi_rf_mean = perm_results['Random Forest']['mean']
pi_rf_std = perm_results['Random Forest']['std']
pi_rf_pos = pi_rf_mean[pi_rf_mean > 0].sort_values()
pi_rf_std_pos = pi_rf_std[pi_rf_pos.index]
 
if len(pi_rf_pos) == 0:
    # Si todas son negativas o cero, mostrar todas ordenadas
    pi_rf_pos = pi_rf_mean.sort_values()
    pi_rf_std_pos = pi_rf_std[pi_rf_pos.index]
    print("  [Aviso] Todas las importancias RF ≤ 0 —> mostrando todas]") # dataset test muy pequeño
 
axes9[0].barh(pi_rf_pos.index, pi_rf_pos.values,
              xerr=pi_rf_std_pos.values,
              color='#4e79a7', alpha=0.85,
              capsize=4, error_kw={'lw': 1.5})
axes9[0].axvline(0, color='black', lw=0.8)
axes9[0].set_xlabel('ΔAUC-OvR (permutation)', fontsize=14)
axes9[0].set_title('RF: Permutation Importance\n(AUC-OvR, ±std, 30 repeticiones)', fontsize=13)
axes9[0].tick_params(labelsize=12)
axes9[0].grid(alpha=0.3, axis='x')
 
# RB Experta: sensibilidad marginal
sens_exp    = sens_por_red.get('Experta', {})
sens_sorted = sens_exp.get('sorted', [])
detalle_sens = sens_exp.get('detalle', {})
if sens_sorted:
    sens_ser = pd.Series(dict(sens_sorted)).sort_values()
    # Colores distintos para disc vs cont
    colores_sens = ['#f28e2b' if detalle_sens.get(n, {}).get('tipo') == 'disc'
                    else '#e05c4b'
                    for n in sens_ser.index]
    axes9[1].barh(sens_ser.index, sens_ser.values,
                  color=colores_sens, alpha=0.85)
    # Leyenda manual
    import matplotlib.patches as mpatches
    axes9[1].legend(handles=[
        mpatches.Patch(color='#f28e2b', label='Discreta'),
        mpatches.Patch(color='#e05c4b', label='Continua'),
    ], fontsize=13, loc='lower right')
else:
    axes9[1].text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                  transform=axes9[1].transAxes)
 
axes9[1].set_xlabel('ΔP(target>0)', fontsize=13)
axes9[1].set_title('RB Experta: sensibilidad marginal\n'
                   '(naranja=discreta, rojo=continua)', fontsize=14)
axes9[1].tick_params(labelsize=12)
axes9[1].grid(alpha=0.3, axis='x')
 
 
# RB Común sensibilidad marginal
sens_com = sens_por_red.get('Común', {}).get('sorted', [])
det_com  = sens_por_red.get('Común', {}).get('detalle', {})
if sens_com:
    sens_ser_c = pd.Series(dict(sens_com)).sort_values()
    cols_c = ['#f28e2b' if det_com.get(n, {}).get('tipo') == 'disc'
              else '#e05c4b' for n in sens_ser_c.index]
    axes9[2].barh(sens_ser_c.index, sens_ser_c.values, color=cols_c, alpha=0.85)
    axes9[2].legend(handles=[
        mpatches.Patch(color='#f28e2b', label='Discreta'),
        mpatches.Patch(color='#e05c4b', label='Continua'),
    ], fontsize=13, loc='lower right')
axes9[2].set_xlabel('ΔP(target>0)', fontsize=13)
axes9[2].set_title('RB Común: sensibilidad marginal\n(naranja=discreta, rojo=continua)', fontsize=14)
axes9[2].tick_params(labelsize=12)
axes9[2].grid(alpha=0.3, axis='x')
plt.suptitle('Importancia de variables: RF permutation vs RB sensibilidad marginal',
             fontsize=15)
fig9.set_size_inches(22, 7)
plt.tight_layout()
plt.savefig('uci_feature_importance.png', dpi=120, bbox_inches='tight')
plt.show()
plt.close()


# =============================================================================
# BLOQUE 10: DIFERENTES PRUEBAS
# =============================================================================
print("\n" + "=" * 70)
print("BLOQUE 10: DIFERENTES PRUEBAS")
print("=" * 70)

colores_k = ['#4e79a7', '#76b7b2', '#f28e2b', '#e05c4b', '#59a14f']
 
# 10.1 Distribuciones condicionales CLG 
print("\n--- 10.1 Distribución condicional P(thalach | target=k) ---")
# thalach es la variable continua MÁS discriminativa (buscamos inversa de target ->thalach)

fig_cond, axes_cond = plt.subplots(2, 2, figsize=(14, 10))
t_range_plot  = np.linspace(60, 220, 400)
thalach_range = np.arange(80, 200, 1)
post_matrices = {}

for row_i, (nombre_m, m) in enumerate([('Experta', modelo_expert),
                                         ('Común',   modelo_comun)]):
    estados_t = _estados(m, TARGET)
    priors_t  = np.array([float(m.p_disc(TARGET, k, {})) for k in estados_t])
    clg_thal  = m.clg.get('thalach', {})
    mus_t = []; sigmas_t = []
    print(f"\n  ── Red {nombre_m} ──")
    print(f"  {'target=k':>9}  {'μₖ (lpm)':>10}  {'σₖ':>8}  {'P(target=k)':>12}")
    print("  " + "─" * 46)
    for i_k, k in enumerate(estados_t):
        entry = clg_thal.get((k,), clg_thal.get(k, None))
        if entry:
            mu_k, _, sig2_k = entry
            sig_k = float(np.sqrt(max(sig2_k, 1e-6)))
        else:
            mu_k, sig_k = 150.0, 20.0
        mus_t.append(float(mu_k)); sigmas_t.append(sig_k)
        print(f"  {k:>9}  {mu_k:>10.2f}  {sig_k:>8.2f}  {priors_t[i_k]:>12.4f}")
    # Inversión bayesiana
    post_matrix = np.zeros((len(thalach_range), len(estados_t)))
    for i_t, t in enumerate(thalach_range):
        lk   = np.array([sp_norm.pdf(t, loc=mus_t[i_k], scale=sigmas_t[i_k])
                         for i_k in range(len(estados_t))])
        post = lk * priors_t
        s    = post.sum()
        post_matrix[i_t] = post / s if s > 1e-300 else priors_t
    post_matrices[nombre_m] = (post_matrix, estados_t, mus_t, sigmas_t)
    print(f"\n  P(target=k | thalach=t): valores clave:")
    print(f"  {'thalach':>8}  " + "  ".join([f"P(k={k})" for k in estados_t]))
    print("  " + "─" * 58)
    for t_check in [85, 110, 130, 150, 158, 175, 190]:
        i_t  = np.argmin(np.abs(thalach_range - t_check))
        vals = "  ".join([f"{post_matrix[i_t, i_k]:>7.4f}" for i_k in range(len(estados_t))])
        print(f"  {t_check:>8}  {vals}")
    # Graficas
    ax_l = axes_cond[row_i, 0]
    for i_k, k in enumerate(estados_t):
        dens = sp_norm.pdf(t_range_plot, loc=mus_t[i_k], scale=sigmas_t[i_k])
        ax_l.plot(t_range_plot, dens, color=colores_k[i_k], lw=2,
                  label=f'target={k}  μ={mus_t[i_k]:.0f}')
    ax_l.set_xlabel('thalach (lpm)', fontsize=13); ax_l.set_ylabel('Densidad', fontsize=13)
    ax_l.set_title(f'Red {nombre_m}: P(thalach|target=k)', fontsize=13)
    ax_l.legend(fontsize=11); ax_l.tick_params(labelsize=12); ax_l.grid(alpha=0.3)
    ax_r = axes_cond[row_i, 1]
    for i_k, k in enumerate(estados_t):
        ax_r.plot(thalach_range, post_matrix[:, i_k],
                  color=colores_k[i_k], lw=2, label=f'P(target={k}|t)')
    ax_r.axvline(158, color='gray', linestyle='--', lw=1.2, label='μ(sano)=158')
    ax_r.set_xlabel('thalach (lpm)', fontsize=13); ax_r.set_ylabel('P(target=k | thalach=t)', fontsize=13)
    ax_r.set_title(f'Red {nombre_m}: inversión bayesiana CLG', fontsize=13)
    ax_r.legend(fontsize=11); ax_r.tick_params(labelsize=12); ax_r.grid(alpha=0.3)

# Variables para uso posterior (experta como referencia)
estados_t = _estados(modelo_expert, TARGET)
post_matrix = post_matrices['Experta'][0]
thalach_range = np.arange(80, 200, 1)

plt.suptitle('Análisis CLG: P(thalach|target) y P(target|thalach)\n'
             'Experta (fila sup.) vs Común (fila inf.)', fontsize=15)
plt.tight_layout()
plt.savefig('uci_condicional_clg.png', dpi=120, bbox_inches='tight')
plt.show(); plt.close()

# 10.2 Diagnóstico en pacientes reales del test set 
print("\n--- 10.2 Diagnóstico en pacientes reales del test set ---")

# selección de pacientes representativos:
idx_sano = y_test[y_test == 0].index[:2].tolist()
idx_mod = y_test[y_test == 2].index[:1].tolist()
idx_grave = y_test[y_test >= 3].index[:1].tolist()
idx_casos = (idx_sano + idx_mod + idx_grave)[:5]
 
print(f"\n  {'#':>2}  {'Real':>5}  {'Exp':>6}  {'Com':>6}  "
      f"{'P(k=0)':>7}  {'P(k=1)':>7}  {'P(k=2)':>7}  "
      f"{'P(k=3)':>7}  {'P(k=4)':>7}  {'N_obs':>5}  {'age':>8}")
print("  " + "─" * 80)
 
for i, idx in enumerate(idx_casos): #recorremos pacientes
    real = int(y_test[idx])
    fila = X_test_full.loc[idx]
    # separamos evidencia discreta y continua:
    ev_d_raw = {col: fila[col] for col in VARS_DISCRETAS_RB
                if col in fila.index and not pd.isna(fila[col])}
    ev_d_raw = discretizar_fila_rb(ev_d_raw)
    ev_d_raw = {k: int(v) for k, v in ev_d_raw.items() if k != TARGET}
    ev_c_raw = {col: float(fila[col]) for col in VARS_CONTINUAS_RB
                if col in fila.index and not pd.isna(fila[col])}
    n_obs = len(ev_d_raw) + len(ev_c_raw)
    age_disc = ev_d_raw.get('age', '?') #discrtizamos age
    age_str = f"{age_disc}({['<40','40-55','55-70','>70'][age_disc]})" \
               if isinstance(age_disc, int) and age_disc < 4 else str(age_disc)
 
    preds = []
    dist_exp = None
    for m in [modelo_expert, modelo_comun]: #evaluamos
        ev_d = filtrar_evidencia_disc(ev_d_raw, m)
        ev_c = {k: v for k, v in ev_c_raw.items() if k in m.nodes_cont}
        try:
            dist = np.array(ve_hibrido(m, TARGET, ev_d, ev_c), dtype=float) #inferencia probabilística
            dist /= dist.sum()
            est = _estados(m, TARGET)
            preds.append(est[int(np.argmax(dist))])
            if m is modelo_expert:
                dist_exp = dist
        except Exception:
            preds.append(-1)
 
    ks = "  ".join([f"{v:.3f}" for v in (dist_exp if dist_exp is not None
                                            else [0]*5)])
    ok_e = "BIEN" if preds[0] == real else "MAL"
    ok_c = "BIEN" if preds[1] == real else "MAL"
    print(f"  {i+1:>2}  {real:>5}  {preds[0]:>4}{ok_e}  {preds[1]:>4}{ok_c}  "
          f"{ks}  {n_obs:>5}  {age_str:>8}")
 
    
#  10.3 Robustez: datos faltantes simulados (discretos y continuos) 
print("\n--- 10.3 Robustez: datos faltantes simulados (RB vs RF) ---")
rf_clf_full = resultados_ml['Random Forest']['clf']
 
# A) NaN discretos
print("\n  A) NaN en variables discretas:")
print(f"  {'% faltante':>12}  {'RB Exp':>10}  {'RB Com':>10}  {'RF (imp)':>12}  {'Δ(Exp-RF)':>9}")
print("  " + "─" * 52)
fracs_miss  = [0.0, 0.1, 0.3, 0.5]
acc_rb_miss = []
acc_rf_miss = []
 
for frac_miss in fracs_miss:
    rng_m = np.random.default_rng(123)
    y_p_rb = []
    X_miss = X_test_norm.copy()
    vars_d_idx = [c for c in VARS_DISCRETAS_RB
                  if c != TARGET and c in X_miss.columns]
    for idx in range(len(X_miss)):
        n_miss = int(len(vars_d_idx) * frac_miss)
        if n_miss > 0:
            cols_miss = rng_m.choice(vars_d_idx, size=n_miss, replace=False)
            X_miss.iloc[idx, [X_miss.columns.get_loc(c)
                               for c in cols_miss]] = np.nan
    y_p_rb_exp_m = []; y_p_rb_com_m = []
    for idx in range(len(X_test_rb)):
        fila2    = X_test_rb.iloc[idx]
        cols_nan = X_miss.iloc[idx][X_miss.iloc[idx].isna()].index.tolist()
        ev_d2    = {col: fila2[col] for col in VARS_DISCRETAS_RB
                    if col in fila2.index and not pd.isna(fila2[col])
                    and col not in cols_nan and col != TARGET}
        ev_d2    = discretizar_fila_rb(ev_d2)
        ev_d2    = {k: int(v) for k, v in ev_d2.items()}
        ev_c2    = {col: float(fila2[col]) for col in VARS_CONTINUAS_RB
                    if col in fila2.index and not pd.isna(fila2[col])}
        for m_nan, pred_list_nan in [(modelo_expert, y_p_rb_exp_m),
                                      (modelo_comun,  y_p_rb_com_m)]:
            ev_df2 = filtrar_evidencia_disc(ev_d2, m_nan)
            ev_cf2 = {k: v for k, v in ev_c2.items() if k in m_nan.nodes_cont}
            try:
                d2 = np.array(ve_hibrido(m_nan, TARGET, ev_df2, ev_cf2), float)
                d2 /= d2.sum()
                pred_list_nan.append(_estados(m_nan, TARGET)[int(np.argmax(d2))])
            except Exception:
                pred_list_nan.append(0)
    X_imp  = X_miss.fillna(X_train_norm.mean())
    y_p_rf = rf_clf_full.predict(X_imp).tolist()
    a_rb = accuracy_score(y_test_arr, y_p_rb_exp_m)
    a_rbc = accuracy_score(y_test_arr, y_p_rb_com_m)
    a_rf  = accuracy_score(y_test_arr, y_p_rf)
    acc_rb_miss.append(a_rb); acc_rf_miss.append(a_rf)
    print(f"  {frac_miss*100:>11.0f}%  {a_rb:>10.4f}  {a_rbc:>10.4f}  "
          f"{a_rf:>12.4f}  {a_rb-a_rf:>+9.4f}")
 
# B) Ruido gaussiano en continuas
print("\n  B) Ruido gaussiano en variables continuas: ")
print(f"  {'σ_ruido':>9}  {'RB Exp':>10}  {'RB Com':>10}  {'RF (ruido)':>12}  {'Δ(Exp-RF)':>9}")
print("  " + "─" * 52)
 
# Desviaciones típicas relativas al rango clínico de cada variable
SIGMAS_RUIDO = [0, 5, 10, 20]   # lpm / mg·dL (misma escala para las 3)
acc_rb_noise = []
acc_rf_noise = []
 
for sigma_r in SIGMAS_RUIDO:
    rng_n   = np.random.default_rng(99)
    y_p_rb2 = []
    y_p_rf2_rows = []
    y_p_rb2_com_n = []
 
    for idx in range(len(X_test_rb)):
        fila2 = X_test_rb.iloc[idx]
        ev_d2 = {col: fila2[col] for col in VARS_DISCRETAS_RB
                   if col in fila2.index and not pd.isna(fila2[col])
                   and col != TARGET}
        ev_d2 = discretizar_fila_rb(ev_d2)
        ev_d2 = {k: int(v) for k, v in ev_d2.items()}
        # Añadir ruido gaussiano a las variables continuas
        ev_c2 = {}
        row_rf = X_test_norm.iloc[idx].copy()
        for col in VARS_CONTINUAS_RB:
            if col in fila2.index and not pd.isna(fila2[col]):
                ruido = float(rng_n.normal(0, sigma_r)) if sigma_r > 0 else 0.0
                ev_c2[col] = float(fila2[col]) + ruido
                # Para RF: también añadir ruido al valor normalizado
                if col in row_rf.index:
                    std_col = X_train_norm[col].std()
                    row_rf[col] += ruido / (std_col + 1e-9)
        y_p_rb2_exp_n = y_p_rb2  # experta
        # Calcular también para común
        ev_df2_c = filtrar_evidencia_disc(ev_d2, modelo_comun)
        ev_cf2_c = {k: v for k, v in ev_c2.items() if k in modelo_comun.nodes_cont}
        try:
            d2c = np.array(ve_hibrido(modelo_comun, TARGET, ev_df2_c, ev_cf2_c), float)
            d2c /= d2c.sum()
            y_p_rb2_com_n.append(_estados(modelo_comun, TARGET)[int(np.argmax(d2c))])
        except Exception:
            y_p_rb2_com_n.append(0)
        ev_df2 = filtrar_evidencia_disc(ev_d2, modelo_expert)
        ev_cf2 = {k: v for k, v in ev_c2.items() if k in modelo_expert.nodes_cont}
        try:
            d2 = np.array(ve_hibrido(modelo_expert, TARGET, ev_df2, ev_cf2), float)
            d2 /= d2.sum()
            y_p_rb2.append(_estados(modelo_expert, TARGET)[int(np.argmax(d2))])
        except Exception:
            y_p_rb2.append(0)
        y_p_rf2_rows.append(row_rf.values)

    X_rf_noisy = pd.DataFrame(y_p_rf2_rows, columns=X_test_norm.columns)
    y_p_rf2 = rf_clf_full.predict(X_rf_noisy).tolist()
    a_rb2 = accuracy_score(y_test_arr, y_p_rb2)
    a_rbc2 = accuracy_score(y_test_arr, y_p_rb2_com_n) if len(y_p_rb2_com_n) == len(y_test_arr) else float('nan')
    a_rf2 = accuracy_score(y_test_arr, y_p_rf2)
    acc_rb_noise.append(a_rb2); acc_rf_noise.append(a_rf2)
    print(f"  {sigma_r:>9}  {a_rb2:>10.4f}  {a_rbc2:>10.4f}  "
          f"{a_rf2:>12.4f}  {a_rb2-a_rf2:>+9.4f}")
 
#  C) dataset reducido (fbs disc + trestbps cont) 
print("\n--- 10.4 Robustez: dataset reducido (sin fbs y trestbps) ---")
"""
  Por B9.2 elimino las vars menos influyentes: fbs y trestbps.
 """
VARS_ELIM  = ['fbs', 'trestbps']  #eliminamos 
# construimos  dataset reducido
VARS_RED = [v for v in X_train_norm.columns if v not in VARS_ELIM]
VARS_DISC_RED = [v for v in VARS_DISCRETAS_RB
                 if v not in VARS_ELIM + [TARGET]]
VARS_CONT_RED = [v for v in VARS_CONTINUAS_RB if v not in VARS_ELIM]
 
print(f"  Eliminadas: {VARS_ELIM}  (fbs=discreta ΔP=0.113, trestbps=continua ΔP bajo)")
 
X_train_red = X_train_norm[VARS_RED]
X_test_red = X_test_norm[VARS_RED]
 
res_red = {}
for nombre, clf_orig in modelos_ml.items():
    clf_r = clf_orig.__class__(**clf_orig.get_params())
    clf_r.fit(X_train_red, y_train) # reentrenamos ML desde cero
    y_pr = clf_r.predict(X_test_red)
    res_red[nombre] = {'acc': accuracy_score(y_test, y_pr),
                        'f1':  f1_score(y_test, y_pr,
                                         average='macro', zero_division=0)}
 
for nombre_rb_red, m_red in [('RB Experta (VE)', modelo_expert),
                               ('RB Común  (VE)', modelo_comun)]:
    y_pred_rb_red2 = []
    for idx in range(len(X_test_rb)):
        fila   = X_test_rb.iloc[idx]
        ev_d_r = {col: fila[col] for col in VARS_DISC_RED
                  if col in fila.index and not pd.isna(fila[col])}
        ev_d_r = discretizar_fila_rb(ev_d_r)
        ev_d_r = {k: int(v) for k, v in ev_d_r.items()}
        ev_c_r = {col: float(fila[col]) for col in VARS_CONT_RED
                  if col in fila.index and not pd.isna(fila[col])}
        ev_d_f = filtrar_evidencia_disc(ev_d_r, m_red)
        ev_c_f = {k: v for k, v in ev_c_r.items() if k in m_red.nodes_cont}
        try:
            d = np.array(ve_hibrido(m_red, TARGET, ev_d_f, ev_c_f), float)
            d /= d.sum()
            y_pred_rb_red2.append(_estados(m_red, TARGET)[int(np.argmax(d))])
        except Exception:
            y_pred_rb_red2.append(0)
    res_red[nombre_rb_red] = {
        'acc': accuracy_score(y_test_arr, y_pred_rb_red2),
        'f1':  f1_score(y_test_arr, y_pred_rb_red2, average='macro', zero_division=0)
    }
 
print(f"\n  {'Modelo':<22} {'Acc full':>10} {'Acc red':>10} "
      f"{'ΔAcc':>8} {'F1 full':>9} {'F1 red':>9} {'ΔF1':>7}")
print("  " + "─" * 78)
for nombre in list(modelos_ml.keys()) + ['RB Experta (VE)', 'RB Común  (VE)']:
    af = resultados_ml.get(nombre, {}).get('acc', float('nan'))
    ff = resultados_ml.get(nombre, {}).get('f1',  float('nan'))
    ar = res_red.get(nombre, {}).get('acc', float('nan'))
    fr = res_red.get(nombre, {}).get('f1',  float('nan'))
    mk = " #" if nombre.startswith('RB') else ""
    print(f"  {nombre:<22} {af:>10.4f} {ar:>10.4f} {ar-af:>+8.4f} "
          f"{ff:>9.4f} {fr:>9.4f} {fr-ff:>+7.4f}{mk}")

# Gráficas
fig_rob, axes_rob = plt.subplots(1, 3, figsize=(18, 5))
 
# NaN discretos
ax1 = axes_rob[0]
ax1.plot([f*100 for f in fracs_miss], acc_rb_miss,
         'o-', color='#b07aa1', lw=2, markersize=7, label='RB Experta')
ax1.plot([f*100 for f in fracs_miss], acc_rf_miss,
         's--', color='#4e79a7', lw=2, markersize=7, label='RF (imputed)')
ax1.set_xlabel('% vars. discretas con NaN', fontsize=11)
ax1.set_ylabel('Accuracy (test)', fontsize=11)
ax1.set_title('A) NaN discretos\n(RB marginaliza, RF imputa)', fontsize=15)
ax1.legend(fontsize=10); ax1.grid(alpha=0.3)
 
# Ruido continuas
ax2 = axes_rob[1]
ax2.plot(SIGMAS_RUIDO, acc_rb_noise,
         'o-', color='#b07aa1', lw=2, markersize=7, label='RB Experta')
ax2.plot(SIGMAS_RUIDO, acc_rf_noise,
         's--', color='#4e79a7', lw=2, markersize=7, label='RF (ruidoso)')
ax2.set_xlabel('σ ruido gaussiano (unidades originales)', fontsize=11)
ax2.set_ylabel('Accuracy (test)', fontsize=11)
ax2.set_title('B) Ruido en continuas\n(thalach, chol, trestbps)', fontsize=15)
ax2.legend(fontsize=10); ax2.grid(alpha=0.3)
 
# Dataset reducido
ax3 = axes_rob[2]
nombres_c = list(modelos_ml.keys()) + ['RB Experta (VE)', 'RB Común  (VE)']
accs_f2 = [resultados_ml.get(n, {}).get('acc', 0) for n in nombres_c]
accs_r2 = [res_red.get(n, {}).get('acc', 0) for n in nombres_c]
x_p2 = np.arange(len(nombres_c)); w2 = 0.35
ax3.bar(x_p2 - w2/2, accs_f2, w2, label='Full (13 vars)',
        color='#4e79a7', alpha=0.85)
ax3.bar(x_p2 + w2/2, accs_r2, w2, label='Sin fbs+trestbps',
        color='#f28e2b', alpha=0.85)
ax3.set_xticks(x_p2)
ax3.set_xticklabels([n.replace(' ', '\n') for n in nombres_c], fontsize=7)
ax3.set_ylabel('Accuracy'); ax3.set_ylim(0, 1)
ax3.set_title('C) Dataset reducido\n(sin fbs disc + trestbps cont)', fontsize=15)
ax3.legend(fontsize=10); ax3.grid(alpha=0.3, axis='y')
for i_rb, n in enumerate(nombres_c):
    if n.startswith('RB'):
        ax3.axvspan(i_rb - 0.5, i_rb + 0.5, alpha=0.08, color='green')
 
plt.suptitle('Bloque 10: Análisis de robustez: RB Experta vs RF',
             fontsize=13)
plt.tight_layout()
plt.savefig('uci_robustez.png', dpi=120, bbox_inches='tight')
plt.show(); plt.close()