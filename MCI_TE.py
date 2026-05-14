# -*- codi.ng: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import os
from sklearn.preprocessing import RobustScaler
from matplotlib import rcParams
import warnings

# === 字体配置 ===
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# ====== Step 0. 归一化函数 ======
def robust_scale_dataframe(df: pd.DataFrame, quantile_range=(25, 75)) -> pd.DataFrame:
    scaler = RobustScaler(quantile_range=quantile_range)
    X = scaler.fit_transform(df.values).astype(np.float32)  # 全量拟合即可
    return pd.DataFrame(X, columns=df.columns)

# ====== Step 1. 数据读取 ======
df = pd.read_csv("D:/DISTANCE/DATA/train_wave1.csv")

# 删除不需要的列
drop_cols = [ ]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
var_names = df.columns.tolist()
data = robust_scale_dataframe(df)
n_vars = len(var_names)  # 变量总数

# 转换为 Tigramite DataFrame
dataframe = pp.DataFrame(
    data.values,
    datatime=np.arange(len(data)),
    var_names=var_names
)

# ====== Step 2. 定义独立性检验 ======
cond_test = ParCorr(significance='analytic')

# ====== Step 3. 设置 link_assumptions（关键修正） ======
tau_min = 0
tau_max = 4
target_idx = [0, 1]  # JulianDay, TIME

# 1) 先给所有节点放开“可探索”的默认规则：
#    - 滞后 (tau>0): '-?>'  表示若存在则定向 i -> j
#    - 同现 (tau=0): 'o?o' 表示同现方向未知
link_assumptions = {
    j: {
        (i, -tau): ('-?>' if tau > 0 else 'o?o')
        for i in range(n_vars)
        for tau in range(tau_min, tau_max + 1)
        # 不包含自环的同现 (i==j, tau=0)
        if (tau > 0) or (i != j)
    }
    for j in range(n_vars)
}

# 2) 仅保留 target 的“自身滞后”入边；删除其它所有指向 target 的入边（含同现与滞后）
for j in target_idx:          # 被指向者（effect）
    for i in range(n_vars):   # 潜在原因（cause）
        for tau in range(tau_min, tau_max + 1):
            # 只有 (i==j 且 tau>0) 才保留 = 自身滞后 → 当期
            if not (i == j and tau > 0):
                link_assumptions[j].pop((i, -tau), None)

print("仅保留以下变量的自身滞后作为其父节点：", [var_names[k] for k in target_idx])
print("var_names:", var_names)
print("target_idx:", target_idx)

# ====== Step 4. 运行 PCMCI+ ======
pcmci = PCMCI(
    dataframe=dataframe,
    cond_ind_test=cond_test,
    verbosity=1
)

pc_alpha = 0.05
results = pcmci.run_pcmciplus(
    tau_min=0,
    tau_max=tau_max,
    pc_alpha=pc_alpha,
    link_assumptions=link_assumptions
)

# # FDR 校正
# q_matrix = pcmci.get_corrected_pvalues(
#     p_matrix=results['p_matrix'],
#     tau_max=tau_max,
#     fdr_method='fdr_bh'
# )

# ====== Step 5. 输出显著邻接矩阵 ======
adj = (results['p_matrix'] <= pc_alpha).astype(float)
adj = adj[:, :, 1:].max(axis=2)
adj_df = pd.DataFrame(adj, index=var_names, columns=var_names)

print("\n✅ 显著性因果邻接矩阵（0/1）：")
print(adj_df)

# ====== Step 6. 绘制因果图 ======
tp.plot_graph(
    graph=results['graph'],
    val_matrix=results['val_matrix'],
    var_names=var_names,
    figsize=(14, 12),
    link_colorbar_label='MCI',
    node_colorbar_label='auto-MCI',
    link_label_fontsize=14,
    label_fontsize=14,
    tick_label_size=10,
    node_label_size=14,
    edge_ticks=0.5,
    node_ticks=0.5,
    node_size=0.5
)

tp.plot_time_series_graph(
    figsize=(5, 5),
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    var_names=var_names,
    link_colorbar_label='MCI',
    label_fontsize=14,
    tick_label_size=12
)

plt.tight_layout()
plt.show()

# ====== 构建并保存 “时滞展开”的超矩阵 (包含 tau=0) ======
val_matrix = results['val_matrix']   # 形状: [i, j, tau], tau=0..tau_max
p_matrix   = results['p_matrix']
n_vars     = len(var_names)
L          = val_matrix.shape[2]     # = tau_max + 1
assert L == tau_max + 1

# 行/列标签：tau 在外层，var 在内层（tau-major）
cross_temporal_labels = [f"{var}_tau{tau}" for tau in range(tau_max+1) for var in var_names]
total_nodes = n_vars * (tau_max + 1)

# 是否使用 |MCI|（如需绝对值，把这个开关改为 True）
USE_ABS_MCI = False

cross_val_matrix = np.zeros((total_nodes, total_nodes), dtype=float)
cross_p_matrix   = np.ones((total_nodes, total_nodes), dtype=float)

# 填充规则：
#  val_matrix[i,j,tau_diff] 表示 X_i(t - tau_diff) → X_j(t)
#  若把每个 (var, tau) 当作一个“层节点”，则允许：tau_src >= tau_tgt（过去→现在/更近过去）
for tau_src in range(tau_max + 1):
    for tau_tgt in range(tau_max + 1):
        tau_diff = tau_src - tau_tgt
        if tau_diff < 0 or tau_diff > tau_max:
            continue  # 禁止未来→过去；也防越界

        for i in range(n_vars):
            for j in range(n_vars):
                src_idx = tau_src * n_vars + i   # 行：i 在 tau_src 层
                tgt_idx = tau_tgt * n_vars + j   # 列：j 在 tau_tgt 层
                mci = val_matrix[i, j, tau_diff]
                if USE_ABS_MCI:
                    mci = abs(mci)
                cross_val_matrix[src_idx, tgt_idx] = float(mci)
                cross_p_matrix[src_idx, tgt_idx]   = float(p_matrix[i, j, tau_diff])

# 显著性邻接矩阵（包含 tau=0）
cross_adj_matrix = (cross_p_matrix <= pc_alpha).astype(float)

# ====== Step 7-Plot. 可视化与 DIST 相关的特征 ======
import matplotlib.pyplot as plt
import numpy as np

# ====== Step 7. 提取“与当前 DIST 相关”的显著父特征 ======
TARGET_NAME = "DIST"          # 目标列名（如与你数据不符，请改成实际列名）
INCLUDE_TAU0 = True           # 是否包含同期因果 tau=0
P_THRESH = pc_alpha           # 显著性阈值，沿用上面的 pc_alpha

if TARGET_NAME not in var_names:
    print(f"[WARN] 目标列 {TARGET_NAME} 不在数据列中，跳过特征提取。")
else:
    j = var_names.index(TARGET_NAME)
    selected_edges = []   # [(src_name, tau, mci, pval), ...]

    # 从 p_matrix / val_matrix 中筛选 (i, j, tau) 的显著父边
    tau_start = 0 if INCLUDE_TAU0 else 1
    for i, src_name in enumerate(var_names):
        for tau in range(tau_start, tau_max + 1):
            pval = float(results['p_matrix'][i, j, tau])
            if np.isfinite(pval) and (pval <= P_THRESH):
                mci = float(results['val_matrix'][i, j, tau])
                selected_edges.append((src_name, tau, mci, pval))

    # 列出选中的父边
    if not selected_edges:
        print(f"[INFO] 未发现显著父边指向 {TARGET_NAME}（阈值 p<={P_THRESH}，tau0包含={INCLUDE_TAU0}）。")
    else:
        sel_df = pd.DataFrame(selected_edges, columns=["source", "tau", "MCI", "pvalue"])
        sel_df = sel_df.sort_values(["tau", "pvalue", "source"]).reset_index(drop=True)
        print(f"\n✅ 与 {TARGET_NAME} 当期相关的显著父边（p<= {P_THRESH}）：")
        print(sel_df)

        # 基于“原始 df”（未缩放）构造滞后特征矩阵
        feat_df = pd.DataFrame(index=df.index)
        for src, tau, _, _ in selected_edges:
            col_name = f"{src}_lag{tau}"
            feat_df[col_name] = df[src].shift(tau)  # X_src(t - tau)

        # 目标列：与特征对齐（去掉前 tau_max 行的 NaN）
        target_aligned = df[TARGET_NAME]

        # 对齐与清洗
        design = pd.concat([feat_df, target_aligned.rename(TARGET_NAME)], axis=1)
        design = design.iloc[tau_max:].dropna(axis=0, how="any").reset_index(drop=True)

        print(f"\n✅ 已构造与 {TARGET_NAME} 对齐的特征表：shape = {design.shape}")
        print("前几行：")
        print(design.head())

# 注意：这里不再画单独的 MCI 柱状图，而是留到后面和 TE 一起画

if TARGET_NAME in var_names and 'design' in locals() and 'sel_df' in globals() and not sel_df.empty:

    # --- 图 2：DIST 与所选特征的相关性热力图 ---
    # 只取特征列，跳过目标列
    feat_cols = [c for c in design.columns if c != TARGET_NAME]
    if len(feat_cols) > 0:
        corr_row = design[feat_cols + [TARGET_NAME]].corr().loc[TARGET_NAME, feat_cols].to_frame().T
        plt.figure(figsize=(min(12, 0.45*len(feat_cols)+2), 2.6))
        im = plt.imshow(corr_row, aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Pearson r")
        plt.yticks([0], [TARGET_NAME])
        plt.xticks(range(len(feat_cols)), feat_cols, rotation=60, ha='right')
        plt.title(f"{TARGET_NAME} 与显著父特征的相关性")
        plt.tight_layout()
        plt.show()

    # --- 图 3：时间序列叠加（Top-3 特征，标准化后与目标同图） ---
    top_k = min(3, len(sel_df))
    if top_k > 0:
        # 选择 |MCI| 最大的前 K 个特征
        top_idx = np.argsort(np.abs(sel_df['MCI'].values))[::-1][:top_k]
        top_feats = [f"{sel_df.loc[i, 'source']}_lag{int(sel_df.loc[i, 'tau'])}" for i in sel_df.index[top_idx]]

        # 标准化函数（为了叠加展示）
        def zscore(x):
            x = x.astype(float)
            return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

        plt.figure(figsize=(10, 4.5))
        t = np.arange(len(design))
        plt.plot(t, zscore(design[TARGET_NAME].values), label=f"{TARGET_NAME} (z)", linewidth=2)

        for c in top_feats:
            if c in design.columns:
                plt.plot(t, zscore(design[c].values), label=f"{c} (z)", alpha=0.9)

        plt.title(f"{TARGET_NAME} 与 Top-{top_k} 显著父特征（标准化后）")
        plt.xlabel("样本索引（已与最大滞后对齐）")
        plt.ylabel("标准化值 (z-score)")
        plt.legend(ncol=1, fontsize=9)
        plt.tight_layout()
        plt.show()
else:
    print("[INFO] 无可绘制对象：可能是未找到目标列、没有显著父边，或 Step 7 未运行完成。")

# ====== 提取并保存 “带延迟的 MCI 矩阵” ======

# 从 PCMCI+ 结果中获取原始的三维 MCI 矩阵
# val_matrix[i, j, tau] 表示 变量i(t-tau) 对 变量j(t) 的 MCI 值
val_matrix_3d = results['val_matrix']  # 形状: [n_vars, n_vars, tau_max + 1]
p_matrix_3d = results['p_matrix']  # 形状: [n_vars, n_vars, tau_max + 1]

n_vars = len(var_names)
tau_max = val_matrix_3d.shape[2] - 1  # 因为 tau 从 0 开始

# 1. 构建新的行标签（带时滞）和列标签
# 行标签：例如 "DIST_tau0", "DIST_tau1", ..., "FIRE_tau0", ...
row_labels = []
for var in var_names:
    for tau in range(tau_max + 1):
        row_labels.append(f"{var}_tau{tau}")

# 列标签：就是原始的变量名
col_labels = var_names

# 2. 初始化一个新的二维 MCI 矩阵
# 行数 = 变量数 * (时滞数 + 1)
# 列数 = 变量数
mci_matrix_with_lag = np.zeros((n_vars * (tau_max + 1), n_vars), dtype=float)
p_matrix_with_lag = np.ones((n_vars * (tau_max + 1), n_vars), dtype=float)

# 3. 填充新的二维矩阵
for i in range(n_vars):  # 遍历原因变量
    for j in range(n_vars):  # 遍历结果变量
        for tau in range(tau_max + 1):  # 遍历所有时滞

            # 计算在新矩阵中的行索引
            row_idx = i * (tau_max + 1) + tau

            # 将 3D 矩阵中的值赋给 2D 矩阵
            mci_matrix_with_lag[row_idx, j] = val_matrix_3d[i, j, tau]
            p_matrix_with_lag[row_idx, j] = p_matrix_3d[i, j, tau]

# 4. 转换为 Pandas DataFrame，方便查看和保存
mci_df = pd.DataFrame(mci_matrix_with_lag, index=row_labels, columns=col_labels)
p_df = pd.DataFrame(p_matrix_with_lag, index=row_labels, columns=col_labels)

# 5. (可选) 生成一个“显著性掩码”矩阵，不显著的 MCI 值标记为 NaN
#    这样在查看或绘图时，可以只关注显著的因果关系
pc_alpha = 0.05
significance_mask = (p_df <= pc_alpha)
mci_df_significant = mci_df.where(significance_mask, np.nan)

# =====================================================================================
# =====================       TE + MCI 组合图部分（上下两幅）     =====================
# =====================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm
from pyinform.transferentropy import transfer_entropy
import textwrap

# =============================
# 1. Font & Style
# =============================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

target = "SL"

edges_from_pcmci = [
    ("TIME", 4), ("SLOPE", 0), ("SL", 2), ("SL", 1), ("SLOPE", 4), ("TAIR", 0),
    ("TIME", 0), ("RHUM", 0), ("ELEV", 0), ("TD", 0), ("FIRE", 0), ("SL", 3), ("SL", 4)
]

# =============================
# 2. Data
# =============================
data_te = pd.read_csv("D:/DISTANCE/DATA/train_wave.csv")  # 单独读取，避免和前面 df 混淆

lagged_vars = {}
for var, lag in edges_from_pcmci:
    colname = f"{var}_t-{lag}"
    data_te[colname] = data_te[var].shift(lag)
    lagged_vars[colname] = (var, lag)

data_te = data_te.dropna()

def discretize(series, bins=10):
    return pd.cut(series, bins=bins, labels=False).astype(int)

discrete_data = pd.DataFrame()
discrete_data[target] = discretize(data_te[target])
for new_col in lagged_vars.keys():
    discrete_data[new_col] = discretize(data_te[new_col])

# =============================
# 3. Transfer Entropy
# =============================
te_values, ndte_values = {}, {}
for new_col, (var, lag) in lagged_vars.items():
    try:
        src = discrete_data[new_col].values
        tgt = discrete_data[target].values
        te_xy = transfer_entropy(src, tgt, k=1)
        te_yx = transfer_entropy(tgt, src, k=1)
        denom = te_xy + te_yx
        ndte = te_xy / denom if denom > 0 else 0.0
        te_values[new_col] = te_xy
        ndte_values[new_col] = ndte
    except Exception:
        te_values[new_col], ndte_values[new_col] = 0.0, 0.0

# =============================
# 4. Build Graph (可选 TE 网络图)
# =============================
G = nx.DiGraph()
G.add_node(target)
for new_col in lagged_vars.keys():
    G.add_node(new_col)
    if te_values[new_col] > 0:
        G.add_edge(new_col, target, weight=te_values[new_col], ndte=ndte_values[new_col])

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

num_nodes = len(lagged_vars)
radius = 4
pos = {target: (0, 0)}
angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
for i, node in enumerate(lagged_vars.keys()):
    pos[node] = (radius * np.cos(angles[i]), radius * np.sin(angles[i]))

edges = G.edges()
if len(edges) > 0:
    weights = np.array([G[u][v]['weight'] for u, v in edges])
    norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
else:
    weights = np.array([0.0])
    norm = plt.Normalize(vmin=0, vmax=1)
cmap = plt.colormaps.get_cmap('Blues')

nx.draw_networkx_nodes(G, pos, nodelist=[target],
                       node_size=1200, node_color='#C3E6CB',
                       edgecolors='black', linewidths=1.5, ax=ax)

nx.draw_networkx_nodes(G, pos, nodelist=lagged_vars.keys(),
                       node_size=1000, node_color='#A9CCE3',
                       edgecolors='black', linewidths=1.2, ax=ax)

nx.draw_networkx_edges(G, pos, edgelist=edges,
                       width=2.0, alpha=0.9,
                       edge_color=[G[u][v]['weight'] for u, v in edges] if len(edges) > 0 else 'k',
                       edge_cmap=cmap,
                       arrows=True, arrowstyle='-|>', arrowsize=12, ax=ax)

wrapped_labels = {}
for node in G.nodes():
    if node == target:
        wrapped_labels[node] = node
    else:
        parts = node.split('_t-')
        if len(parts) == 2:
            name_part = parts[0]
            time_part = f"(t-{parts[1]})"
            wrapped_labels[node] = f"{name_part}\n{time_part}"
        else:
            wrapped_labels[node] = node

nx.draw_networkx_labels(G, pos, labels=wrapped_labels, font_size=4, font_weight='bold', ax=ax)

edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                             font_size=6, font_color='black',
                             bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8),
                             label_pos=0.5, ax=ax)

ax.axis('off')
plt.tight_layout()

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.03)
cbar.set_label('Transfer Entropy', fontsize=8, fontweight='bold')

plt.show()

# =============================
# 5. Bar Plot: MCI + TE 上下拼接
# =============================

# 先构造 TE DataFrame（主要为了检查/排序）
te_df = pd.DataFrame({
    "Variable": list(te_values.keys()),
    "TE_Value": list(te_values.values())
})

# 自定义特征顺序（名字和 "TIME_t-4" 这种一致）
feature_order = [
    "SL_t-1",
    "SL_t-2",
    "SL_t-3",
    "SL_t-4",
    "SLOPE_t-0",
    "SLOPE_t-4",
    "TD_t-0",
    "FIRE_t-0",
    "ELEV_t-0",
    "TIME_t-4",
    "TIME_t-0",
    "RHUM_t-0",
    "TAIR_t-0",
]

# 格式化成 “变量名(τ=lag)” 的标签
def format_label(var):
    if "_t-" in var:
        name, lag = var.split("_t-")
        return f"{name}(τ={lag})"
    return var

# 从 sel_df 中提取 MCI（带符号）
mci_dict = {}
if 'sel_df' in globals() and not sel_df.empty:
    for _, row in sel_df.iterrows():
        key = f"{row['source']}_t-{int(row['tau'])}"
        mci_dict[key] = row['MCI']

# 只保留既在 feature_order 中又在 TE/MCI 中出现的特征
plot_features = []
for f in feature_order:
    if (f in te_values) or (f in mci_dict):
        plot_features.append(f)

if len(plot_features) == 0:
    print("[INFO] 没有可用于绘制的共同特征，跳过 MCI+TE 柱状图。")
else:
    x = np.arange(len(plot_features))

    # 构造 MCI 和 TE 的数组
    mci_signed = np.array([mci_dict.get(f, 0.0) for f in plot_features], dtype=float)
    mci_abs = np.abs(mci_signed)
    te_arr = np.array([te_values.get(f, 0.0) for f in plot_features], dtype=float)

    labels = [format_label(f) for f in plot_features]

    # 颜色：MCI 正负区分，TE 统一颜色
    pos_color = "#edc0bd"
    neg_color = "#9bbcde"
    mci_colors = [pos_color if v >= 0 else neg_color for v in mci_signed]

    te_color = "#D6EDF9"

    # 建立上下两个子图，共用 x 轴
    fig2, (ax_mci, ax_te) = plt.subplots(
        2, 1, figsize=(7, 14), dpi=300, sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.0], 'hspace': 0.1}
    )

    # ---------- 上图：|MCI| ----------
    max_mci = mci_abs.max()
    bars_mci = ax_mci.bar(x, mci_abs, color=mci_colors,
                          edgecolor="black", linewidth=1.0, width=0.6)

    for bar, v in zip(bars_mci, mci_abs):
        x_bar = bar.get_x() + bar.get_width() / 2
        y_bar = bar.get_height()
        ax_mci.text(x_bar, y_bar + max_mci * 0.03, f"{v:.3f}",
                    ha='center', va='bottom', fontsize=7)
    ax_mci.set_ylim(0, max_mci * 1.15)

    ax_mci.set_ylabel("|MCI|", fontsize=10, fontweight='bold')
    # 只显示 y 轴刻度，隐藏上图 x 轴刻度标签
    ax_mci.tick_params(axis='x', labelbottom=False)
    ax_mci.tick_params(axis='y', labelsize=9)

    for side in ['top', 'right', 'bottom', 'left']:
        ax_mci.spines[side].set_visible(True)
    ax_mci.grid(False)

    # ---------- 下图：TE ----------
    max_te = te_arr.max()
    bars_te = ax_te.bar(x, te_arr, color=te_color,
                        edgecolor="black", linewidth=1.0, width=0.6)

    for bar in bars_te:
        h = bar.get_height()
        ax_te.text(bar.get_x() + bar.get_width() / 2, h + max_te * 0.05,
                   f"{h:.3f}", ha='center', va='bottom', fontsize=7)

    # 同样拉高 y 轴上限
    ax_te.set_ylim(0, max_te * 1.25)

    ax_te.set_ylabel("Transfer Entropy", fontsize=10, fontweight='bold')
    ax_te.set_xlabel("Feature", fontsize=11, fontweight='bold')

    ax_te.set_xticks(x)
    ax_te.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax_te.tick_params(axis='y', labelsize=9)

    for side in ['top', 'right', 'bottom', 'left']:
        ax_te.spines[side].set_visible(True)
    ax_te.grid(False)

    plt.tight_layout()
    plt.show()
