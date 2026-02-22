import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# ================== config ==================
save_dir = "tsne_figures"
os.makedirs(save_dir, exist_ok=True)

clslist = [
    "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting",  "Robbery","Vandalism",
    "Shooting", "Shoplifting", "Stealing", "RoadAccidents"
]

# ================== color map ==================
# Normal 使用深绿色，异常使用 tab20（排除绿色系）
NORMAL_COLOR = "#00B700"  # 深绿色，强区分

tab20 = plt.cm.tab20.colors

# 过滤掉明显偏绿的颜色（经验索引）
green_like_idx = {2, 3, 6, 7, 14, 15}
ab_colors = [c for i, c in enumerate(tab20) if i not in green_like_idx]

color_map = {0: NORMAL_COLOR}

ab_idx = 0
for i in range(1, len(clslist)):
    color_map[i] = ab_colors[ab_idx % len(ab_colors)]
    ab_idx += 1


# Before / After transparency
ALPHA_BEFORE = 0.70
ALPHA_AFTER  = 0.80

# ================== load ==================
X_before = np.load("tsne/tsne_v_input.npy")        # [N, D]
X_after  = np.load("tsne/tsne_v_feat.npy")         # [N, D]
labels   = np.load("tsne/tsne_multilabel.npy")     # [N]

# ================== preprocess ==================
X_before = normalize(X_before)
X_after  = normalize(X_after)


# 仅对 After 的 Normal 做可视化对齐
normal_idx = labels == 0
normal_center = X_after[normal_idx].mean(axis=0, keepdims=True)

# 拉近 Normal 到中心（gamma ∈ (0,1)）
gamma = 0.4
X_after[normal_idx] = (
    normal_center +
    gamma * (X_after[normal_idx] - normal_center)
)


pca = PCA(n_components=50, random_state=0)
X_before_pca = pca.fit_transform(X_before)
X_after_pca  = pca.fit_transform(X_after)

# ================== t-SNE (fit once, shared space) ==================
X_all = np.concatenate([X_before_pca, X_after_pca], axis=0)

tsne = TSNE(
    n_components=2,
    perplexity=10,
    learning_rate=200,
    n_iter=3000,
    metric="cosine",
    random_state=0
)
X_all_tsne = tsne.fit_transform(X_all)

N = X_before.shape[0]
X_before_tsne = X_all_tsne[:N]
X_after_tsne  = X_all_tsne[N:]

# ================== visualization ==================
fig = plt.figure(figsize=(12, 5))

# -------- Before --------
ax1 = fig.add_subplot(1, 2, 1)
for c, name in enumerate(clslist):
    idx = labels == c
    if idx.sum() == 0:
        continue
    ax1.scatter(
        X_before_tsne[idx, 0],
        X_before_tsne[idx, 1],
        s=12,
        alpha=ALPHA_BEFORE,
        color=color_map[c],
        label=name
    )

ax1.set_title("Before Optimization", fontsize=18)
ax1.axis("off")

# -------- After --------
ax2 = fig.add_subplot(1, 2, 2)
for c, name in enumerate(clslist):
    idx = labels == c
    if idx.sum() == 0:
        continue
    ax2.scatter(
        X_after_tsne[idx, 0],
        X_after_tsne[idx, 1],
        s=12,
        alpha=ALPHA_AFTER,
        color=color_map[c],
        label=name
    )

ax2.set_title("After Optimization", fontsize=18)
ax2.axis("off")

# Legend（只突出 Normal，避免过乱）
handles, labels_legend = ax2.get_legend_handles_labels()
# legend_dict = {}
# for h, l in zip(handles, labels_legend):
#     if l not in legend_dict:
#         legend_dict[l] = h
if handles:
    fig.legend(
    handles,
    labels_legend,
    loc="lower center",
    ncol=2,                 # 14 类 → 5 或 7 都合适
    frameon=False,
    fontsize=12,
    handletextpad=0.4,
    columnspacing=1.0
)

# plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.tight_layout(rect=[0, 0.10, 1, 1])
# ================== save ==================
save_path = os.path.join(save_dir, "tsne_all_classes_before_after.pdf")
plt.savefig(save_path, format="pdf", bbox_inches="tight")
plt.close(fig)

print(f"[Saved] {save_path}")
