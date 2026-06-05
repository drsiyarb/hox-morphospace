"""
Figure 3: Internal Geometry and Cell Identity in the Morphospace
Generates a 4-panel figure (A-D) saved as figures/fig3_geometry_celltypes.png
"""
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull
from itertools import combinations

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
H5AD_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s_fca_biohub_body_10x.h5ad")
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

HOX_GENES = ['lab', 'pb', 'Dfd', 'Scr', 'Antp', 'Ubx', 'abd-A', 'Abd-B']

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading h5ad ...")
adata = sc.read_h5ad(H5AD_PATH)
gn = list(adata.var_names)
hox_idx = [gn.index(g) for g in HOX_GENES]

X_raw = adata.X
if hasattr(X_raw, 'toarray'):
    X_raw = X_raw.toarray()

hox_mask = X_raw[:, hox_idx].sum(axis=1) > 0
X_hox = X_raw[hox_mask][:, hox_idx]
print(f"  Hox-expressing cells: {X_hox.shape[0]}")

scaler = MinMaxScaler()
X_hox_norm = scaler.fit_transform(X_hox)

Antp_v = X_hox_norm[:, 4]; Ubx_v = X_hox_norm[:, 5]
abdA_v = X_hox_norm[:, 6]; AbdB_v = X_hox_norm[:, 7]

H1 = Antp_v - Ubx_v
H2 = abdA_v - AbdB_v
H3 = (Antp_v + Ubx_v) - (abdA_v + AbdB_v)

ann_broad = adata.obs['annotation_broad'].values[hox_mask]
ann_specific = adata.obs['annotation'].values[hox_mask]
print(f"  H1 range: [{H1.min():.3f}, {H1.max():.3f}]")
print(f"  H2 range: [{H2.min():.3f}, {H2.max():.3f}]")
print(f"  H3 range: [{H3.min():.3f}, {H3.max():.3f}]")

# ── Load boxcentered CSV for cluster labels ─────────────────────────────────
print("Loading cluster labels ...")
df = pd.read_csv(os.path.join(OUTPUT_DIR, "hox_morphospace_boxcentered.csv"))
labels = df['leiden_label'].values.copy()
U = df['U'].values; V = df['V'].values; W = df['W'].values

# Cluster splits
for i in np.where(labels == 20)[0]:
    if W[i] < -0.22: labels[i] = 30
    elif W[i] > 0.05: labels[i] = 31
    else: labels[i] = 32
for i in np.where(labels == 7)[0]:
    if W[i] < -0.13: labels[i] = 33
    elif W[i] > -0.10: labels[i] = 34
    else: labels[i] = 35
for cl in [12, 15, 16, 27]:
    labels[labels == cl] = 36

# ── Build zonotope (possibility space hull) ─────────────────────────────────
print("Computing possibility space hull ...")
n_grid = 25
gv = np.linspace(0, 1, n_grid)
gAntp, gUbx, gabdA, gAbdB = np.meshgrid(gv, gv, gv, gv, indexing='ij')
gAntp = gAntp.ravel(); gUbx = gUbx.ravel()
gabdA = gabdA.ravel(); gAbdB = gAbdB.ravel()
gH1 = gAntp - gUbx
gH2 = gabdA - gAbdB
gH3 = (gAntp + gUbx) - (gabdA + gAbdB)
pts_h = np.column_stack([gH1, gH2, gH3])

ps_hull = ConvexHull(pts_h)
ps_verts = pts_h[ps_hull.vertices]
print(f"  Hull: {len(ps_hull.vertices)} vertices, {len(ps_hull.simplices)} faces")

# True edges (merge coplanar faces)
face_normals = []
ps_vert_map = {old: new for new, old in enumerate(ps_hull.vertices)}
ps_fi, ps_fj, ps_fk = [], [], []
for tri in ps_hull.simplices:
    mapped = [ps_vert_map.get(v) for v in tri]
    if all(m is not None for m in mapped):
        ps_fi.append(mapped[0]); ps_fj.append(mapped[1]); ps_fk.append(mapped[2])
        v0 = ps_verts[mapped[0]]; v1 = ps_verts[mapped[1]]; v2 = ps_verts[mapped[2]]
        n = np.cross(v1 - v0, v2 - v0)
        nl = np.linalg.norm(n)
        n = n / nl if nl > 1e-12 else n
        face_normals.append(n)
face_normals = np.array(face_normals)

def normal_key(n):
    n = n.copy()
    for i in range(3):
        if abs(n[i]) > 0.1:
            if n[i] < 0: n = -n
            break
    return tuple(np.round(n, 2))

face_groups = {}
for fi_idx in range(len(ps_fi)):
    key = normal_key(face_normals[fi_idx])
    face_groups.setdefault(key, []).append(fi_idx)

edge_to_faces = {}
for fi_idx in range(len(ps_fi)):
    verts = [ps_fi[fi_idx], ps_fj[fi_idx], ps_fk[fi_idx]]
    for a, b in combinations(verts, 2):
        edge = (min(a, b), max(a, b))
        edge_to_faces.setdefault(edge, []).append(fi_idx)

true_edges = set()
for edge, face_list in edge_to_faces.items():
    if len(face_list) >= 2:
        keys = set()
        for fi_idx in face_list:
            keys.add(normal_key(face_normals[fi_idx]))
        if len(keys) > 1:
            true_edges.add(edge)
    else:
        true_edges.add(edge)
print(f"  True edges: {len(true_edges)}")

# ── Vertex dictionary ──────────────────────────────────────────────────────
verts_dict = {f'V{i+1}': ps_verts[i] for i in range(len(ps_hull.vertices))}
verts_dict['C*'] = np.array([0., 0., 0.])

def lerp3(a, b, t):
    return a + t * (b - a)

# ── Diamond planes ─────────────────────────────────────────────────────────
DIAMOND_PLANES = [
    ('C*_V3_V2_V1',        'C*', 'V3',  'V2', 'V1',  'V7',  0),
    ('C*_V3_V2_V1_s20',    'C*', 'V3',  'V2', 'V1',  'V7',  20),
    ('C*_V5_V1_V4',        'C*', 'V5',  'V1', 'V4',  'V13', 10),
    ('C*_V6_V4_V2',        'C*', 'V6',  'V4', 'V2',  'V14', 10),
    ('C*_V10_V2_V8',       'C*', 'V10', 'V2', 'V8',  'V14', 20),
    ('C*_V9_V8_V1',        'C*', 'V9',  'V8', 'V1',  'V13', 20),
    ('C*_V12_V8_V4',       'C*', 'V12', 'V8', 'V4',  'V14', 0),
]

PLANE_COLORS = [
    (0.2, 0.6, 1.0),    # blue
    (0.3, 0.8, 0.9),    # cyan
    (0.9, 0.3, 0.3),    # red
    (0.9, 0.6, 0.2),    # orange
    (0.6, 0.3, 0.8),    # purple
    (0.3, 0.8, 0.3),    # green
    (0.9, 0.8, 0.2),    # yellow
]

def build_diamond(pname, apex_k, base_k, left_k, right_k, shift_k, shift_pct):
    apex  = verts_dict[apex_k].copy()
    base  = verts_dict[base_k].copy()
    left  = verts_dict[left_k].copy()
    right = verts_dict[right_k].copy()
    if shift_pct > 0:
        target = verts_dict[shift_k].copy()
        base = lerp3(base, target, shift_pct / 100.0)
    return apex, base, left, right

# ── Helper: draw zonotope wireframe ────────────────────────────────────────
def draw_wireframe(ax, color='gray', alpha=0.3, lw=0.5):
    for a, b in true_edges:
        ax.plot3D(
            [ps_verts[a, 0], ps_verts[b, 0]],
            [ps_verts[a, 1], ps_verts[b, 1]],
            [ps_verts[a, 2], ps_verts[b, 2]],
            color=color, alpha=alpha, lw=lw
        )

# ══════════════════════════════════════════════════════════════════════════════
# BUILD FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("Building figure ...")
fig = plt.figure(figsize=(16, 14))

# Layout: top row = A, B ; bottom row = C (3 sub-panels), D
gs_top = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25,
                           top=0.95, bottom=0.05, left=0.05, right=0.95,
                           height_ratios=[1, 1])

# ── Panel A: Internal Diamond Planes ──────────────────────────────────────
print("  Panel A ...")
ax_a = fig.add_subplot(gs_top[0, 0], projection='3d')
ax_a.view_init(elev=20, azim=-60)

# Cells as faint background
ax_a.scatter(H1, H2, H3, c='gray', alpha=0.03, s=0.3, rasterized=True)

# Wireframe
draw_wireframe(ax_a, color='gray', alpha=0.4, lw=0.5)

# Diamond planes as colored surfaces
for pi, (pname, apex_k, base_k, left_k, right_k, shift_k, shift_pct) in enumerate(DIAMOND_PLANES):
    apex, base, left, right = build_diamond(pname, apex_k, base_k, left_k, right_k, shift_k, shift_pct)
    r, g, b = PLANE_COLORS[pi]
    # Two triangles: apex-left-right and left-base-right
    tri1 = [apex, left, right]
    tri2 = [left, base, right]
    for tri in [tri1, tri2]:
        poly = Poly3DCollection([tri], alpha=0.35)
        poly.set_facecolor((r, g, b, 0.35))
        poly.set_edgecolor((r, g, b, 0.6))
        ax_a.add_collection3d(poly)

ax_a.set_xlabel('H1', fontsize=8)
ax_a.set_ylabel('H2', fontsize=8)
ax_a.set_zlabel('H3', fontsize=8)
ax_a.set_title('A. Seven internal diamond planes', fontweight='bold', fontsize=11)
ax_a.tick_params(labelsize=6)

# ── Panel B: 2D Plane Projection ──────────────────────────────────────────
print("  Panel B ...")
ax_b = fig.add_subplot(gs_top[0, 1])

# Use the second plane (C*_V3_V2_V1_s20)
pi_sel = 1
pname, apex_k, base_k, left_k, right_k, shift_k, shift_pct = DIAMOND_PLANES[pi_sel]
apex, base, left, right = build_diamond(pname, apex_k, base_k, left_k, right_k, shift_k, shift_pct)

# Build orthonormal basis
e1 = right - left
e1 = e1 / np.linalg.norm(e1)
e2_raw = base - apex
e2_raw = e2_raw - np.dot(e2_raw, e1) * e1  # orthogonalize
e2 = e2_raw / np.linalg.norm(e2_raw)
normal = np.cross(e1, e2)

# Project all cells
cell_pts = np.column_stack([H1, H2, H3])
rel = cell_pts - apex
proj_x = rel @ e1
proj_y = rel @ e2
proj_d = rel @ normal  # distance to plane

# Filter cells near the plane
dist_thresh = 0.3
near_mask = np.abs(proj_d) < dist_thresh

# Diamond corners in 2D
corners = {'apex': apex, 'left': left, 'base': base, 'right': right}
corner_2d = {}
for name, pt in corners.items():
    r = pt - apex
    corner_2d[name] = (np.dot(r, e1), np.dot(r, e2))

# Draw diamond outline
outline_x = [corner_2d['apex'][0], corner_2d['left'][0], corner_2d['base'][0],
             corner_2d['right'][0], corner_2d['apex'][0]]
outline_y = [corner_2d['apex'][1], corner_2d['left'][1], corner_2d['base'][1],
             corner_2d['right'][1], corner_2d['apex'][1]]
ax_b.plot(outline_x, outline_y, 'k-', lw=1.5)

# Grid inside the diamond (10x10)
n_grid_lines = 10
a2d = np.array(corner_2d['apex'])
l2d = np.array(corner_2d['left'])
b2d = np.array(corner_2d['base'])
r2d = np.array(corner_2d['right'])

# Direction 1: lines parallel to apex->left / right->base
for si in range(1, n_grid_lines):
    t = si / n_grid_lines
    p1 = a2d + t * (r2d - a2d)   # point on apex->right
    p2 = l2d + t * (b2d - l2d)   # point on left->base
    ax_b.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', alpha=0.3, lw=0.5)

# Direction 2: lines parallel to apex->right / left->base
for si in range(1, n_grid_lines):
    t = si / n_grid_lines
    p1 = a2d + t * (l2d - a2d)   # point on apex->left
    p2 = r2d + t * (b2d - r2d)   # point on right->base
    ax_b.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', alpha=0.3, lw=0.5)

# Plot cells
ax_b.scatter(proj_x[near_mask], proj_y[near_mask], c='gray', s=1, alpha=0.3, rasterized=True)

# Label vertices
for name, (cx, cy) in corner_2d.items():
    label = name.capitalize()
    if name == 'apex':
        label = apex_k
    elif name == 'base':
        label = base_k + f" (shifted {shift_pct}%)" if shift_pct > 0 else base_k
    elif name == 'left':
        label = left_k
    elif name == 'right':
        label = right_k
    ax_b.annotate(label, (cx, cy), fontsize=8, fontweight='bold',
                  xytext=(5, 5), textcoords='offset points')

ax_b.set_aspect('equal')
ax_b.set_xlabel('e1 (along right-left)', fontsize=8)
ax_b.set_ylabel('e2 (along base-apex)', fontsize=8)
ax_b.set_title('B. 2D projection onto internal plane', fontweight='bold', fontsize=11)
ax_b.tick_params(labelsize=7)
near_count = near_mask.sum()
ax_b.text(0.02, 0.02, f'{near_count} cells within d={dist_thresh}',
          transform=ax_b.transAxes, fontsize=7, color='gray')

# ── Panel C: Cell Types (3 sub-panels) ────────────────────────────────────
print("  Panel C ...")
gs_c = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_top[1, 0], wspace=0.05)

muscle_mask = ann_broad == 'muscle cell'
neuron_mask = ann_broad == 'neuron'
print(f"    Muscle: {muscle_mask.sum()}, Neurons: {neuron_mask.sum()}")

views = [
    (20, -60, 'default 3/4'),
    (15, 30, 'rotated'),
    (5, -120, 'other side'),
]

for vi, (elev, azim, view_label) in enumerate(views):
    ax_c = fig.add_subplot(gs_c[0, vi], projection='3d')
    ax_c.view_init(elev=elev, azim=azim)

    # All cells faint
    ax_c.scatter(H1, H2, H3, c='gray', alpha=0.02, s=0.2, rasterized=True)

    # Wireframe
    draw_wireframe(ax_c, color='gray', alpha=0.2, lw=0.3)

    # Muscle in red
    ax_c.scatter(H1[muscle_mask], H2[muscle_mask], H3[muscle_mask],
                 c='red', alpha=0.5, s=2, rasterized=True)

    # Neurons in blue
    ax_c.scatter(H1[neuron_mask], H2[neuron_mask], H3[neuron_mask],
                 c='blue', alpha=0.5, s=2, rasterized=True)

    ax_c.set_xlabel('H1', fontsize=5, labelpad=-2)
    ax_c.set_ylabel('H2', fontsize=5, labelpad=-2)
    ax_c.set_zlabel('H3', fontsize=5, labelpad=-2)
    ax_c.tick_params(labelsize=4)

    if vi == 1:
        ax_c.set_title('C. Muscle (red) vs neuron (blue) territories',
                       fontweight='bold', fontsize=10)

# ── Panel D: Hox Expression Heatmap ───────────────────────────────────────
print("  Panel D ...")
ax_d = fig.add_subplot(gs_top[1, 1])

PAIRS_RAW = [(4, 11), (5, 0), (26, 18), (1, 36), (13, 8), (2, 9), (6, 10), (14, 19)]
TRIS = [(30, 31, 32), (33, 34, 35)]

def mean_h1(cl):
    return H1[labels == cl].mean()

def cl_name(c):
    if c == 36: return "C12/15/16/27"
    return f"C{c}"

# Build row order
all_cl_in_groups = set()
row_info = []  # list of (pair_id, row_label, cluster_label)

for pi, (a, b) in enumerate(PAIRS_RAW):
    wa, wb = mean_h1(a), mean_h1(b)
    pos, neg = (a, b) if wa >= wb else (b, a)
    row_info.append((f'P{pi+1}', f'{cl_name(pos)} (pos)', pos))
    row_info.append((f'P{pi+1}', f'{cl_name(neg)} (neg)', neg))
    all_cl_in_groups.update([a, b])

for ti, (neg_l, pos_l, ctr_l) in enumerate(TRIS):
    orig_name = {30: 'C20', 31: 'C20', 32: 'C20', 33: 'C7', 34: 'C7', 35: 'C7'}[neg_l]
    trips = [(neg_l, mean_h1(neg_l), 'neg'), (pos_l, mean_h1(pos_l), 'pos'),
             (ctr_l, mean_h1(ctr_l), 'ctr')]
    trips.sort(key=lambda x: -x[1])
    pair_id = f'P{len(PAIRS_RAW)+ti+1}'
    for lbl, mw, side in trips:
        row_info.append((pair_id, f'{orig_name}-{side}', lbl))
        all_cl_in_groups.add(lbl)

# Compute mean Hox expression per cluster (renormalized within group cells)
corr_mask_h = np.isin(labels, list(all_cl_in_groups))
corr_idx_h = np.where(corr_mask_h)[0]
scaler2 = MinMaxScaler()
X_renorm = scaler2.fit_transform(X_hox[corr_idx_h])
idx_map = {o: n for n, o in enumerate(corr_idx_h)}

def get_expr(cl):
    ci = [idx_map[i] for i in np.where(labels == cl)[0] if i in idx_map]
    return X_renorm[ci].mean(0) if ci else np.zeros(8)

# Build heatmap matrix
heatmap_data = []
row_labels = []
pair_ids = []
for pair_id, row_label, cl in row_info:
    expr = get_expr(cl)
    heatmap_data.append(expr)
    row_labels.append(row_label)
    pair_ids.append(pair_id)

heatmap_matrix = np.array(heatmap_data)

# Plot
im = ax_d.imshow(heatmap_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax_d.set_xticks(range(8))
ax_d.set_xticklabels(HOX_GENES, fontsize=8, rotation=45, ha='right')
ax_d.set_yticks(range(len(row_labels)))
ax_d.set_yticklabels(row_labels, fontsize=6)

# Annotate cells with values
for i in range(len(row_labels)):
    for j in range(8):
        val = heatmap_matrix[i, j]
        color = 'white' if val > 0.6 else 'black'
        ax_d.text(j, i, f'{val:.2f}', ha='center', va='center',
                  fontsize=4.5, color=color)

# Add pair IDs as group labels on the left
prev_pair = None
for i, pid in enumerate(pair_ids):
    if pid != prev_pair:
        # Find span of this pair
        span = sum(1 for p in pair_ids if p == pid)
        mid = i + span / 2 - 0.5
        ax_d.text(-1.5, mid, pid, ha='center', va='center',
                  fontsize=7, fontweight='bold', color='navy')
        prev_pair = pid

# Horizontal lines between pairs
prev_pair = None
for i, pid in enumerate(pair_ids):
    if pid != prev_pair and i > 0:
        ax_d.axhline(y=i - 0.5, color='black', lw=0.5)
    prev_pair = pid

plt.colorbar(im, ax=ax_d, fraction=0.03, pad=0.04, label='Expression (0-1)')
ax_d.set_title('D. Mean Hox expression by cluster pair', fontweight='bold', fontsize=11)

# ── Save ────────────────────────────────────────────────────────────────────
out_path = os.path.join(FIG_DIR, "fig3_geometry_celltypes.png")
print(f"Saving to {out_path} ...")
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Done!")
