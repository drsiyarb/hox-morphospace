"""
Figure 1 — The Hox Morphospace: From PCA to Hox Coordinates
Static 2x2 panel figure (matplotlib, 300 DPI PNG).

Uses the same color conventions as the Hox Morphospace Explorer.
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
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import to_rgba

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
H5AD_PATH  = os.path.join(os.path.dirname(OUTPUT_DIR), "s_fca_biohub_body_10x.h5ad")
CSV_PATH   = os.path.join(os.path.dirname(OUTPUT_DIR), "flycellatlas_hox_output", "hox_morphospace_boxcentered.csv")
POLY_PATH  = os.path.join(os.path.dirname(OUTPUT_DIR), "flycellatlas_hox_output", "blender_polytope.csv")
EDGE_PATH  = os.path.join(os.path.dirname(OUTPUT_DIR), "flycellatlas_hox_output", "blender_polytope_edges.csv")
HOX_GENES  = ['lab', 'pb', 'Dfd', 'Scr', 'Antp', 'Ubx', 'abd-A', 'Abd-B']

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading CSV ...")
df = pd.read_csv(CSV_PATH)
pc1 = df['pc1'].values; pc2 = df['pc2'].values; pc3 = df['pc3'].values
U = df['U'].values; V = df['V'].values; W = df['W'].values
u_raw = df['u_coord'].values; v_raw = df['v_coord'].values; w_raw = df['w_coord'].values
labels = df['leiden_label'].values.copy()
broad  = df['broad'].values

plane_W = -((w_raw.min() + w_raw.max()) / 2)

# Splits & merge — same as explorer
for i in np.where(labels==20)[0]:
    if W[i]<-0.22: labels[i]=30
    elif W[i]>0.05: labels[i]=31
    else: labels[i]=32
for i in np.where(labels==7)[0]:
    if W[i]<-0.13: labels[i]=33
    elif W[i]>-0.10: labels[i]=34
    else: labels[i]=35
for cl in [12,15,16,27]: labels[labels==cl]=36

# Load Hox expression
print("Loading h5ad ...")
adata = sc.read_h5ad(H5AD_PATH)
gn = list(adata.var_names)
hox_idx = [gn.index(g) for g in HOX_GENES]
X_raw = adata.X
if hasattr(X_raw, 'toarray'): X_raw = X_raw.toarray()
X_raw = X_raw.astype(float)
hox_mask = X_raw[:, hox_idx].sum(1) > 0
X_hox = X_raw[hox_mask][:, hox_idx]
X_hox_norm = MinMaxScaler().fit_transform(X_hox)

Antp = X_hox_norm[:, 4]; Ubx = X_hox_norm[:, 5]
abdA = X_hox_norm[:, 6]; AbdB = X_hox_norm[:, 7]
H1 = Antp - Ubx
H2 = abdA - AbdB
H3 = (Antp + Ubx) - (abdA + AbdB)
print(f"  {len(H1)} cells loaded")

# ── Polytope ─────────────────────────────────────────────────────────────────
poly_df = pd.read_csv(POLY_PATH)
poly_verts = poly_df[['H1','H2','H3']].values          # (14,3)
poly_labels = poly_df['vertex_id'].values               # V1..V14
edge_df = pd.read_csv(EDGE_PATH)
edges = list(zip(edge_df['v_start'].values, edge_df['v_end'].values))

# ── Color scheme — same as explorer ──────────────────────────────────────────
def hex_to_rgb01(h):
    h = h.lstrip('#')
    return (int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255)

def lighten(rgb, f=0.40):
    return tuple(c + (1-c)*f for c in rgb)

def darken(rgb, f=0.15):
    return tuple(c*(1-f) for c in rgb)

def mean_h1(cl): return H1[labels==cl].mean()

PAIRS_RAW = [
    (4, 11, '#0000CD'), (5, 0, '#CC0000'), (26,18, '#006400'),
    (1, 36, '#C71585'), (13, 8, '#7B2D8E'), (2, 9, '#CC8400'),
    (6, 10, '#008B8B'), (14,19, '#D2691E'),
]
TRIS = [(30,31,32, '#FF1493'), (33,34,35, '#9ACD32')]
UNMATCHED_RGB = (0.53, 0.53, 0.53)

color_map = {}
pair_legend = []   # (label, color) for legend

for pi, (a, b, hexc) in enumerate(PAIRS_RAW):
    rgb = hex_to_rgb01(hexc)
    pos, neg = (a, b) if mean_h1(a) >= mean_h1(b) else (b, a)
    color_map[pos] = darken(rgb)
    color_map[neg] = lighten(rgb)
    clA = f"C{a}" if a != 36 else "C12/15/16/27"
    clB = f"C{b}" if b != 36 else "C12/15/16/27"
    pair_legend.append((f"P{pi+1} ({clA} & {clB})", rgb))

for ti, (neg_l, pos_l, ctr_l, hexc) in enumerate(TRIS):
    rgb = hex_to_rgb01(hexc)
    orig = {30:'C20',31:'C20',32:'C20',33:'C7',34:'C7',35:'C7'}[neg_l]
    trips = [(neg_l, mean_h1(neg_l), 'neg'), (pos_l, mean_h1(pos_l), 'pos'),
             (ctr_l, mean_h1(ctr_l), 'ctr')]
    for lbl, mw, side in trips:
        color_map[lbl] = rgb if side == 'ctr' else lighten(rgb)
    pair_legend.append((f"P{len(PAIRS_RAW)+ti+1} ({orig} 3-way)", rgb))

paired_cls = set()
for a, b, _ in PAIRS_RAW: paired_cls.update([a, b])
# remap to split labels
paired_cls.discard(20); paired_cls.discard(7)
paired_cls.update([30,31,32,33,34,35,36])
for a, b, _ in PAIRS_RAW:
    paired_cls.add(a); paired_cls.add(b)

all_labels_set = sorted(set(labels))
for cl in all_labels_set:
    if cl not in color_map:
        color_map[cl] = UNMATCHED_RGB

# Build per-cell color arrays
cell_colors = np.array([color_map[l] for l in labels])

# ── G-plane normals in H-space ───────────────────────────────────────────────
print("Computing G vs H angles ...")
H_stack  = np.column_stack([H1, H2, H3, np.ones(len(H1))])
UVW_stack = np.column_stack([U, V, W])
M_full, _, _, _ = np.linalg.lstsq(H_stack, UVW_stack, rcond=None)
M_lin = M_full[:3, :]
M_inv_T = np.linalg.inv(M_lin).T

def angle_deg(n1, n2):
    c = abs(np.dot(n1, n2) / (np.linalg.norm(n1)*np.linalg.norm(n2)))
    return np.degrees(np.arccos(np.clip(c, 0, 1)))

g_normals_h = {}
for name, uvw_n in [('W', [0,0,1]), ('U', [1,0,0]), ('V', [0,1,0])]:
    n_h = M_inv_T @ np.array(uvw_n, dtype=float)
    g_normals_h[name] = n_h / np.linalg.norm(n_h)

ang_WH1 = angle_deg(g_normals_h['W'], [1,0,0])
ang_UH2 = angle_deg(g_normals_h['U'], [0,1,0])
ang_VH3 = angle_deg(g_normals_h['V'], [0,0,1])
print(f"  W-H1={ang_WH1:.1f} deg  U-H2={ang_UH2:.1f} deg  V-H3={ang_VH3:.1f} deg")

# ── Heatmap data ─────────────────────────────────────────────────────────────
all_cl = [4,11,5,0,26,18,1,36,13,8,2,9,6,10,14,19,30,31,32,33,34,35]
corr_mask = np.isin(labels, all_cl)
corr_idx = np.where(corr_mask)[0]
X_renorm = MinMaxScaler().fit_transform(X_hox[corr_mask])
idx_map = {o: n for n, o in enumerate(corr_idx)}

def get_expr(cl):
    ci = [idx_map[i] for i in np.where(labels==cl)[0] if i in idx_map]
    return X_renorm[ci].mean(0) if ci else np.zeros(8)

hm_labels = []
hm_data = []
hm_colors = []
for pi, (a, b, hexc) in enumerate(PAIRS_RAW):
    pos, neg = (a, b) if mean_h1(a) >= mean_h1(b) else (b, a)
    for cl in [pos, neg]:
        nm = f"C{cl}" if cl != 36 else "C12+"
        hm_labels.append(nm)
        hm_data.append(get_expr(cl))
        hm_colors.append(hex_to_rgb01(hexc))
for ti, (neg_l, pos_l, ctr_l, hexc) in enumerate(TRIS):
    orig = {30:'C20',31:'C20',32:'C20',33:'C7',34:'C7',35:'C7'}[neg_l]
    trips = [(neg_l, mean_h1(neg_l), 'neg'), (pos_l, mean_h1(pos_l), 'pos'),
             (ctr_l, mean_h1(ctr_l), 'ctr')]
    trips.sort(key=lambda x: -x[1])
    for lbl, mw, side in trips:
        hm_labels.append(f"{orig[1:]}{side[0]}")
        hm_data.append(get_expr(lbl))
        hm_colors.append(hex_to_rgb01(hexc))
hm_data = np.array(hm_data)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def draw_scatter(ax, x, y, z, elev, azim):
    """Scatter colored by cluster pair, randomized order to avoid overplotting."""
    rng = np.random.RandomState(42)
    order = rng.permutation(len(x))
    ax.scatter(x[order], y[order], z[order],
               c=cell_colors[order], s=0.3, alpha=0.7,
               rasterized=True, linewidths=0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1,1,1])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0.85,0.85,0.9,0.5))
    ax.yaxis.pane.set_edgecolor((0.85,0.9,0.85,0.5))
    ax.zaxis.pane.set_edgecolor((0.9,0.85,0.85,0.5))
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=6, pad=0)
    for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
        spine.label.set_size(7)

def draw_wireframe(ax, verts, edges, color='k', lw=1.0, alpha=0.8):
    """Draw polytope wireframe."""
    segs = []
    for a, b in edges:
        segs.append([verts[a], verts[b]])
    lc = Line3DCollection(segs, colors=color, linewidths=lw, alpha=alpha)
    ax.add_collection3d(lc)

def draw_vertex_labels(ax, verts, labels, fontsize=5):
    """Draw vertex markers and labels."""
    ax.scatter(verts[:,0], verts[:,1], verts[:,2],
               c='black', s=12, marker='D', zorder=10, depthshade=False)
    for i, lbl in enumerate(labels):
        ax.text(verts[i,0], verts[i,1], verts[i,2]+0.08,
                lbl, fontsize=fontsize, ha='center', va='bottom',
                fontweight='bold', color='black', zorder=11)

def draw_plane_disc(ax, centroid, normal, color, alpha=0.3, radius=None):
    """Draw a semi-transparent plane disc in 3D."""
    if abs(normal[0]) < 0.9:
        tmp = np.array([1.,0.,0.])
    else:
        tmp = np.array([0.,1.,0.])
    e1 = np.cross(normal, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1); e2 /= np.linalg.norm(e2)

    if radius is None:
        radius = 1.5

    u_g = np.linspace(-radius, radius, 30)
    v_g = np.linspace(-radius, radius, 30)
    UU, VV = np.meshgrid(u_g, v_g)
    # Clip to circle
    mask = UU**2 + VV**2 <= radius**2
    XX = centroid[0] + UU*e1[0] + VV*e2[0]
    YY = centroid[1] + UU*e1[1] + VV*e2[1]
    ZZ = centroid[2] + UU*e1[2] + VV*e2[2]
    XX[~mask] = np.nan; YY[~mask] = np.nan; ZZ[~mask] = np.nan
    ax.plot_surface(XX, YY, ZZ, color=color, alpha=alpha, shade=False,
                    linewidth=0, antialiased=True, zorder=1)


# ══════════════════════════════════════════════════════════════════════════════
# BUILD FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("Building figure ...")

fig = plt.figure(figsize=(16, 13), facecolor='white', dpi=150)

ELEV, AZIM = 22, -55   # consistent camera for A, B, C

# ── Panel A: PCA ─────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(2, 2, 1, projection='3d')
draw_scatter(ax_a, pc1, pc2, pc3, ELEV, AZIM)
ax_a.set_xlabel('PC1 (29.7%)')
ax_a.set_ylabel('PC2 (20.7%)')
ax_a.set_zlabel('PC3 (19.0%)')

# ── Panel B: Hox Morphospace + zonotope ──────────────────────────────────────
ax_b = fig.add_subplot(2, 2, 2, projection='3d')
draw_scatter(ax_b, H1, H2, H3, ELEV, AZIM)
draw_wireframe(ax_b, poly_verts, edges, color='black', lw=1.2, alpha=0.85)
draw_vertex_labels(ax_b, poly_verts, poly_labels, fontsize=5)
ax_b.set_xlabel('H1 (Antp-Ubx)')
ax_b.set_ylabel('H2 (abdA-AbdB)')
ax_b.set_zlabel('H3 (thorax-abd)')

# ── Panel C: H-planes vs G-planes ───────────────────────────────────────────
ax_c = fig.add_subplot(2, 2, 3, projection='3d')
# Lighter cells
rng = np.random.RandomState(42)
order = rng.permutation(len(H1))
faded = np.clip(cell_colors[order] * 0.6 + 0.4, 0, 1)
ax_c.scatter(H1[order], H2[order], H3[order],
             c=faded, s=0.2, alpha=0.25, rasterized=True, linewidths=0)
draw_wireframe(ax_c, poly_verts, edges, color=(0.3,0.3,0.3,0.35), lw=0.8)

# H-planes (solid, vibrant)
origin = np.array([0.,0.,0.])
draw_plane_disc(ax_c, origin, np.array([1.,0.,0.]), color=(0.12,0.39,0.86), alpha=0.35, radius=1.8)  # blue  H1=0
draw_plane_disc(ax_c, origin, np.array([0.,1.,0.]), color=(0.86,0.20,0.20), alpha=0.35, radius=1.8)  # red   H2=0
draw_plane_disc(ax_c, origin, np.array([0.,0.,1.]), color=(0.86,0.71,0.0),  alpha=0.35, radius=1.8)  # gold  H3=0

# G-planes (dashed outline only — to distinguish from H)
# Compute centroids of G-planes in H-space
for g_name, uvw_n, uvw_cv in [('W', [0,0,1], plane_W), ('U', [1,0,0], 0.0), ('V', [0,1,0], 0.0)]:
    n_h = g_normals_h[g_name]
    uvw_vals = UVW_stack @ np.array(uvw_n, dtype=float)
    near = np.abs(uvw_vals - uvw_cv) < np.percentile(np.abs(uvw_vals - uvw_cv), 10)
    cent = np.array([H1[near].mean(), H2[near].mean(), H3[near].mean()])
    g_color = {'W': (0.12,0.39,0.86,0.12), 'U': (0.86,0.20,0.20,0.12), 'V': (0.86,0.71,0.0,0.12)}[g_name]
    draw_plane_disc(ax_c, cent, n_h, color=g_color[:3], alpha=0.15, radius=1.8)

ax_c.view_init(elev=ELEV, azim=AZIM)
ax_c.set_box_aspect([1,1,1])
ax_c.xaxis.pane.fill = False; ax_c.yaxis.pane.fill = False; ax_c.zaxis.pane.fill = False
ax_c.xaxis.pane.set_edgecolor((0.85,0.85,0.9,0.5))
ax_c.yaxis.pane.set_edgecolor((0.85,0.9,0.85,0.5))
ax_c.zaxis.pane.set_edgecolor((0.9,0.85,0.85,0.5))
ax_c.grid(True, alpha=0.2)
ax_c.tick_params(labelsize=6, pad=0)
ax_c.set_xlabel('H1', fontsize=7); ax_c.set_ylabel('H2', fontsize=7); ax_c.set_zlabel('H3', fontsize=7)

# ── Panel D: Heatmap ────────────────────────────────────────────────────────
ax_d = fig.add_subplot(2, 2, 4)
im = ax_d.imshow(hm_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax_d.set_xticks(range(8))
ax_d.set_xticklabels(HOX_GENES, fontsize=7, rotation=45, ha='right')
ax_d.set_yticks(range(len(hm_labels)))
ax_d.set_yticklabels(hm_labels, fontsize=6)
# Color y-tick labels by pair
for i, (lbl, col) in enumerate(zip(ax_d.get_yticklabels(), hm_colors)):
    lbl.set_color(col)
    lbl.set_fontweight('bold')
# Add values
for i in range(hm_data.shape[0]):
    for j in range(hm_data.shape[1]):
        v = hm_data[i, j]
        tc = 'white' if v > 0.55 else 'black'
        ax_d.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5, color=tc)
cbar = fig.colorbar(im, ax=ax_d, fraction=0.03, pad=0.02)
cbar.set_label('Norm. expression', fontsize=7)
cbar.ax.tick_params(labelsize=6)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL LABELS & ANNOTATION ARROWS
# ══════════════════════════════════════════════════════════════════════════════

# Create invisible overlay axes spanning the full figure for annotations
ax_ov = fig.add_axes([0, 0, 1, 1], facecolor='none')
ax_ov.set_xlim(0, 1); ax_ov.set_ylim(0, 1)
ax_ov.axis('off')

# Panel letter labels (bold, top-left of each panel)
for ax, letter in [(ax_a, 'A'), (ax_b, 'B'), (ax_c, 'C'), (ax_d, 'D')]:
    bbox = ax.get_position()
    ax_ov.text(bbox.x0 - 0.01, bbox.y1 + 0.01, letter,
               fontsize=18, fontweight='bold', va='bottom', ha='right')

# ── Annotation arrows on overlay axes ────────────────────────────────────────

# Panel A
bbox_a = ax_a.get_position()
ax_ov.annotate('3D structure\n"looks like a fly"',
    xy=(bbox_a.x0 + bbox_a.width*0.65, bbox_a.y0 + bbox_a.height*0.72),
    xytext=(bbox_a.x1 + 0.01, bbox_a.y1 - 0.02),
    fontsize=7, color='#444', fontstyle='italic', ha='left', va='top',
    arrowprops=dict(arrowstyle='->', color='#666', lw=0.8,
                    connectionstyle='arc3,rad=-0.15'))

ax_ov.annotate('PCA axes\n(not biological)',
    xy=(bbox_a.x0 + bbox_a.width*0.25, bbox_a.y0 + bbox_a.height*0.18),
    xytext=(bbox_a.x0 - 0.01, bbox_a.y0 + 0.02),
    fontsize=7, color='#444', fontstyle='italic', ha='right', va='bottom',
    arrowprops=dict(arrowstyle='->', color='#666', lw=0.8,
                    connectionstyle='arc3,rad=0.15'))

# Panel B
bbox_b = ax_b.get_position()
ax_ov.annotate('Zonotope wireframe\n(14 vertices, 24 edges)',
    xy=(bbox_b.x0 + bbox_b.width*0.72, bbox_b.y0 + bbox_b.height*0.80),
    xytext=(bbox_b.x1 + 0.01, bbox_b.y1 - 0.02),
    fontsize=7, color='#444', fontstyle='italic', ha='left', va='top',
    arrowprops=dict(arrowstyle='->', color='#666', lw=0.8,
                    connectionstyle='arc3,rad=-0.15'))

ax_ov.annotate('Cells in Hox\ncoordinates',
    xy=(bbox_b.x0 + bbox_b.width*0.40, bbox_b.y0 + bbox_b.height*0.45),
    xytext=(bbox_b.x0 + 0.01, bbox_b.y0 + 0.02),
    fontsize=7, color='#444', fontstyle='italic', ha='left', va='bottom',
    arrowprops=dict(arrowstyle='->', color='#666', lw=0.8,
                    connectionstyle='arc3,rad=0.15'))

# Panel C
bbox_c = ax_c.get_position()
ax_ov.annotate('H-planes (solid)\nexpression-defined',
    xy=(bbox_c.x0 + bbox_c.width*0.55, bbox_c.y0 + bbox_c.height*0.60),
    xytext=(bbox_c.x0 - 0.01, bbox_c.y1 - 0.02),
    fontsize=7, color=(0.12,0.39,0.86), fontweight='bold', ha='right', va='top',
    arrowprops=dict(arrowstyle='->', color=(0.12,0.39,0.86), lw=1.0,
                    connectionstyle='arc3,rad=0.2'))

ax_ov.annotate('G-planes (faint)\ngeometry-derived',
    xy=(bbox_c.x0 + bbox_c.width*0.48, bbox_c.y0 + bbox_c.height*0.52),
    xytext=(bbox_c.x0 - 0.01, bbox_c.y0 + bbox_c.height*0.35),
    fontsize=7, color='#888', fontweight='bold', ha='right', va='top',
    arrowprops=dict(arrowstyle='->', color='#888', lw=1.0,
                    connectionstyle='arc3,rad=0.2'))

ax_ov.annotate(f'Aligned within\n{ang_WH1:.1f}-{ang_VH3:.1f} deg',
    xy=(bbox_c.x0 + bbox_c.width*0.52, bbox_c.y0 + bbox_c.height*0.56),
    xytext=(bbox_c.x1 + 0.01, bbox_c.y0 + 0.03),
    fontsize=8, color='#222', fontweight='bold', ha='left', va='bottom',
    arrowprops=dict(arrowstyle='->', color='#222', lw=1.2,
                    connectionstyle='arc3,rad=-0.2'),
    bbox=dict(boxstyle='round,pad=0.3', fc='#ffffcc', ec='#cccc00', alpha=0.9))

# Panel D
bbox_d = ax_d.get_position()
ax_ov.annotate('Mirror-image pairs\n(Antp/Ubx swap)',
    xy=(bbox_d.x0 + bbox_d.width*0.52, bbox_d.y1 - 0.01),
    xytext=(bbox_d.x1 + 0.01, bbox_d.y1 - 0.01),
    fontsize=7, color='#444', fontstyle='italic', ha='left', va='top',
    arrowprops=dict(arrowstyle='->', color='#666', lw=0.8,
                    connectionstyle='arc3,rad=-0.15'))

ax_ov.annotate('abd-A / Abd-B\ngradient',
    xy=(bbox_d.x0 + bbox_d.width*0.78, bbox_d.y0 + bbox_d.height*0.5),
    xytext=(bbox_d.x1 + 0.01, bbox_d.y0 + bbox_d.height*0.4),
    fontsize=7, color='#444', fontstyle='italic', ha='left', va='top',
    arrowprops=dict(arrowstyle='->', color='#666', lw=0.8,
                    connectionstyle='arc3,rad=-0.1'))

# ── Title ────────────────────────────────────────────────────────────────────
fig.suptitle('Figure 1: The Hox Morphospace - From PCA to Hox Coordinates',
             fontsize=14, fontweight='bold', y=0.98)
fig.text(0.5, 0.955,
         f'38,227 Hox-expressing cells  |  8 Hox genes  |  '
         f'G-H alignment: {ang_WH1:.1f}, {ang_UH2:.1f}, {ang_VH3:.1f} deg',
         ha='center', fontsize=9, color='#666')

# ── Cluster pair legend (compact, below title) ──────────────────────────────
legend_handles = []
for label, rgb in pair_legend:
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=rgb, markersize=5, label=label))
legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=UNMATCHED_RGB, markersize=5, label='Unmatched'))
fig.legend(handles=legend_handles, loc='upper center',
           bbox_to_anchor=(0.5, 0.945), ncol=5, fontsize=7,
           frameon=True, fancybox=True, framealpha=0.9,
           edgecolor='#ddd', handletextpad=0.3, columnspacing=1.0)

plt.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.04,
                    wspace=0.12, hspace=0.15)

# ── Save ─────────────────────────────────────────────────────────────────────
out_png = os.path.join(OUTPUT_DIR, 'figures', 'fig1_pca_to_planes.png')
os.makedirs(os.path.dirname(out_png), exist_ok=True)
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {out_png}")

# Also save a quick-view copy next to the script
out_quick = os.path.join(OUTPUT_DIR, 'fig1_preview.png')
fig.savefig(out_quick, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Preview: {out_quick}")
