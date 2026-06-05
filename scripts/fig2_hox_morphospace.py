"""
Figure 2: The Hox Morphospace and its Zonotope Envelope
4 panels (A-D), publication quality, 300 DPI
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
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, art3d
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
H5AD_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s_fca_biohub_body_10x.h5ad")
HOX_GENES = ['lab', 'pb', 'Dfd', 'Scr', 'Antp', 'Ubx', 'abd-A', 'Abd-B']

os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

# ── Load leiden labels and do cluster splits ────────────────────────────────
print("Loading CSV ...")
df = pd.read_csv(os.path.join(OUTPUT_DIR, "hox_morphospace_boxcentered.csv"))
labels = df['leiden_label'].values.copy()
W_coord = df['W'].values

# Splits & merge (same as main script)
for i in np.where(labels == 20)[0]:
    if W_coord[i] < -0.22: labels[i] = 30
    elif W_coord[i] > 0.05: labels[i] = 31
    else: labels[i] = 32
for i in np.where(labels == 7)[0]:
    if W_coord[i] < -0.13: labels[i] = 33
    elif W_coord[i] > -0.10: labels[i] = 34
    else: labels[i] = 35
for cl in [12, 15, 16, 27]:
    labels[labels == cl] = 36

# ── Load Hox expression ────────────────────────────────────────────────────
print("Loading h5ad ...")
adata = sc.read_h5ad(H5AD_PATH)
gn = list(adata.var_names)
hox_idx = [gn.index(g) for g in HOX_GENES]
X_raw = adata.X
X_raw = X_raw.toarray() if hasattr(X_raw, 'toarray') else X_raw
hox_mask = X_raw[:, hox_idx].sum(1) > 0
X_hox = X_raw[hox_mask][:, hox_idx]
scaler = MinMaxScaler()
X_hox_norm = scaler.fit_transform(X_hox)

Antp = X_hox_norm[:, 4]
Ubx  = X_hox_norm[:, 5]
abdA = X_hox_norm[:, 6]
AbdB = X_hox_norm[:, 7]

H1 = Antp - Ubx
H2 = abdA - AbdB
H3 = (Antp + Ubx) - (abdA + AbdB)

n_cells = len(H1)
print(f"  {n_cells} Hox+ cells loaded")

# ── Color scheme ───────────────────────────────────────────────────────────
def hex_to_rgb01(h):
    h = h.lstrip('#')
    return (int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255)

def lighten(h, factor=0.40):
    r, g, b = hex_to_rgb01(h)
    return (r + (1-r)*factor, g + (1-g)*factor, b + (1-b)*factor)

def darken(h, factor=0.15):
    r, g, b = hex_to_rgb01(h)
    return (r*(1-factor), g*(1-factor), b*(1-factor))

def mean_h1(cl):
    return H1[labels == cl].mean()

PAIRS_RAW = [
    (4, 11, '#0000CD'), (5, 0, '#CC0000'), (26, 18, '#006400'),
    (1, 36, '#C71585'), (13, 8, '#7B2D8E'), (2, 9, '#CC8400'),
    (6, 10, '#008B8B'), (14, 19, '#D2691E'),
]
TRIS = [(30, 31, 32, '#FF1493'), (33, 34, 35, '#9ACD32')]
UNMATCHED_COLOR = (0.53, 0.53, 0.53)

color_map = {}
for a, b, color in PAIRS_RAW:
    pos, neg = (a, b) if mean_h1(a) >= mean_h1(b) else (b, a)
    color_map[pos] = darken(color, 0.15)
    color_map[neg] = lighten(color, 0.40)

for neg_l, pos_l, ctr_l, color in TRIS:
    trips = [(neg_l, mean_h1(neg_l)), (pos_l, mean_h1(pos_l)), (ctr_l, mean_h1(ctr_l))]
    trips.sort(key=lambda x: -x[1])
    # darkest to lightest by H1
    color_map[trips[0][0]] = darken(color, 0.15)
    color_map[trips[1][0]] = hex_to_rgb01(color)
    color_map[trips[2][0]] = lighten(color, 0.40)

paired_cls = set()
for a, b, _ in PAIRS_RAW:
    paired_cls.update([a, b])
for a, b, c, _ in TRIS:
    paired_cls.update([a, b, c])
for cl in sorted(set(labels)):
    if cl not in color_map:
        color_map[cl] = UNMATCHED_COLOR

# Build per-cell color array
cell_colors = np.array([color_map[l] for l in labels])

# ── Zonotope ───────────────────────────────────────────────────────────────
print("Building zonotope ...")
grid = np.linspace(0, 1, 25)
g_antp, g_ubx, g_abda, g_abdb = np.meshgrid(grid, grid, grid, grid, indexing='ij')
g_antp = g_antp.ravel(); g_ubx = g_ubx.ravel()
g_abda = g_abda.ravel(); g_abdb = g_abdb.ravel()

z_h1 = g_antp - g_ubx
z_h2 = g_abda - g_abdb
z_h3 = (g_antp + g_ubx) - (g_abda + g_abdb)

pts_3d = np.column_stack([z_h1, z_h2, z_h3])
hull = ConvexHull(pts_3d)
vertices = pts_3d[hull.vertices]

print(f"  Zonotope has {len(vertices)} vertices")

# Get edges from hull simplices
edges = set()
for simplex in hull.simplices:
    for i in range(len(simplex)):
        for j in range(i+1, len(simplex)):
            e = tuple(sorted([simplex[i], simplex[j]]))
            edges.add(e)

# Filter to actual hull edges (shared by exactly 2 faces or on boundary)
hull_pts = pts_3d[hull.vertices]
hull_3d = ConvexHull(hull_pts)

hull_edges = set()
for simplex in hull_3d.simplices:
    for i in range(len(simplex)):
        for j in range(i+1, len(simplex)):
            e = tuple(sorted([simplex[i], simplex[j]]))
            hull_edges.add(e)

# ── Vertex labels ──────────────────────────────────────────────────────────
# The zonotope vertices correspond to pure gene states (corners of [0,1]^4)
# Map each vertex to nearest corner and label it
corners_4d = []
corner_labels = []
for a in [0, 1]:
    for u in [0, 1]:
        for d in [0, 1]:
            for b in [0, 1]:
                corners_4d.append((a, u, d, b))
                parts = []
                if a: parts.append('Antp')
                if u: parts.append('Ubx')
                if d: parts.append('abdA')
                if b: parts.append('AbdB')
                if not parts:
                    corner_labels.append('none')
                else:
                    corner_labels.append('+'.join(parts))

corners_h = []
for a, u, d, b in corners_4d:
    corners_h.append((a - u, d - b, (a + u) - (d + b)))
corners_h = np.array(corners_h, dtype=float)

# Find unique corners in H-space (some map to same point)
unique_verts = {}
for i, (lbl, h) in enumerate(zip(corner_labels, corners_h)):
    key = tuple(np.round(h, 6))
    if key not in unique_verts:
        unique_verts[key] = []
    unique_verts[key].append(lbl)

vertex_labels = {}
for key, lbls in unique_verts.items():
    vertex_labels[key] = ' / '.join(lbls)

# ── Figure ─────────────────────────────────────────────────────────────────
print("Generating figure ...")
fig = plt.figure(figsize=(16, 14))

# Use gridspec for layout: top row = A, B; bottom row = C, D
gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25,
                       left=0.05, right=0.97, top=0.95, bottom=0.05)

# ═══════════════════════════════════════════════════════════════════════════
# Panel A: Axis Definition Diagram
# ═══════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_xlim(-0.5, 6.5)
ax_a.set_ylim(-1.0, 5.5)
ax_a.set_aspect('equal')
ax_a.axis('off')
ax_a.set_title('A. Hox coordinate system definition', fontsize=14, fontweight='bold',
               loc='left', pad=10)

# Gene boxes
gene_box_style = dict(boxstyle='round,pad=0.4', facecolor='#E8E8F0', edgecolor='#333333', linewidth=1.5)
axis_colors = {'H1': '#CC0000', 'H2': '#0066CC', 'H3': '#228B22'}

# Antp and Ubx on top row
ax_a.text(1.5, 4.5, 'Antp', fontsize=13, fontweight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFD0D0', edgecolor='#CC0000', linewidth=1.5))
ax_a.text(4.5, 4.5, 'Ubx', fontsize=13, fontweight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFD0D0', edgecolor='#CC0000', linewidth=1.5))

# abdA and AbdB on bottom row
ax_a.text(1.5, 2.0, 'abdA', fontsize=13, fontweight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#D0D0FF', edgecolor='#0066CC', linewidth=1.5))
ax_a.text(4.5, 2.0, 'AbdB', fontsize=13, fontweight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#D0D0FF', edgecolor='#0066CC', linewidth=1.5))

# H1 arrow: Antp - Ubx (horizontal)
ax_a.annotate('', xy=(4.0, 4.5), xytext=(2.1, 4.5),
             arrowprops=dict(arrowstyle='->', color=axis_colors['H1'], lw=2.5))
ax_a.text(3.0, 4.9, 'H1 = Antp $-$ Ubx', fontsize=11, ha='center', va='bottom',
         color=axis_colors['H1'], fontweight='bold')
ax_a.text(2.5, 4.15, '$-$', fontsize=14, ha='center', va='center', color=axis_colors['H1'],
         fontweight='bold')

# H2 arrow: abdA - AbdB (horizontal on lower row)
ax_a.annotate('', xy=(4.0, 2.0), xytext=(2.1, 2.0),
             arrowprops=dict(arrowstyle='->', color=axis_colors['H2'], lw=2.5))
ax_a.text(3.0, 2.4, 'H2 = abdA $-$ AbdB', fontsize=11, ha='center', va='bottom',
         color=axis_colors['H2'], fontweight='bold')
ax_a.text(2.5, 1.65, '$-$', fontsize=14, ha='center', va='center', color=axis_colors['H2'],
         fontweight='bold')

# H3 arrow: vertical, connecting top sum to bottom sum
ax_a.annotate('', xy=(3.0, 2.7), xytext=(3.0, 3.9),
             arrowprops=dict(arrowstyle='->', color=axis_colors['H3'], lw=2.5))

# Bracket for top pair
ax_a.plot([1.5, 1.5, 4.5, 4.5], [3.95, 3.8, 3.8, 3.95], color=axis_colors['H3'],
         lw=1.5, solid_capstyle='round')
ax_a.plot([3.0, 3.0], [3.8, 3.9], color=axis_colors['H3'], lw=1.5)

# Bracket for bottom pair
ax_a.plot([1.5, 1.5, 4.5, 4.5], [2.55, 2.7, 2.7, 2.55], color=axis_colors['H3'],
         lw=1.5, solid_capstyle='round')
ax_a.plot([3.0, 3.0], [2.7, 2.7], color=axis_colors['H3'], lw=1.5)

ax_a.text(3.55, 3.3, 'H3 = (Antp+Ubx) $-$ (abdA+AbdB)', fontsize=10, ha='left', va='center',
         color=axis_colors['H3'], fontweight='bold')

# Summary box
summary_text = ('Thorax vs. Abdomen\n'
                'H1: anterior thorax identity\n'
                'H2: anterior abdomen identity\n'
                'H3: thorax-abdomen axis')
ax_a.text(3.0, 0.5, summary_text, fontsize=9, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5DC', edgecolor='#999999',
                  linewidth=1), linespacing=1.6)

# ═══════════════════════════════════════════════════════════════════════════
# Panel B: The Morphospace — Main 3D View
# ═══════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[0, 1], projection='3d')
ax_b.set_title(f'B. {n_cells:,} cells in Hox morphospace', fontsize=14, fontweight='bold',
               loc='left', pad=10)

# Scatter cells
ax_b.scatter(H1, H2, H3, c=cell_colors, s=1, alpha=0.5, rasterized=True)

# Zonotope wireframe
for e in hull_edges:
    p1, p2 = hull_pts[e[0]], hull_pts[e[1]]
    ax_b.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color='black', linewidth=1.5, alpha=0.8)

ax_b.set_xlabel('H1 (Antp$-$Ubx)', fontsize=9, labelpad=5)
ax_b.set_ylabel('H2 (abdA$-$AbdB)', fontsize=9, labelpad=5)
ax_b.set_zlabel('H3 (thorax$-$abdomen)', fontsize=9, labelpad=5)
ax_b.view_init(elev=20, azim=-60)
ax_b.tick_params(labelsize=7)

# ═══════════════════════════════════════════════════════════════════════════
# Panel C: Zonotope with Vertex Labels
# ═══════════════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[1, 0], projection='3d')
ax_c.set_title(f'C. Zonotope: {len(hull_pts)}-vertex possibility space', fontsize=14,
               fontweight='bold', loc='left', pad=10)

# Wireframe
for e in hull_edges:
    p1, p2 = hull_pts[e[0]], hull_pts[e[1]]
    ax_c.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color='black', linewidth=1.5, alpha=0.6)

# Vertices as diamonds
ax_c.scatter(hull_pts[:, 0], hull_pts[:, 1], hull_pts[:, 2],
            marker='D', s=50, c='#333333', edgecolors='black', zorder=10)

# Label vertices
for vi in range(len(hull_pts)):
    pt = hull_pts[vi]
    key = tuple(np.round(pt, 1))
    # Find closest vertex label
    best_lbl = ''
    best_dist = 1e9
    for vk, vl in vertex_labels.items():
        d = sum((a - b)**2 for a, b in zip(key, vk))
        if d < best_dist:
            best_dist = d
            best_lbl = vl
    if best_lbl and best_dist < 0.5:
        # Shorten label
        short = best_lbl.split(' / ')[0]
        if len(short) > 20:
            short = short.replace('+', '+\n')
        ax_c.text(pt[0], pt[1], pt[2] + 0.08, short, fontsize=6,
                 ha='center', va='bottom', color='#333333',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='#CCCCCC', alpha=0.85, linewidth=0.5))

ax_c.set_xlabel('H1', fontsize=9, labelpad=5)
ax_c.set_ylabel('H2', fontsize=9, labelpad=5)
ax_c.set_zlabel('H3', fontsize=9, labelpad=5)
ax_c.view_init(elev=20, azim=-60)
ax_c.tick_params(labelsize=7)

# ═══════════════════════════════════════════════════════════════════════════
# Panel D: Three Orthogonal Projections
# ═══════════════════════════════════════════════════════════════════════════
gs_d = gs[1, 1].subgridspec(1, 3, wspace=0.35)

proj_configs = [
    ('H1', 'H2', H1, H2, 0, 1, 'H1 vs H2\n(top-down)'),
    ('H1', 'H3', H1, H3, 0, 2, 'H1 vs H3\n(front)'),
    ('H2', 'H3', H2, H3, 1, 2, 'H2 vs H3\n(side)'),
]

# Add panel D title
fig.text(0.525, 0.47, 'D. Three orthogonal projections', fontsize=14, fontweight='bold',
        ha='left', va='bottom')

for idx, (xlabel, ylabel, xdata, ydata, ci, cj, subtitle) in enumerate(proj_configs):
    ax_d = fig.add_subplot(gs_d[0, idx])

    # Scatter cells
    ax_d.scatter(xdata, ydata, c=cell_colors, s=0.3, alpha=0.15, rasterized=True)

    # Zonotope boundary (2D projection)
    proj_pts = hull_pts[:, [ci, cj]]
    try:
        hull_2d = ConvexHull(proj_pts)
        boundary = np.append(hull_2d.vertices, hull_2d.vertices[0])
        ax_d.plot(proj_pts[boundary, 0], proj_pts[boundary, 1],
                 color='black', linewidth=1.5, zorder=5)
    except Exception:
        pass

    ax_d.set_xlabel(xlabel, fontsize=8)
    ax_d.set_ylabel(ylabel, fontsize=8)
    ax_d.set_title(subtitle, fontsize=9, pad=3)
    ax_d.tick_params(labelsize=6)
    ax_d.set_aspect('equal')

# ── Save ───────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "figures", "fig2_hox_morphospace.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out_path}")
print("Done.")
