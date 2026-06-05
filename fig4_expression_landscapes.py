"""
Figure 4: Gene Expression Landscapes on the Morphospace
"""
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
H5AD_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s_fca_biohub_body_10x.h5ad")
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

HOX_GENES = ['lab', 'pb', 'Dfd', 'Scr', 'Antp', 'Ubx', 'abd-A', 'Abd-B']
SIGNAL_GENES = ['dpp', 'Dll', 'hh', 'vg']

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("Loading h5ad ...")
adata = sc.read_h5ad(H5AD_PATH)
gn = list(adata.var_names)
hox_idx = [gn.index(g) for g in HOX_GENES]

X_raw = adata.X
if hasattr(X_raw, 'toarray'):
    X_dense = X_raw.toarray()
else:
    X_dense = np.array(X_raw)

# Hox+ mask
hox_sum = X_dense[:, hox_idx].sum(axis=1)
hox_mask = hox_sum > 0
print(f"  {hox_mask.sum()} Hox+ cells out of {len(hox_mask)}")

X_hox_raw = X_dense[hox_mask][:, hox_idx]
scaler = MinMaxScaler()
X_hox_norm = scaler.fit_transform(X_hox_raw)

Antp = X_hox_norm[:, 4]
Ubx  = X_hox_norm[:, 5]
abdA = X_hox_norm[:, 6]
AbdB = X_hox_norm[:, 7]

H1 = Antp - Ubx
H2 = abdA - AbdB
H3 = (Antp + Ubx) - (abdA + AbdB)

n_cells = hox_mask.sum()
pts_3d = np.column_stack([H1, H2, H3])

# Extract signal/TF gene expression for Hox+ cells
signal_expr = {}
for gene in SIGNAL_GENES:
    gi = gn.index(gene)
    raw = X_dense[hox_mask, gi].astype(float)
    vmax = raw.max()
    signal_expr[gene] = raw / vmax if vmax > 0 else raw

# ═══════════════════════════════════════════════════════════════════════════
# BUILD ZONOTOPE
# ═══════════════════════════════════════════════════════════════════════════
print("Building zonotope ...")
n_pg = 25
gv = np.linspace(0, 1, n_pg)
gAntp, gUbx, gabdA, gAbdB = np.meshgrid(gv, gv, gv, gv, indexing='ij')
gH1 = gAntp.ravel() - gUbx.ravel()
gH2 = gabdA.ravel() - gAbdB.ravel()
gH3 = (gAntp.ravel() + gUbx.ravel()) - (gabdA.ravel() + gAbdB.ravel())
pts_h = np.column_stack([gH1, gH2, gH3])
hull = ConvexHull(pts_h)
ps_verts = pts_h[hull.vertices]
print(f"  {len(ps_verts)} vertices")

# Build vertex dict
verts_dict = {f'V{i+1}': ps_verts[i].copy() for i in range(len(hull.vertices))}
verts_dict['C*'] = np.array([0., 0., 0.])

# Compute true edges via face normal test
ps_vert_map = {old: new for new, old in enumerate(hull.vertices)}

# Recompute hull on just vertices for clean edges
hull2 = ConvexHull(ps_verts)
hull_edges = set()
for simplex in hull2.simplices:
    for i in range(len(simplex)):
        for j in range(i + 1, len(simplex)):
            hull_edges.add(tuple(sorted([simplex[i], simplex[j]])))

# Filter to true edges (shared by faces with different normals)
def normal_key(n, tol=4):
    return tuple(np.round(n / np.linalg.norm(n), tol))

face_normals = []
for simplex in hull2.simplices:
    v0, v1, v2 = ps_verts[simplex[0]], ps_verts[simplex[1]], ps_verts[simplex[2]]
    n = np.cross(v1 - v0, v2 - v0)
    nl = np.linalg.norm(n)
    face_normals.append(n / nl if nl > 0 else n)

edge_to_faces = {}
for fi, simplex in enumerate(hull2.simplices):
    for i in range(3):
        for j in range(i + 1, 3):
            e = tuple(sorted([simplex[i], simplex[j]]))
            edge_to_faces.setdefault(e, []).append(fi)

true_edges = set()
for edge, fl in edge_to_faces.items():
    if len(fl) >= 2:
        keys = set(normal_key(face_normals[fi]) for fi in fl)
        if len(keys) > 1:
            true_edges.add(edge)
    else:
        true_edges.add(edge)

print(f"  {len(true_edges)} true edges")


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: draw zonotope wireframe
# ═══════════════════════════════════════════════════════════════════════════
def draw_wireframe(ax):
    """Draw zonotope wireframe on a 3D axes."""
    for a, b in true_edges:
        p1, p2 = ps_verts[a], ps_verts[b]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color='gray', linewidth=0.4, alpha=0.5, zorder=1)


def setup_3d_ax(ax, elev=20, azim=-60):
    """Common setup for 3D scatter axes."""
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('H1 (Antp-Ubx)', fontsize=7, labelpad=1)
    ax.set_ylabel('H2 (abdA-AbdB)', fontsize=7, labelpad=1)
    ax.set_zlabel('H3 (trunk-abd)', fontsize=7, labelpad=1)
    ax.tick_params(labelsize=5, pad=0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, alpha=0.2, linewidth=0.3)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL A: Four Cardinal Hox Gradients (2x2 grid)
# ═══════════════════════════════════════════════════════════════════════════
print("Building Panel A ...")

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('white')

# Panel A: rows 0-1, cols 0-1 (2x2 in a 4x3-ish layout)
# We'll use gridspec for fine control
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(2, 2, figure=fig,
                       left=0.02, right=0.48, top=0.95, bottom=0.52,
                       wspace=0.05, hspace=0.12)

panel_a_genes = [('Antp', Antp), ('Ubx', Ubx), ('abdA', abdA), ('AbdB', AbdB)]

for idx, (name, expr) in enumerate(panel_a_genes):
    row, col = divmod(idx, 2)
    ax = fig.add_subplot(gs[row, col], projection='3d')
    mask = expr > 0
    draw_wireframe(ax)
    sc_plot = ax.scatter(H1[mask], H2[mask], H3[mask],
                         c=expr[mask], cmap='jet', s=0.6, alpha=1.0,
                         vmin=0, vmax=1, zorder=5, rasterized=True)
    setup_3d_ax(ax, elev=20, azim=-60)
    ax.set_title(name, fontsize=11, fontweight='bold', pad=2)

# Panel A overall title
fig.text(0.25, 0.965, 'A. Four trunk Hox genes define the morphospace gradients',
         fontsize=13, fontweight='bold', ha='center', va='bottom')

# Colorbar for panel A
cax_a = fig.add_axes([0.12, 0.505, 0.22, 0.012])
sm_a = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
sm_a.set_array([])
cb_a = fig.colorbar(sm_a, cax=cax_a, orientation='horizontal')
cb_a.set_label('Normalized expression', fontsize=7)
cb_a.ax.tick_params(labelsize=6)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL B: dpp expression - 3 views
# ═══════════════════════════════════════════════════════════════════════════
print("Building Panel B ...")

gs_b = gridspec.GridSpec(1, 3, figure=fig,
                         left=0.52, right=0.98, top=0.95, bottom=0.52,
                         wspace=0.05)

dpp_expr = signal_expr['dpp']
dpp_mask = dpp_expr > 0

views_b = [(20, -60), (15, 30), (5, -120)]
view_labels = ['Standard 3/4 view', 'Antp-Ubx plane', 'Opposite side']

for vi, (elev, azim) in enumerate(views_b):
    ax = fig.add_subplot(gs_b[0, vi], projection='3d')
    draw_wireframe(ax)
    ax.scatter(H1[dpp_mask], H2[dpp_mask], H3[dpp_mask],
               c=dpp_expr[dpp_mask], cmap='jet', s=1.0, alpha=1.0,
               vmin=0, vmax=1, zorder=5, rasterized=True)
    setup_3d_ax(ax, elev=elev, azim=azim)
    ax.set_title(view_labels[vi], fontsize=8, pad=2)

fig.text(0.75, 0.965, 'B. dpp (BMP signal) expression \u2014 3 views',
         fontsize=13, fontweight='bold', ha='center', va='bottom')

# Colorbar for panel B
cax_b = fig.add_axes([0.62, 0.505, 0.22, 0.012])
sm_b = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
sm_b.set_array([])
cb_b = fig.colorbar(sm_b, cax=cax_b, orientation='horizontal')
cb_b.set_label('dpp expression', fontsize=7)
cb_b.ax.tick_params(labelsize=6)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL C: Gene Expression on 2D Plane Projection
# ═══════════════════════════════════════════════════════════════════════════
print("Building Panel C ...")

# Plane: C*_V3_V2_V1, shifted 20% toward V7
apex  = verts_dict['C*'].copy()
base  = verts_dict['V3'].copy()
left  = verts_dict['V2'].copy()
right = verts_dict['V1'].copy()
shift_target = verts_dict['V7'].copy()
shift_pct = 20

# Apply shift to base
base = base + (shift_pct / 100.0) * (shift_target - base)

# Build orthonormal basis
e1_raw = right - left
e1 = e1_raw / np.linalg.norm(e1_raw)
e2_raw = base - apex
e2_raw = e2_raw - np.dot(e2_raw, e1) * e1
e2 = e2_raw / np.linalg.norm(e2_raw)
normal = np.cross(e1, e2)
normal = normal / np.linalg.norm(normal)

# Project all cells onto this plane
offsets = pts_3d - apex[None, :]
proj_e1 = offsets @ e1
proj_e2 = offsets @ e2
proj_dist = np.abs(offsets @ normal)

PLANE_TOL = 0.3
near_mask = proj_dist < PLANE_TOL

# 2D diamond corners
def to_2d(pt):
    d = pt - apex
    return (np.dot(d, e1), np.dot(d, e2))

apex_2d = to_2d(apex)
base_2d = to_2d(base)
left_2d = to_2d(left)
right_2d = to_2d(right)

diamond_x = [apex_2d[0], right_2d[0], base_2d[0], left_2d[0], apex_2d[0]]
diamond_y = [apex_2d[1], right_2d[1], base_2d[1], left_2d[1], apex_2d[1]]

gs_c = gridspec.GridSpec(1, 2, figure=fig,
                         left=0.04, right=0.48, top=0.46, bottom=0.04,
                         wspace=0.15)

for ci, (gene, gene_label) in enumerate([('Dll', 'Dll'), ('dpp', 'dpp')]):
    ax = fig.add_subplot(gs_c[0, ci])
    expr = signal_expr[gene]
    expr_mask = expr > 0
    combined_mask = near_mask & expr_mask

    ax.plot(diamond_x, diamond_y, 'k-', linewidth=1.2, zorder=10)

    # Label diamond vertices
    for label, pt2d in [("C*", apex_2d), ("V3'", base_2d), ("V2", left_2d), ("V1", right_2d)]:
        ax.annotate(label, xy=pt2d, fontsize=7, fontweight='bold',
                    ha='center', va='bottom' if pt2d[1] > 0 else 'top',
                    xytext=(0, 5 if pt2d[1] >= base_2d[1]/2 else -8),
                    textcoords='offset points')

    sc_c = ax.scatter(proj_e1[combined_mask], proj_e2[combined_mask],
                      c=expr[combined_mask], cmap='jet', s=1.5, alpha=0.9,
                      vmin=0, vmax=1, zorder=5, rasterized=True)

    ax.set_aspect('equal')
    ax.set_xlabel('e1', fontsize=8)
    ax.set_ylabel('e2', fontsize=8)
    ax.set_title(f'{gene_label} expression', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=6)

fig.text(0.26, 0.475, 'C. Gene expression on internal plane projection',
         fontsize=13, fontweight='bold', ha='center', va='bottom')

# Colorbar for panel C
cax_c = fig.add_axes([0.15, 0.025, 0.20, 0.012])
sm_c = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
sm_c.set_array([])
cb_c = fig.colorbar(sm_c, cax=cax_c, orientation='horizontal')
cb_c.set_label('Normalized expression', fontsize=7)
cb_c.ax.tick_params(labelsize=6)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL D: Top Variable Non-Hox Gene
# ═══════════════════════════════════════════════════════════════════════════
print("Computing top variable gene ...")

# Sparse-aware variance across Hox+ cells
X_hox_all = adata.X[hox_mask]
KNOWN = set(HOX_GENES) | set(SIGNAL_GENES)
_mean = np.array(X_hox_all.mean(axis=0)).ravel()
if hasattr(X_hox_all, 'multiply'):
    _msq = np.array(X_hox_all.multiply(X_hox_all).mean(axis=0)).ravel()
else:
    _msq = np.mean(np.array(X_hox_all) ** 2, axis=0)
_var = _msq - _mean ** 2

# Rank and pick top non-Hox, non-signal gene
ranked = np.argsort(-_var)
top_gene = None
top_gi = None
for gi in ranked:
    g = gn[gi]
    if g not in KNOWN:
        top_gene = g
        top_gi = gi
        break

print(f"  Top variable gene: {top_gene} (variance={_var[top_gi]:.4f})")

# Get expression
top_raw = X_dense[hox_mask, top_gi].astype(float)
top_vmax = top_raw.max()
top_norm = top_raw / top_vmax if top_vmax > 0 else top_raw
top_mask = top_raw > 0

print(f"  {top_mask.sum()} expressing cells")

# Panel D: 3 views
gs_d = gridspec.GridSpec(1, 3, figure=fig,
                         left=0.52, right=0.98, top=0.46, bottom=0.04,
                         wspace=0.05)

for vi, (elev, azim) in enumerate(views_b):
    ax = fig.add_subplot(gs_d[0, vi], projection='3d')
    draw_wireframe(ax)
    ax.scatter(H1[top_mask], H2[top_mask], H3[top_mask],
               c=top_norm[top_mask], cmap='jet', s=1.0, alpha=1.0,
               vmin=0, vmax=1, zorder=5, rasterized=True)
    setup_3d_ax(ax, elev=elev, azim=azim)
    ax.set_title(view_labels[vi], fontsize=8, pad=2)

fig.text(0.75, 0.475, f'D. {top_gene} \u2014 top variable gene',
         fontsize=13, fontweight='bold', ha='center', va='bottom')

# Colorbar for panel D
cax_d = fig.add_axes([0.62, 0.025, 0.22, 0.012])
sm_d = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
sm_d.set_array([])
cb_d = fig.colorbar(sm_d, cax=cax_d, orientation='horizontal')
cb_d.set_label(f'{top_gene} expression', fontsize=7)
cb_d.ax.tick_params(labelsize=6)


# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
out_path = os.path.join(FIG_DIR, "fig4_expression_landscapes.png")
print(f"Saving to {out_path} ...")
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Done.")
