import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

PALETTE = [
    '#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
    '#42d4f4','#f032e6','#bfef45','#469990','#9a6324',
    '#800000','#aaffc3','#808000','#000075','#a9a9a9',
    '#ffe119','#008080','#dcbeff','#fabed4','#ffd8b1',
    '#e6beff','#fffac8','#4169e1','#ff6347','#2e8b57',
    '#da70d6','#cd853f','#40e0d0','#ff1493','#00ced1',
    '#ff8c00','#8b0000','#006400','#00008b','#8b008b',
    '#b8860b','#556b2f','#ff4500','#2f4f4f','#9932cc',
]
PLANE_CLUSTERS = {2, 6, 7, 9, 10}

# ── Load plane-relative coordinates ──────────────────────────────────────────
print("Loading morphospace CSV ...")
df = pd.read_csv(f"{OUTPUT_DIR}/hox_morphospace_coordinates.csv")
print(f"  {len(df)} cells")

u_raw = df['u_coord'].values
v_raw = df['v_coord'].values
w_raw = df['w_coord'].values

# ── Bounding box ─────────────────────────────────────────────────────────────
u_min, u_max = u_raw.min(), u_raw.max()
v_min, v_max = v_raw.min(), v_raw.max()
w_min, w_max = w_raw.min(), w_raw.max()

# ── Box center = new origin ──────────────────────────────────────────────────
u_center = (u_min + u_max) / 2
v_center = (v_min + v_max) / 2
w_center = (w_min + w_max) / 2

print(f"\n  Bounding box (plane-relative):")
print(f"    u: [{u_min:.4f}, {u_max:.4f}]  center: {u_center:.4f}")
print(f"    v: [{v_min:.4f}, {v_max:.4f}]  center: {v_center:.4f}")
print(f"    w: [{w_min:.4f}, {w_max:.4f}]  center: {w_center:.4f}")

# The original plane sat at w=0 in plane-relative coords.
# In box-centered coords, the plane is at w = -w_center
plane_w_in_box = -w_center
print(f"\n  Fitted plane sits at w = {plane_w_in_box:.4f} in box-centered coords")
print(f"  (offset from box center by {abs(plane_w_in_box):.4f})")

# ── Recenter to box center ───────────────────────────────────────────────────
U = u_raw - u_center
V = v_raw - v_center
W = w_raw - w_center

# box half-extents (for drawing the box)
u_half = (u_max - u_min) / 2
v_half = (v_max - v_min) / 2
w_half = (w_max - w_min) / 2

print(f"\n  Box-centered coordinate ranges:")
print(f"    U: [{U.min():.4f}, {U.max():.4f}]  (half-extent: {u_half:.4f})")
print(f"    V: [{V.min():.4f}, {V.max():.4f}]  (half-extent: {v_half:.4f})")
print(f"    W: [{W.min():.4f}, {W.max():.4f}]  (half-extent: {w_half:.4f})")

labels = df['leiden_label'].values
annots = df['annotation'].values
broad  = df['broad'].values
sex    = df['sex'].values
n_clusters = labels.max() + 1

def dominant(idx):
    return pd.Series(broad[idx]).value_counts().index[0]

# ── Save box-centered coordinates ────────────────────────────────────────────
df['U'] = U
df['V'] = V
df['W'] = W
csv_out = f"{OUTPUT_DIR}/hox_morphospace_boxcentered.csv"
df.to_csv(csv_out, index=False)
print(f"\n  Box-centered CSV saved: {csv_out}")

# =====================================================================
# FIGURE — 3D Hox Morphospace (box-centered)
# =====================================================================
print("\nBuilding 3D Hox Morphospace ...")
traces = []

# ── Bounding box edges ───────────────────────────────────────────────────────
# 8 corners of the box
box_corners = np.array([
    [-u_half, -v_half, -w_half],
    [+u_half, -v_half, -w_half],
    [+u_half, +v_half, -w_half],
    [-u_half, +v_half, -w_half],
    [-u_half, -v_half, +w_half],
    [+u_half, -v_half, +w_half],
    [+u_half, +v_half, +w_half],
    [-u_half, +v_half, +w_half],
])
# 12 edges
edges = [
    (0,1),(1,2),(2,3),(3,0),  # bottom face
    (4,5),(5,6),(6,7),(7,4),  # top face
    (0,4),(1,5),(2,6),(3,7),  # verticals
]
for i, (a, b) in enumerate(edges):
    traces.append(go.Scatter3d(
        x=[box_corners[a,0], box_corners[b,0]],
        y=[box_corners[a,1], box_corners[b,1]],
        z=[box_corners[a,2], box_corners[b,2]],
        mode='lines', line=dict(color='rgba(60,60,80,0.6)', width=2),
        showlegend=(i==0), name='Bounding box', hoverinfo='skip'))

# ── Fitted plane (now at W = plane_w_in_box) ────────────────────────────────
plane_quad = np.array([
    [-u_half, -v_half, plane_w_in_box],
    [+u_half, -v_half, plane_w_in_box],
    [+u_half, +v_half, plane_w_in_box],
    [-u_half, +v_half, plane_w_in_box],
])
traces.append(go.Mesh3d(
    x=plane_quad[:,0], y=plane_quad[:,1], z=plane_quad[:,2],
    i=[0,0], j=[1,2], k=[2,3],
    color='rgba(100,100,255,0.22)', flatshading=True,
    name=f'Fitted plane (W={plane_w_in_box:.3f})', showlegend=True, hoverinfo='skip'))
# plane border
pbx = list(plane_quad[:,0]) + [plane_quad[0,0]]
pby = list(plane_quad[:,1]) + [plane_quad[0,1]]
pbz = list(plane_quad[:,2]) + [plane_quad[0,2]]
traces.append(go.Scatter3d(x=pbx, y=pby, z=pbz, mode='lines',
    line=dict(color='navy', width=2.5), name='Plane border', hoverinfo='skip'))

# ── W=0 plane (box center) — subtle reference ───────────────────────────────
center_quad = np.array([
    [-u_half, -v_half, 0],
    [+u_half, -v_half, 0],
    [+u_half, +v_half, 0],
    [-u_half, +v_half, 0],
])
traces.append(go.Mesh3d(
    x=center_quad[:,0], y=center_quad[:,1], z=center_quad[:,2],
    i=[0,0], j=[1,2], k=[2,3],
    color='rgba(200,200,200,0.10)', flatshading=True,
    name='Box center (W=0)', showlegend=True, hoverinfo='skip'))

# ── Subsample for normal lines (~4000, stratified by cluster) ────────────────
np.random.seed(42)
max_lines = 4000
n_cells   = len(df)
sample_idx = []
for cl in range(n_clusters):
    cl_idx = np.where(labels == cl)[0]
    n_sample = max(10, int(max_lines * len(cl_idx) / n_cells))
    if n_sample >= len(cl_idx):
        sample_idx.append(cl_idx)
    else:
        sample_idx.append(np.random.choice(cl_idx, n_sample, replace=False))
sample_idx = np.concatenate(sample_idx)
sample_set = set(sample_idx)
print(f"  {len(sample_idx)} normal lines will be drawn")

# ── Cluster scatter + normal lines (per cluster, linked by legendgroup) ──────
for cl in range(n_clusters):
    idx   = np.where(labels == cl)[0]
    color = PALETTE[cl % len(PALETTE)]
    in_pl = cl in PLANE_CLUSTERS
    mw    = float(np.mean(W[idx]))
    tag   = ' ** plane' if in_pl else f' W={mw:+.3f}'

    # points
    hover_cl = [
        f"<b>C{cl}</b>{'  (plane)' if in_pl else ''}"
        f"<br>{annots[i]}<br>{broad[i]}"
        f"<br>U={U[i]:.3f}  V={V[i]:.3f}  W={W[i]:+.3f}"
        for i in idx
    ]
    traces.append(go.Scatter3d(
        x=U[idx], y=V[idx], z=W[idx],
        mode='markers',
        name=f"C{cl} {dominant(idx)} ({len(idx):,}){tag}",
        marker=dict(size=3 if in_pl else 2, color=color,
                    opacity=0.88 if in_pl else 0.55),
        text=hover_cl, hovertemplate="%{text}<extra></extra>",
        legendgroup=f'c{cl}'))

    # normal lines for this cluster's subsample
    cl_sample = [i for i in idx if i in sample_set]
    if cl_sample:
        x_lines, y_lines, z_lines = [], [], []
        for i in cl_sample:
            x_lines.extend([U[i], U[i], None])
            y_lines.extend([V[i], V[i], None])
            z_lines.extend([W[i], plane_w_in_box, None])
        traces.append(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines', line=dict(color=color, width=1),
            legendgroup=f'c{cl}', showlegend=False,
            hoverinfo='skip', opacity=0.4))

fig = go.Figure(data=traces)
fig.update_layout(
    title=dict(
        text=(
            "Hox Morphospace  |  Cells + normals to plane"
            f"<br><sup>Each line = perpendicular from cell to plane (W={plane_w_in_box:.3f})"
            f"  |  {len(sample_idx)} lines shown  |  toggle clusters in legend</sup>"
        ),
        x=0.5, xanchor='center', font=dict(size=13)),
    scene=dict(
        xaxis=dict(title='U', showbackground=True,
                   backgroundcolor='rgba(235,235,248,0.7)', gridcolor='rgba(180,180,205,0.7)',
                   zeroline=True, zerolinecolor='rgba(0,0,0,0.3)', zerolinewidth=2),
        yaxis=dict(title='V', showbackground=True,
                   backgroundcolor='rgba(235,248,235,0.7)', gridcolor='rgba(180,205,180,0.7)',
                   zeroline=True, zerolinecolor='rgba(0,0,0,0.3)', zerolinewidth=2),
        zaxis=dict(title='W', showbackground=True,
                   backgroundcolor='rgba(248,235,235,0.7)', gridcolor='rgba(205,180,180,0.7)',
                   zeroline=True, zerolinecolor='rgba(0,0,0,0.3)', zerolinewidth=2),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)),
        aspectmode='data',
    ),
    legend=dict(
        title=dict(text="Cluster (**=plane-defining)", font=dict(size=10)),
        itemsizing='constant', font=dict(size=7),
        x=1.01, y=0.99, xanchor='left',
        bgcolor='rgba(255,255,255,0.88)', bordercolor='rgba(0,0,0,0.2)', borderwidth=1),
    width=1300, height=850, paper_bgcolor='white',
    margin=dict(l=10, r=320, t=85, b=10))
out1 = f"{OUTPUT_DIR}/hox_morphospace_pure_3d.html"
fig.write_html(out1, include_plotlyjs='cdn', full_html=True)
print(f"  Saved: {out1}")

# =====================================================================
# FIGURE 2 — 2D: U vs V (looking down the W axis)
# =====================================================================
print("Building 2D (U, V) ...")
traces2 = []
for cl in range(n_clusters):
    idx   = np.where(labels == cl)[0]
    color = PALETTE[cl % len(PALETTE)]
    in_pl = cl in PLANE_CLUSTERS
    hover = [
        f"<b>C{cl}</b><br>{annots[i]}<br>{broad[i]}"
        f"<br>U={U[i]:.3f}  V={V[i]:.3f}  W={W[i]:+.3f}"
        for i in idx
    ]
    traces2.append(go.Scatter(
        x=U[idx], y=V[idx], mode='markers',
        name=f"C{cl} {dominant(idx)} ({len(idx):,})",
        marker=dict(size=3 if in_pl else 2.5, color=color,
                    opacity=0.85 if in_pl else 0.45),
        text=hover, hovertemplate="%{text}<extra></extra>"))
fig2 = go.Figure(data=traces2)
# draw box boundary
fig2.add_shape(type='rect', x0=-u_half, x1=u_half, y0=-v_half, y1=v_half,
               line=dict(color='rgba(60,60,80,0.5)', width=2, dash='dot'))
fig2.update_layout(
    title=dict(
        text="Hox Morphospace: U vs V  (looking down W axis)"
        "<br><sup>Dotted box = bounding box edges | hover shows W</sup>",
        x=0.5, xanchor='center', font=dict(size=14)),
    xaxis=dict(title='U', scaleanchor='y', scaleratio=1,
               zeroline=True, zerolinecolor='rgba(0,0,0,0.3)', zerolinewidth=1),
    yaxis=dict(title='V',
               zeroline=True, zerolinecolor='rgba(0,0,0,0.3)', zerolinewidth=1),
    legend=dict(title=dict(text="Cluster", font=dict(size=10)),
                itemsizing='constant', font=dict(size=7),
                x=1.01, y=0.99, xanchor='left',
                bgcolor='rgba(255,255,255,0.88)', bordercolor='rgba(0,0,0,0.2)', borderwidth=1),
    width=1150, height=850, paper_bgcolor='white',
    plot_bgcolor='rgba(250,250,255,1)',
    margin=dict(l=10, r=280, t=80, b=50))
out2 = f"{OUTPUT_DIR}/hox_morphospace_pure_uv.html"
fig2.write_html(out2, include_plotlyjs='cdn', full_html=True)
print(f"  Saved: {out2}")

print(f"\nDone!")
print(f"  3D: {out1}")
print(f"  2D: {out2}")
print(f"  CSV: {csv_out}")
