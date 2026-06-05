import warnings
warnings.filterwarnings('ignore')

import os
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
H5AD_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s_fca_biohub_body_10x.h5ad")
HOX_GENES  = ['lab', 'pb', 'Dfd', 'Scr', 'Antp', 'Ubx', 'abd-A', 'Abd-B']
PLANE_CLUSTERS = {2, 6, 7, 9, 10}

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

# ── Load + prep (same pipeline as before) ────────────────────────────────────
print("Loading h5ad ...")
adata = sc.read_h5ad(H5AD_PATH)
present = [g for g in HOX_GENES if g in adata.var_names]
X = adata[:, present].X
if hasattr(X, 'toarray'):
    X = X.toarray()
X = X.astype(float)
mask   = X.sum(axis=1) > 0
X      = X[mask]
obs    = adata.obs[mask].copy()
X_norm = MinMaxScaler().fit_transform(X)
print(f"  {X.shape[0]} Hox-expressing cells")

# ── Leiden clustering (same params as before) ─────────────────────────────────
print("Clustering ...")
mini = ad.AnnData(X=X_norm, obs=obs.copy())
sc.pp.neighbors(mini, n_neighbors=15, use_rep='X', metric='euclidean')
sc.tl.leiden(mini, resolution=0.1, key_added='leiden', random_state=42)
labels_raw = mini.obs['leiden'].astype(int).values
sizes      = np.bincount(labels_raw)
rank       = np.argsort(-sizes)
remap      = {old: new for new, old in enumerate(rank)}
labels     = np.array([remap[l] for l in labels_raw])
n_clusters = labels.max() + 1
print(f"  {n_clusters} clusters")

# ── 3D PCA ────────────────────────────────────────────────────────────────────
pca    = PCA(n_components=3)
coords = pca.fit_transform(X_norm)
ve     = pca.explained_variance_ratio_ * 100
pc1, pc2, pc3 = coords[:, 0], coords[:, 1], coords[:, 2]
pts = coords   # (n, 3)

# ── Metadata ──────────────────────────────────────────────────────────────────
annotation_col = 'annotation'      if 'annotation'       in obs.columns else obs.columns[0]
broad          = obs['annotation_broad'].astype(str).values if 'annotation_broad' in obs.columns else obs[annotation_col].astype(str).values
annotations    = obs[annotation_col].astype(str).values
sex            = obs['sex'].astype(str).values if 'sex' in obs.columns else ['?']*len(annotations)

def dominant_type(idx):
    return pd.Series(broad[idx]).value_counts().index[0]

# cluster centroids & dominant labels
cluster_info = {}
for cl in range(n_clusters):
    idx = np.where(labels == cl)[0]
    cluster_info[cl] = {
        'idx':      idx,
        'centroid': pts[idx].mean(axis=0),
        'dom':      dominant_type(idx),
        'n':        len(idx),
    }

# ── Fit plane through clusters 2, 6, 7, 9, 10 via SVD ───────────────────────
print(f"Fitting plane through clusters {sorted(PLANE_CLUSTERS)} ...")
plane_idx = np.concatenate([cluster_info[c]['idx'] for c in PLANE_CLUSTERS])
plane_pts = pts[plane_idx]

centroid  = plane_pts.mean(axis=0)
centered  = plane_pts - centroid
_, _, Vt  = np.linalg.svd(centered, full_matrices=False)
normal    = Vt[-1]            # last right-singular vector = normal to best-fit plane
normal   /= np.linalg.norm(normal)   # unit normal
print(f"  Plane centroid: {centroid.round(4)}")
print(f"  Plane normal:   {normal.round(4)}")

# ── In-plane axes from SVD (data-aligned, not arbitrary cross products) ──────
u_axis = Vt[0]   # max variance within the plane
v_axis = Vt[1]   # 2nd variance within the plane
print(f"  u-axis:         {u_axis.round(4)}  (max in-plane variance)")
print(f"  v-axis:         {v_axis.round(4)}  (2nd in-plane variance)")

# ── Transform ALL cells into Hox morphospace (u, v, w) ──────────────────────
#    u, v = coordinates within the plane
#    w    = signed perpendicular distance from plane (w=0 is the plane)
delta   = pts - centroid
u_coord = delta @ u_axis
v_coord = delta @ v_axis
w_coord = delta @ normal   # = signed_dist
signed_dist = w_coord

# ── Bounding box in morphospace ──────────────────────────────────────────────
bbox = {
    'u_min': float(u_coord.min()), 'u_max': float(u_coord.max()),
    'v_min': float(v_coord.min()), 'v_max': float(v_coord.max()),
    'w_min': float(w_coord.min()), 'w_max': float(w_coord.max()),
}
print(f"\n  Hox morphospace bounding box:")
print(f"    u: [{bbox['u_min']:.4f}, {bbox['u_max']:.4f}]")
print(f"    v: [{bbox['v_min']:.4f}, {bbox['v_max']:.4f}]")
print(f"    w: [{bbox['w_min']:.4f}, {bbox['w_max']:.4f}]  (w=0 is the plane)")

# ── Per-cluster statistics ─────────────────────────────────────────────────────
rows = []
for cl in range(n_clusters):
    idx  = cluster_info[cl]['idx']
    d    = signed_dist[idx]
    rows.append({
        'cluster':    cl,
        'dom':        cluster_info[cl]['dom'],
        'n':          cluster_info[cl]['n'],
        'mean_w':     d.mean(),
        'std_w':      d.std(),
        'mean_u':     float(np.mean(u_coord[idx])),
        'mean_v':     float(np.mean(v_coord[idx])),
        'in_plane':   cl in PLANE_CLUSTERS,
    })
df = pd.DataFrame(rows).sort_values('mean_w').reset_index(drop=True)

print("\n-- Cluster positions in Hox morphospace --------------------------------")
print(df[['cluster','dom','n','mean_w','std_w','mean_u','mean_v','in_plane']].to_string(index=False))

# ── Find equidistant pairs (exclude plane-defining clusters) ──────────────────
free    = df[~df['in_plane']].copy()
tol     = 0.05
pairs   = []
dists   = free['mean_w'].values
ids     = free['cluster'].values
doms    = free['dom'].values
ns      = free['n'].values

for i in range(len(free)):
    for j in range(i+1, len(free)):
        d_i, d_j = dists[i], dists[j]
        if d_i * d_j < 0:
            if abs(abs(d_i) - abs(d_j)) < tol:
                pairs.append({
                    'cl_A': int(ids[i]), 'dom_A': doms[i], 'n_A': int(ns[i]), 'w_A': round(d_i,4),
                    'cl_B': int(ids[j]), 'dom_B': doms[j], 'n_B': int(ns[j]), 'w_B': round(d_j,4),
                    'delta': round(abs(abs(d_i)-abs(d_j)), 4),
                })

pairs_df = pd.DataFrame(pairs).sort_values('delta') if pairs else pd.DataFrame()
print("\n-- Equidistant pairs (opposite sides, |delta| < 0.05) -----------------")
if len(pairs_df):
    print(pairs_df.to_string(index=False))
else:
    print("  None found within tolerance")

# ── Save morphospace CSV ─────────────────────────────────────────────────────
cell_df = pd.DataFrame({
    'cell_id': obs.index, 'annotation': annotations, 'broad': broad, 'sex': sex,
    'leiden_label': labels,
    'pc1': pc1, 'pc2': pc2, 'pc3': pc3,
    'u_coord': u_coord, 'v_coord': v_coord, 'w_coord': w_coord,
})
csv_out = f"{OUTPUT_DIR}/hox_morphospace_coordinates.csv"
cell_df.to_csv(csv_out, index=False)
print(f"\nMorphospace CSV saved: {csv_out}")

# ── Build plane quad sized to bounding box ───────────────────────────────────
margin_uv = 0.05
u_lo, u_hi = bbox['u_min'] - margin_uv, bbox['u_max'] + margin_uv
v_lo, v_hi = bbox['v_min'] - margin_uv, bbox['v_max'] + margin_uv
w_lo, w_hi = bbox['w_min'], bbox['w_max']

def make_quad_3d(w_offset):
    return np.array([
        centroid + u_lo*u_axis + v_lo*v_axis + w_offset*normal,
        centroid + u_hi*u_axis + v_lo*v_axis + w_offset*normal,
        centroid + u_hi*u_axis + v_hi*v_axis + w_offset*normal,
        centroid + u_lo*u_axis + v_hi*v_axis + w_offset*normal,
    ])

center_corners = make_quad_3d(0)
top_corners    = make_quad_3d(w_hi)
bot_corners    = make_quad_3d(w_lo)
px, py, pz = center_corners[:,0], center_corners[:,1], center_corners[:,2]

# =====================================================================
# FIGURE 1 — 3D scatter + plane + bounding box
# =====================================================================
print("\nBuilding Figure 1: 3D Hox morphospace ...")
traces = []

# center plane
traces.append(go.Mesh3d(
    x=px, y=py, z=pz, i=[0,0], j=[1,2], k=[2,3],
    color='rgba(100,100,255,0.28)', flatshading=True,
    name='Morphospace plane (w=0)', showlegend=True, hoverinfo='skip'))
bx = list(px)+[px[0]]; by = list(py)+[py[0]]; bz = list(pz)+[pz[0]]
traces.append(go.Scatter3d(x=bx, y=by, z=bz, mode='lines',
    line=dict(color='navy', width=3), name='Plane border (w=0)', hoverinfo='skip'))

# top/bottom bounding planes
traces.append(go.Mesh3d(
    x=top_corners[:,0], y=top_corners[:,1], z=top_corners[:,2],
    i=[0,0], j=[1,2], k=[2,3], color='rgba(255,100,100,0.10)',
    flatshading=True, name=f'w_max = +{w_hi:.3f}', showlegend=True, hoverinfo='skip'))
traces.append(go.Mesh3d(
    x=bot_corners[:,0], y=bot_corners[:,1], z=bot_corners[:,2],
    i=[0,0], j=[1,2], k=[2,3], color='rgba(100,255,100,0.10)',
    flatshading=True, name=f'w_min = {w_lo:.3f}', showlegend=True, hoverinfo='skip'))

# vertical edges
for i in range(4):
    traces.append(go.Scatter3d(
        x=[bot_corners[i,0],top_corners[i,0]],
        y=[bot_corners[i,1],top_corners[i,1]],
        z=[bot_corners[i,2],top_corners[i,2]],
        mode='lines', line=dict(color='grey', width=1.5, dash='dot'),
        showlegend=(i==0), name='Bounding box edges', hoverinfo='skip'))

# u/v/w axis arrows
arrow_s = 0.35
for ax_vec, ax_name, ax_col, ax_dash in [
    (u_axis, 'u', 'crimson', None),
    (v_axis, 'v', 'darkgreen', None),
    (normal, 'w', 'blue', 'dash'),
]:
    tip = centroid + arrow_s * ax_vec
    traces.append(go.Scatter3d(
        x=[centroid[0],tip[0]], y=[centroid[1],tip[1]], z=[centroid[2],tip[2]],
        mode='lines+text', line=dict(color=ax_col, width=5, dash=ax_dash),
        text=['', ax_name], textfont=dict(size=13, color=ax_col),
        name=f'{ax_name}-axis', hoverinfo='skip'))

# cluster scatter
for cl in range(n_clusters):
    idx    = cluster_info[cl]['idx']
    color  = PALETTE[cl % len(PALETTE)]
    dom    = cluster_info[cl]['dom']
    mw     = df.loc[df['cluster']==cl, 'mean_w'].values[0]
    in_pl  = cl in PLANE_CLUSTERS
    tag    = '  ** plane' if in_pl else f'  w={mw:+.3f}'

    hover_cl = [
        f"<b>C{cl}</b>{'  (plane)' if in_pl else ''}"
        f"<br>{ann}<br>{br}"
        f"<br>u={u_coord[i]:.3f}  v={v_coord[i]:.3f}  w={w_coord[i]:+.3f}"
        for ann, br, i in zip(annotations[idx], broad[idx], idx)
    ]
    traces.append(go.Scatter3d(
        x=pc1[idx], y=pc2[idx], z=pc3[idx],
        mode='markers',
        name=f"C{cl} {dom} ({cluster_info[cl]['n']:,}){tag}",
        marker=dict(size=3 if in_pl else 2, color=color,
                    opacity=0.88 if in_pl else 0.55),
        text=hover_cl, hovertemplate="%{text}<extra></extra>"))

fig1 = go.Figure(data=traces)
fig1.update_layout(
    title=dict(
        text=(
            "Hox Morphospace  |  Plane through C2,C6,C7,C9,C10  |  Bounding box in (u,v,w)"
            f"<br><sup>normal=[{normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}]"
            f"  |  w=[{w_lo:.3f}, {w_hi:.3f}]"
            f"  |  PC1 {ve[0]:.1f}% PC2 {ve[1]:.1f}% PC3 {ve[2]:.1f}%</sup>"
        ),
        x=0.5, xanchor='center', font=dict(size=12)),
    scene=dict(
        xaxis=dict(title=f"PC1 ({ve[0]:.1f}%)", showbackground=True,
                   backgroundcolor='rgba(235,235,248,0.7)', gridcolor='rgba(180,180,205,0.7)'),
        yaxis=dict(title=f"PC2 ({ve[1]:.1f}%)", showbackground=True,
                   backgroundcolor='rgba(235,248,235,0.7)', gridcolor='rgba(180,205,180,0.7)'),
        zaxis=dict(title=f"PC3 ({ve[2]:.1f}%)", showbackground=True,
                   backgroundcolor='rgba(248,235,235,0.7)', gridcolor='rgba(205,180,180,0.7)'),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)),
    ),
    legend=dict(
        title=dict(text="Cluster (**=plane-defining, w=mean dist)", font=dict(size=10)),
        itemsizing='constant', font=dict(size=7),
        x=1.01, y=0.99, xanchor='left',
        bgcolor='rgba(255,255,255,0.88)', bordercolor='rgba(0,0,0,0.2)', borderwidth=1),
    width=1300, height=850, paper_bgcolor='white',
    margin=dict(l=10, r=320, t=90, b=10))
out1 = f"{OUTPUT_DIR}/hox_morphospace_3d.html"
fig1.write_html(out1, include_plotlyjs='cdn', full_html=True)
print(f"  Saved: {out1}")

# =====================================================================
# FIGURE 2 — top-down view: u vs v (looking along w)
# =====================================================================
print("Building Figure 2: top-down (u, v) ...")
traces2 = []
for cl in range(n_clusters):
    idx   = cluster_info[cl]['idx']
    color = PALETTE[cl % len(PALETTE)]
    in_pl = cl in PLANE_CLUSTERS
    hover = [
        f"<b>C{cl}</b><br>{annotations[i]}<br>u={u_coord[i]:.3f} v={v_coord[i]:.3f} w={w_coord[i]:+.3f}"
        for i in idx
    ]
    traces2.append(go.Scatter(
        x=u_coord[idx], y=v_coord[idx], mode='markers',
        name=f"C{cl} {dominant_type(idx)} ({len(idx):,})",
        marker=dict(size=3 if in_pl else 2.5, color=color,
                    opacity=0.85 if in_pl else 0.45),
        text=hover, hovertemplate="%{text}<extra></extra>"))
fig2 = go.Figure(data=traces2)
fig2.update_layout(
    title=dict(
        text="Hox Morphospace: top-down (u, v)  |  looking along plane normal"
        "<br><sup>Hover shows w (height above/below plane)</sup>",
        x=0.5, xanchor='center', font=dict(size=13)),
    xaxis=dict(title='u', scaleanchor='y', scaleratio=1, zeroline=True, zerolinecolor='grey'),
    yaxis=dict(title='v', zeroline=True, zerolinecolor='grey'),
    legend=dict(title=dict(text="Cluster", font=dict(size=10)),
                itemsizing='constant', font=dict(size=7),
                x=1.01, y=0.99, xanchor='left',
                bgcolor='rgba(255,255,255,0.88)', bordercolor='rgba(0,0,0,0.2)', borderwidth=1),
    width=1150, height=800, paper_bgcolor='white',
    plot_bgcolor='rgba(248,248,252,1)',
    margin=dict(l=10, r=280, t=90, b=50))
out2 = f"{OUTPUT_DIR}/hox_morphospace_uv.html"
fig2.write_html(out2, include_plotlyjs='cdn', full_html=True)
print(f"  Saved: {out2}")

# =====================================================================
# FIGURE 3 — side view: u vs w (plane edge-on)
# =====================================================================
print("Building Figure 3: side view (u, w) ...")
traces3 = []
for cl in range(n_clusters):
    idx   = cluster_info[cl]['idx']
    color = PALETTE[cl % len(PALETTE)]
    in_pl = cl in PLANE_CLUSTERS
    hover = [f"<b>C{cl}</b><br>{annotations[i]}<br>u={u_coord[i]:.3f} w={w_coord[i]:+.3f}" for i in idx]
    traces3.append(go.Scatter(
        x=u_coord[idx], y=w_coord[idx], mode='markers',
        name=f"C{cl} {dominant_type(idx)} ({len(idx):,})",
        marker=dict(size=3 if in_pl else 2.5, color=color,
                    opacity=0.85 if in_pl else 0.45),
        text=hover, hovertemplate="%{text}<extra></extra>"))
fig3 = go.Figure(data=traces3)
fig3.add_hline(y=0, line_width=2, line_dash='dash', line_color='navy',
               annotation_text='morphospace plane (w=0)', annotation_position='bottom right')
fig3.add_hline(y=w_hi, line_width=1, line_dash='dot', line_color='red',
               annotation_text=f'w_max={w_hi:.3f}', annotation_position='top right')
fig3.add_hline(y=w_lo, line_width=1, line_dash='dot', line_color='green',
               annotation_text=f'w_min={w_lo:.3f}', annotation_position='bottom right')
fig3.update_layout(
    title=dict(
        text="Hox Morphospace: side view (u, w)  |  plane seen edge-on"
        "<br><sup>w=0 is the plane | positive/negative = opposite sides | dashed = bounding box</sup>",
        x=0.5, xanchor='center', font=dict(size=13)),
    xaxis=dict(title='u (in-plane)', zeroline=True, zerolinecolor='grey'),
    yaxis=dict(title='w (distance from plane)', zeroline=True, zerolinecolor='navy', zerolinewidth=2),
    legend=dict(title=dict(text="Cluster", font=dict(size=10)),
                itemsizing='constant', font=dict(size=7),
                x=1.01, y=0.99, xanchor='left',
                bgcolor='rgba(255,255,255,0.88)', bordercolor='rgba(0,0,0,0.2)', borderwidth=1),
    width=1150, height=700, paper_bgcolor='white',
    plot_bgcolor='rgba(248,248,252,1)',
    margin=dict(l=10, r=280, t=90, b=50))
out3 = f"{OUTPUT_DIR}/hox_morphospace_uw.html"
fig3.write_html(out3, include_plotlyjs='cdn', full_html=True)
print(f"  Saved: {out3}")

# =====================================================================
# FIGURE 4 — side view: v vs w
# =====================================================================
print("Building Figure 4: side view (v, w) ...")
traces4 = []
for cl in range(n_clusters):
    idx   = cluster_info[cl]['idx']
    color = PALETTE[cl % len(PALETTE)]
    in_pl = cl in PLANE_CLUSTERS
    hover = [f"<b>C{cl}</b><br>{annotations[i]}<br>v={v_coord[i]:.3f} w={w_coord[i]:+.3f}" for i in idx]
    traces4.append(go.Scatter(
        x=v_coord[idx], y=w_coord[idx], mode='markers',
        name=f"C{cl} {dominant_type(idx)} ({len(idx):,})",
        marker=dict(size=3 if in_pl else 2.5, color=color,
                    opacity=0.85 if in_pl else 0.45),
        text=hover, hovertemplate="%{text}<extra></extra>"))
fig4 = go.Figure(data=traces4)
fig4.add_hline(y=0, line_width=2, line_dash='dash', line_color='navy',
               annotation_text='morphospace plane (w=0)', annotation_position='bottom right')
fig4.add_hline(y=w_hi, line_width=1, line_dash='dot', line_color='red',
               annotation_text=f'w_max={w_hi:.3f}', annotation_position='top right')
fig4.add_hline(y=w_lo, line_width=1, line_dash='dot', line_color='green',
               annotation_text=f'w_min={w_lo:.3f}', annotation_position='bottom right')
fig4.update_layout(
    title=dict(
        text="Hox Morphospace: side view (v, w)  |  rotated 90 deg"
        "<br><sup>Orthogonal side view</sup>",
        x=0.5, xanchor='center', font=dict(size=13)),
    xaxis=dict(title='v (in-plane)', zeroline=True, zerolinecolor='grey'),
    yaxis=dict(title='w (distance from plane)', zeroline=True, zerolinecolor='navy', zerolinewidth=2),
    legend=dict(title=dict(text="Cluster", font=dict(size=10)),
                itemsizing='constant', font=dict(size=7),
                x=1.01, y=0.99, xanchor='left',
                bgcolor='rgba(255,255,255,0.88)', bordercolor='rgba(0,0,0,0.2)', borderwidth=1),
    width=1150, height=700, paper_bgcolor='white',
    plot_bgcolor='rgba(248,248,252,1)',
    margin=dict(l=10, r=280, t=90, b=50))
out4 = f"{OUTPUT_DIR}/hox_morphospace_vw.html"
fig4.write_html(out4, include_plotlyjs='cdn', full_html=True)
print(f"  Saved: {out4}")

# =====================================================================
# FIGURE 5 — distance bar chart
# =====================================================================
print("Building Figure 5: distance bar chart ...")
df_plot = df.sort_values('mean_w').reset_index(drop=True)
bar_colors = []
for _, r in df_plot.iterrows():
    if r['cluster'] in PLANE_CLUSTERS: bar_colors.append('rgba(80,80,200,0.8)')
    elif r['mean_w'] > 0:              bar_colors.append('rgba(210,70,50,0.8)')
    else:                               bar_colors.append('rgba(50,150,70,0.8)')
tick_labels = [
    f"C{int(r['cluster'])} {r['dom'][:20]}{'  **' if r['cluster'] in PLANE_CLUSTERS else ''}"
    for _, r in df_plot.iterrows()
]
fig5 = go.Figure()
fig5.add_trace(go.Bar(
    x=df_plot['mean_w'].values, y=tick_labels, orientation='h',
    marker_color=bar_colors,
    error_x=dict(type='data', array=df_plot['std_w'].values, visible=True, color='rgba(0,0,0,0.3)'),
    customdata=np.stack([
        df_plot['cluster'].values, df_plot['n'].values,
        df_plot['mean_w'].round(4).values, df_plot['std_w'].round(4).values,
        df_plot['mean_u'].round(4).values, df_plot['mean_v'].round(4).values,
    ], axis=1),
    hovertemplate=(
        "<b>C%{customdata[0]}</b><br>Cells: %{customdata[1]:,}<br>"
        "w: %{customdata[2]:.4f} +/- %{customdata[3]:.4f}<br>"
        "u: %{customdata[4]:.4f}  v: %{customdata[5]:.4f}<extra></extra>")))
fig5.add_vline(x=0, line_width=2, line_dash='dash', line_color='navy')
fig5.update_layout(
    title=dict(
        text="Hox Morphospace: cluster distances (w) from plane"
        "<br><sup>Blue=plane-defining | Green=negative | Red=positive | error bars=+/-1 SD</sup>",
        x=0.5, xanchor='center', font=dict(size=12)),
    xaxis=dict(title='w (signed distance from plane)'),
    yaxis=dict(automargin=True, tickfont=dict(size=8)),
    height=max(550, n_clusters*22), width=950,
    paper_bgcolor='white', plot_bgcolor='rgba(245,245,252,1)',
    margin=dict(l=10, r=20, t=90, b=40))
out5 = f"{OUTPUT_DIR}/hox_morphospace_distances.html"
fig5.write_html(out5, include_plotlyjs='cdn', full_html=True)
print(f"  Saved: {out5}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n-- Equidistant pairs -----------------------------------------------")
if len(pairs_df):
    for _, p in pairs_df.iterrows():
        print(f"  C{int(p['cl_A'])} ({p['dom_A']}, w={p['w_A']:+.4f})  <->  "
              f"C{int(p['cl_B'])} ({p['dom_B']}, w={p['w_B']:+.4f})   |delta|={p['delta']:.4f}")
else:
    print("  None within tolerance 0.05")

print("\nAll done!")
print(f"  1. 3D morphospace:   {out1}")
print(f"  2. Top-down (u,v):   {out2}")
print(f"  3. Side view (u,w):  {out3}")
print(f"  4. Side view (v,w):  {out4}")
print(f"  5. Distance bar:     {out5}")
print(f"  6. Coordinates CSV:  {csv_out}")
