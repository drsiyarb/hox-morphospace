import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull
from itertools import combinations

OUTPUT_DIR = r"C:\Users\siyar\Downloads\flycellatlas_hox_output"
H5AD_PATH  = r"C:\Users\siyar\Downloads\s_fca_biohub_body_10x.h5ad"
HOX_GENES = ['lab', 'pb', 'Dfd', 'Scr', 'Antp', 'Ubx', 'abd-A', 'Abd-B']

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading ...")
df = pd.read_csv(f"{OUTPUT_DIR}/hox_morphospace_boxcentered.csv")
pc1 = df['pc1'].values; pc2 = df['pc2'].values; pc3 = df['pc3'].values
U = df['U'].values; V = df['V'].values; W = df['W'].values
u_raw = df['u_coord'].values; v_raw = df['v_coord'].values; w_raw = df['w_coord'].values
labels = df['leiden_label'].values.copy()
broad = df['broad'].values

plane_W = -((w_raw.min() + w_raw.max()) / 2)

# Splits & merge
for i in np.where(labels==20)[0]:
    if W[i]<-0.22: labels[i]=30
    elif W[i]>0.05: labels[i]=31
    else: labels[i]=32
for i in np.where(labels==7)[0]:
    if W[i]<-0.13: labels[i]=33
    elif W[i]>-0.10: labels[i]=34
    else: labels[i]=35
for cl in [12,15,16,27]: labels[labels==cl]=36

# ── Load Hox expression ────────────────────────────────────────────────────
print("Loading h5ad ...")
adata = sc.read_h5ad(H5AD_PATH)
gn = list(adata.var_names)
hox_idx = [gn.index(g) for g in HOX_GENES]
X_raw = adata.X; X_raw = X_raw.toarray() if hasattr(X_raw,'toarray') else X_raw
X_hox = X_raw[X_raw[:,hox_idx].sum(1)>0][:,hox_idx]
scaler = MinMaxScaler()
X_hox_norm = scaler.fit_transform(X_hox)

Antp = X_hox_norm[:, 4]; Ubx = X_hox_norm[:, 5]
abdA = X_hox_norm[:, 6]; AbdB = X_hox_norm[:, 7]

# Extract annotation_broad for Hox-expressing cells
hox_mask = X_raw[:,hox_idx].sum(1) > 0
ann_broad = adata.obs['annotation_broad'].values[hox_mask]
ann_specific = adata.obs['annotation'].values[hox_mask]
print(f"  annotation_broad loaded: {len(ann_broad)} cells")

H1 = Antp - Ubx
H2 = abdA - AbdB
H3 = (Antp + Ubx) - (abdA + AbdB)

# ── In H1/H2/H3 space, the coordinate planes are trivially axis-aligned ─────
# H1=0 → normal is [1,0,0], H2=0 → [0,1,0], H3=0 → [0,0,1]
data_center_h = np.array([H1.mean(), H2.mean(), H3.mean()])
print(f"  Data center in H space: {data_center_h.round(4)}")
print(f"  H1 range: [{H1.min():.3f}, {H1.max():.3f}]")
print(f"  H2 range: [{H2.min():.3f}, {H2.max():.3f}]")
print(f"  H3 range: [{H3.min():.3f}, {H3.max():.3f}]")

# ── Extract top 200 variable genes for Gene Search panel ─────────────────
print("Building gene search bank (top 200 variable genes) ...")
KNOWN_GENES = set(HOX_GENES) | {'inv', 'ap', 'vg', 'Dll', 'dpp', 'wg', 'hh'}

X_sp = adata.X[hox_mask]  # sparse, 38k × n_genes
_mean = np.array(X_sp.mean(axis=0)).ravel()
_msq  = np.array(X_sp.multiply(X_sp).mean(axis=0)).ravel() if hasattr(X_sp,'multiply') \
        else np.mean(np.array(X_sp)**2, axis=0)
_var  = _msq - _mean**2

ranked_gi = np.argsort(-_var)
gene_bank = []
for gi in ranked_gi:
    g = gn[gi]
    if g in KNOWN_GENES or len(gene_bank) >= 200:
        continue
    col = X_sp[:, gi]
    vals = col.toarray().ravel().astype(float) if hasattr(col,'toarray') else np.array(col).ravel().astype(float)
    em = vals > 0
    ne = int(em.sum())
    if ne < 10:
        continue
    vmax = vals.max()
    if vmax <= 0:
        continue
    nv = vals / vmax
    gene_bank.append({
        'name': g, 'n': ne,
        'idx': np.where(em)[0].tolist(),
        'val': np.round(nv[em], 3).tolist(),
    })
    if len(gene_bank) >= 200:
        break

print(f"  Gene bank: {len(gene_bank)} genes (top by variance, excl. Hox/TF)")
del X_sp, _mean, _msq, _var, ranked_gi  # free memory

# ── Normalise for heatmap ────────────────────────────────────────────────────
all_cl = [4,11,5,0,26,18,1,36,13,8,2,9,6,10,14,19,30,31,32,33,34,35]
corr_mask = np.isin(labels, all_cl)
corr_idx = np.where(corr_mask)[0]
scaler2 = MinMaxScaler()
X_renorm = scaler2.fit_transform(X_hox[corr_mask])
idx_map = {o:n for n,o in enumerate(corr_idx)}

def mean_w(cl): return H1[labels==cl].mean()   # sort by H1 (was W, ~same axis)
def get_expr(cl):
    ci = [idx_map[i] for i in np.where(labels==cl)[0] if i in idx_map]
    return X_renorm[ci].mean(0) if ci else np.zeros(8)

# ── Helpers ──────────────────────────────────────────────────────────────────
def hex_to_rgb(h):
    h = h.lstrip('#')
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def lighten_hex(h, factor=0.45):
    r, g, b = hex_to_rgb(h)
    return f'#{int(r+(255-r)*factor):02x}{int(g+(255-g)*factor):02x}{int(b+(255-b)*factor):02x}'

def darken_hex(h, factor=0.3):
    r, g, b = hex_to_rgb(h)
    return f'#{int(r*(1-factor)):02x}{int(g*(1-factor)):02x}{int(b*(1-factor)):02x}'

def make_colorscale(hex_color):
    r, g, b = hex_to_rgb(hex_color)
    return [[0, 'rgb(255,255,255)'], [1, f'rgb({r},{g},{b})']]

def cl_name(c):
    if c==36: return "C12/15/16/27"
    return f"C{c}"

# ── Pairs with dark/light shades ─────────────────────────────────────────────
PAIRS_RAW = [
    (4, 11, '#0000CD'), (5, 0, '#CC0000'), (26,18, '#006400'),
    (1, 36, '#C71585'), (13, 8, '#7B2D8E'), (2, 9, '#CC8400'),
    (6, 10, '#008B8B'), (14,19, '#D2691E'),
]
TRIS = [(30,31,32, '#FF1493'), (33,34,35, '#9ACD32')]
UNMATCHED_COLOR = '#888888'

# Assign dark to pos side, light to neg side
color_map = {}
shade_map = {}  # 'dark' or 'light' per cluster
pairs_info = []
for pi, (a, b, color) in enumerate(PAIRS_RAW):
    pos, neg = (a, b) if mean_w(a) >= mean_w(b) else (b, a)  # H1-pos side = dark
    dark = darken_hex(color, 0.15)
    light = lighten_hex(color, 0.40)
    color_map[pos] = dark
    color_map[neg] = light
    shade_map[pos] = 'dark'
    shade_map[neg] = 'light'
    pairs_info.append({
        'id': f"P{pi+1}", 'label': f"P{pi+1} ({cl_name(a)} & {cl_name(b)})",
        'pos': pos, 'neg': neg, 'color': color, 'dark': dark, 'light': light,
        'clusters': [pos, neg],
    })

tris_info = []
for ti, (neg_l, pos_l, ctr_l, color) in enumerate(TRIS):
    orig_name = {30:'C20',31:'C20',32:'C20',33:'C7',34:'C7',35:'C7'}[neg_l]
    trips = [(neg_l, mean_w(neg_l), 'neg'), (pos_l, mean_w(pos_l), 'pos'), (ctr_l, mean_w(ctr_l), 'ctr')]
    trips.sort(key=lambda x: -x[1])
    tris_info.append({
        'id': f"P{len(PAIRS_RAW)+ti+1}", 'label': f"P{len(PAIRS_RAW)+ti+1} ({orig_name} 3-way)",
        'color': color, 'trips': trips, 'clusters': [t[0] for t in trips], 'orig': orig_name,
    })
    for lbl, mw, side in trips:
        if side == 'ctr':
            color_map[lbl] = color
            shade_map[lbl] = 'full'
        else:
            color_map[lbl] = lighten_hex(color, 0.40)
            shade_map[lbl] = 'light'

paired_cls = set()
for p in pairs_info: paired_cls.update(p['clusters'])
for t in tris_info: paired_cls.update(t['clusters'])
all_labels_set = sorted(set(labels))
for cl in all_labels_set:
    if cl not in color_map: color_map[cl] = UNMATCHED_COLOR

# ── Build figure ─────────────────────────────────────────────────────────────
print("Building figure ...")
fig = go.Figure()

# ── 3D scatter ───────────────────────────────────────────────────────────────
legend_added = set()
for p in pairs_info:
    for cl in [p['pos'], p['neg']]:
        idx = np.where(labels == cl)[0]
        if len(idx) == 0: continue
        show = (p['id'] not in legend_added); legend_added.add(p['id'])
        side_tag = "pos" if cl == p['pos'] else "neg"
        fig.add_trace(go.Scatter3d(
            x=H1[idx], y=H2[idx], z=H3[idx], mode='markers',
            name=p['label'], legendgroup=p['id'], showlegend=show,
            marker=dict(size=3, color=color_map[cl], opacity=0.8),
            customdata=np.column_stack([U[idx], V[idx], W[idx]]),
            hovertemplate=(f"<b>{cl_name(cl)}</b> ({side_tag}) [{p['id']}]<br>"
                f"H1=%{{x:.3f}} H2=%{{y:.3f}} H3=%{{z:.3f}}<br>"
                f"U=%{{customdata[0]:.3f}} V=%{{customdata[1]:.3f}} W=%{{customdata[2]:.3f}}<extra></extra>"),
            scene='scene',
            legend='legend',
        ))

for t in tris_info:
    for lbl, mw, side in t['trips']:
        idx = np.where(labels == lbl)[0]
        if len(idx) == 0: continue
        show = (t['id'] not in legend_added); legend_added.add(t['id'])
        fig.add_trace(go.Scatter3d(
            x=H1[idx], y=H2[idx], z=H3[idx], mode='markers',
            name=t['label'], legendgroup=t['id'], showlegend=show,
            marker=dict(size=3, color=color_map[lbl], opacity=0.8),
            customdata=np.column_stack([U[idx], V[idx], W[idx]]),
            hovertemplate=(f"<b>{t['orig']}-{side}</b> [{t['id']}]<br>"
                f"H1=%{{x:.3f}} H2=%{{y:.3f}} H3=%{{z:.3f}}<br>"
                f"U=%{{customdata[0]:.3f}} V=%{{customdata[1]:.3f}} W=%{{customdata[2]:.3f}}<extra></extra>"),
            scene='scene',
            legend='legend',
        ))

unmatched_first = True
for cl in [c for c in all_labels_set if c not in paired_cls]:
    idx = np.where(labels == cl)[0]
    if len(idx) == 0: continue
    fig.add_trace(go.Scatter3d(
        x=H1[idx], y=H2[idx], z=H3[idx], mode='markers',
        name='Unmatched', legendgroup='unmatched', showlegend=unmatched_first,
        marker=dict(size=2.5, color=UNMATCHED_COLOR, opacity=0.4),
        customdata=np.column_stack([U[idx], V[idx], W[idx]]),
        hovertemplate=f"<b>C{cl}</b><br>H1=%{{x:.3f}} H2=%{{y:.3f}} H3=%{{z:.3f}}<br>U=%{{customdata[0]:.3f}} V=%{{customdata[1]:.3f}} W=%{{customdata[2]:.3f}}<extra></extra>",
        scene='scene',
        legend='legend',
    ))
    unmatched_first = False

# ── 6 planes (toggleable) ───────────────────────────────────────────────────
# Now working in H1/H2/H3 space
h_corners = np.array([[h1,h2,h3]
    for h1 in [H1.min(), H1.max()]
    for h2 in [H2.min(), H2.max()]
    for h3 in [H3.min(), H3.max()]])

def add_plane(fig, centroid, normal, name, color_str, opacity=0.7, bounds_pts=None):
    bp = bounds_pts if bounds_pts is not None else h_corners
    if abs(normal[0]) < 0.9:
        tmp = np.array([1.,0.,0.])
    else:
        tmp = np.array([0.,1.,0.])
    e1 = np.cross(normal, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1);  e2 /= np.linalg.norm(e2)
    cc = bp - centroid
    s_p = cc @ e1; t_p = cc @ e2
    n_g = 20
    SS, TT = np.meshgrid(np.linspace(s_p.min(), s_p.max(), n_g),
                          np.linspace(t_p.min(), t_p.max(), n_g))
    grid = centroid[None,None,:] + SS[:,:,None]*e1[None,None,:] + TT[:,:,None]*e2[None,None,:]
    gx, gy, gz = grid[:,:,0], grid[:,:,1], grid[:,:,2]
    # Clip to data range in H-space
    h1r = [H1.min(), H1.max()]; h2r = [H2.min(), H2.max()]; h3r = [H3.min(), H3.max()]
    ib = ((gx>=h1r[0])&(gx<=h1r[1])&(gy>=h2r[0])&(gy<=h2r[1])&(gz>=h3r[0])&(gz<=h3r[1]))
    gx = np.where(ib, gx, np.nan); gy = np.where(ib, gy, np.nan); gz = np.where(ib, gz, np.nan)
    fig.add_trace(go.Surface(
        x=gx, y=gy, z=gz,
        colorscale=[[0, color_str], [1, color_str]],
        surfacecolor=np.zeros_like(gx),
        showscale=False, showlegend=True, name=name, hoverinfo='skip',
        opacity=opacity, scene='scene',
        legend='legend3', visible='legendonly',
    ))

# 3 Hox coordinate planes — in H-space these are trivially axis-aligned
origin_h = np.array([0., 0., 0.])
add_plane(fig, origin_h, np.array([1.,0.,0.]), 'H1=0: Antp=Ubx', 'rgba(30,100,220,0.7)')
add_plane(fig, origin_h, np.array([0.,1.,0.]), 'H2=0: abdA=AbdB', 'rgba(220,50,50,0.7)')
add_plane(fig, origin_h, np.array([0.,0.,1.]), 'H3=0: thoracic=abdominal', 'rgba(220,180,0,0.7)')

# 3 Geometric morphospace (U/V/W) planes projected into H-space
# Fit H1/H2/H3 → U/V/W mapping, then find normals in H-space
H_stack = np.column_stack([H1, H2, H3, np.ones(len(H1))])
UVW_stack = np.column_stack([U, V, W])
M_h2uvw_full, _, _, _ = np.linalg.lstsq(H_stack, UVW_stack, rcond=None)
M_lin = M_h2uvw_full[:3, :]   # 3x3 linear part
M_off = M_h2uvw_full[3, :]    # 1x3 offset

# W=plane_W in UVW → find centroid and normal in H-space
# U=0 normal in UVW is [1,0,0] → in H-space: M_lin^(-T) @ [1,0,0]
M_lin_invT = np.linalg.inv(M_lin).T

for uvw_name, uvw_normal, uvw_cent_val, geom_color in [
    ('W=plane (geom)', np.array([0.,0.,1.]), plane_W, 'rgba(30,100,220,0.3)'),
    ('U=0 (geom)',     np.array([1.,0.,0.]), 0.0,     'rgba(220,50,50,0.3)'),
    ('V=0 (geom)',     np.array([0.,1.,0.]), 0.0,     'rgba(220,180,0,0.3)'),
]:
    # Normal in H-space
    n_h = M_lin_invT @ uvw_normal
    n_h = n_h / np.linalg.norm(n_h)
    # Find centroid: a point in H-space that maps to the UVW plane
    # For U=0: solve M_lin @ h + M_off = [0, ?, ?], use data mean for other dims
    # Simpler: find the H-point closest to the plane for each cell, use mean of near cells
    uvw_vals = UVW_stack @ uvw_normal
    target = uvw_cent_val
    near_mask = np.abs(uvw_vals - target) < np.percentile(np.abs(uvw_vals - target), 10)
    cent_h_geom = np.array([H1[near_mask].mean(), H2[near_mask].mean(), H3[near_mask].mean()])
    add_plane(fig, cent_h_geom, n_h, uvw_name, geom_color, opacity=0.5)

# ── Possibility space hull (natively in H1/H2/H3) ──────────────────────────
print("Computing possibility space ...")
n_grid = 25   # 25^4 = 390,625 combinations
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
print(f"  Hull H1: [{ps_verts[:,0].min():.2f},{ps_verts[:,0].max():.2f}]  "
      f"H2: [{ps_verts[:,1].min():.2f},{ps_verts[:,1].max():.2f}]  "
      f"H3: [{ps_verts[:,2].min():.2f},{ps_verts[:,2].max():.2f}]")

# Remap indices
ps_vert_map = {old: new for new, old in enumerate(ps_hull.vertices)}
ps_fi, ps_fj, ps_fk = [], [], []
for tri in ps_hull.simplices:
    mapped = [ps_vert_map.get(v) for v in tri]
    if all(m is not None for m in mapped):
        ps_fi.append(mapped[0]); ps_fj.append(mapped[1]); ps_fk.append(mapped[2])

# ── Merge coplanar triangles to find TRUE polytope edges ────────────────────
# Compute face normals for each triangle
face_normals = []
for fi_idx in range(len(ps_fi)):
    v0 = ps_verts[ps_fi[fi_idx]]
    v1 = ps_verts[ps_fj[fi_idx]]
    v2 = ps_verts[ps_fk[fi_idx]]
    n = np.cross(v1-v0, v2-v0)
    nl = np.linalg.norm(n)
    n = n / nl if nl > 1e-12 else n
    face_normals.append(n)
face_normals = np.array(face_normals)

# Group triangles by normal direction (coplanar faces share same normal)
# Round normals to cluster them
def normal_key(n):
    # Canonicalize: make first non-zero component positive
    for i in range(3):
        if abs(n[i]) > 0.1:
            if n[i] < 0: n = -n
            break
    return tuple(np.round(n, 2))

face_groups = {}
for fi_idx in range(len(ps_fi)):
    key = normal_key(face_normals[fi_idx].copy())
    face_groups.setdefault(key, []).append(fi_idx)

print(f"  Polytope has {len(face_groups)} true faces (from {len(ps_fi)} triangles)")

# An edge is a TRUE polytope edge only if its two adjacent triangles
# belong to DIFFERENT face groups (different normals)
edge_to_faces = {}  # edge → list of face indices
for fi_idx in range(len(ps_fi)):
    verts = [ps_fi[fi_idx], ps_fj[fi_idx], ps_fk[fi_idx]]
    for a, b in combinations(verts, 2):
        edge = (min(a,b), max(a,b))
        edge_to_faces.setdefault(edge, []).append(fi_idx)

true_edges = set()
for edge, face_list in edge_to_faces.items():
    if len(face_list) >= 2:
        # Check if adjacent faces have different normals
        keys = set()
        for fi_idx in face_list:
            keys.add(normal_key(face_normals[fi_idx].copy()))
        if len(keys) > 1:
            true_edges.add(edge)
    else:
        # Boundary edge (only 1 face) — always a true edge
        true_edges.add(edge)

print(f"  True edges: {len(true_edges)} (from {len(edge_to_faces)} triangle edges)")

# ── Color faces by which constraint surface they belong to ──────────────────
# Group normals into face types for coloring
# The zonotope has faces from: H1=±1, H2=±1, and the 4 diagonal constraint planes
face_type_colors = {}
for key, group in face_groups.items():
    n = np.array(key)
    # Classify by dominant normal direction
    if abs(n[0]) > 0.8 and abs(n[1]) < 0.2 and abs(n[2]) < 0.2:
        col = 'rgb(30,100,220)'    # H1 boundary (blue)
    elif abs(n[1]) > 0.8 and abs(n[0]) < 0.2 and abs(n[2]) < 0.2:
        col = 'rgb(220,50,50)'     # H2 boundary (red)
    else:
        col = 'rgb(180,160,40)'    # Diagonal constraint (gold)
    for fi_idx in group:
        face_type_colors[fi_idx] = col

face_colors = [face_type_colors.get(i, 'rgb(150,150,150)') for i in range(len(ps_fi))]

# Surface with face-type coloring (toggleable)
fig.add_trace(go.Mesh3d(
    x=ps_verts[:,0], y=ps_verts[:,1], z=ps_verts[:,2],
    i=ps_fi, j=ps_fj, k=ps_fk,
    facecolor=face_colors,
    opacity=0.45,
    name='Possibility space (surface)',
    showlegend=True,
    showscale=False,
    flatshading=True,
    legendgroup='poss_space',
    scene='scene',
    hoverinfo='skip',
    legend='legend2', visible='legendonly',
))

# Wireframe — TRUE edges only (no internal diagonals)
ps_ex, ps_ey, ps_ez = [], [], []
for a, b in true_edges:
    ps_ex += [ps_verts[a,0], ps_verts[b,0], None]
    ps_ey += [ps_verts[a,1], ps_verts[b,1], None]
    ps_ez += [ps_verts[a,2], ps_verts[b,2], None]

fig.add_trace(go.Scatter3d(
    x=ps_ex, y=ps_ey, z=ps_ez, mode='lines',
    line=dict(color='rgba(40,40,40,0.9)', width=3),
    name='Possibility space (wireframe)',
    showlegend=True, hoverinfo='skip',
    legendgroup='poss_wire',
    scene='scene',
    legend='legend2',
))

# ── Vertex labels ───────────────────────────────────────────────────────────
# Each vertex is an extremal gene state. Find which (Antp,Ubx,abdA,AbdB) produces it.
# For each hull vertex, find the closest sampled point and recover its gene values.
ps_orig_idx = ps_hull.vertices  # indices into pts_h, which maps back to flat grid
print("\n  Polytope vertices:")
v_labels = []
for vi, oidx in enumerate(ps_orig_idx):
    h1v, h2v, h3v = pts_h[oidx]
    a_v, u_v, c_v, d_v = gAntp[oidx], gUbx[oidx], gabdA[oidx], gAbdB[oidx]
    gene_str = f"A{a_v:.0f}U{u_v:.0f}a{c_v:.0f}B{d_v:.0f}"
    label = f"V{vi+1}"
    v_labels.append(label)
    print(f"    {label}: H=({h1v:+.0f},{h2v:+.0f},{h3v:+.0f})  "
          f"genes=(Antp={a_v:.0f},Ubx={u_v:.0f},abdA={c_v:.0f},AbdB={d_v:.0f})")

# Add vertex markers with labels
fig.add_trace(go.Scatter3d(
    x=ps_verts[:,0], y=ps_verts[:,1], z=ps_verts[:,2],
    mode='markers+text',
    marker=dict(size=5, color='black', symbol='diamond'),
    text=v_labels,
    textposition='top center',
    textfont=dict(size=10, color='black', family='Arial Black'),
    name='Polytope vertices',
    showlegend=True,
    legendgroup='poss_verts',
    scene='scene',
    legend='legend2',
    customdata=np.column_stack([
        [gAntp[oi] for oi in ps_orig_idx],
        [gUbx[oi] for oi in ps_orig_idx],
        [gabdA[oi] for oi in ps_orig_idx],
        [gAbdB[oi] for oi in ps_orig_idx],
    ]),
    hovertemplate=(
        '<b>%{text}</b><br>'
        'H1=%{x:+.0f} H2=%{y:+.0f} H3=%{z:+.0f}<br>'
        'Antp=%{customdata[0]:.0f} Ubx=%{customdata[1]:.0f}<br>'
        'abdA=%{customdata[2]:.0f} AbdB=%{customdata[3]:.0f}'
        '<extra></extra>'
    ),
))

# ── Center vertex + internal axes ───────────────────────────────────────────
center = np.array([0., 0., 0.])
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0], mode='markers+text',
    marker=dict(size=6, color='white', symbol='diamond',
                line=dict(width=2, color='black')),
    text=['C'], textposition='top center',
    textfont=dict(size=11, color='black', family='Arial Black'),
    name='Center (0,0,0)',
    showlegend=True, legendgroup='center',
    scene='scene',
    legend='legend2', visible='legendonly',
    hovertemplate='<b>Center</b><br>H1=0 H2=0 H3=0<extra></extra>',
))

# V4 (idx 3): (-1,0,+1) Ubx only   ←→  V8 (idx 7): (+1,0,+1) Antp only  — GREEN (H1 axis)
v4 = ps_verts[3]; v8 = ps_verts[7]
fig.add_trace(go.Scatter3d(
    x=[v4[0], center[0], v8[0]], y=[v4[1], center[1], v8[1]], z=[v4[2], center[2], v8[2]],
    mode='lines',
    line=dict(color='rgba(0,160,0,0.9)', width=5),
    name='V4↔C↔V8 (Ubx↔Antp)',
    showlegend=True, legendgroup='axis_h1',
    scene='scene',
    legend='legend2', visible='legendonly',
    hoverinfo='skip',
))

# V1 (idx 0): (0,-1,-1) AbdB only  ←→  V2 (idx 1): (0,+1,-1) abdA only  — ORANGE (H2 axis)
v1 = ps_verts[0]; v2 = ps_verts[1]
fig.add_trace(go.Scatter3d(
    x=[v1[0], center[0], v2[0]], y=[v1[1], center[1], v2[1]], z=[v1[2], center[2], v2[2]],
    mode='lines',
    line=dict(color='rgba(230,140,0,0.9)', width=5),
    name='V1↔C↔V2 (AbdB↔abdA)',
    showlegend=True, legendgroup='axis_h2',
    scene='scene',
    legend='legend2', visible='legendonly',
    hoverinfo='skip',
))

# ── Polytope center (origin) + shifted internal axes ─────────────────────
data_c = np.array([0., 0., 0.])   # polytope center = origin of H-space
fig.add_trace(go.Scatter3d(
    x=[data_c[0]], y=[data_c[1]], z=[data_c[2]], mode='markers+text',
    marker=dict(size=6, color='yellow', symbol='diamond',
                line=dict(width=2, color='black')),
    text=['C*'], textposition='top center',
    textfont=dict(size=11, color='black', family='Arial Black'),
    name='Polytope center C*',
    showlegend=True, legendgroup='data_center',
    scene='scene',
    legend='legend2', visible='legendonly',
    hovertemplate='<b>Polytope Center C*</b><br>H1=0 H2=0 H3=0<extra></extra>',
))

# Shifted green arms: 3 lines from C* with endpoints interpolated toward V12
v12 = ps_verts[11]  # (0,0,+2) pure thoracic

# t=0.0: original V4/V8 level
v4_t0 = v4; v8_t0 = v8
# t=0.10: 10% toward V12
v4_t1 = v4 + 0.10 * (v12 - v4); v8_t1 = v8 + 0.10 * (v12 - v8)
# t=0.40: 40% toward V12
v4_t2 = v4 + 0.40 * (v12 - v4); v8_t2 = v8 + 0.40 * (v12 - v8)

for ti, (v4t, v8t, alpha, tint, lbl) in enumerate([
    (v4_t0, v8_t0, 0.85, 'rgba(100,220,100,0.85)', 'V4↔C*↔V8 (t=0)'),
    (v4_t1, v8_t1, 0.80, 'rgba(60,190,60,0.80)',   'V4↔C*↔V8 (t=0.1)'),
    (v4_t2, v8_t2, 0.75, 'rgba(30,150,30,0.75)',   'V4↔C*↔V8 (t=0.4)'),
]):
    fig.add_trace(go.Scatter3d(
        x=[v4t[0], data_c[0], v8t[0]], y=[v4t[1], data_c[1], v8t[1]], z=[v4t[2], data_c[2], v8t[2]],
        mode='lines',
        line=dict(color=tint, width=5),
        name=lbl,
        showlegend=True, legendgroup=f'axis_h1_shifted_{ti}',
        scene='scene',
        hoverinfo='skip',
        legend='legend2', visible='legendonly',
    ))

# Shifted orange (lighter tint): C* → V1 and C* → V2 directions
fig.add_trace(go.Scatter3d(
    x=[v1[0], data_c[0], v2[0]], y=[v1[1], data_c[1], v2[1]], z=[v1[2], data_c[2], v2[2]],
    mode='lines',
    line=dict(color='rgba(255,190,80,0.85)', width=5),
    name='V1↔C*↔V2 (shifted)',
    showlegend=True, legendgroup='axis_h2_shifted',
    scene='scene',
    legend='legend2', visible='legendonly',
    hoverinfo='skip',
))

# Shifted blue: V1 moved 10% toward V5, V2 moved 10% toward V6, through C*
v5 = ps_verts[4]   # (-1,-1,0) Ubx+AbdB
v6 = ps_verts[5]   # (-1,+1,0) Ubx+abdA
v1_b = v1 + 0.10 * (v5 - v1)
v2_b = v2 + 0.10 * (v6 - v2)
fig.add_trace(go.Scatter3d(
    x=[v1_b[0], data_c[0], v2_b[0]],
    y=[v1_b[1], data_c[1], v2_b[1]],
    z=[v1_b[2], data_c[2], v2_b[2]],
    mode='lines',
    line=dict(color='rgba(80,140,255,0.85)', width=5),
    name='V1→V5↔C*↔V2→V6 (t=0.1)',
    showlegend=True, legendgroup='axis_h2_blue',
    scene='scene',
    legend='legend2', visible='legendonly',
    hoverinfo='skip',
))

# Full theoretical bounding box [-1,1]×[-1,1]×[-2,2] (dotted, to show unreachable corners)
bb = [[-1,1],[-1,1],[-2,2]]
bb_edges = []
for a in bb[0]:
    for b in bb[1]: bb_edges.append(([a,a],[b,b],[bb[2][0],bb[2][1]]))
for a in bb[0]:
    for c in bb[2]: bb_edges.append(([a,a],[bb[1][0],bb[1][1]],[c,c]))
for b in bb[1]:
    for c in bb[2]: bb_edges.append(([bb[0][0],bb[0][1]],[b,b],[c,c]))
for i,(ex,ey,ez) in enumerate(bb_edges):
    fig.add_trace(go.Scatter3d(
        x=ex,y=ey,z=ez,mode='lines',
        line=dict(color='rgba(180,0,0,0.35)',width=1.5, dash='dot'),
        showlegend=(i==0),
        name='Theoretical bbox (unreachable corners)' if i==0 else '',
        legendgroup='bbox_full', hoverinfo='skip', scene='scene',
        legend='legend2', visible='legendonly',
    ))

# Data bounding box
h1_ext=[H1.min(),H1.max()]; h2_ext=[H2.min(),H2.max()]; h3_ext=[H3.min(),H3.max()]
edges = []
for a in h1_ext:
    for b in h2_ext: edges.append(([a,a],[b,b],[h3_ext[0],h3_ext[1]]))
for a in h1_ext:
    for c in h3_ext: edges.append(([a,a],[h2_ext[0],h2_ext[1]],[c,c]))
for b in h2_ext:
    for c in h3_ext: edges.append(([h1_ext[0],h1_ext[1]],[b,b],[c,c]))
for i,(ex,ey,ez) in enumerate(edges):
    fig.add_trace(go.Scatter3d(
        x=ex,y=ey,z=ez,mode='lines',
        line=dict(color='rgba(0,0,0,0.3)',width=2),
        showlegend=(i==0), name='Data bounding box' if i==0 else '',
        legendgroup='bbox', hoverinfo='skip', scene='scene',
        legend='legend2', visible='legendonly',
    ))

# ── Diamond stripe planes ─────────────────────────────────────────────────
# Build a vertex lookup: V1..V14 → coordinates
print("Adding diamond stripe planes ...")
verts_dict = {}
for vi in range(len(ps_orig_idx)):
    verts_dict[f'V{vi+1}'] = ps_verts[vi].copy()
verts_dict['C*'] = data_c.copy()

GRID_CELL_SIZE = 0.15   # side length of each square grid cell

def lerp3(a, b, t):
    return a + t * (b - a)

# Each plane: (name, apex, base, left, right, shift_target, shift_pct)
DIAMOND_PLANES = [
    ('C*_V3_V2_V1_V7_s0',   'C*', 'V3',  'V2', 'V1',  'V7',  0),
    ('C*_V3_V2_V1_V7_s20',  'C*', 'V3',  'V2', 'V1',  'V7',  20),
    ('C*_V5_V1_V4_V13_s10', 'C*', 'V5',  'V1', 'V4',  'V13', 10),
    ('C*_V6_V4_V2_V14_s10', 'C*', 'V6',  'V4', 'V2',  'V14', 10),
    ('C*_V10_V2_V8_V14_s20','C*', 'V10', 'V2', 'V8',  'V14', 20),
    ('C*_V9_V8_V1_V13_s20', 'C*', 'V9',  'V8', 'V1',  'V13', 20),
    ('C*_V12_V8_V4_V14_s0', 'C*', 'V12', 'V8', 'V4',  'V14', 0),
]

# Distinct colors for each plane
PLANE_COLORS = [
    (0.2, 0.6, 1.0),    # blue
    (0.3, 0.8, 0.9),    # cyan
    (0.9, 0.3, 0.3),    # red
    (0.9, 0.6, 0.2),    # orange
    (0.6, 0.3, 0.8),    # purple
    (0.3, 0.8, 0.3),    # green
    (0.9, 0.8, 0.2),    # yellow
]

for pi, (pname, apex_k, base_k, left_k, right_k, shift_k, shift_pct) in enumerate(DIAMOND_PLANES):
    apex  = verts_dict[apex_k].copy()
    base  = verts_dict[base_k].copy()
    left  = verts_dict[left_k].copy()
    right = verts_dict[right_k].copy()

    # Apply shift: only the base moves toward shift target
    if shift_pct > 0:
        target = verts_dict[shift_k].copy()
        base = lerp3(base, target, shift_pct / 100.0)

    # The diamond has 4 corners: apex (top), left, right, base (bottom)
    # Two triangles: apex-left-right and left-base-right
    quad_pts = np.array([apex, left, base, right])  # 0=apex, 1=left, 2=base, 3=right

    r, g, b = PLANE_COLORS[pi]
    face_col = f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.50)'
    line_col = f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.85)'
    grid_col = f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.55)'
    legend_grp = f'dplane_{pi}'

    # Filled surface (two triangles for the diamond quad)
    fig.add_trace(go.Mesh3d(
        x=quad_pts[:,0], y=quad_pts[:,1], z=quad_pts[:,2],
        i=[0, 1], j=[1, 2], k=[3, 3],
        color=face_col,
        opacity=0.50,
        name=f'Plane {pname}',
        showlegend=True,
        legendgroup=legend_grp,
        scene='scene',
        hoverinfo='skip',
        flatshading=True,
        legend='legend4', visible='legendonly',
    ))

    # Diamond outline
    outline_x, outline_y, outline_z = [], [], []
    for a_idx, b_idx in [(0,1),(1,2),(2,3),(3,0)]:
        outline_x += [quad_pts[a_idx,0], quad_pts[b_idx,0], None]
        outline_y += [quad_pts[a_idx,1], quad_pts[b_idx,1], None]
        outline_z += [quad_pts[a_idx,2], quad_pts[b_idx,2], None]
    fig.add_trace(go.Scatter3d(
        x=outline_x, y=outline_y, z=outline_z, mode='lines',
        line=dict(color=line_col, width=3),
        name=f'Plane {pname} (outline)',
        showlegend=False,
        legendgroup=legend_grp,
        scene='scene',
        hoverinfo='skip',
        legend='legend4', visible='legendonly',
    ))

    # ── Square grid wireframe ──────────────────────────────────────────────
    # The diamond has two sets of parallel edges:
    #   Set A: apex→left  and  right→base  (left diagonal pair)
    #   Set B: apex→right and  left→base   (right diagonal pair)
    # We parameterise lines parallel to Set A at intervals along Set B, and vice versa.
    #
    # For each direction, we compute edge lengths and divide by GRID_CELL_SIZE
    # to get the number of grid lines.

    # Direction 1: lines parallel to apex→left / right→base
    # These lines connect points on apex→right with points on left→base
    edge_len_B = np.linalg.norm(right - apex)  # length along Set B direction
    n_steps_B = max(1, int(np.round(edge_len_B / GRID_CELL_SIZE)))

    grid_x, grid_y, grid_z = [], [], []
    for si in range(1, n_steps_B):
        t = si / n_steps_B
        p1 = lerp3(apex, right, t)   # point on apex→right edge
        p2 = lerp3(left, base, t)    # corresponding point on left→base edge
        grid_x += [p1[0], p2[0], None]
        grid_y += [p1[1], p2[1], None]
        grid_z += [p1[2], p2[2], None]

    # Direction 2: lines parallel to apex→right / left→base
    # These lines connect points on apex→left with points on right→base
    edge_len_A = np.linalg.norm(left - apex)  # length along Set A direction
    n_steps_A = max(1, int(np.round(edge_len_A / GRID_CELL_SIZE)))

    for si in range(1, n_steps_A):
        t = si / n_steps_A
        p1 = lerp3(apex, left, t)    # point on apex→left edge
        p2 = lerp3(right, base, t)   # corresponding point on right→base edge
        grid_x += [p1[0], p2[0], None]
        grid_y += [p1[1], p2[1], None]
        grid_z += [p1[2], p2[2], None]

    if grid_x:
        fig.add_trace(go.Scatter3d(
            x=grid_x, y=grid_y, z=grid_z, mode='lines',
            line=dict(color=grid_col, width=1.5),
            name=f'Plane {pname} (grid)',
            showlegend=False,
            legendgroup=legend_grp,
            scene='scene',
            hoverinfo='skip',
            legend='legend4', visible='legendonly',
        ))

    print(f"  Plane {pname}: {n_steps_A}x{n_steps_B} grid (cell~{GRID_CELL_SIZE})")

# ── Precompute plane corner data for JS embedding ─────────────────────────
plane_data_js = []
for pi, (pname, apex_k, base_k, left_k, right_k, shift_k, shift_pct) in enumerate(DIAMOND_PLANES):
    a = verts_dict[apex_k].copy()
    b = verts_dict[base_k].copy()
    l = verts_dict[left_k].copy()
    r = verts_dict[right_k].copy()
    if shift_pct > 0:
        t = verts_dict[shift_k].copy()
        b = lerp3(b, t, shift_pct / 100.0)
    plane_data_js.append({
        'name': pname,
        'apex': np.round(a, 6).tolist(),
        'base': np.round(b, 6).tolist(),
        'left': np.round(l, 6).tolist(),
        'right': np.round(r, 6).tolist(),
        'color': [round(c, 3) for c in PLANE_COLORS[pi]],
    })

# ── Cell type overlays (muscle + neuron) ──────────────────────────────────
print("Adding cell type overlays ...")
muscle_mask = ann_broad == 'muscle cell'
neuron_mask = ann_broad == 'neuron'
print(f"  Muscle cells: {muscle_mask.sum()}")
print(f"  Neurons: {neuron_mask.sum()}")

fig.add_trace(go.Scatter3d(
    x=H1[muscle_mask], y=H2[muscle_mask], z=H3[muscle_mask],
    mode='markers',
    marker=dict(size=4, color='rgba(220,50,50,0.7)', symbol='circle',
                line=dict(width=0.5, color='rgba(150,0,0,0.5)')),
    name=f'Muscle cells ({muscle_mask.sum()})',
    showlegend=True,
    legendgroup='overlay_muscle',
    visible='legendonly',
    scene='scene',
    legend='legend5',
    customdata=np.column_stack([
        U[muscle_mask], V[muscle_mask], W[muscle_mask]
    ]),
    hovertemplate=(
        '<b>Muscle cell</b><br>'
        'H1=%{x:.3f} H2=%{y:.3f} H3=%{z:.3f}<br>'
        'U=%{customdata[0]:.3f} V=%{customdata[1]:.3f} W=%{customdata[2]:.3f}'
        '<extra></extra>'
    ),
))

fig.add_trace(go.Scatter3d(
    x=H1[neuron_mask], y=H2[neuron_mask], z=H3[neuron_mask],
    mode='markers',
    marker=dict(size=4, color='rgba(50,120,220,0.7)', symbol='circle',
                line=dict(width=0.5, color='rgba(0,50,150,0.5)')),
    name=f'Neurons ({neuron_mask.sum()})',
    showlegend=True,
    legendgroup='overlay_neuron',
    visible='legendonly',
    scene='scene',
    legend='legend5',
    customdata=np.column_stack([
        U[neuron_mask], V[neuron_mask], W[neuron_mask]
    ]),
    hovertemplate=(
        '<b>Neuron</b><br>'
        'H1=%{x:.3f} H2=%{y:.3f} H3=%{z:.3f}<br>'
        'U=%{customdata[0]:.3f} V=%{customdata[1]:.3f} W=%{customdata[2]:.3f}'
        '<extra></extra>'
    ),
))

# Additional specific cell type overlays
SPECIFIC_OVERLAYS = [
    ('sensory neuron',                       'rgba(0,200,180,0.8)',  'rgba(0,140,120,0.6)', 'diamond'),
    ('leg muscle motor neuron',              'rgba(255,140,0,0.8)', 'rgba(180,90,0,0.6)',  'circle'),
    ('leg taste bristle chemosensory neuron', 'rgba(180,0,220,0.8)', 'rgba(120,0,150,0.6)', 'circle'),
    ('follicle cell',                        'rgba(220,180,40,0.8)','rgba(160,120,0,0.6)', 'square'),
]

for cell_type, color, line_color, symbol in SPECIFIC_OVERLAYS:
    # sensory neuron is in annotation_broad, others in annotation (specific)
    if cell_type == 'sensory neuron':
        ct_mask = ann_broad == cell_type
    else:
        ct_mask = ann_specific == cell_type
    ct_count = ct_mask.sum()
    print(f"  {cell_type}: {ct_count}")

    if ct_count == 0:
        continue

    fig.add_trace(go.Scatter3d(
        x=H1[ct_mask], y=H2[ct_mask], z=H3[ct_mask],
        mode='markers',
        marker=dict(size=5, color=color, symbol=symbol,
                    line=dict(width=0.5, color=line_color)),
        name=f'{cell_type} ({ct_count})',
        showlegend=True,
        legendgroup=f'overlay_{cell_type[:10]}',
        legend='legend5',
        visible='legendonly',
        scene='scene',
        customdata=np.column_stack([U[ct_mask], V[ct_mask], W[ct_mask]]),
        hovertemplate=(
            f'<b>{cell_type}</b><br>'
            'H1=%{x:.3f} H2=%{y:.3f} H3=%{z:.3f}<br>'
            'U=%{customdata[0]:.3f} V=%{customdata[1]:.3f} W=%{customdata[2]:.3f}'
            '<extra></extra>'
        ),
    ))

# ── Default colorscale for gene overlays (JS universal selector overrides) ──
DEFAULT_GENE_CS = 'Jet'

# ── Hox gene expression overlays ──────────────────────────────────────────
HOX_GENE_INFO = {
    'lab':   'labial',        'pb':   'proboscipedia',
    'Dfd':   'Deformed',      'Scr':  'Sex combs reduced',
    'Antp':  'Antennapedia',  'Ubx':  'Ultrabithorax',
    'abd-A': 'abdominal-A',   'Abd-B':'Abdominal-B',
}

print("Adding Hox gene expression overlays ...")
for hi, (gene, full_name) in enumerate(HOX_GENE_INFO.items()):
    hox_vals = X_hox_norm[:, hi].copy()
    expr_mask_hox = hox_vals > 0
    n_expr = int(expr_mask_hox.sum())
    print(f"  {gene} ({full_name}): {n_expr} expressing cells")
    if n_expr == 0:
        continue
    cscale_name = DEFAULT_GENE_CS
    fig.add_trace(go.Scatter3d(
        x=H1[expr_mask_hox], y=H2[expr_mask_hox], z=H3[expr_mask_hox],
        mode='markers',
        marker=dict(
            size=2.5,
            color=hox_vals[expr_mask_hox],
            colorscale=cscale_name,
            cmin=0, cmax=1,
            showscale=False,
            opacity=1.0,
        ),
        name=f'{gene} ({full_name}) ({n_expr})',
        showlegend=True,
        legendgroup=f'gene_{gene}',
        visible='legendonly',
        scene='scene',
        legend='legend6',
        customdata=np.column_stack([
            hox_vals[expr_mask_hox],
            U[expr_mask_hox], V[expr_mask_hox], W[expr_mask_hox]
        ]),
        hovertemplate=(
            f'<b>{gene}</b> ({full_name}) [{cscale_name}]<br>'
            'expr=%{customdata[0]:.3f}<br>'
            'H1=%{x:.3f} H2=%{y:.3f} H3=%{z:.3f}<br>'
            'U=%{customdata[1]:.3f} V=%{customdata[2]:.3f} W=%{customdata[3]:.3f}'
            '<extra></extra>'
        ),
    ))

# ── TF / signaling gene expression overlays ────────────────────────────────
# Each gene shown as a 3D scatter, colored by normalized expression intensity
# All start hidden (legendonly), toggleable in legend
OVERLAY_GENES = {
    'inv':  'invected (post. compartment)',
    'ap':   'apterous (dorsal)',
    'vg':   'vestigial (wing/appendage)',
    'Dll':  'Distal-less (distal appendage)',
    'dpp':  'decapentaplegic (BMP signal)',
    'wg':   'wingless (Wnt signal)',
    'hh':   'hedgehog (Hh signal)',
}

print("Adding TF/signaling gene overlays ...")
for gene, full_name in OVERLAY_GENES.items():
    if gene not in gn:
        print(f"  WARNING: {gene} not in dataset")
        continue
    gi = gn.index(gene)
    raw_vals = X_raw[hox_mask, gi].copy()
    if hasattr(raw_vals, 'toarray'):
        raw_vals = raw_vals.toarray().ravel()
    raw_vals = raw_vals.astype(float)

    # Normalize to [0,1]
    vmin, vmax = raw_vals.min(), raw_vals.max()
    if vmax > vmin:
        norm_vals = (raw_vals - vmin) / (vmax - vmin)
    else:
        norm_vals = np.zeros_like(raw_vals)

    # Only show expressing cells (non-zero) for cleaner viz
    expr_mask = raw_vals > 0
    n_expr = int(expr_mask.sum())
    print(f"  {gene} ({full_name}): {n_expr} expressing cells")

    if n_expr == 0:
        continue

    cscale_name = DEFAULT_GENE_CS

    fig.add_trace(go.Scatter3d(
        x=H1[expr_mask], y=H2[expr_mask], z=H3[expr_mask],
        mode='markers',
        marker=dict(
            size=2.5,
            color=norm_vals[expr_mask],
            colorscale=cscale_name,
            cmin=0, cmax=1,
            showscale=False,
            opacity=1.0,
        ),
        name=f'{gene} ({full_name}) ({n_expr})',
        showlegend=True,
        legendgroup=f'gene_{gene}',
        visible='legendonly',
        scene='scene',
        legend='legend6',
        customdata=np.column_stack([
            raw_vals[expr_mask], norm_vals[expr_mask],
            U[expr_mask], V[expr_mask], W[expr_mask]
        ]),
        hovertemplate=(
            f'<b>{gene}</b> ({full_name}) [{cscale_name}]<br>'
            'raw=%{customdata[0]:.2f} norm=%{customdata[1]:.3f}<br>'
            'H1=%{x:.3f} H2=%{y:.3f} H3=%{z:.3f}<br>'
            'U=%{customdata[2]:.3f} V=%{customdata[3]:.3f} W=%{customdata[4]:.3f}'
            '<extra></extra>'
        ),
    ))

# ── Build HTML heatmap table (separate from Plotly fig) ──────────────────────
def val_to_bg(value, hex_color):
    """Blend white→hex_color based on expression value 0→1."""
    r, g, b = hex_to_rgb(hex_color)
    r2 = int(255 - value * (255 - r))
    g2 = int(255 - value * (255 - g))
    b2 = int(255 - value * (255 - b))
    return f'rgb({r2},{g2},{b2})'

matrix_rows_html = []
# Header row
hdr = '<tr class="mx-hdr"><th></th>'
for g in HOX_GENES:
    short = g[:3] if len(g) > 3 else g
    hdr += f'<th title="{g}">{short}</th>'
hdr += '</tr>'
matrix_rows_html.append(hdr)

for p in pairs_info:
    pos_expr = get_expr(p['pos']); neg_expr = get_expr(p['neg'])
    for ri, (row_label, row_expr, shade) in enumerate([
        (cl_name(p['pos']), pos_expr, p['dark']),
        (cl_name(p['neg']), neg_expr, p['light']),
    ]):
        cells = ''
        for vi, v in enumerate(row_expr):
            bg = val_to_bg(v, shade)
            cells += f'<td style="background:{bg}" title="{HOX_GENES[vi]}={v:.2f}">{v:.2f}</td>'
        lbl_td = ''
        if ri == 0:
            lbl_td = f'<th rowspan="2" class="mx-pair" style="color:{p["color"]}">{p["id"]}</th>'
        matrix_rows_html.append(f'<tr>{lbl_td}{cells}</tr>')

for t in tris_info:
    rows_data = []
    for lbl, mw, side in t['trips']:
        sc = t['color'] if side == 'ctr' else lighten_hex(t['color'], 0.40)
        rows_data.append((f"{t['orig']}-{side}", get_expr(lbl), sc))
    for ri, (row_label, row_expr, shade) in enumerate(rows_data):
        cells = ''
        for vi, v in enumerate(row_expr):
            bg = val_to_bg(v, shade)
            cells += f'<td style="background:{bg}" title="{HOX_GENES[vi]}={v:.2f}">{v:.2f}</td>'
        lbl_td = ''
        if ri == 0:
            lbl_td = f'<th rowspan="{len(rows_data)}" class="mx-pair" style="color:{t["color"]}">{t["id"]}</th>'
        matrix_rows_html.append(f'<tr>{lbl_td}{cells}</tr>')

matrix_table_html = '<table class="mx-table">' + ''.join(matrix_rows_html) + '</table>'

# ── Layout (Plotly legend disabled — custom HTML panels instead) ──────────
fig.update_layout(
    scene=dict(
        domain=dict(x=[0, 1], y=[0, 1]),
        xaxis=dict(title='H1 (Antp−Ubx)', showgrid=True, gridcolor='rgba(200,200,220,0.4)'),
        yaxis=dict(title='H2 (abdA−AbdB)', showgrid=True, gridcolor='rgba(200,200,220,0.4)'),
        zaxis=dict(title='H3 (Antp+Ubx−abdA−AbdB)', showgrid=True, gridcolor='rgba(200,200,220,0.4)'),
        aspectmode='data',
    ),
    showlegend=False,
    autosize=True,
    height=960,
    paper_bgcolor='white',
    margin=dict(l=20, r=20, t=30, b=20),
)

# ── Build trace registry for custom HTML legend panels ────────────────────
import json as json_module
import re as re_module

PANELS = [
    ('pairs',           'Cluster Pairs'),
    ('polytope',        'Polytope'),
    ('coord_planes',    'Coord Planes'),
    ('internal_planes', 'Internal Planes'),
    ('cell_types',      'Cell Types'),
    ('gene_expr',       'Gene Expression'),
]

def classify_trace(trace):
    lg = getattr(trace, 'legendgroup', '') or ''
    name = getattr(trace, 'name', '') or ''
    if trace.type == 'heatmap':
        return None
    if lg and lg[0] == 'P' and lg[1:].isdigit():
        return 'pairs'
    if lg == 'unmatched':
        return 'pairs'
    if lg in ('poss_space','poss_wire','poss_verts','center','data_center','bbox','bbox_full'):
        return 'polytope'
    if lg.startswith('axis_'):
        return 'polytope'
    if 'H1=0' in name or 'H2=0' in name or 'H3=0' in name or '(geom)' in name:
        return 'coord_planes'
    if lg.startswith('dplane_'):
        return 'internal_planes'
    if lg.startswith('overlay_'):
        return 'cell_types'
    if lg.startswith('gene_'):
        return 'gene_expr'
    return None

CS_SWATCH = {
    'Jet':     'linear-gradient(90deg,#00f,#0ff,#0f0,#ff0,#f00)',
    'Viridis': 'linear-gradient(90deg,#440154,#31688e,#35b779,#fde725)',
    'Inferno': 'linear-gradient(90deg,#000004,#932567,#f98e09,#fcffa4)',
    'Turbo':   'linear-gradient(90deg,#30123b,#28bbec,#a2fc3c,#fb8022,#7a0403)',
    'Cividis': 'linear-gradient(90deg,#002051,#7b7b78,#fdea45)',
}

def get_trace_color(trace):
    try:
        if trace.type == 'scatter3d':
            if trace.mode and 'lines' in trace.mode and trace.line and trace.line.color:
                return str(trace.line.color)
            if trace.marker:
                # Named colorscale → return gradient swatch
                cs = trace.marker.colorscale
                if cs is not None and isinstance(cs, str) and cs in CS_SWATCH:
                    return CS_SWATCH[cs]
                c = trace.marker.color
                if c is not None and isinstance(c, str):
                    return c
                if cs and not isinstance(cs, str) and len(cs) > 0:
                    return str(cs[-1][1])
        elif trace.type == 'mesh3d':
            if trace.color:
                return str(trace.color)
            fc = trace.facecolor
            if fc is not None and len(fc) > 0:
                return str(fc[0])
        elif trace.type == 'surface':
            cs = trace.colorscale
            if cs and len(cs) > 0:
                return str(cs[0][1])
    except Exception:
        pass
    return '#888888'

# Group traces by legendgroup within each panel
panel_items = {pid: {} for pid, _ in PANELS}

for i, trace in enumerate(fig.data):
    group = classify_trace(trace)
    if group is None:
        continue
    lg = getattr(trace, 'legendgroup', '') or f'_solo_{i}'
    name = getattr(trace, 'name', '') or ''
    color = get_trace_color(trace)
    show = getattr(trace, 'showlegend', True)
    if lg not in panel_items[group]:
        panel_items[group][lg] = {'name': name, 'color': color, 'indices': [], 'show': show}
    panel_items[group][lg]['indices'].append(i)

# Default visibility rules
DEFAULT_ON = {
    'pairs': lambda lg: True,
    'polytope': lambda lg: lg in ('poss_wire', 'poss_verts'),
    'coord_planes': lambda lg: False,
    'internal_planes': lambda lg: False,
    'cell_types': lambda lg: False,
    'gene_expr': lambda lg: False,
}

for pid, items in panel_items.items():
    for lg, item in items.items():
        item['on'] = DEFAULT_ON[pid](lg)

# Apply visibility to figure traces
for pid, items in panel_items.items():
    for lg, item in items.items():
        for idx in item['indices']:
            fig.data[idx].visible = True if item['on'] else False

# Build JS panel config (only items with showlegend=True appear in UI)
js_panels = []
for pid, title in PANELS:
    items_list = []
    for lg, item in panel_items[pid].items():
        if not item['show']:
            continue
        uid = re_module.sub(r'[^a-zA-Z0-9]', '_', lg)
        items_list.append({
            'lg': lg, 'uid': uid,
            'name': item['name'], 'color': item['color'],
            'indices': item['indices'], 'on': item['on'],
        })
    js_panels.append({'id': pid, 'title': title, 'items': items_list})

# ── Generate Hox Morphospace Explorer HTML ────────────────────────────────
print("Generating Hox Morphospace Explorer ...")
fig_json = fig.to_json()
fig_json_safe = fig_json.replace('</', r'<\/')
panels_json = json_module.dumps(js_panels)

# Cell coordinates for dynamic trace creation + gene bank
coords_json = json_module.dumps({
    'h1': np.round(H1, 4).tolist(),
    'h2': np.round(H2, 4).tolist(),
    'h3': np.round(H3, 4).tolist(),
})
bank_json = json_module.dumps(gene_bank)
planes_data_json = json_module.dumps(plane_data_js)
n_total_traces = len(fig.data)  # so JS knows where dynamic traces start

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hox Morphospace Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #eef0f4; overflow: hidden; }}

#header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: white; padding: 10px 24px;
    display: flex; align-items: baseline; gap: 14px;
    border-bottom: 2px solid #0f3460;
}}
#header h1 {{ font-size: 17px; font-weight: 600; letter-spacing: 0.3px; }}
#header .sub {{ font-size: 11px; color: #8899aa; }}

#main {{ display: flex; height: calc(100vh - 46px); }}

/* ── 3D section (60%) ─────────────────────────────────── */
#sec-3d {{ display: flex; flex: 6; min-width: 0; }}
#chart {{ flex: 1; min-width: 0; background: white; }}
#panels-3d {{
    width: 220px; min-width: 220px;
    background: #f4f5f9; border-left: 2px solid #d0d3db;
    overflow-y: auto; padding: 5px 5px 16px;
}}
#panels-3d::-webkit-scrollbar {{ width: 5px; }}
#panels-3d::-webkit-scrollbar-thumb {{ background: #bbb; border-radius: 3px; }}

/* ── 2D Plane Projector section (40%) ─────────────────── */
#sec-2d {{ display: flex; flex: 4; min-width: 0; border-left: 2px solid #d0d3db; }}
#plane2d {{
    flex: 1; min-width: 0;
    background: white;
    display: flex; flex-direction: column;
}}
#plane2d-chart {{ flex: 1; min-height: 0; }}
#plane2d-placeholder {{
    flex: 1; display: flex; align-items: center; justify-content: center;
    color: #aaa; font-size: 12px; font-style: italic; text-align: center;
    padding: 20px;
}}
#panels-2d {{
    width: 185px; min-width: 185px;
    background: #f4f5f9; border-left: 2px solid #d0d3db;
    overflow-y: auto; padding: 5px 5px 16px;
}}
#panels-2d::-webkit-scrollbar {{ width: 5px; }}
#panels-2d::-webkit-scrollbar-thumb {{ background: #bbb; border-radius: 3px; }}

/* ── Matrix section ───────────────────────────────────── */
#sec-matrix {{
    width: 220px; min-width: 220px;
    background: #f4f5f9; border-left: 2px solid #d0d3db;
    overflow-y: auto; padding: 6px;
}}
#sec-matrix::-webkit-scrollbar {{ width: 5px; }}
#sec-matrix::-webkit-scrollbar-thumb {{ background: #bbb; border-radius: 3px; }}
#sec-matrix h3 {{
    font-size: 11px; color: #2c3e50; margin: 0 0 6px;
    padding: 6px 8px; background: #2c3e50; color: white;
    border-radius: 4px; text-align: center; letter-spacing: 0.3px;
}}
.mx-table {{
    width: 100%; border-collapse: collapse; font-size: 8px;
    table-layout: fixed;
}}
.mx-table th, .mx-table td {{
    padding: 2px 1px; text-align: center; border: 1px solid #e0e0e0;
}}
.mx-hdr th {{
    background: #2c3e50; color: white; font-size: 8px;
    font-weight: 600; padding: 3px 1px;
}}
.mx-pair {{
    font-weight: 700; font-size: 9px;
    writing-mode: vertical-lr; text-orientation: mixed;
    background: #f0f0f0; min-width: 18px; padding: 2px;
}}
.mx-table td {{
    font-size: 7.5px; color: #333; font-weight: 500;
}}

.panel {{
    background: white; border-radius: 5px;
    margin-bottom: 6px; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    border: 1px solid #dce0e8;
}}
.p-head {{
    background: #2c3e50; color: white;
    padding: 7px 10px; font-size: 11.5px; font-weight: 600;
    cursor: pointer; display: flex; align-items: center;
    user-select: none; letter-spacing: 0.3px;
}}
.p-head:hover {{ background: #34495e; }}
.p-head .arr {{
    margin-right: 6px; font-size: 9px;
    transition: transform 0.15s; display: inline-block;
}}
.p-head.shut .arr {{ transform: rotate(-90deg); }}
.p-head .ttl {{ flex: 1; }}
.p-head .ba {{
    font-size: 9px; padding: 2px 7px; border-radius: 3px;
    border: 1px solid rgba(255,255,255,0.3); color: white;
    background: rgba(255,255,255,0.08); cursor: pointer;
    margin-left: 3px;
}}
.p-head .ba:hover {{ background: rgba(255,255,255,0.22); }}

.p-body {{ padding: 3px 6px 5px; }}
.p-body.shut {{ display: none; }}

.li {{
    display: flex; align-items: center;
    padding: 2.5px 3px; font-size: 11px; color: #333;
    border-radius: 3px; cursor: pointer;
}}
.li:hover {{ background: #eef2ff; }}
.li input {{ margin-right: 5px; cursor: pointer; accent-color: #2c3e50; }}
.sw {{
    display: inline-block; width: 10px; height: 10px;
    border-radius: 2px; margin-right: 5px; flex-shrink: 0;
    border: 1px solid rgba(0,0,0,0.12);
}}
.li .lb {{
    flex: 1; cursor: pointer;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.li.off {{ opacity: 0.4; }}

/* ── Gene Search panel ──────────────────────────────────────────────── */
.search-row {{
    display: flex; gap: 4px; padding: 4px 6px; align-items: center;
}}
.search-row input[type="text"] {{
    flex: 1; font-size: 11px; padding: 4px 6px;
    border: 1px solid #ccc; border-radius: 3px; outline: none;
    font-family: inherit;
}}
.search-row input[type="text"]:focus {{ border-color: #2c3e50; }}
.search-row button {{
    font-size: 10px; padding: 4px 10px; border-radius: 3px;
    border: 1px solid #2c3e50; background: #2c3e50; color: white;
    cursor: pointer; white-space: nowrap;
}}
.search-row button:hover {{ background: #1a252f; }}
.gene-hint {{
    font-size: 9px; color: #888; padding: 0 8px 2px;
    font-style: italic;
}}
.gene-list {{
    max-height: 180px; overflow-y: auto; padding: 0 4px;
    scrollbar-width: thin;
}}
.gene-list::-webkit-scrollbar {{ width: 5px; }}
.gene-list::-webkit-scrollbar-thumb {{ background: #ccc; border-radius: 2px; }}
.gl-item {{
    display: flex; align-items: center; padding: 2px 4px;
    font-size: 10.5px; color: #444; cursor: pointer; border-radius: 3px;
    gap: 4px;
}}
.gl-item:hover {{ background: #e8ecf4; }}
.gl-item.hl {{ background: #d4e2f7; font-weight: 600; }}
.gl-num {{ color: #999; font-size: 9.5px; min-width: 20px; text-align: right; }}
.gl-name {{ flex: 1; }}
.gl-cnt {{ color: #aaa; font-size: 9px; }}
/* ── cutoff slider ───────────────────────────────────────────────── */
.cutoff-row {{
    display: flex; align-items: center; gap: 5px;
    padding: 4px 6px; font-size: 10px; color: #555;
}}
.cutoff-row label {{ white-space: nowrap; }}
.cutoff-row input[type="range"] {{
    flex: 1; height: 4px; cursor: pointer;
    accent-color: #2c3e50;
}}
.cutoff-row .cv {{ font-weight: 600; min-width: 30px; text-align: right; }}

.loaded-sep {{
    font-size: 9px; color: #888; padding: 4px 8px 2px;
    border-top: 1px solid #e0e0e0; margin-top: 4px;
}}
.loaded-item {{
    display: flex; align-items: center; padding: 2px 4px;
    font-size: 10.5px; gap: 4px;
}}
.loaded-item input {{ accent-color: #2c3e50; cursor: pointer; }}
.loaded-item .lname {{ flex: 1; }}
.loaded-item .lx {{
    color: #c00; cursor: pointer; font-size: 12px;
    font-weight: bold; padding: 0 3px;
}}
.loaded-item .lx:hover {{ color: #f00; }}

/* ── Plane Projector panel ─────────────────────────────────────────── */
.proj-row {{
    display: flex; align-items: center; gap: 5px;
    padding: 4px 6px; font-size: 10px; color: #555;
}}
.proj-row label {{ white-space: nowrap; min-width: 60px; }}
.proj-row select {{
    flex: 1; font-size: 10px; padding: 2px 4px;
    border: 1px solid #ccc; border-radius: 3px; outline: none;
    font-family: inherit; background: white;
}}
.proj-row select:focus {{ border-color: #2c3e50; }}
.proj-row input[type="range"] {{
    flex: 1; height: 4px; cursor: pointer; accent-color: #2c3e50;
}}
.proj-row .pv {{ font-weight: 600; min-width: 30px; text-align: right; }}

/* ── Colorscheme selector ──────────────────────────────────────────── */
.cs-panel {{ padding: 5px 6px; }}
.cs-panel .cs-title {{
    font-size: 10px; font-weight: 600; color: #555;
    margin-bottom: 4px;
}}
.cs-swatches {{
    display: flex; flex-direction: column; gap: 3px;
}}
.cs-opt {{
    display: flex; align-items: center; gap: 5px;
    padding: 3px 5px; border-radius: 4px; cursor: pointer;
    border: 2px solid transparent; font-size: 10px; color: #444;
}}
.cs-opt:hover {{ background: #eef2ff; }}
.cs-opt.active {{ border-color: #2c3e50; background: #e8ecf4; font-weight: 600; }}
.cs-opt .cs-bar {{
    width: 50px; height: 10px; border-radius: 2px; flex-shrink: 0;
    border: 1px solid rgba(0,0,0,0.15);
}}
.cs-opt .cs-lbl {{ flex: 1; }}
</style>
</head>
<body>

<div id="header">
    <h1>Hox Morphospace Explorer</h1>
    <span class="sub">Fly Cell Atlas &middot; 38,227 Hox-expressing cells &middot; Possibility space polytope &middot; 15 gene overlays + 200 searchable genes</span>
</div>

<div id="main">
    <div id="sec-3d">
        <div id="chart"></div>
        <div id="panels-3d"></div>
    </div>
    <div id="sec-2d">
        <div id="plane2d">
            <div id="plane2d-placeholder">Select a plane and<br>data source in the<br>Plane Projector panel →</div>
            <div id="plane2d-chart" style="display:none;"></div>
        </div>
        <div id="panels-2d"></div>
    </div>
    <div id="sec-matrix">
        <h3>Hox Expression Matrix</h3>
        {matrix_table_html}
    </div>
</div>

<script>
var defined   = {fig_json_safe};
var panels    = {panels_json};
var coords    = {coords_json};
var geneBank  = {bank_json};
var planesDef = {planes_data_json};
var nBase     = {n_total_traces};  /* index where dynamic traces start */

Plotly.newPlot('chart', defined.data, defined.layout, {{
    responsive: true, scrollZoom: true, displayModeBar: true,
    modeBarButtonsToRemove: ['toImage'],
}});

/* ══════════════════════════════════════════════════════════════════════
   Universal Colorscheme Selector
   ══════════════════════════════════════════════════════════════════════ */
var CS_DEFS = [
    {{ name: 'Jet',      gradient: 'linear-gradient(90deg,#00f,#0ff,#0f0,#ff0,#f00)' }},
    {{ name: 'Viridis',  gradient: 'linear-gradient(90deg,#440154,#31688e,#35b779,#fde725)' }},
    {{ name: 'Inferno',  gradient: 'linear-gradient(90deg,#000004,#932567,#f98e09,#fcffa4)' }},
    {{ name: 'Turbo',    gradient: 'linear-gradient(90deg,#30123b,#28bbec,#a2fc3c,#fb8022,#7a0403)' }},
    {{ name: 'Cividis',  gradient: 'linear-gradient(90deg,#002051,#7b7b78,#fdea45)' }},
];
var activeCS = 'Jet';  /* default */

/* Build the colorscheme selector panel in panels-2d */
(function() {{
    var p2d = document.getElementById('panels-2d');
    var pan = document.createElement('div'); pan.className = 'panel';
    var hd = document.createElement('div'); hd.className = 'p-head';
    hd.innerHTML = '<span class="arr">&#9660;</span><span class="ttl">Colorscheme</span>';
    var bd = document.createElement('div'); bd.className = 'p-body cs-panel';
    bd.id = 'bd_colorscheme';
    hd.addEventListener('click', function() {{
        hd.classList.toggle('shut'); bd.classList.toggle('shut');
    }});

    var sw = document.createElement('div'); sw.className = 'cs-swatches';
    CS_DEFS.forEach(function(cs, ci) {{
        var opt = document.createElement('div');
        opt.className = 'cs-opt' + (cs.name === activeCS ? ' active' : '');
        opt.dataset.cs = cs.name;

        var bar = document.createElement('span'); bar.className = 'cs-bar';
        bar.style.background = cs.gradient;
        var lbl = document.createElement('span'); lbl.className = 'cs-lbl';
        lbl.textContent = cs.name;
        opt.appendChild(bar); opt.appendChild(lbl);

        opt.onclick = function() {{
            activeCS = cs.name;
            /* update active class */
            sw.querySelectorAll('.cs-opt').forEach(function(el) {{
                el.classList.toggle('active', el.dataset.cs === activeCS);
            }});
            /* restyle all visible gene_expr traces */
            restyleAllToCS();
        }};

        sw.appendChild(opt);
    }});
    bd.appendChild(sw);
    pan.appendChild(hd); pan.appendChild(bd);
    p2d.appendChild(pan);
}})();

/* Restyle all gene expression traces + dynamically loaded genes to activeCS */
function restyleAllToCS() {{
    var chart = document.getElementById('chart');
    /* static gene_expr traces */
    var gePan = panels.find(function(p) {{ return p.id === 'gene_expr'; }});
    if (gePan) {{
        gePan.items.forEach(function(it) {{
            it.indices.forEach(function(idx) {{
                Plotly.restyle('chart', {{ 'marker.colorscale': activeCS }}, [idx]);
            }});
        }});
    }}
    /* dynamically loaded search genes */
    var sl = window._geneSearchLoaded || [];
    sl.forEach(function(lg) {{
        lg.cs = activeCS;
        Plotly.restyle('chart', {{ 'marker.colorscale': activeCS }}, [lg.traceIdx]);
    }});
    /* update the plane projector if active */
    if (typeof window._updateProjection === 'function') window._updateProjection();
}}

/* ── build legend panels ─────────────────────────────────────────────── */
var $p = document.getElementById('panels-3d');

panels.forEach(function(pan) {{
    var d = document.createElement('div'); d.className = 'panel';

    /* header */
    var hd = document.createElement('div'); hd.className = 'p-head';
    hd.innerHTML = '<span class="arr">&#9660;</span><span class="ttl">'
                   + pan.title + ' (' + pan.items.length + ')</span>';

    var bA = document.createElement('button'); bA.className = 'ba'; bA.textContent = 'All';
    bA.onclick = function(e) {{ e.stopPropagation(); togAll(pan.id, true); }};
    var bN = document.createElement('button'); bN.className = 'ba'; bN.textContent = 'None';
    bN.onclick = function(e) {{ e.stopPropagation(); togAll(pan.id, false); }};
    hd.appendChild(bA); hd.appendChild(bN);

    var bd = document.createElement('div'); bd.className = 'p-body';
    bd.id = 'bd_' + pan.id;

    hd.addEventListener('click', function() {{
        hd.classList.toggle('shut'); bd.classList.toggle('shut');
    }});

    /* items */
    pan.items.forEach(function(it, ii) {{
        var row = document.createElement('div');
        row.className = 'li' + (it.on ? '' : ' off');
        row.id = 'r_' + pan.id + '_' + ii;

        var cb = document.createElement('input');
        cb.type = 'checkbox'; cb.checked = it.on;
        cb.id = 'c_' + pan.id + '_' + ii;
        cb.onchange = function() {{
            Plotly.restyle('chart', {{visible: cb.checked}}, it.indices);
            row.classList.toggle('off', !cb.checked);
        }};

        var sw = document.createElement('span');
        sw.className = 'sw'; sw.style.background = it.color;
        if (it.color && it.color.indexOf('gradient') >= 0) sw.style.width = '28px';

        var lb = document.createElement('span');
        lb.className = 'lb'; lb.textContent = it.name; lb.title = it.name;
        lb.onclick = function() {{ cb.click(); }};

        row.appendChild(cb); row.appendChild(sw); row.appendChild(lb);
        bd.appendChild(row);
    }});

    d.appendChild(hd); d.appendChild(bd);
    $p.appendChild(d);
}});

/* ── toggle all in a panel ───────────────────────────────────────────── */
function togAll(pid, on) {{
    var pan = panels.find(function(p) {{ return p.id === pid; }});
    if (!pan) return;
    var allIdx = [];
    pan.items.forEach(function(it, ii) {{
        allIdx = allIdx.concat(it.indices);
        var cb = document.getElementById('c_' + pid + '_' + ii);
        if (cb) cb.checked = on;
        var rw = document.getElementById('r_' + pid + '_' + ii);
        if (rw) rw.classList.toggle('off', !on);
    }});
    if (allIdx.length) Plotly.restyle('chart', {{visible: on}}, allIdx);
}}

/* ══════════════════════════════════════════════════════════════════════
   Gene Expression panel – cutoff slider
   ══════════════════════════════════════════════════════════════════════ */
(function() {{
    /* find the gene_expr panel and its body */
    var gePan = panels.find(function(p) {{ return p.id === 'gene_expr'; }});
    if (!gePan) return;
    var geBd = document.getElementById('bd_gene_expr');
    if (!geBd) return;

    /* snapshot original trace data before any filtering */
    var chart = document.getElementById('chart');
    var origData = {{}};
    gePan.items.forEach(function(it) {{
        it.indices.forEach(function(idx) {{
            var d = chart.data[idx];
            origData[idx] = {{
                x: d.x.slice(), y: d.y.slice(), z: d.z.slice(),
                color: d.marker.color.slice(),
                cd: d.customdata.map(function(r) {{ return r.slice(); }})
            }};
        }});
    }});

    /* add cutoff slider at top of panel body */
    var cRow = document.createElement('div'); cRow.className = 'cutoff-row';
    var cLbl = document.createElement('label'); cLbl.textContent = 'Expr cutoff:';
    var cSlider = document.createElement('input');
    cSlider.type = 'range'; cSlider.min = '0'; cSlider.max = '0.95';
    cSlider.step = '0.05'; cSlider.value = '0';
    var cVal = document.createElement('span'); cVal.className = 'cv'; cVal.textContent = '0';
    cRow.appendChild(cLbl); cRow.appendChild(cSlider); cRow.appendChild(cVal);
    geBd.insertBefore(cRow, geBd.firstChild);

    var geCutoff = 0;

    cSlider.oninput = function() {{
        geCutoff = parseFloat(cSlider.value);
        cVal.textContent = geCutoff.toFixed(2);
        applyGEThreshold();
    }};

    function applyGEThreshold() {{
        gePan.items.forEach(function(it) {{
            it.indices.forEach(function(idx) {{
                var od = origData[idx];
                if (!od) return;
                if (geCutoff <= 0) {{
                    /* restore full data */
                    Plotly.restyle('chart', {{
                        x: [od.x], y: [od.y], z: [od.z],
                        'marker.color': [od.color],
                        customdata: [od.cd]
                    }}, [idx]);
                    return;
                }}
                /* filter by threshold — color array holds normalised expression */
                var xF = [], yF = [], zF = [], cF = [], cdF = [];
                for (var k = 0; k < od.color.length; k++) {{
                    if (od.color[k] >= geCutoff) {{
                        xF.push(od.x[k]);
                        yF.push(od.y[k]);
                        zF.push(od.z[k]);
                        cF.push(od.color[k]);
                        cdF.push(od.cd[k]);
                    }}
                }}
                Plotly.restyle('chart', {{
                    x: [xF], y: [yF], z: [zF],
                    'marker.color': [cF],
                    customdata: [cdF]
                }}, [idx]);
            }});
        }});
    }}
}})();

/* ══════════════════════════════════════════════════════════════════════
   Gene Search panel
   ══════════════════════════════════════════════════════════════════════ */
(function() {{
    var curMatches = geneBank.slice();  /* current filtered list */
    var hlIdx = -1;                     /* highlighted index in curMatches */
    var loaded = [];                    /* {{name, traceIdx, hue, n}} — visible items */
    window._geneSearchLoaded = loaded;  /* expose for Plane Projector */
    var hidden = {{}};                   /* geneName → traceIdx  (hidden, not deleted) */
    var dynCount = 0;

    /* ── build panel DOM ──────────────────────────────────────────────── */
    var pan = document.createElement('div'); pan.className = 'panel';

    var hd = document.createElement('div'); hd.className = 'p-head';
    hd.innerHTML = '<span class="arr">&#9660;</span>'
        + '<span class="ttl">Gene Search (' + geneBank.length + ')</span>';
    var bd = document.createElement('div'); bd.className = 'p-body';
    bd.id = 'bd_genesearch';
    hd.addEventListener('click', function() {{
        hd.classList.toggle('shut'); bd.classList.toggle('shut');
    }});

    /* search row */
    var sr = document.createElement('div'); sr.className = 'search-row';
    var inp = document.createElement('input'); inp.type = 'text';
    inp.placeholder = 'Gene name or #...';
    var btn = document.createElement('button'); btn.textContent = 'Load';

    sr.appendChild(inp); sr.appendChild(btn);
    bd.appendChild(sr);

    var hint = document.createElement('div'); hint.className = 'gene-hint';
    hint.textContent = 'Type name to filter, or # to pick by rank';
    bd.appendChild(hint);

    /* cutoff slider */
    var cutoff = 0;
    var cRow = document.createElement('div'); cRow.className = 'cutoff-row';
    var cLbl = document.createElement('label'); cLbl.textContent = 'Expr cutoff:';
    var cSlider = document.createElement('input');
    cSlider.type = 'range'; cSlider.min = '0'; cSlider.max = '0.95';
    cSlider.step = '0.05'; cSlider.value = '0';
    var cVal = document.createElement('span'); cVal.className = 'cv'; cVal.textContent = '0';
    cRow.appendChild(cLbl); cRow.appendChild(cSlider); cRow.appendChild(cVal);
    bd.appendChild(cRow);

    cSlider.oninput = function() {{
        cutoff = parseFloat(cSlider.value);
        cVal.textContent = cutoff.toFixed(2);
        applyThreshold();
    }};

    /* gene list */
    var glist = document.createElement('div'); glist.className = 'gene-list';
    glist.id = 'glist';
    bd.appendChild(glist);

    /* loaded genes area */
    var lsep = document.createElement('div'); lsep.className = 'loaded-sep';
    lsep.textContent = 'Loaded genes';
    lsep.style.display = 'none';
    bd.appendChild(lsep);
    var ldiv = document.createElement('div'); ldiv.id = 'loaded-genes';
    bd.appendChild(ldiv);

    pan.appendChild(hd); pan.appendChild(bd);
    document.getElementById('panels-3d').appendChild(pan);

    /* ── render gene list ─────────────────────────────────────────────── */
    function renderList() {{
        glist.innerHTML = '';
        curMatches.forEach(function(g, i) {{
            var row = document.createElement('div');
            row.className = 'gl-item' + (i === hlIdx ? ' hl' : '');
            row.innerHTML =
                '<span class="gl-num">' + (i + 1) + '</span>' +
                '<span class="gl-name">' + g.name + '</span>' +
                '<span class="gl-cnt">(' + g.n.toLocaleString() + ')</span>';
            row.onclick = function() {{ loadGene(g); }};
            row.onmouseenter = function() {{
                hlIdx = i; renderList();
            }};
            glist.appendChild(row);
        }});
        if (hlIdx >= 0 && hlIdx < curMatches.length) {{
            var hlRow = glist.children[hlIdx];
            if (hlRow) hlRow.scrollIntoView({{ block: 'nearest' }});
        }}
    }}

    function filterGenes(q) {{
        q = q.toLowerCase();
        curMatches = geneBank.filter(function(g) {{
            return g.name.toLowerCase().indexOf(q) >= 0;
        }});
        hlIdx = curMatches.length > 0 ? 0 : -1;
        renderList();
    }}

    /* ── input handling ───────────────────────────────────────────────── */
    inp.addEventListener('input', function() {{
        var v = inp.value.trim();
        if (/^[0-9]+$/.test(v)) {{
            /* pure number: highlight that rank in current matches */
            var num = parseInt(v);
            if (num >= 1 && num <= curMatches.length) {{
                hlIdx = num - 1;
                renderList();
                hint.textContent = '→ ' + curMatches[hlIdx].name
                    + ' (' + curMatches[hlIdx].n.toLocaleString() + ' cells)';
            }} else {{
                hint.textContent = '# out of range (1–' + curMatches.length + ')';
            }}
        }} else if (v.length > 0) {{
            filterGenes(v);
            if (curMatches.length > 0) {{
                hint.textContent = curMatches.length + ' matches';
            }} else {{
                hint.textContent = 'No matches';
            }}
        }} else {{
            curMatches = geneBank.slice();
            hlIdx = -1;
            renderList();
            hint.textContent = 'Type name to filter, or # to pick by rank';
        }}
    }});

    inp.addEventListener('keydown', function(e) {{
        if (e.key === 'ArrowDown') {{
            e.preventDefault();
            if (hlIdx < curMatches.length - 1) {{ hlIdx++; renderList(); }}
            if (hlIdx >= 0) hint.textContent = '→ ' + curMatches[hlIdx].name;
        }} else if (e.key === 'ArrowUp') {{
            e.preventDefault();
            if (hlIdx > 0) {{ hlIdx--; renderList(); }}
            if (hlIdx >= 0) hint.textContent = '→ ' + curMatches[hlIdx].name;
        }} else if (e.key === 'Enter') {{
            e.preventDefault();
            if (hlIdx >= 0 && hlIdx < curMatches.length) {{
                loadGene(curMatches[hlIdx]);
            }}
        }}
    }});

    btn.onclick = function() {{
        if (hlIdx >= 0 && hlIdx < curMatches.length) {{
            loadGene(curMatches[hlIdx]);
        }}
    }};

    /* ── load a gene as a new Plotly trace ─────────────────────────────── */
    function loadGene(g) {{
        /* already visible? */
        for (var i = 0; i < loaded.length; i++) {{
            if (loaded[i].name === g.name) {{
                hint.textContent = g.name + ' already loaded';
                return;
            }}
        }}

        /* was previously loaded then hidden? just re-show it */
        if (g.name in hidden) {{
            var h = hidden[g.name];
            Plotly.restyle('chart', {{visible: true}}, [h.traceIdx]);
            loaded.push({{ name: g.name, traceIdx: h.traceIdx, cs: h.cs, n: g.n, raw: g }});
            delete hidden[g.name];
            hint.textContent = '✓ Re-shown ' + g.name + ' (' + g.n.toLocaleString() + ' cells)';
            applyThresholdSingle(loaded[loaded.length - 1]);
            renderLoaded();
            return;
        }}

        var csName = activeCS;  /* use universal colorscheme */

        /* build coordinate arrays for expressing cells */
        var xArr = [], yArr = [], zArr = [], cArr = [];
        for (var k = 0; k < g.idx.length; k++) {{
            var ci = g.idx[k];
            xArr.push(coords.h1[ci]);
            yArr.push(coords.h2[ci]);
            zArr.push(coords.h3[ci]);
            cArr.push(g.val[k]);
        }}

        var trace = {{
            type: 'scatter3d', mode: 'markers',
            x: xArr, y: yArr, z: zArr,
            marker: {{
                size: 2.5, color: cArr, colorscale: csName,
                cmin: 0, cmax: 1, showscale: false,
                opacity: 1.0
            }},
            name: g.name + ' (' + g.n + ')',
            showlegend: false,
            scene: 'scene',
            hovertemplate:
                '<b>' + g.name + '</b><br>' +
                'expr=%{{customdata[0]:.3f}}<br>' +
                'H1=%{{x:.3f}} H2=%{{y:.3f}} H3=%{{z:.3f}}<extra></extra>',
            customdata: cArr.map(function(v) {{ return [v]; }}),
            visible: true
        }};

        Plotly.addTraces('chart', [trace]);
        /* get the actual index from the live chart */
        var traceIdx = document.getElementById('chart').data.length - 1;

        loaded.push({{ name: g.name, traceIdx: traceIdx, cs: csName, n: g.n, raw: g }});
        hint.textContent = '✓ Loaded ' + g.name + ' [' + csName + '] (' + g.n.toLocaleString() + ' cells)';
        if (cutoff > 0) applyThresholdSingle(loaded[loaded.length - 1]);
        renderLoaded();
    }}

    /* ── render loaded genes ──────────────────────────────────────────── */
    function renderLoaded() {{
        lsep.style.display = loaded.length > 0 ? '' : 'none';
        ldiv.innerHTML = '';
        loaded.forEach(function(lg, li) {{
            var row = document.createElement('div'); row.className = 'loaded-item';

            var cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = true;
            cb.onchange = function() {{
                Plotly.restyle('chart', {{ visible: cb.checked }}, [lg.traceIdx]);
            }};

            /* show swatch for current active colorscheme */
            var csGrad = CS_DEFS.find(function(c) {{ return c.name === activeCS; }});
            var sw = document.createElement('span'); sw.className = 'sw';
            sw.style.background = csGrad ? csGrad.gradient : '#888';
            sw.style.width = '28px';

            var nm = document.createElement('span'); nm.className = 'lname';
            nm.textContent = lg.name + ' (' + lg.n.toLocaleString() + ')';
            nm.style.fontSize = '10.5px';

            var xb = document.createElement('span'); xb.className = 'lx';
            xb.textContent = '×';
            xb.title = 'Remove';
            xb.onclick = function() {{
                /* hide trace (don't delete — avoids index shifting bugs) */
                Plotly.restyle('chart', {{visible: false}}, [lg.traceIdx]);
                hidden[lg.name] = {{ traceIdx: lg.traceIdx, cs: lg.cs }};
                loaded.splice(li, 1);
                renderLoaded();
            }};

            row.appendChild(cb); row.appendChild(sw);
            row.appendChild(nm); row.appendChild(xb);
            ldiv.appendChild(row);
        }});
    }}

    /* ── expression threshold filtering ─────────────────────────────── */
    function applyThresholdSingle(lg) {{
        var g = lg.raw;
        if (!g) return;
        var xArr = [], yArr = [], zArr = [], cArr = [], cdArr = [];
        for (var k = 0; k < g.idx.length; k++) {{
            if (g.val[k] >= cutoff) {{
                var ci = g.idx[k];
                xArr.push(coords.h1[ci]);
                yArr.push(coords.h2[ci]);
                zArr.push(coords.h3[ci]);
                cArr.push(g.val[k]);
                cdArr.push([g.val[k]]);
            }}
        }}
        Plotly.restyle('chart', {{
            x: [xArr], y: [yArr], z: [zArr],
            'marker.color': [cArr],
            customdata: [cdArr]
        }}, [lg.traceIdx]);
    }}

    function applyThreshold() {{
        for (var i = 0; i < loaded.length; i++) {{
            applyThresholdSingle(loaded[i]);
        }}
    }}

    /* initial render */
    renderList();
}})();

/* ══════════════════════════════════════════════════════════════════════
   Plane Projector
   ══════════════════════════════════════════════════════════════════════ */
(function() {{
    /* ── vector helpers ───────────────────────────────────────────────── */
    function vsub(a,b) {{ return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]; }}
    function vdot(a,b) {{ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }}
    function vcross(a,b) {{ return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]; }}
    function vscale(a,s) {{ return [a[0]*s,a[1]*s,a[2]*s]; }}
    function vlen(a) {{ return Math.sqrt(vdot(a,a)); }}
    function vnorm(a) {{ var l=vlen(a); return [a[0]/l,a[1]/l,a[2]/l]; }}

    var selPlane = -1;
    var selSource = '';   /* 'panel:pid:itemIdx' or 'search:geneName' */
    var gridRes = 10;
    var distThresh = 0.5;

    /* ── build panel DOM ─────────────────────────────────────────────── */
    var pan = document.createElement('div'); pan.className = 'panel';

    var hd = document.createElement('div'); hd.className = 'p-head';
    hd.innerHTML = '<span class="arr">&#9660;</span>'
        + '<span class="ttl">Plane Projector</span>';
    var bd = document.createElement('div'); bd.className = 'p-body';
    bd.id = 'bd_projector';
    hd.addEventListener('click', function() {{
        hd.classList.toggle('shut'); bd.classList.toggle('shut');
    }});

    /* plane selector */
    var r1 = document.createElement('div'); r1.className = 'proj-row';
    var l1 = document.createElement('label'); l1.textContent = 'Plane:';
    var selP = document.createElement('select');
    var opt0 = document.createElement('option'); opt0.value = ''; opt0.textContent = '— select plane —';
    selP.appendChild(opt0);
    planesDef.forEach(function(pd, pi) {{
        var o = document.createElement('option'); o.value = pi;
        o.textContent = pd.name; selP.appendChild(o);
    }});
    r1.appendChild(l1); r1.appendChild(selP);
    bd.appendChild(r1);

    /* data source selector */
    var r2 = document.createElement('div'); r2.className = 'proj-row';
    var l2 = document.createElement('label'); l2.textContent = 'Data:';
    var selD = document.createElement('select');
    selD.id = 'proj-data-sel';
    var od0 = document.createElement('option'); od0.value = ''; od0.textContent = '— select data —';
    selD.appendChild(od0);

    /* populate from panels: cell_types and gene_expr */
    var srcPanels = ['cell_types', 'gene_expr'];
    srcPanels.forEach(function(pid) {{
        var p = panels.find(function(pp) {{ return pp.id === pid; }});
        if (!p) return;
        var og = document.createElement('optgroup');
        og.label = p.title;
        p.items.forEach(function(it, ii) {{
            var o = document.createElement('option');
            o.value = 'panel:' + pid + ':' + ii;
            o.textContent = it.name;
            og.appendChild(o);
        }});
        selD.appendChild(og);
    }});

    /* optgroup for dynamically loaded search genes (populated on click) */
    var ogSearch = document.createElement('optgroup');
    ogSearch.label = 'Gene Search (loaded)';
    selD.appendChild(ogSearch);
    selD.addEventListener('mousedown', function() {{
        /* refresh search gene options each time dropdown opens */
        ogSearch.innerHTML = '';
        var sl = window._geneSearchLoaded || [];
        sl.forEach(function(lg) {{
            var o = document.createElement('option');
            o.value = 'search:' + lg.name;
            o.textContent = lg.name + ' (search)';
            ogSearch.appendChild(o);
        }});
    }});

    r2.appendChild(l2); r2.appendChild(selD);
    bd.appendChild(r2);

    /* grid resolution slider */
    var r3 = document.createElement('div'); r3.className = 'proj-row';
    var l3 = document.createElement('label'); l3.textContent = 'Grid:';
    var sGrid = document.createElement('input');
    sGrid.type = 'range'; sGrid.min = '2'; sGrid.max = '30';
    sGrid.step = '1'; sGrid.value = '10';
    var vGrid = document.createElement('span'); vGrid.className = 'pv'; vGrid.textContent = '10';
    r3.appendChild(l3); r3.appendChild(sGrid); r3.appendChild(vGrid);
    bd.appendChild(r3);

    /* distance threshold slider */
    var r4 = document.createElement('div'); r4.className = 'proj-row';
    var l4 = document.createElement('label'); l4.textContent = 'Dist max:';
    var sDist = document.createElement('input');
    sDist.type = 'range'; sDist.min = '0.05'; sDist.max = '2.0';
    sDist.step = '0.05'; sDist.value = '0.50';
    var vDist = document.createElement('span'); vDist.className = 'pv'; vDist.textContent = '0.50';
    r4.appendChild(l4); r4.appendChild(sDist); r4.appendChild(vDist);
    bd.appendChild(r4);

    /* info line */
    var info = document.createElement('div'); info.className = 'gene-hint';
    info.textContent = 'Select a plane and data source above.';
    bd.appendChild(info);

    pan.appendChild(hd); pan.appendChild(bd);
    document.getElementById('panels-2d').appendChild(pan);

    /* ── event handlers ──────────────────────────────────────────────── */
    selP.onchange = function() {{
        selPlane = selP.value === '' ? -1 : parseInt(selP.value);
        updateProjection();
    }};
    selD.onchange = function() {{
        selSource = selD.value;
        updateProjection();
    }};
    sGrid.oninput = function() {{
        gridRes = parseInt(sGrid.value);
        vGrid.textContent = gridRes;
        updateProjection();
    }};
    sDist.oninput = function() {{
        distThresh = parseFloat(sDist.value);
        vDist.textContent = distThresh.toFixed(2);
        updateProjection();
    }};

    /* ── projection logic ────────────────────────────────────────────── */
    var chart2d = document.getElementById('plane2d-chart');
    var placeholder = document.getElementById('plane2d-placeholder');
    var chart3d = document.getElementById('chart');

    window._updateProjection = updateProjection;
    function updateProjection() {{
        if (selPlane < 0) {{
            chart2d.style.display = 'none';
            placeholder.style.display = '';
            return;
        }}
        placeholder.style.display = 'none';
        chart2d.style.display = '';

        var pd = planesDef[selPlane];
        var apex = pd.apex, base = pd.base, left = pd.left, right = pd.right;

        /* build orthonormal basis */
        var e1 = vnorm(vsub(right, left));
        var e2_raw = vsub(base, apex);
        e2_raw = vsub(e2_raw, vscale(e1, vdot(e2_raw, e1)));
        var e2 = vnorm(e2_raw);
        var normal = vnorm(vcross(e1, e2));

        function to2d(pt) {{
            var off = vsub(pt, apex);
            return [vdot(off, e1), vdot(off, e2)];
        }}

        /* diamond corners in 2D */
        var cA = to2d(apex), cL = to2d(left), cB = to2d(base), cR = to2d(right);

        /* ── background: all cells within distance ───────────────────── */
        var bgX = [], bgY = [];
        for (var i = 0; i < coords.h1.length; i++) {{
            var off = [coords.h1[i]-apex[0], coords.h2[i]-apex[1], coords.h3[i]-apex[2]];
            var d = Math.abs(vdot(off, normal));
            if (d <= distThresh) {{
                bgX.push(vdot(off, e1));
                bgY.push(vdot(off, e2));
            }}
        }}

        /* ── foreground: selected data source ────────────────────────── */
        var fgX = [], fgY = [], fgC = [], fgText = [];
        var fgIsGene = false;
        var fgColor = '#e74c3c';
        var fgCS = activeCS;
        var fgName = 'Data';

        if (selSource) {{
            var parts = selSource.split(':');
            if (parts[0] === 'panel') {{
                var pid = parts[1], idx = parseInt(parts[2]);
                var p = panels.find(function(pp) {{ return pp.id === pid; }});
                if (p && p.items[idx]) {{
                    var it = p.items[idx];
                    fgName = it.name;
                    /* read data from all trace indices of this item */
                    it.indices.forEach(function(ti) {{
                        var td = chart3d.data[ti];
                        if (!td || td.type !== 'scatter3d') return;
                        var xs = td.x || [], ys = td.y || [], zs = td.z || [];
                        /* detect colorscale (gene) vs single color */
                        var mc = td.marker ? td.marker.color : null;
                        var hasCS = td.marker && td.marker.colorscale;
                        if (hasCS && Array.isArray(mc)) {{
                            fgIsGene = true;
                            if (typeof td.marker.colorscale === 'string') fgCS = td.marker.colorscale;
                        }} else if (typeof mc === 'string') {{
                            fgColor = mc;
                        }}
                        /* use the ORIGINAL full data if available (before cutoff filtering) */
                        for (var k = 0; k < xs.length; k++) {{
                            var off = [xs[k]-apex[0], ys[k]-apex[1], zs[k]-apex[2]];
                            var d = Math.abs(vdot(off, normal));
                            if (d <= distThresh) {{
                                fgX.push(vdot(off, e1));
                                fgY.push(vdot(off, e2));
                                if (fgIsGene && Array.isArray(mc)) fgC.push(mc[k]);
                                fgText.push(fgName);
                            }}
                        }}
                    }});
                }}
            }} else if (parts[0] === 'search') {{
                var gname = parts.slice(1).join(':');
                var sl = window._geneSearchLoaded || [];
                var lg = null;
                for (var j = 0; j < sl.length; j++) {{
                    if (sl[j].name === gname) {{ lg = sl[j]; break; }}
                }}
                if (lg) {{
                    fgName = lg.name;
                    fgIsGene = true;
                    fgCS = activeCS;
                    var td = chart3d.data[lg.traceIdx];
                    if (td) {{
                        /* use raw data from gene bank for full unfiltered set */
                        var raw = lg.raw;
                        if (raw) {{
                            for (var k = 0; k < raw.idx.length; k++) {{
                                var ci = raw.idx[k];
                                var off = [coords.h1[ci]-apex[0], coords.h2[ci]-apex[1], coords.h3[ci]-apex[2]];
                                var d = Math.abs(vdot(off, normal));
                                if (d <= distThresh) {{
                                    fgX.push(vdot(off, e1));
                                    fgY.push(vdot(off, e2));
                                    fgC.push(raw.val[k]);
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}

        /* ── diamond grid lines ──────────────────────────────────────── */
        function lerp2(a, b, t) {{ return [a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])]; }}
        var gridXs = [], gridYs = [];

        /* set 1: lines parallel to apex→base, at intervals along apex→left / apex→right */
        for (var s = 1; s < gridRes; s++) {{
            var t = s / gridRes;
            /* left side: from lerp(apex,left,t) to lerp(base,left,t) */
            var p1L = lerp2(cA, cL, t), p2L = lerp2(cB, cL, t);
            gridXs.push(p1L[0], p2L[0], null); gridYs.push(p1L[1], p2L[1], null);
            /* right side: from lerp(apex,right,t) to lerp(base,right,t) */
            var p1R = lerp2(cA, cR, t), p2R = lerp2(cB, cR, t);
            gridXs.push(p1R[0], p2R[0], null); gridYs.push(p1R[1], p2R[1], null);
        }}
        /* central line */
        gridXs.push(cA[0], cB[0], null); gridYs.push(cA[1], cB[1], null);

        /* set 2: cross lines (parallel to left→right) at intervals along apex→base */
        for (var s = 1; s < gridRes; s++) {{
            var t = s / gridRes;
            /* from lerp(apex,left,t) + lerp adjustment to lerp(apex,right,t) */
            var pL = lerp2(cA, cL, t);
            var pR = lerp2(cA, cR, t);
            /* extend to base side: lerp(base,left,t) ↔ lerp(base,right,t) */
            /* cross line at fraction t along apex→base:
               left end = lerp( lerp(apex,left,s), lerp(base,left,s), ... ) — hmm
               Simpler: at fraction t along the central axis, the cross line spans
               from the left edge to the right edge */
            var leftEdge1 = lerp2(cA, cL, 1);   /* = cL */
            var leftEdge2 = lerp2(cB, cL, 1);   /* = cL, no — need parameterize differently */
            /* The diamond edges: apex→left, left→base, base→right, right→apex
               At fraction t from apex toward base:
               - left boundary = lerp(apex→left @ t) if t<1, or lerp(left→base @ t-...) */
            /* Simplify: treat diamond as two triangles. Upper: apex-left-right. Lower: left-base-right.
               At fraction t (0=apex, 1=base), in upper half (t<=0.5 mapped):
               Actually, parametrize by e2 coordinate. */
        }}

        /* Better approach for cross lines: use the diamond parameterization directly.
           For fraction t along apex→base direction:
           - the left point is on edge apex→left (if in upper half) or left→base (lower half)
           - the right point is on edge apex→right (if in upper half) or right→base (lower half)
           Diamond: apex(top), left, base(bottom), right.
           Upper half = apex→left + apex→right, lower half = left→base + right→base. */
        /* Actually cleaner: just parametrize by e2 fraction from apex to base */
        var e2_apex = cA[1], e2_base = cB[1];
        /* Clear old cross lines attempt and redo */
        gridXs = []; gridYs = [];

        /* Set 1: lines parallel to central line (apex→base direction) */
        /* Along the left side of diamond */
        for (var s = 1; s < gridRes; s++) {{
            var t = s / gridRes;
            var p1 = lerp2(cA, cL, t), p2 = lerp2(cB, cL, t);
            gridXs.push(p1[0], p2[0], null); gridYs.push(p1[1], p2[1], null);
        }}
        /* Along the right side */
        for (var s = 1; s < gridRes; s++) {{
            var t = s / gridRes;
            var p1 = lerp2(cA, cR, t), p2 = lerp2(cB, cR, t);
            gridXs.push(p1[0], p2[0], null); gridYs.push(p1[1], p2[1], null);
        }}
        /* central line */
        gridXs.push(cA[0], cB[0], null); gridYs.push(cA[1], cB[1], null);

        /* Set 2: cross lines — parallel to left↔right at intervals along apex→base.
           At fraction t: left edge point = lerp(apex, left, t) going to lerp(apex, right, t)
           AND continuing the "lower" portion: lerp(left, base, t) to lerp(right, base, t) */
        for (var s = 1; s < gridRes; s++) {{
            var t = s / gridRes;
            /* upper cross: connects point on apex→left with point on apex→right */
            var pL = lerp2(cA, cL, t);
            var pR = lerp2(cA, cR, t);
            gridXs.push(pL[0], pR[0], null); gridYs.push(pL[1], pR[1], null);
            /* lower cross: connects point on left→base with point on right→base */
            var qL = lerp2(cL, cB, t);
            var qR = lerp2(cR, cB, t);
            gridXs.push(qL[0], qR[0], null); gridYs.push(qL[1], qR[1], null);
        }}

        /* ── build 2D plot traces ────────────────────────────────────── */
        var pc = pd.color;
        var pcStr = 'rgba(' + Math.round(pc[0]*255) + ',' + Math.round(pc[1]*255) + ','
                    + Math.round(pc[2]*255) + ',';
        var traces = [];

        /* background cells */
        traces.push({{
            x: bgX, y: bgY, mode: 'markers', type: 'scatter',
            marker: {{ size: 1.5, color: '#ddd', opacity: 0.4 }},
            name: 'All cells', hoverinfo: 'skip', showlegend: false
        }});

        /* grid lines */
        traces.push({{
            x: gridXs, y: gridYs, mode: 'lines', type: 'scatter',
            line: {{ color: pcStr + '0.25)', width: 0.8 }},
            name: 'Grid', hoverinfo: 'skip', showlegend: false
        }});

        /* diamond outline */
        traces.push({{
            x: [cA[0], cL[0], cB[0], cR[0], cA[0]],
            y: [cA[1], cL[1], cB[1], cR[1], cA[1]],
            mode: 'lines', type: 'scatter',
            line: {{ color: pcStr + '0.9)', width: 2.5 }},
            name: 'Plane', hoverinfo: 'skip', showlegend: false
        }});

        /* vertex labels: name format is "C*_V3_V2_V1_V7_s0" = apex_base_left_right_shift_pct */
        var vlbls = pd.name.split('_');
        /* vlbls: [apex, base, left, right, shiftTarget, sPct] */
        var vertNames = [vlbls[0] || 'Apex', vlbls[2] || 'Left', vlbls[1] || 'Base', vlbls[3] || 'Right'];
        traces.push({{
            x: [cA[0], cL[0], cB[0], cR[0]],
            y: [cA[1], cL[1], cB[1], cR[1]],
            mode: 'markers+text', type: 'scatter',
            marker: {{ size: 6, color: pcStr + '1)', symbol: 'diamond' }},
            text: vertNames,
            textposition: ['top center', 'middle left', 'bottom center', 'middle right'],
            textfont: {{ size: 9, color: '#333' }},
            name: 'Vertices', hoverinfo: 'skip', showlegend: false
        }});

        /* foreground data */
        if (fgX.length > 0) {{
            var fgTrace = {{
                x: fgX, y: fgY, mode: 'markers', type: 'scatter',
                name: fgName, showlegend: false,
                hovertemplate: '<b>' + fgName + '</b><br>e1=%{{x:.3f}} e2=%{{y:.3f}}<extra></extra>'
            }};
            if (fgIsGene && fgC.length > 0) {{
                fgTrace.marker = {{
                    size: 3, color: fgC, colorscale: fgCS,
                    cmin: 0, cmax: 1, showscale: true, opacity: 1.0,
                    colorbar: {{ len: 0.4, thickness: 10, x: 1.02,
                                 title: {{ text: 'expr', font: {{ size: 9 }} }} }}
                }};
            }} else {{
                fgTrace.marker = {{ size: 3, color: fgColor, opacity: 0.7 }};
            }}
            traces.push(fgTrace);
        }}

        /* ── layout ──────────────────────────────────────────────────── */
        var layout2d = {{
            margin: {{ l: 35, r: 50, t: 30, b: 35 }},
            paper_bgcolor: 'white', plot_bgcolor: '#fafafa',
            title: {{ text: fgName + ' → ' + pd.name, font: {{ size: 11 }}, x: 0.5 }},
            xaxis: {{
                title: {{ text: 'e1 (lateral)', font: {{ size: 9 }} }},
                scaleanchor: 'y', scaleratio: 1,
                showgrid: false, zeroline: false,
                tickfont: {{ size: 8 }}
            }},
            yaxis: {{
                title: {{ text: 'e2 (central)', font: {{ size: 9 }} }},
                showgrid: false, zeroline: false,
                tickfont: {{ size: 8 }}
            }},
            showlegend: false,
            hovermode: 'closest'
        }};

        info.textContent = fgX.length > 0
            ? fgName + ': ' + fgX.length + ' cells within dist ' + distThresh.toFixed(2)
                + ' | bg: ' + bgX.length + ' cells'
            : bgX.length + ' cells within dist ' + distThresh.toFixed(2);

        Plotly.newPlot('plane2d-chart', traces, layout2d, {{
            responsive: true, displayModeBar: false
        }});
    }}
}})();
</script>
</body>
</html>'''

out = f"{OUTPUT_DIR}/hox_morphospace_explorer.html"
with open(out, 'w', encoding='utf-8') as fout:
    fout.write(html)
print(f"\nSaved: {out}")
print("Done!")
