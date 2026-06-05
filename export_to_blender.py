"""
Export Hox morphospace data for Blender import.
Outputs:
  - blender_cells.csv        (cell coordinates, colors, labels, h5ad row index)
  - blender_polytope.csv     (polytope vertices)
  - blender_polytope_edges.csv (edge pairs)
  - blender_lines.csv        (named lines with start/end coords + color)
  - blender_import.py        (Blender Python script to rebuild the scene)
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull
from itertools import combinations
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent
H5AD_PATH  = str(Path(__file__).parent / "s_fca_biohub_body_10x.h5ad")
HOX_GENES = ['lab', 'pb', 'Dfd', 'Scr', 'Antp', 'Ubx', 'abd-A', 'Abd-B']

# ── Load and compute (same as generate_morphospace_hoxaxes.py) ──────────────
print("Loading ...")
df = pd.read_csv(OUTPUT_DIR / "hox_morphospace_boxcentered.csv")
U = df['U'].values; V = df['V'].values; W = df['W'].values
labels = df['leiden_label'].values.copy()
broad = df['broad'].values

# Cluster splits (same as main script)
for i in np.where(labels==20)[0]:
    if W[i]<-0.22: labels[i]=30
    elif W[i]>0.05: labels[i]=31
    else: labels[i]=32
for i in np.where(labels==7)[0]:
    if W[i]<-0.13: labels[i]=33
    elif W[i]>-0.10: labels[i]=34
    else: labels[i]=35
for cl in [12,15,16,27]: labels[labels==cl]=36

print("Loading h5ad ...")
adata = sc.read_h5ad(H5AD_PATH)
gn = list(adata.var_names)
hox_idx = [gn.index(g) for g in HOX_GENES]
X_raw = adata.X; X_raw = X_raw.toarray() if hasattr(X_raw,'toarray') else X_raw
hox_mask = X_raw[:,hox_idx].sum(1)>0
h5ad_row_indices = np.where(hox_mask)[0]
X_hox = X_raw[hox_mask][:,hox_idx]
scaler = MinMaxScaler()
X_hox_norm = scaler.fit_transform(X_hox)

Antp = X_hox_norm[:,4]; Ubx = X_hox_norm[:,5]
abdA = X_hox_norm[:,6]; AbdB = X_hox_norm[:,7]
H1 = Antp - Ubx
H2 = abdA - AbdB
H3 = (Antp + Ubx) - (abdA + AbdB)

# ── Color assignment (same logic as main script) ───────────────────────────
def hex_to_rgb(h):
    h = h.lstrip('#')
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def lighten_hex(h, factor=0.45):
    r,g,b = hex_to_rgb(h)
    return f'#{int(r+(255-r)*factor):02x}{int(g+(255-g)*factor):02x}{int(b+(255-b)*factor):02x}'

def darken_hex(h, factor=0.3):
    r,g,b = hex_to_rgb(h)
    return f'#{int(r*(1-factor)):02x}{int(g*(1-factor)):02x}{int(b*(1-factor)):02x}'

def mean_h1(cl): return H1[labels==cl].mean()

PAIRS_RAW = [
    (4,11,'#0000CD'),(5,0,'#CC0000'),(26,18,'#006400'),
    (1,36,'#C71585'),(13,8,'#7B2D8E'),(2,9,'#CC8400'),
    (6,10,'#008B8B'),(14,19,'#D2691E'),
]
TRIS = [(30,31,32,'#FF1493'),(33,34,35,'#9ACD32')]
UNMATCHED_COLOR = '#888888'

color_map = {}
for a,b,color in PAIRS_RAW:
    pos,neg = (a,b) if mean_h1(a) >= mean_h1(b) else (b,a)
    color_map[pos] = darken_hex(color, 0.15)
    color_map[neg] = lighten_hex(color, 0.40)

for neg_l,pos_l,ctr_l,color in TRIS:
    trips = [(neg_l,mean_h1(neg_l),'neg'),(pos_l,mean_h1(pos_l),'pos'),(ctr_l,mean_h1(ctr_l),'ctr')]
    for lbl,mw,side in trips:
        if side=='ctr': color_map[lbl] = color
        else: color_map[lbl] = lighten_hex(color, 0.40)

all_labels_set = sorted(set(labels))
for cl in all_labels_set:
    if cl not in color_map: color_map[cl] = UNMATCHED_COLOR

cell_colors = [color_map[l] for l in labels]

# ── 1. Export cells ─────────────────────────────────────────────────────────
print("Exporting cells ...")
cells_df = pd.DataFrame({
    'h5ad_row': h5ad_row_indices,
    'H1': H1, 'H2': H2, 'H3': H3,
    'U': U, 'V': V, 'W': W,
    'leiden': labels,
    'broad': broad,
    'color_hex': cell_colors,
    'Antp': Antp, 'Ubx': Ubx, 'abdA': abdA, 'AbdB': AbdB,
    'lab': X_hox_norm[:,0], 'pb': X_hox_norm[:,1],
    'Dfd': X_hox_norm[:,2], 'Scr': X_hox_norm[:,3],
})
cells_df.to_csv(OUTPUT_DIR / "blender_cells.csv", index=False)
print(f"  {len(cells_df)} cells → blender_cells.csv")

# ── 2. Export polytope ──────────────────────────────────────────────────────
print("Computing polytope ...")
n_grid = 25
gv = np.linspace(0, 1, n_grid)
gAntp,gUbx,gabdA,gAbdB = np.meshgrid(gv,gv,gv,gv, indexing='ij')
gAntp=gAntp.ravel(); gUbx=gUbx.ravel(); gabdA=gabdA.ravel(); gAbdB=gAbdB.ravel()
pts_h = np.column_stack([gAntp-gUbx, gabdA-gAbdB, (gAntp+gUbx)-(gabdA+gAbdB)])
hull = ConvexHull(pts_h)
ps_verts = pts_h[hull.vertices]
ps_vert_map = {old:new for new,old in enumerate(hull.vertices)}

# Vertex data
vert_genes = []
for oidx in hull.vertices:
    vert_genes.append({
        'vertex_id': f'V{len(vert_genes)+1}',
        'H1': pts_h[oidx,0], 'H2': pts_h[oidx,1], 'H3': pts_h[oidx,2],
        'Antp': gAntp[oidx], 'Ubx': gUbx[oidx], 'abdA': gabdA[oidx], 'AbdB': gAbdB[oidx],
    })
vert_df = pd.DataFrame(vert_genes)
vert_df.to_csv(OUTPUT_DIR / "blender_polytope.csv", index=False)
print(f"  {len(vert_df)} vertices → blender_polytope.csv")

# True edges (coplanar merging)
face_normals = []
fi_list, fj_list, fk_list = [], [], []
for tri in hull.simplices:
    mapped = [ps_vert_map.get(v) for v in tri]
    if all(m is not None for m in mapped):
        fi_list.append(mapped[0]); fj_list.append(mapped[1]); fk_list.append(mapped[2])
        v0,v1,v2 = ps_verts[mapped[0]], ps_verts[mapped[1]], ps_verts[mapped[2]]
        n = np.cross(v1-v0, v2-v0)
        nl = np.linalg.norm(n)
        face_normals.append(n/nl if nl>1e-12 else n)

def normal_key(n):
    n = n.copy()
    for i in range(3):
        if abs(n[i]) > 0.1:
            if n[i] < 0: n = -n
            break
    return tuple(np.round(n, 2))

face_groups = {}
for fi_idx in range(len(fi_list)):
    key = normal_key(face_normals[fi_idx])
    face_groups.setdefault(key, []).append(fi_idx)

edge_to_faces = {}
for fi_idx in range(len(fi_list)):
    verts = [fi_list[fi_idx], fj_list[fi_idx], fk_list[fi_idx]]
    for a,b in combinations(verts,2):
        edge = (min(a,b),max(a,b))
        edge_to_faces.setdefault(edge,[]).append(fi_idx)

true_edges = []
for edge, face_list in edge_to_faces.items():
    if len(face_list) >= 2:
        keys = set(normal_key(face_normals[fi]) for fi in face_list)
        if len(keys) > 1:
            true_edges.append(edge)
    else:
        true_edges.append(edge)

edges_df = pd.DataFrame(true_edges, columns=['v_start','v_end'])
edges_df.to_csv(OUTPUT_DIR / "blender_polytope_edges.csv", index=False)
print(f"  {len(edges_df)} true edges → blender_polytope_edges.csv")

# ── 3. Export named lines ───────────────────────────────────────────────────
print("Exporting lines ...")
center = np.array([0., 0., 0.])
data_c = np.array([-0.121, -0.014, 0.248])
v1=ps_verts[0]; v2=ps_verts[1]; v4=ps_verts[3]; v5=ps_verts[4]
v6=ps_verts[5]; v8=ps_verts[7]; v12=ps_verts[11]

lines = []
def add_line(name, p1, p2, color, group=''):
    lines.append({
        'name': name, 'group': group,
        'x1':p1[0],'y1':p1[1],'z1':p1[2],
        'x2':p2[0],'y2':p2[1],'z2':p2[2],
        'color_hex': color,
    })

# Geometric center axes
add_line('V4-C (dark green)', v4, center, '#00A000', 'geom_center_h1')
add_line('C-V8 (dark green)', center, v8, '#00A000', 'geom_center_h1')
add_line('V1-C (dark orange)', v1, center, '#E68C00', 'geom_center_h2')
add_line('C-V2 (dark orange)', center, v2, '#E68C00', 'geom_center_h2')

# Data center axes — green fan (3 levels)
for ti, t in enumerate([0.0, 0.10, 0.40]):
    v4t = v4 + t*(v12-v4); v8t = v8 + t*(v12-v8)
    shade = max(30, 100-ti*35)
    col = f'#{shade:02x}{min(220,150+ti*35):02x}{shade:02x}'
    add_line(f'V4→V12(t={t})-C*', v4t, data_c, col, f'data_h1_t{t}')
    add_line(f'C*-V8→V12(t={t})', data_c, v8t, col, f'data_h1_t{t}')

# Data center — orange
add_line('V1-C* (light orange)', v1, data_c, '#FFBE50', 'data_h2')
add_line('C*-V2 (light orange)', data_c, v2, '#FFBE50', 'data_h2')

# Data center — blue (shifted toward V5/V6)
v1_b = v1 + 0.10*(v5-v1); v2_b = v2 + 0.10*(v6-v2)
add_line('V1→V5(t=0.1)-C* (blue)', v1_b, data_c, '#508CFF', 'data_h2_blue')
add_line('C*-V2→V6(t=0.1) (blue)', data_c, v2_b, '#508CFF', 'data_h2_blue')

lines_df = pd.DataFrame(lines)
lines_df.to_csv(OUTPUT_DIR / "blender_lines.csv", index=False)
print(f"  {len(lines_df)} line segments → blender_lines.csv")

# ── 4. Generate Blender import script ───────────────────────────────────────
print("Writing Blender import script ...")
blender_script = r'''"""
Blender Import Script — Hox Morphospace
Run this inside Blender: Scripting tab → Open → Run Script
Tested with Blender 3.x / 4.x

After import you can:
  - Draw new lines: select two vertices, Shift+click to make edge, name the object
  - Save lines back: run save_lines_to_csv() from the Blender Python console
"""

import bpy
import bmesh
import csv
import os
from mathutils import Vector

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(bpy.data.filepath) or r"."
CELL_SIZE = 0.008          # radius of each cell sphere instance
POLYTOPE_EDGE_WIDTH = 0.006
LINE_WIDTH = 0.008
VERTEX_SIZE = 0.025

# ── Helpers ─────────────────────────────────────────────────────────────────
def hex_to_rgb(h):
    h = h.lstrip('#')
    return (int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255, 1.0)

def get_or_create_material(name, hex_color):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = hex_to_rgb(hex_color)
        bsdf.inputs['Roughness'].default_value = 0.8
    return mat

def create_curve_line(name, points, hex_color, width=0.005):
    """Create a bevel curve between points."""
    curve_data = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = width
    curve_data.bevel_resolution = 4

    spline = curve_data.splines.new('POLY')
    spline.points.add(len(points) - 1)
    for i, p in enumerate(points):
        spline.points[i].co = (p[0], p[1], p[2], 1)

    obj = bpy.data.objects.new(name, curve_data)
    mat = get_or_create_material(f"mat_{name}", hex_color)
    obj.data.materials.append(mat)
    bpy.context.collection.objects.link(obj)
    return obj

# ── Clear scene ─────────────────────────────────────────────────────────────
print("Clearing scene ...")
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ── 1. Import cells ─────────────────────────────────────────────────────────
print("Importing cells ...")
cells_path = os.path.join(DATA_DIR, "blender_cells.csv")

# Group cells by color for instancing
color_groups = {}
with open(cells_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        color = row['color_hex']
        pos = (float(row['H1']), float(row['H2']), float(row['H3']))
        leiden = row['leiden']
        h5ad_row = row['h5ad_row']
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append((pos, leiden, h5ad_row))

print(f"  {sum(len(v) for v in color_groups.values())} cells in {len(color_groups)} color groups")

# Create instancer mesh per color group (MUCH faster than individual objects)
cells_collection = bpy.data.collections.new("Cells")
bpy.context.scene.collection.children.link(cells_collection)

# Base sphere for instancing
bpy.ops.mesh.primitive_ico_sphere_add(radius=CELL_SIZE, subdivisions=2)
base_sphere = bpy.context.active_object
base_sphere.name = "_cell_template"
base_sphere.hide_set(True)
base_sphere.hide_render = True

for color_hex, cell_list in color_groups.items():
    # Create a mesh with one vertex per cell
    mesh = bpy.data.meshes.new(f"cells_{color_hex}")
    verts = [c[0] for c in cell_list]
    mesh.from_pydata(verts, [], [])
    mesh.update()

    obj = bpy.data.objects.new(f"cells_{color_hex}", mesh)
    cells_collection.objects.link(obj)

    # Material
    mat = get_or_create_material(f"mat_cell_{color_hex}", color_hex)
    obj.data.materials.append(mat)

    # Instance base sphere on vertices
    obj.instance_type = 'VERTS'
    base_copy = base_sphere.copy()
    base_copy.data = base_sphere.data.copy()
    base_copy.name = f"_sphere_{color_hex}"
    base_copy.parent = obj
    mat_copy = get_or_create_material(f"mat_sphere_{color_hex}", color_hex)
    base_copy.data.materials.clear()
    base_copy.data.materials.append(mat_copy)
    cells_collection.objects.link(base_copy)

print("  Cells imported via vertex instancing")

# ── 2. Import polytope ──────────────────────────────────────────────────────
print("Importing polytope ...")
polytope_collection = bpy.data.collections.new("Polytope")
bpy.context.scene.collection.children.link(polytope_collection)

# Vertices
verts_path = os.path.join(DATA_DIR, "blender_polytope.csv")
poly_verts = []
poly_labels = []
with open(verts_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        poly_verts.append((float(row['H1']), float(row['H2']), float(row['H3'])))
        poly_labels.append(row['vertex_id'])

# Vertex markers
for i, (pos, label) in enumerate(zip(poly_verts, poly_labels)):
    bpy.ops.mesh.primitive_ico_sphere_add(radius=VERTEX_SIZE, location=pos, subdivisions=2)
    obj = bpy.context.active_object
    obj.name = f"polytope_{label}"
    mat = get_or_create_material("mat_polytope_vertex", "#111111")
    obj.data.materials.append(mat)
    # Move to polytope collection
    bpy.context.scene.collection.objects.unlink(obj)
    polytope_collection.objects.link(obj)

    # Text label
    bpy.ops.object.text_add(location=(pos[0], pos[1], pos[2] + 0.04))
    txt = bpy.context.active_object
    txt.data.body = label
    txt.data.size = 0.04
    txt.name = f"label_{label}"
    bpy.context.scene.collection.objects.unlink(txt)
    polytope_collection.objects.link(txt)

# Edges
edges_path = os.path.join(DATA_DIR, "blender_polytope_edges.csv")
with open(edges_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        a = int(row['v_start']); b = int(row['v_end'])
        pa = poly_verts[a]; pb = poly_verts[b]
        create_curve_line(f"polytope_edge_{a}_{b}", [pa, pb], "#282828", POLYTOPE_EDGE_WIDTH)
        # Move to collection
        obj = bpy.context.scene.collection.objects.get(f"polytope_edge_{a}_{b}")
        if obj:
            bpy.context.scene.collection.objects.unlink(obj)
            polytope_collection.objects.link(obj)

print(f"  {len(poly_verts)} vertices, edges imported")

# ── 3. Import named lines ──────────────────────────────────────────────────
print("Importing lines ...")
lines_collection = bpy.data.collections.new("Lines")
bpy.context.scene.collection.children.link(lines_collection)

lines_path = os.path.join(DATA_DIR, "blender_lines.csv")
with open(lines_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        p1 = (float(row['x1']), float(row['y1']), float(row['z1']))
        p2 = (float(row['x2']), float(row['y2']), float(row['z2']))
        name = row['name']
        color = row['color_hex']
        obj = create_curve_line(name, [p1, p2], color, LINE_WIDTH)
        bpy.context.scene.collection.objects.unlink(obj)
        lines_collection.objects.link(obj)

print("  Lines imported")

# ── 4. Import center markers ───────────────────────────────────────────────
centers_collection = bpy.data.collections.new("Centers")
bpy.context.scene.collection.children.link(centers_collection)

for name, pos, color in [
    ("Center_C", (0, 0, 0), "#FFFFFF"),
    ("Center_C_star", (-0.121, -0.014, 0.248), "#FFFF00"),
]:
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.02, location=pos, subdivisions=3)
    obj = bpy.context.active_object
    obj.name = name
    mat = get_or_create_material(f"mat_{name}", color)
    obj.data.materials.append(mat)
    bpy.context.scene.collection.objects.unlink(obj)
    centers_collection.objects.link(obj)

# ── 5. Camera and lighting ─────────────────────────────────────────────────
bpy.ops.object.light_add(type='SUN', location=(2, -2, 3))
sun = bpy.context.active_object
sun.data.energy = 3

bpy.ops.object.camera_add(location=(3, -3, 2))
cam = bpy.context.active_object
cam.rotation_euler = (1.1, 0, 0.8)
bpy.context.scene.camera = cam

# ── Utility: save user-created lines back to CSV ───────────────────────────
def save_lines_to_csv(filepath=None):
    """
    Call from Blender Python console:
        exec(open('blender_import.py').read())
        save_lines_to_csv()

    Saves all objects in the 'Lines' collection to blender_lines.csv.
    To add new lines: create curve objects in the Lines collection.
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "blender_lines.csv")

    lines_col = bpy.data.collections.get("Lines")
    if not lines_col:
        print("No 'Lines' collection found!")
        return

    rows = []
    for obj in lines_col.objects:
        if obj.type != 'CURVE':
            continue
        spline = obj.data.splines[0]
        pts = [p.co[:3] for p in spline.points]
        if len(pts) < 2:
            continue
        # Get color from material
        color = '#888888'
        if obj.data.materials:
            mat = obj.data.materials[0]
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                c = bsdf.inputs['Base Color'].default_value
                color = f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}'

        # For multi-segment lines, save each segment
        for i in range(len(pts)-1):
            rows.append({
                'name': obj.name,
                'group': obj.name.split('(')[0].strip() if '(' in obj.name else obj.name,
                'x1': pts[i][0], 'y1': pts[i][1], 'z1': pts[i][2],
                'x2': pts[i+1][0], 'y2': pts[i+1][1], 'z2': pts[i+1][2],
                'color_hex': color,
            })

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name','group','x1','y1','z1','x2','y2','z2','color_hex'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} line segments to {filepath}")

print("\n" + "="*60)
print("IMPORT COMPLETE")
print("="*60)
print(f"  Collections: Cells, Polytope, Lines, Centers")
print(f"  To save lines: save_lines_to_csv()")
print(f"  New lines you draw in the Lines collection will be exported")
'''

with open(OUTPUT_DIR / "blender_import.py", 'w') as f:
    f.write(blender_script)

# ── 5. Export gene expression CSV (for Blender addon Gene Painter) ─────────
print("Exporting gene expression CSV ...")

EXTRA_GENES = ['inv', 'ap', 'vg', 'Dll', 'dpp', 'wg', 'hh']
hox_mask = X_raw[:, hox_idx].sum(1) > 0

gene_expr_df = pd.DataFrame({
    'h1': H1, 'h2': H2, 'h3': H3,
})
# Add Hox genes
for gi, gname in enumerate(HOX_GENES):
    gene_expr_df[gname] = X_hox_norm[:, gi]

# Add extra genes (normalize each)
for gname in EXTRA_GENES:
    if gname in gn:
        gi = gn.index(gname)
        raw_vals = X_raw[hox_mask, gi].astype(float)
        if hasattr(raw_vals, 'toarray'):
            raw_vals = raw_vals.toarray().ravel()
        vmin, vmax = raw_vals.min(), raw_vals.max()
        if vmax > vmin:
            gene_expr_df[gname] = (raw_vals - vmin) / (vmax - vmin)
        else:
            gene_expr_df[gname] = 0.0

gene_expr_df.to_csv(OUTPUT_DIR / "blender_gene_expression.csv", index=False)
print(f"  {len(gene_expr_df)} cells, {len(gene_expr_df.columns)-3} genes → blender_gene_expression.csv")

# ── 6. Export cell metadata CSV (for Blender addon Cell Layers) ───────────
print("Exporting cell metadata CSV ...")

obs = adata.obs.iloc[h5ad_row_indices]
meta_df = pd.DataFrame({
    'h1': H1, 'h2': H2, 'h3': H3,
    'annotation': obs['annotation'].values if 'annotation' in obs.columns else 'unknown',
    'annotation_broad': obs['annotation_broad'].values if 'annotation_broad' in obs.columns else 'unknown',
})
# Add any additional metadata columns that exist
for col_name in ['sex', 'R_annotation', 'R_annotation_broad']:
    if col_name in obs.columns:
        meta_df[col_name] = obs[col_name].values

meta_df.to_csv(OUTPUT_DIR / "blender_cell_metadata.csv", index=False)
print(f"  {len(meta_df)} cells, {len(meta_df.columns)-3} columns → blender_cell_metadata.csv")

print(f"\nAll exports complete:")
print(f"  {OUTPUT_DIR / 'blender_cells.csv'}")
print(f"  {OUTPUT_DIR / 'blender_polytope.csv'}")
print(f"  {OUTPUT_DIR / 'blender_polytope_edges.csv'}")
print(f"  {OUTPUT_DIR / 'blender_lines.csv'}")
print(f"  {OUTPUT_DIR / 'blender_gene_expression.csv'}")
print(f"  {OUTPUT_DIR / 'blender_cell_metadata.csv'}")
print(f"  {OUTPUT_DIR / 'blender_import.py'}")
print("Done!")
