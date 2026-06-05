"""
Blender Import Script — Hox Morphospace
Run inside Blender: Scripting tab → Open → Run Script

Collections created:
  Cells/          — sub-collections per pair (P1_C4_C11, P2_C5_C0, etc.)
  Polytope/       — vertices (V1-V14), wireframe edges, face mesh
  Planes/         — H1=0, H2=0, H3=0, W=0, U=0, V=0
  Lines/          — all named internal axes
  Centers/        — C and C*
  Lighting/       — sun + camera

To save new lines:  save_lines_to_csv()  from Blender Python console
"""

import bpy
import bmesh
import csv
import os
import math

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR = r"C:\Users\siyar\Downloads\flycellatlas_hox_output"
CELL_SIZE = 0.008
POLYTOPE_EDGE_WIDTH = 0.005
LINE_WIDTH = 0.007
VERTEX_SIZE = 0.02
PLANE_SIZE = 2.5        # half-extent of plane quads
PLANE_ALPHA = 0.3

# ── Pair definitions (must match generate_morphospace_hoxaxes.py) ──────────
PAIRS = [
    {'id':'P1', 'clusters':[4,11],  'color':'#0000CD', 'label':'P1 (C4 & C11)'},
    {'id':'P2', 'clusters':[5,0],   'color':'#CC0000', 'label':'P2 (C5 & C0)'},
    {'id':'P3', 'clusters':[26,18], 'color':'#006400', 'label':'P3 (C26 & C18)'},
    {'id':'P4', 'clusters':[1,36],  'color':'#C71585', 'label':'P4 (C1 & C36)'},
    {'id':'P5', 'clusters':[13,8],  'color':'#7B2D8E', 'label':'P5 (C13 & C8)'},
    {'id':'P6', 'clusters':[2,9],   'color':'#CC8400', 'label':'P6 (C2 & C9)'},
    {'id':'P7', 'clusters':[6,10],  'color':'#008B8B', 'label':'P7 (C6 & C10)'},
    {'id':'P8', 'clusters':[14,19], 'color':'#D2691E', 'label':'P8 (C14 & C19)'},
]
TRIS = [
    {'id':'P9',  'clusters':[30,31,32], 'color':'#FF1493', 'label':'P9 (C20 3-way)'},
    {'id':'P10', 'clusters':[33,34,35], 'color':'#9ACD32', 'label':'P10 (C7 3-way)'},
]
UNMATCHED_COLOR = '#888888'

# ── Helpers ─────────────────────────────────────────────────────────────────
def hex_to_rgba(h):
    h = h.lstrip('#')
    return (int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255, 1.0)

def make_material(name, hex_color, alpha=1.0, emit=0.0):
    """Create or reuse a material with proper node setup."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name=name)
    r, g, b, _ = hex_to_rgba(hex_color)
    # Set viewport display color (for Solid mode)
    mat.diffuse_color = (r, g, b, alpha)
    # Set up nodes (for Material Preview / Rendered mode)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    # Clear default nodes
    for node in nodes:
        nodes.remove(node)
    # Create output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    if alpha < 1.0:
        # Transparent material: mix shader
        mat.blend_method = 'BLEND'  # for EEVEE
        mat.shadow_method = 'NONE'
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)
        bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
        bsdf.inputs['Alpha'].default_value = alpha
        bsdf.inputs['Roughness'].default_value = 0.8
        if emit > 0:
            bsdf.inputs['Emission Strength'].default_value = emit
            bsdf.inputs['Emission Color'].default_value = (r, g, b, 1.0)
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    else:
        # Opaque: simple principled BSDF
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)
        bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.7
        if emit > 0:
            bsdf.inputs['Emission Strength'].default_value = emit
            bsdf.inputs['Emission Color'].default_value = (r, g, b, 1.0)
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return mat

def make_collection(name, parent=None):
    """Create a collection and link it."""
    col = bpy.data.collections.new(name)
    if parent is None:
        bpy.context.scene.collection.children.link(col)
    else:
        parent.children.link(col)
    return col

def link_to_collection(obj, collection):
    """Move object from scene root to target collection."""
    # Unlink from all current collections
    for col in obj.users_collection:
        col.objects.unlink(obj)
    collection.objects.link(obj)

def create_curve(name, points, hex_color, width=0.005, collection=None):
    """Create a bevel curve between points."""
    curve_data = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = width
    curve_data.bevel_resolution = 3
    spline = curve_data.splines.new('POLY')
    spline.points.add(len(points) - 1)
    for i, p in enumerate(points):
        spline.points[i].co = (p[0], p[1], p[2], 1)
    obj = bpy.data.objects.new(name, curve_data)
    mat = make_material(f"mat_{name}", hex_color, emit=0.3)
    obj.data.materials.append(mat)
    target = collection if collection else bpy.context.scene.collection
    target.objects.link(obj)
    return obj

def create_plane_quad(name, center, normal, hex_color, alpha=0.3, size=2.0, collection=None):
    """Create a flat quad plane at given center/normal with semi-transparent material."""
    n = [normal[0], normal[1], normal[2]]
    nl = math.sqrt(sum(x*x for x in n))
    n = [x/nl for x in n]
    # Find two tangent vectors
    if abs(n[0]) < 0.9:
        tmp = [1, 0, 0]
    else:
        tmp = [0, 1, 0]
    # Cross products
    e1 = [n[1]*tmp[2]-n[2]*tmp[1], n[2]*tmp[0]-n[0]*tmp[2], n[0]*tmp[1]-n[1]*tmp[0]]
    e1l = math.sqrt(sum(x*x for x in e1))
    e1 = [x/e1l for x in e1]
    e2 = [n[1]*e1[2]-n[2]*e1[1], n[2]*e1[0]-n[0]*e1[2], n[0]*e1[1]-n[1]*e1[0]]

    # 4 corners
    verts = []
    for s in [-size, size]:
        for t in [-size, size]:
            verts.append((
                center[0] + s*e1[0] + t*e2[0],
                center[1] + s*e1[1] + t*e2[1],
                center[2] + s*e1[2] + t*e2[2],
            ))
    faces = [(0, 1, 3, 2)]
    mesh = bpy.data.meshes.new(f"mesh_{name}")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    mat = make_material(f"mat_{name}", hex_color, alpha=alpha, emit=0.1)
    obj.data.materials.append(mat)
    target = collection if collection else bpy.context.scene.collection
    target.objects.link(obj)
    return obj

# ══════════════════════════════════════════════════════════════════════════════
# CLEAR SCENE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CLEARING SCENE ...")
print("="*60)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
# Remove orphan data
for block in bpy.data.meshes:
    if block.users == 0: bpy.data.meshes.remove(block)
for block in bpy.data.materials:
    if block.users == 0: bpy.data.materials.remove(block)
for block in bpy.data.curves:
    if block.users == 0: bpy.data.curves.remove(block)
# Remove non-default collections
for col in list(bpy.data.collections):
    bpy.data.collections.remove(col)

# ══════════════════════════════════════════════════════════════════════════════
# 1. IMPORT CELLS — grouped by leiden cluster, organized by pair
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Importing cells ──")
cells_path = os.path.join(DATA_DIR, "blender_cells.csv")

# Read all cells, group by leiden cluster
cluster_cells = {}  # leiden -> list of (pos, color_hex, h5ad_row, broad)
with open(cells_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cl = int(row['leiden'])
        pos = (float(row['H1']), float(row['H2']), float(row['H3']))
        color = row['color_hex']
        h5ad = row['h5ad_row']
        broad = row['broad']
        if cl not in cluster_cells:
            cluster_cells[cl] = []
        cluster_cells[cl].append((pos, color, h5ad, broad))

total_cells = sum(len(v) for v in cluster_cells.values())
print(f"  {total_cells} cells in {len(cluster_cells)} clusters")

cells_root = make_collection("Cells")

# Track which clusters are assigned to pairs
assigned_clusters = set()

# Create base sphere template (hidden)
bpy.ops.mesh.primitive_ico_sphere_add(radius=CELL_SIZE, subdivisions=2)
base_sphere = bpy.context.active_object
base_sphere.name = "_cell_template"
base_sphere.hide_set(True)
base_sphere.hide_render = True
link_to_collection(base_sphere, cells_root)

def add_cluster_to_collection(cluster_id, cell_list, parent_col, display_name=None):
    """Add a cluster's cells as an instanced mesh object."""
    if not cell_list:
        return
    name = display_name or f"C{cluster_id}"
    color_hex = cell_list[0][1]  # all cells in a cluster share the same color

    # Create mesh with one vertex per cell
    mesh = bpy.data.meshes.new(f"mesh_{name}")
    verts = [c[0] for c in cell_list]
    mesh.from_pydata(verts, [], [])
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    parent_col.objects.link(obj)

    # Instance spheres on vertices
    obj.instance_type = 'VERTS'

    # Create child sphere with material
    child = base_sphere.copy()
    child.data = base_sphere.data.copy()
    child.name = f"_sphere_{name}"
    child.parent = obj
    child.hide_set(False)
    child.hide_render = False

    # Material with emission so it's visible in Material Preview
    mat = make_material(f"mat_{name}", color_hex, emit=0.4)
    child.data.materials.clear()
    child.data.materials.append(mat)
    parent_col.objects.link(child)

    print(f"    {name}: {len(cell_list)} cells, color={color_hex}")

# Process pairs
for p in PAIRS:
    pair_col = make_collection(f"{p['id']}_{p['label']}", parent=cells_root)
    for cl in p['clusters']:
        if cl in cluster_cells:
            add_cluster_to_collection(cl, cluster_cells[cl], pair_col,
                                      display_name=f"{p['id']}_C{cl}")
            assigned_clusters.add(cl)

# Process 3-way splits
for t in TRIS:
    tri_col = make_collection(f"{t['id']}_{t['label']}", parent=cells_root)
    for cl in t['clusters']:
        if cl in cluster_cells:
            add_cluster_to_collection(cl, cluster_cells[cl], tri_col,
                                      display_name=f"{t['id']}_C{cl}")
            assigned_clusters.add(cl)

# Unmatched clusters
unmatched_cls = [cl for cl in cluster_cells if cl not in assigned_clusters]
if unmatched_cls:
    unm_col = make_collection("Unmatched", parent=cells_root)
    for cl in sorted(unmatched_cls):
        add_cluster_to_collection(cl, cluster_cells[cl], unm_col)
        assigned_clusters.add(cl)

print(f"  Total: {len(assigned_clusters)} clusters imported")

# ══════════════════════════════════════════════════════════════════════════════
# 2. POLYTOPE — vertices, edges, faces
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Importing polytope ──")
polytope_col = make_collection("Polytope")

# Read vertices
verts_path = os.path.join(DATA_DIR, "blender_polytope.csv")
poly_verts = []
poly_labels = []
with open(verts_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        poly_verts.append((float(row['H1']), float(row['H2']), float(row['H3'])))
        poly_labels.append(row['vertex_id'])

# Vertex spheres with labels
for i, (pos, label) in enumerate(zip(poly_verts, poly_labels)):
    bpy.ops.mesh.primitive_ico_sphere_add(radius=VERTEX_SIZE, location=pos, subdivisions=2)
    obj = bpy.context.active_object
    obj.name = f"polytope_{label}"
    mat = make_material("mat_polytope_vert", "#222222", emit=0.5)
    obj.data.materials.append(mat)
    link_to_collection(obj, polytope_col)

    # 3D text label
    bpy.ops.object.text_add(location=(pos[0]+0.03, pos[1]+0.03, pos[2]+0.03))
    txt = bpy.context.active_object
    txt.data.body = label
    txt.data.size = 0.05
    txt.data.align_x = 'LEFT'
    txt.name = f"label_{label}"
    mat_txt = make_material("mat_label", "#FFFFFF", emit=1.0)
    txt.data.materials.append(mat_txt)
    link_to_collection(txt, polytope_col)

print(f"  {len(poly_verts)} vertices with labels")

# Wireframe edges
edges_path = os.path.join(DATA_DIR, "blender_polytope_edges.csv")
edge_count = 0
with open(edges_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        a = int(row['v_start']); b = int(row['v_end'])
        pa = poly_verts[a]; pb = poly_verts[b]
        create_curve(f"edge_{poly_labels[a]}_{poly_labels[b]}", [pa, pb],
                     "#404040", POLYTOPE_EDGE_WIDTH, collection=polytope_col)
        edge_count += 1

print(f"  {edge_count} wireframe edges")

# Polytope face mesh (semi-transparent)
# Reconstruct the 6 faces from the convex hull of the 14 vertices
# Use all verts as a single mesh, let Blender handle the hull via convex hull modifier
poly_mesh = bpy.data.meshes.new("polytope_faces_mesh")
poly_mesh.from_pydata(poly_verts, [], [])
poly_mesh.update()
poly_face_obj = bpy.data.objects.new("polytope_faces", poly_mesh)
polytope_col.objects.link(poly_face_obj)

# Add convex hull via bmesh
bm = bmesh.new()
bm.from_mesh(poly_mesh)
bmesh.ops.convex_hull(bm, input=bm.verts)
bm.to_mesh(poly_mesh)
bm.free()
poly_mesh.update()

mat_face = make_material("mat_polytope_face", "#6496FF", alpha=0.2, emit=0.05)
poly_face_obj.data.materials.append(mat_face)
poly_face_obj.show_transparent = True

print(f"  Polytope face mesh created")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PLANES — H1=0, H2=0, H3=0 (expression) + W=0, U=0, V=0 (geometric)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Creating planes ──")
planes_col = make_collection("Planes")

plane_defs = [
    # Expression planes (axis-aligned in H-space)
    ("H1=0 (Antp=Ubx)",        (0,0,0), (1,0,0), "#1E64DC", 0.25),
    ("H2=0 (abdA=AbdB)",       (0,0,0), (0,1,0), "#DC3232", 0.25),
    ("H3=0 (thoracic=abdom)",  (0,0,0), (0,0,1), "#DCB400", 0.25),
    # Geometric morphospace planes (approximate — these are nearly axis-aligned
    # since H1~W, H2~U, H3~V with <7 deg offset, shown as faded references)
    # Using the slight tilts from the angle analysis
    ("W=0 (geom, ~H1)",       (-0.05, 0.01, -0.02), (0.99, -0.07, 0.10), "#1E64DC", 0.12),
    ("U=0 (geom, ~H2)",       (0.01, -0.04, 0.02),  (0.07, 0.99, -0.04), "#DC3232", 0.12),
    ("V=0 (geom, ~H3)",       (0.01, 0.01, 0.05),   (0.04, 0.10, 0.99),  "#DCB400", 0.12),
]

for pname, pcenter, pnormal, pcolor, palpha in plane_defs:
    create_plane_quad(pname, pcenter, pnormal, pcolor, alpha=palpha,
                      size=PLANE_SIZE, collection=planes_col)
    print(f"  {pname}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. NAMED LINES — internal axes from C and C*
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Importing lines ──")
lines_col = make_collection("Lines")

lines_path = os.path.join(DATA_DIR, "blender_lines.csv")
line_count = 0
with open(lines_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        p1 = (float(row['x1']), float(row['y1']), float(row['z1']))
        p2 = (float(row['x2']), float(row['y2']), float(row['z2']))
        name = row['name']
        color = row['color_hex']
        create_curve(name, [p1, p2], color, LINE_WIDTH, collection=lines_col)
        line_count += 1

print(f"  {line_count} line segments")

# ══════════════════════════════════════════════════════════════════════════════
# 5. CENTER MARKERS — C (0,0,0) and C* (data centroid)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Creating centers ──")
centers_col = make_collection("Centers")

for cname, cpos, ccolor in [
    ("Center_C",      (0, 0, 0),              "#FFFFFF"),
    ("Center_C_star", (-0.121, -0.014, 0.248), "#FFFF00"),
]:
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.025, location=cpos, subdivisions=3)
    obj = bpy.context.active_object
    obj.name = cname
    mat = make_material(f"mat_{cname}", ccolor, emit=0.8)
    obj.data.materials.append(mat)
    link_to_collection(obj, centers_col)
    print(f"  {cname} at {cpos}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. BOUNDING BOXES
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Creating bounding boxes ──")
bbox_col = make_collection("BoundingBoxes")

def create_wireframe_box(name, mins, maxs, hex_color, width=0.003, collection=None):
    """Create a wireframe box from min/max corners."""
    corners = []
    for x in [mins[0], maxs[0]]:
        for y in [mins[1], maxs[1]]:
            for z in [mins[2], maxs[2]]:
                corners.append((x, y, z))
    # 12 edges of a box
    edge_pairs = [
        (0,1),(2,3),(4,5),(6,7),  # z-edges
        (0,2),(1,3),(4,6),(5,7),  # y-edges
        (0,4),(1,5),(2,6),(3,7),  # x-edges
    ]
    for i, (a, b) in enumerate(edge_pairs):
        create_curve(f"{name}_edge{i}", [corners[a], corners[b]],
                     hex_color, width, collection=collection)

# Theoretical bbox [-1,1]x[-1,1]x[-2,2]
create_wireframe_box("TheoBBox", (-1,-1,-2), (1,1,2), "#B40000",
                     width=0.002, collection=bbox_col)
print("  Theoretical bbox [-1,1]x[-1,1]x[-2,2]")

# Data bbox (approximate from known ranges)
create_wireframe_box("DataBBox", (-0.87, -0.99, -1.36), (0.79, 0.99, 1.68), "#333333",
                     width=0.003, collection=bbox_col)
print("  Data bbox")

# ══════════════════════════════════════════════════════════════════════════════
# 7. LIGHTING + CAMERA + VIEWPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Setting up scene ──")
lighting_col = make_collection("Lighting")

# Sun light
bpy.ops.object.light_add(type='SUN', location=(3, -3, 5))
sun = bpy.context.active_object
sun.name = "Sun"
sun.data.energy = 2.0
sun.data.angle = 0.5  # softer shadows
link_to_collection(sun, lighting_col)

# Fill light (opposite side)
bpy.ops.object.light_add(type='SUN', location=(-3, 3, 2))
fill = bpy.context.active_object
fill.name = "Fill"
fill.data.energy = 1.0
link_to_collection(fill, lighting_col)

# Area light from above for even illumination
bpy.ops.object.light_add(type='AREA', location=(0, 0, 4))
area = bpy.context.active_object
area.name = "TopArea"
area.data.energy = 50
area.data.size = 6
link_to_collection(area, lighting_col)

# Camera
bpy.ops.object.camera_add(location=(4, -3, 2.5))
cam = bpy.context.active_object
cam.name = "Camera"
cam.rotation_euler = (1.1, 0, 0.8)
bpy.context.scene.camera = cam
link_to_collection(cam, lighting_col)

# Set viewport to Material Preview
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'
                space.shading.use_scene_lights = True
                space.shading.use_scene_world = False
                # Set a neutral studio HDRI background
                space.shading.studio_light = 'studio.exr'
                space.shading.studiolight_rotate_z = 0
                space.shading.studiolight_intensity = 1.0

# Set render engine to EEVEE for transparency support
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT' if bpy.app.version >= (4, 0, 0) else 'BLENDER_EEVEE'

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: Save user-created lines back to CSV
# ══════════════════════════════════════════════════════════════════════════════
def save_lines_to_csv(filepath=None):
    """
    Run from Blender Python console:  save_lines_to_csv()

    Exports all CURVE objects in the 'Lines' collection to blender_lines.csv.
    Draw new lines as curve objects in that collection → they'll be saved.
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "blender_lines.csv")

    lines_c = bpy.data.collections.get("Lines")
    if not lines_c:
        print("No 'Lines' collection found!")
        return

    rows = []
    for obj in lines_c.objects:
        if obj.type != 'CURVE':
            continue
        spline = obj.data.splines[0]
        pts = [p.co[:3] for p in spline.points]
        if len(pts) < 2:
            continue
        # Get color from material diffuse_color
        color = '#888888'
        if obj.data.materials:
            mat = obj.data.materials[0]
            c = mat.diffuse_color
            color = f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}'

        for i in range(len(pts)-1):
            rows.append({
                'name': obj.name,
                'group': obj.name,
                'x1': f'{pts[i][0]:.6f}', 'y1': f'{pts[i][1]:.6f}', 'z1': f'{pts[i][2]:.6f}',
                'x2': f'{pts[i+1][0]:.6f}', 'y2': f'{pts[i+1][1]:.6f}', 'z2': f'{pts[i+1][2]:.6f}',
                'color_hex': color,
            })

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name','group','x1','y1','z1','x2','y2','z2','color_hex'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} line segments → {filepath}")

# Make save function available globally
bpy.app.driver_namespace['save_lines_to_csv'] = save_lines_to_csv

print("\n" + "="*60)
print("IMPORT COMPLETE")
print("="*60)
print("""
Collections:
  Cells/          — organized by pair (P1-P10) + Unmatched
  Polytope/       — V1-V14 vertices, wireframe, face mesh
  Planes/         — 6 planes (3 expression + 3 geometric)
  Lines/          — named internal axes
  Centers/        — C and C*
  BoundingBoxes/  — theoretical + data
  Lighting/       — sun, fill, area, camera

To save lines:  save_lines_to_csv()
Viewport set to Material Preview with scene lights.
""")
