bl_info = {
    "name": "Hox Morphospace Toolkit",
    "author": "Hox Morphospace Project",
    "version": (2, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Hox Morphospace",
    "description": "Scene builder, plane creator, gene expression painting, and cell type layers for Hox morphospace visualization",
    "category": "3D View",
}

import bpy
import csv
import os
import subprocess
import sys
import numpy as np
from bpy.props import (
    PointerProperty, FloatProperty, FloatVectorProperty,
    StringProperty, BoolProperty, IntProperty,
    EnumProperty, CollectionProperty,
)
from bpy.types import Panel, Operator, PropertyGroup, UIList


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_addon_dir = os.path.dirname(os.path.abspath(__file__))


def lerp3(a, b, t):
    return (a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]), a[2]+t*(b[2]-a[2]))


def lerp_color(c0, c1, t):
    return (c0[0]+t*(c1[0]-c0[0]), c0[1]+t*(c1[1]-c0[1]), c0[2]+t*(c1[2]-c0[2]))


def make_material(name, r, g, b, emission=0.5):
    """Create or update a principled BSDF material with emission."""
    mat = bpy.data.materials.get(name)
    if mat:
        mat.diffuse_color = (r, g, b, 1.0)
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                node.inputs['Base Color'].default_value = (r, g, b, 1.0)
                node.inputs['Emission Color'].default_value = (r, g, b, 1.0)
                node.inputs['Emission Strength'].default_value = emission
        return mat

    mat = bpy.data.materials.new(name=name)
    mat.diffuse_color = (r, g, b, 1.0)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.4
    bsdf.inputs['Emission Strength'].default_value = emission
    bsdf.inputs['Emission Color'].default_value = (r, g, b, 1.0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return mat


def remove_collection_recursive(col):
    """Remove a collection and all its objects/children."""
    for child in list(col.children):
        remove_collection_recursive(child)
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(col)


# ── Mesh builder: tiny cubes at cell positions ──────────────────────────────

_CUBE_VERTS = np.array([
    [-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],
    [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1],
], dtype=np.float32)

_CUBE_FACES = np.array([
    (0,1,2,3), (4,5,6,7), (0,1,5,4),
    (2,3,7,6), (0,3,7,4), (1,2,6,5),
], dtype=np.int32)


def build_cubes_mesh(name, centers, size):
    """Build a mesh with tiny cubes at each center position."""
    n = len(centers)
    if n == 0:
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([], [], [])
        return mesh

    scaled = _CUBE_VERTS * size
    all_verts = np.tile(scaled, (n, 1)) + np.repeat(centers, 8, axis=0)
    offsets = (np.arange(n, dtype=np.int32) * 8).reshape(-1, 1)
    all_faces = np.tile(_CUBE_FACES, (n, 1)) + np.repeat(offsets, 6, axis=0)

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(all_verts.tolist(), [], all_faces.tolist())
    mesh.update()
    return mesh


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 1: PLANE CREATOR
# ═══════════════════════════════════════════════════════════════════════════════

class HoxDiamondProperties(PropertyGroup):
    apex: PointerProperty(
        name="Apex", type=bpy.types.Object,
        description="Top point of diamond (e.g., C*)"
    )
    base: PointerProperty(
        name="Base", type=bpy.types.Object,
        description="Bottom point of diamond (e.g., V3)"
    )
    left_wing: PointerProperty(
        name="Left Wing", type=bpy.types.Object,
        description="Left corner (e.g., V1)"
    )
    right_wing: PointerProperty(
        name="Right Wing", type=bpy.types.Object,
        description="Right corner (e.g., V2)"
    )
    shift_target: PointerProperty(
        name="Shift Target", type=bpy.types.Object,
        description="Shift the base toward this point"
    )
    shift_pct: FloatProperty(
        name="Shift %", default=0.0,
        min=0.0, max=100.0, step=100, precision=1,
        subtype='PERCENTAGE',
        description="How far to shift base toward shift target (0%=original, 100%=at target)"
    )
    step_size: FloatProperty(
        name="Step Size", default=0.10,
        min=0.01, max=0.50, step=1, precision=2,
        description="Spacing between stripes"
    )
    line_width: FloatProperty(
        name="Line Width", default=0.005,
        min=0.001, max=0.02, step=1, precision=3,
        description="Bevel thickness of stripe lines"
    )
    left_color: FloatVectorProperty(
        name="Left Color", subtype='COLOR',
        default=(0.3, 0.8, 1.0), min=0.0, max=1.0,
    )
    right_color: FloatVectorProperty(
        name="Right Color", subtype='COLOR',
        default=(1.0, 0.5, 0.4), min=0.0, max=1.0,
    )
    central_color: FloatVectorProperty(
        name="Central Color", subtype='COLOR',
        default=(1.0, 1.0, 1.0), min=0.0, max=1.0,
    )
    draw_outline: BoolProperty(name="Draw Outline", default=True)
    draw_vertices: BoolProperty(name="Draw Vertices", default=True)
    vertex_size: FloatProperty(
        name="Vertex Size", default=0.02,
        min=0.005, max=0.1, step=1, precision=3,
    )
    collection_name: StringProperty(
        name="Collection", default="Diamond_Stripes",
    )


def _make_curve_line(name, p1, p2, r, g, b, width, collection):
    curve_data = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = width
    curve_data.bevel_resolution = 3
    spline = curve_data.splines.new('POLY')
    spline.points.add(1)
    spline.points[0].co = (p1[0], p1[1], p1[2], 1)
    spline.points[1].co = (p2[0], p2[1], p2[2], 1)
    obj = bpy.data.objects.new(name, curve_data)
    mat = make_material(f"mat_ds_{name}", r, g, b, emission=0.3)
    obj.data.materials.append(mat)
    collection.objects.link(obj)
    return obj


class HOX_OT_generate_stripes(Operator):
    bl_idname = "hox.generate_stripes"
    bl_label = "Generate Diamond Stripes"
    bl_description = "Generate stripe pattern on the selected diamond plane"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.hox_diamond

        if not all([props.apex, props.base, props.left_wing, props.right_wing]):
            self.report({'ERROR'}, "Please assign all 4 diamond corners")
            return {'CANCELLED'}

        apex = tuple(props.apex.location)
        base = tuple(props.base.location)
        left = tuple(props.left_wing.location)
        right = tuple(props.right_wing.location)

        t = props.shift_pct / 100.0
        if t > 0 and props.shift_target:
            target = tuple(props.shift_target.location)
            base = lerp3(base, target, t)

        # Get or create Lines collection
        lines_col = bpy.data.collections.get("Lines")
        if not lines_col:
            lines_col = bpy.data.collections.new("Lines")
            bpy.context.scene.collection.children.link(lines_col)

        col_name = props.collection_name
        existing = bpy.data.collections.get(col_name)
        if existing:
            remove_collection_recursive(existing)

        out_col = bpy.data.collections.new(col_name)
        lines_col.children.link(out_col)

        step = props.step_size
        width = props.line_width
        lc, rc, cc = props.left_color, props.right_color, props.central_color
        vsize = props.vertex_size
        do_verts = props.draw_vertices

        if do_verts:
            verts_col = bpy.data.collections.new(f"{col_name}_vertices")
            out_col.children.link(verts_col)

        # Central line
        _make_curve_line(f"{col_name}_central", apex, base,
                         cc[0], cc[1], cc[2], width * 1.3, out_col)

        if do_verts:
            for vn, vp in [("apex", apex), ("base", base), ("left", left), ("right", right)]:
                e = bpy.data.objects.new(f"{col_name}_v_{vn}", None)
                e.empty_display_type = 'SPHERE'
                e.empty_display_size = vsize * 1.5
                e.location = vp
                verts_col.objects.link(e)

        # Left stripes
        s, i = step, 1
        while s < 1.0 - 0.001:
            p_top = lerp3(apex, left, s)
            p_bot = lerp3(base, left, s)
            color = lerp_color(cc, lc, s)
            _make_curve_line(f"{col_name}_L{i}", p_top, p_bot,
                             color[0], color[1], color[2], width, out_col)
            if do_verts:
                for suffix, pos in [("top", p_top), ("bot", p_bot)]:
                    e = bpy.data.objects.new(f"{col_name}_L{i}_{suffix}", None)
                    e.empty_display_type = 'SPHERE'
                    e.empty_display_size = vsize
                    e.location = pos
                    verts_col.objects.link(e)
            s += step
            i += 1
        left_count = i - 1

        # Right stripes
        s, i = step, 1
        while s < 1.0 - 0.001:
            p_top = lerp3(apex, right, s)
            p_bot = lerp3(base, right, s)
            color = lerp_color(cc, rc, s)
            _make_curve_line(f"{col_name}_R{i}", p_top, p_bot,
                             color[0], color[1], color[2], width, out_col)
            if do_verts:
                for suffix, pos in [("top", p_top), ("bot", p_bot)]:
                    e = bpy.data.objects.new(f"{col_name}_R{i}_{suffix}", None)
                    e.empty_display_type = 'SPHERE'
                    e.empty_display_size = vsize
                    e.location = pos
                    verts_col.objects.link(e)
            s += step
            i += 1
        right_count = i - 1

        # Outline
        if props.draw_outline:
            for ename, p1, p2 in [
                ("apex_left", apex, left), ("left_base", left, base),
                ("base_right", base, right), ("right_apex", right, apex),
            ]:
                _make_curve_line(f"{col_name}_outline_{ename}", p1, p2,
                                 0.4, 0.4, 0.4, width * 0.7, out_col)

        shift_str = f" (shifted {props.shift_pct:.0f}%)" if (t > 0 and props.shift_target) else ""
        self.report({'INFO'}, f"{left_count}L + {right_count}R stripes{shift_str}")
        return {'FINISHED'}


class HOX_OT_clear_stripes(Operator):
    bl_idname = "hox.clear_stripes"
    bl_label = "Clear Stripes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        col_name = context.scene.hox_diamond.collection_name
        col = bpy.data.collections.get(col_name)
        if col:
            remove_collection_recursive(col)
            self.report({'INFO'}, f"Cleared: {col_name}")
        else:
            self.report({'WARNING'}, f"'{col_name}' not found")
        return {'FINISHED'}


class HOX_PT_diamond_panel(Panel):
    bl_label = "Plane Creator"
    bl_idname = "HOX_PT_diamond_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Hox Morphospace"
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        props = context.scene.hox_diamond

        box = layout.box()
        box.label(text="Diamond Corners", icon='MESH_PLANE')
        box.prop(props, "apex", icon='TRIA_UP')
        box.prop(props, "base", icon='TRIA_DOWN')
        box.prop(props, "left_wing", icon='TRIA_LEFT')
        box.prop(props, "right_wing", icon='TRIA_RIGHT')

        box = layout.box()
        box.label(text="Shift", icon='TRACKING')
        box.prop(props, "shift_target")
        box.prop(props, "shift_pct", slider=True)

        box = layout.box()
        box.label(text="Stripe Settings", icon='LINENUMBERS_ON')
        box.prop(props, "step_size")
        box.prop(props, "line_width")
        row = box.row()
        row.prop(props, "left_color", text="Left")
        row.prop(props, "central_color", text="Center")
        row.prop(props, "right_color", text="Right")
        box.prop(props, "draw_outline")
        box.prop(props, "draw_vertices")
        if props.draw_vertices:
            box.prop(props, "vertex_size")

        box = layout.box()
        box.label(text="Output", icon='OUTLINER_COLLECTION')
        box.prop(props, "collection_name")
        row = box.row(align=True)
        row.scale_y = 1.5
        row.operator("hox.generate_stripes", icon='PLAY', text="Generate")
        row.operator("hox.clear_stripes", icon='TRASH', text="Clear")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 2: CELL LAYER SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════

_CELL_META_CSV = os.path.join(_addon_dir, "blender_cell_metadata.csv")
_metadata = None
_meta_columns = []
_column_values = {}


def load_metadata():
    global _metadata, _meta_columns, _column_values
    if _metadata is not None:
        return True

    path = _CELL_META_CSV
    if not os.path.exists(path):
        try:
            blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else ""
        except (AttributeError, TypeError):
            blend_dir = ""
        alt = os.path.join(blend_dir, "blender_cell_metadata.csv") if blend_dir else ""
        if alt and os.path.exists(alt):
            path = alt
        else:
            return False

    _metadata = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            _metadata.append(row)

    skip = {'h1', 'h2', 'h3'}
    _meta_columns = [c for c in _metadata[0].keys() if c not in skip]
    _column_values = {}
    for col in _meta_columns:
        _column_values[col] = sorted(set(row[col] for row in _metadata))

    print(f"[HoxLayers] Loaded {len(_metadata)} cells, {len(_meta_columns)} columns")
    return True


class HoxLayerItem(PropertyGroup):
    layer_name: StringProperty(name="Name")
    collection_name: StringProperty(name="Collection")
    column: StringProperty(name="Column")
    value: StringProperty(name="Value")
    cell_count: IntProperty(name="Cells")
    color: FloatVectorProperty(
        name="Color", subtype='COLOR',
        default=(1.0, 0.5, 0.0), min=0.0, max=1.0,
    )
    visible: BoolProperty(name="Visible", default=True)


def get_column_items(self, context):
    if not load_metadata():
        return [('NONE', 'No CSV found', '')]
    return [(c, c, f'{len(_column_values[c])} values') for c in _meta_columns]


def get_value_items(self, context):
    props = context.scene.hox_layers
    col = props.selected_column
    if col == 'NONE' or col not in _column_values:
        return [('NONE', '---', '')]
    items = []
    for v in _column_values[col]:
        count = sum(1 for row in _metadata if row[col] == v)
        items.append((v, f'{v} ({count})', f'{count} cells'))
    return items


class HoxLayerProperties(PropertyGroup):
    selected_column: EnumProperty(name="Column", items=get_column_items)
    selected_value: EnumProperty(name="Value", items=get_value_items)
    layer_color: FloatVectorProperty(
        name="Color", subtype='COLOR',
        default=(1.0, 0.3, 0.1), min=0.0, max=1.0,
    )
    point_size: FloatProperty(
        name="Point Size", default=0.015,
        min=0.005, max=0.05, step=1, precision=3,
    )
    active_layer_index: IntProperty(default=0)
    layers: CollectionProperty(type=HoxLayerItem)


class HOX_OT_load_metadata(Operator):
    bl_idname = "hox.layers_load"
    bl_label = "Load Metadata"

    def execute(self, context):
        global _metadata
        _metadata = None
        if load_metadata():
            self.report({'INFO'}, f"Loaded {len(_metadata)} cells")
        else:
            self.report({'ERROR'}, f"CSV not found")
        return {'FINISHED'}


class HOX_OT_add_layer(Operator):
    bl_idname = "hox.layers_add"
    bl_label = "Add Layer"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not load_metadata():
            self.report({'ERROR'}, "Metadata not loaded")
            return {'CANCELLED'}

        props = context.scene.hox_layers
        col_key = props.selected_column
        val = props.selected_value

        if col_key == 'NONE' or val == 'NONE':
            self.report({'ERROR'}, "Select a column and value first")
            return {'CANCELLED'}

        for layer in props.layers:
            if layer.column == col_key and layer.value == val:
                self.report({'WARNING'}, f"Layer '{val}' already exists")
                return {'CANCELLED'}

        matching = [(i, row) for i, row in enumerate(_metadata) if row[col_key] == val]
        if not matching:
            self.report({'WARNING'}, f"No cells match {col_key}={val}")
            return {'CANCELLED'}

        col_name = f"Layer_{col_key}_{val}"

        parent = bpy.data.collections.get("CellLayers")
        if not parent:
            parent = bpy.data.collections.new("CellLayers")
            bpy.context.scene.collection.children.link(parent)

        existing = bpy.data.collections.get(col_name)
        if existing:
            remove_collection_recursive(existing)

        layer_col = bpy.data.collections.new(col_name)
        parent.children.link(layer_col)

        r, g, b = props.layer_color
        mat = make_material(f"mat_{col_name}", r, g, b, emission=0.5)

        centers = np.array(
            [(float(row['h1']), float(row['h2']), float(row['h3'])) for _, row in matching],
            dtype=np.float32,
        )
        mesh = build_cubes_mesh(f"mesh_{col_name}", centers, props.point_size)
        obj = bpy.data.objects.new(f"cells_{col_name}", mesh)
        obj.data.materials.append(mat)
        layer_col.objects.link(obj)

        layer = props.layers.add()
        layer.layer_name = f"{col_key}: {val}"
        layer.collection_name = col_name
        layer.column = col_key
        layer.value = val
        layer.cell_count = len(matching)
        layer.color = (r, g, b)
        layer.visible = True
        props.active_layer_index = len(props.layers) - 1

        self.report({'INFO'}, f"Added: {val} ({len(matching)} cells)")
        return {'FINISHED'}


class HOX_OT_remove_layer(Operator):
    bl_idname = "hox.layers_remove"
    bl_label = "Remove Layer"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.hox_layers
        idx = props.active_layer_index
        if idx < 0 or idx >= len(props.layers):
            return {'CANCELLED'}
        layer = props.layers[idx]
        col = bpy.data.collections.get(layer.collection_name)
        if col:
            remove_collection_recursive(col)
        props.layers.remove(idx)
        if props.active_layer_index >= len(props.layers):
            props.active_layer_index = max(0, len(props.layers) - 1)
        return {'FINISHED'}


class HOX_OT_clear_all_layers(Operator):
    bl_idname = "hox.layers_clear_all"
    bl_label = "Clear All Layers"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.hox_layers
        for layer in props.layers:
            col = bpy.data.collections.get(layer.collection_name)
            if col:
                remove_collection_recursive(col)
        props.layers.clear()
        props.active_layer_index = 0
        parent = bpy.data.collections.get("CellLayers")
        if parent and len(parent.children) == 0 and len(parent.objects) == 0:
            bpy.data.collections.remove(parent)
        self.report({'INFO'}, "Cleared all layers")
        return {'FINISHED'}


class HOX_OT_toggle_layer(Operator):
    bl_idname = "hox.layers_toggle"
    bl_label = "Toggle Layer"
    layer_index: IntProperty()

    def execute(self, context):
        props = context.scene.hox_layers
        if self.layer_index < 0 or self.layer_index >= len(props.layers):
            return {'CANCELLED'}
        layer = props.layers[self.layer_index]
        col = bpy.data.collections.get(layer.collection_name)
        if col:
            layer.visible = not layer.visible
            col.hide_viewport = not layer.visible
        return {'FINISHED'}


class HOX_OT_update_layer_color(Operator):
    bl_idname = "hox.layers_update_color"
    bl_label = "Update Color"

    def execute(self, context):
        props = context.scene.hox_layers
        idx = props.active_layer_index
        if idx < 0 or idx >= len(props.layers):
            return {'CANCELLED'}
        layer = props.layers[idx]
        r, g, b = layer.color
        mat = bpy.data.materials.get(f"mat_{layer.collection_name}")
        if mat:
            mat.diffuse_color = (r, g, b, 1.0)
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    node.inputs['Base Color'].default_value = (r, g, b, 1.0)
                    node.inputs['Emission Color'].default_value = (r, g, b, 1.0)
        return {'FINISHED'}


class HOX_UL_layer_list(UIList):
    bl_idname = "HOX_UL_layer_list"

    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            vis_icon = 'HIDE_OFF' if item.visible else 'HIDE_ON'
            op = row.operator("hox.layers_toggle", text="", icon=vis_icon, emboss=False)
            op.layer_index = index
            row.prop(item, "color", text="")
            row.label(text=f"{item.value} ({item.cell_count})")


class HOX_PT_cell_layers_panel(Panel):
    bl_label = "Cell Layers"
    bl_idname = "HOX_PT_cell_layers_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Hox Morphospace"
    bl_order = 3

    def draw(self, context):
        layout = self.layout
        props = context.scene.hox_layers

        box = layout.box()
        if _metadata is not None:
            box.label(text=f"Metadata: {len(_metadata)} cells", icon='CHECKMARK')
        else:
            box.label(text="Metadata not loaded", icon='ERROR')
            box.operator("hox.layers_load", icon='FILE_REFRESH')
            return

        box = layout.box()
        box.label(text="Add New Layer", icon='ADD')
        box.prop(props, "selected_column")
        box.prop(props, "selected_value")
        row = box.row(align=True)
        row.prop(props, "layer_color", text="")
        row.prop(props, "point_size")
        row = box.row(align=True)
        row.scale_y = 1.4
        row.operator("hox.layers_add", icon='ADD', text="Add Layer")

        box = layout.box()
        box.label(text=f"Active Layers ({len(props.layers)})", icon='OUTLINER_COLLECTION')

        if len(props.layers) > 0:
            box.template_list(
                "HOX_UL_layer_list", "",
                props, "layers", props, "active_layer_index",
                rows=min(6, len(props.layers)),
            )
            row = box.row(align=True)
            row.operator("hox.layers_update_color", icon='BRUSHES_ALL', text="Apply Color")
            row.operator("hox.layers_remove", icon='REMOVE', text="Remove")
            row = box.row()
            row.operator("hox.layers_clear_all", icon='TRASH', text="Clear All")

            idx = props.active_layer_index
            if 0 <= idx < len(props.layers):
                layer = props.layers[idx]
                info = box.box()
                info.scale_y = 0.8
                info.label(text=f"Column: {layer.column}")
                info.label(text=f"Value: {layer.value}")
                info.label(text=f"Cells: {layer.cell_count}")
        else:
            box.label(text="No layers yet.")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 3: GENE EXPRESSION PAINTER
# ═══════════════════════════════════════════════════════════════════════════════

CURATED_GENES = {
    'lab':   {'full': 'labial (head)',           'rgb': (100, 180, 220), 'group': 'HOX'},
    'pb':    {'full': 'proboscipedia (head)',     'rgb': ( 80, 150, 210), 'group': 'HOX'},
    'Dfd':   {'full': 'Deformed (head)',          'rgb': ( 50, 130, 190), 'group': 'HOX'},
    'Scr':   {'full': 'Sex combs reduced (T1)',   'rgb': ( 10,  80, 160), 'group': 'HOX'},
    'Antp':  {'full': 'Antennapedia (T2)',        'rgb': (230,  50,  50), 'group': 'HOX'},
    'Ubx':   {'full': 'Ultrabithorax (T3)',       'rgb': ( 50, 120, 220), 'group': 'HOX'},
    'abd-A': {'full': 'abdominal-A (A1-A7)',      'rgb': ( 50, 180,  80), 'group': 'HOX'},
    'Abd-B': {'full': 'Abdominal-B (A8+)',        'rgb': (255, 140,   0), 'group': 'HOX'},
    'inv':   {'full': 'invected (posterior)',      'rgb': (231,  76,  60), 'group': 'TF'},
    'ap':    {'full': 'apterous (dorsal)',         'rgb': ( 52, 152, 219), 'group': 'TF'},
    'vg':    {'full': 'vestigial (wing)',          'rgb': ( 46, 204, 113), 'group': 'TF'},
    'Dll':   {'full': 'Distal-less (appendage)',   'rgb': (243, 156,  18), 'group': 'TF'},
    'dpp':   {'full': 'decapentaplegic (BMP)',     'rgb': (155,  89, 182), 'group': 'TF'},
    'wg':    {'full': 'wingless (Wnt)',            'rgb': ( 26, 188, 156), 'group': 'TF'},
    'hh':    {'full': 'hedgehog (Hh)',             'rgb': (230, 126,  34), 'group': 'TF'},
}

CUSTOM_COLORS = [
    (220,  20,  60), (  0, 191, 255), (  0, 250, 154), (255, 215,   0),
    (186,  85, 211), (255, 105, 180), ( 64, 224, 208), (255,  69,   0),
    (  0, 206, 209), (218, 165,  32), (123, 104, 238), (250, 128, 114),
]

N_BINS = 12
PARENT_COL = "GeneExprPaint"

_GENE_CSV = os.path.join(_addon_dir, "blender_gene_expression.csv")
_EXTRACT_SCRIPT = os.path.join(_addon_dir, "extract_gene_helper.py")

_system_python = None
_gene_data = None
_gene_coords = None
_gene_values = {}
_gene_columns = []


def _find_system_python():
    import shutil
    candidates = ['python', 'python3', 'py']
    blender_py = os.path.dirname(sys.executable).lower()

    for name in candidates:
        path = shutil.which(name)
        if path and blender_py not in path.lower():
            try:
                result = subprocess.run(
                    [path, '-c', 'import scanpy; print("ok")'],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 and 'ok' in result.stdout:
                    return path
            except Exception:
                pass

    for base in [
        os.path.expanduser("~") + r"\AppData\Local\Programs\Python",
        r"C:\Python312", r"C:\Python311", r"C:\Python310",
    ]:
        if os.path.isdir(base):
            for root, dirs, files in os.walk(base):
                if 'python.exe' in files:
                    path = os.path.join(root, 'python.exe')
                    try:
                        result = subprocess.run(
                            [path, '-c', 'import scanpy; print("ok")'],
                            capture_output=True, text=True, timeout=10,
                        )
                        if result.returncode == 0 and 'ok' in result.stdout:
                            return path
                    except Exception:
                        pass
    return "python"


def load_gene_data():
    global _gene_data, _gene_coords, _gene_values, _gene_columns
    if _gene_data is not None:
        return True

    path = _GENE_CSV
    if not os.path.exists(path):
        try:
            blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else ""
        except (AttributeError, TypeError):
            blend_dir = ""
        alt = os.path.join(blend_dir, "blender_gene_expression.csv") if blend_dir else ""
        if alt and os.path.exists(alt):
            path = alt
        else:
            return False

    _gene_data = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            _gene_data.append(row)

    _gene_coords = np.array(
        [(float(r['h1']), float(r['h2']), float(r['h3'])) for r in _gene_data],
        dtype=np.float32,
    )
    skip = {'h1', 'h2', 'h3'}
    _gene_columns = [c for c in _gene_data[0].keys() if c not in skip]
    _gene_values = {}
    for gene in _gene_columns:
        _gene_values[gene] = np.array([float(r[gene]) for r in _gene_data], dtype=np.float32)

    print(f"[GenePainter] Loaded {len(_gene_data)} cells, {len(_gene_columns)} genes")
    return True


def reload_gene_data():
    global _gene_data
    _gene_data = None
    return load_gene_data()


def get_gene_rgb(gene):
    if gene in CURATED_GENES:
        return CURATED_GENES[gene]['rgb']
    return CUSTOM_COLORS[hash(gene) % len(CUSTOM_COLORS)]


def gene_color_ramp(gene, t, color_low=None, color_high=None):
    if color_low is None or color_high is None:
        rgb = get_gene_rgb(gene)
        gene_col = (rgb[0]/255, rgb[1]/255, rgb[2]/255)
        if color_low is None:
            color_low = lerp_color(gene_col, (1, 1, 1), 0.65)
        if color_high is None:
            color_high = lerp_color(gene_col, (1, 1, 1), 0.05)
    grey = (0.12, 0.12, 0.12)
    if t < 0.05:
        return grey
    elif t < 0.15:
        s = (t - 0.05) / 0.10
        return lerp_color(grey, color_low, s)
    else:
        s = (t - 0.15) / 0.85
        return lerp_color(color_low, color_high, s)


class GenePaintLayerItem(PropertyGroup):
    layer_name: StringProperty(name="Name")
    collection_name: StringProperty(name="Collection")
    gene: StringProperty(name="Gene")
    n_expressing: IntProperty(name="Expressing")
    n_total: IntProperty(name="Total")
    point_size: FloatProperty(name="Size")
    color: FloatVectorProperty(
        name="Color", subtype='COLOR',
        default=(1.0, 0.3, 0.1), min=0.0, max=1.0,
    )
    visible: BoolProperty(name="Visible", default=True)


def get_gene_items(self, context):
    if not load_gene_data():
        return [('NONE', 'No data loaded', '')]
    items = []
    curated_order = ['lab','pb','Dfd','Scr','Antp','Ubx','abd-A','Abd-B',
                     'inv','ap','vg','Dll','dpp','wg','hh']
    for gene in curated_order:
        if gene in _gene_values:
            info = CURATED_GENES.get(gene, {})
            tag = info.get('group', '??')
            n_pos = int((_gene_values[gene] > 0).sum())
            items.append((gene, f"[{tag}] {gene} ({n_pos})", info.get('full', '')))
    extra = [g for g in _gene_columns if g not in curated_order]
    for gene in sorted(extra):
        n_pos = int((_gene_values[gene] > 0).sum())
        items.append((gene, f"[+] {gene} ({n_pos})", f"Custom: {gene}"))
    return items if items else [('NONE', 'No genes', '')]


def on_gene_changed(self, context):
    gene = self.selected_gene
    rgb = get_gene_rgb(gene) if gene in _gene_values else (230, 50, 50)
    gene_col = (rgb[0]/255, rgb[1]/255, rgb[2]/255)
    self.color_low = lerp_color(gene_col, (1, 1, 1), 0.6)
    self.color_high = lerp_color(gene_col, (0, 0, 0), 0.1)


class HoxGenePaintProps(PropertyGroup):
    selected_gene: EnumProperty(name="Gene", items=get_gene_items, update=on_gene_changed)
    custom_gene_name: StringProperty(name="Custom Gene", default="")
    color_low: FloatVectorProperty(
        name="Low", subtype='COLOR',
        default=(0.95, 0.7, 0.65), min=0.0, max=1.0,
    )
    color_high: FloatVectorProperty(
        name="High", subtype='COLOR',
        default=(0.9, 0.1, 0.1), min=0.0, max=1.0,
    )
    point_size: FloatProperty(
        name="Point Size", default=0.012,
        min=0.003, max=0.04, step=1, precision=3,
    )
    expr_cutoff: FloatProperty(
        name="Expression Cutoff", default=0.0,
        min=0.0, max=0.99, step=1, precision=3,
    )
    range_min: FloatProperty(
        name="Range Min", default=0.0,
        min=0.0, max=1.0, step=1, precision=3,
    )
    range_max: FloatProperty(
        name="Range Max", default=1.0,
        min=0.0, max=1.0, step=1, precision=3,
    )
    show_nonexpr: BoolProperty(name="Show Non-expressing", default=True)
    active_layer_index: IntProperty(default=0)
    layers: CollectionProperty(type=GenePaintLayerItem)
    python_path: StringProperty(name="Python Path", default="")


class GENEPAINT_OT_load(Operator):
    bl_idname = "genepaint.load"
    bl_label = "Load Gene Data"

    def execute(self, context):
        global _gene_data
        _gene_data = None
        if load_gene_data():
            self.report({'INFO'}, f"Loaded {len(_gene_data)} cells, {len(_gene_columns)} genes")
        else:
            self.report({'ERROR'}, f"CSV not found")
        return {'FINISHED'}


class GENEPAINT_OT_add_custom(Operator):
    bl_idname = "genepaint.add_custom"
    bl_label = "Add Custom Gene"
    bl_description = "Extract a custom gene from h5ad and add to CSV"

    def execute(self, context):
        props = context.scene.hox_gene_paint
        gene_name = props.custom_gene_name.strip()
        if not gene_name:
            self.report({'ERROR'}, "Enter a gene name first")
            return {'CANCELLED'}
        if gene_name in _gene_values:
            self.report({'INFO'}, f"'{gene_name}' already available")
            return {'FINISHED'}
        if not os.path.exists(_EXTRACT_SCRIPT):
            self.report({'ERROR'}, f"Helper not found: {_EXTRACT_SCRIPT}")
            return {'CANCELLED'}

        global _system_python
        py = props.python_path.strip()
        if not py:
            if _system_python is None:
                _system_python = _find_system_python()
            py = _system_python

        self.report({'INFO'}, f"Extracting '{gene_name}' ...")
        try:
            result = subprocess.run(
                [py, _EXTRACT_SCRIPT, gene_name],
                capture_output=True, text=True, timeout=120, cwd=_addon_dir,
            )
            if result.returncode != 0:
                err = result.stderr.strip().split('\n')[-1] if result.stderr else "Unknown"
                self.report({'ERROR'}, f"Failed: {err}")
                return {'CANCELLED'}
            if reload_gene_data() and gene_name in _gene_values:
                n_pos = int((_gene_values[gene_name] > 0).sum())
                self.report({'INFO'}, f"Added '{gene_name}': {n_pos} expressing")
                props.custom_gene_name = ""
            else:
                self.report({'WARNING'}, f"Extracted but not found in CSV")
        except FileNotFoundError:
            self.report({'ERROR'}, f"Python not found at '{py}'")
        except subprocess.TimeoutExpired:
            self.report({'ERROR'}, "Timed out")
        except Exception as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}


class GENEPAINT_OT_paint(Operator):
    bl_idname = "genepaint.paint"
    bl_label = "Paint Gene"
    bl_description = "Create a new colored layer for the selected gene"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not load_gene_data():
            self.report({'ERROR'}, "Gene data not loaded")
            return {'CANCELLED'}

        props = context.scene.hox_gene_paint
        gene = props.selected_gene
        if gene == 'NONE' or gene not in _gene_values:
            self.report({'ERROR'}, f"Gene '{gene}' not available")
            return {'CANCELLED'}

        values = _gene_values[gene]
        n_cells = len(values)
        size = props.point_size
        show_zero = props.show_nonexpr
        cutoff = props.expr_cutoff
        r_min, r_max = props.range_min, props.range_max
        if r_max <= r_min:
            r_max = r_min + 0.01
        c_low = tuple(props.color_low)
        c_high = tuple(props.color_high)
        n_expr = int((values > cutoff).sum())

        col_name = f"GeneExpr_{gene}"

        # Remove existing layer for this gene
        for i, layer in enumerate(props.layers):
            if layer.gene == gene:
                existing = bpy.data.collections.get(layer.collection_name)
                if existing:
                    remove_collection_recursive(existing)
                props.layers.remove(i)
                break

        parent = bpy.data.collections.get(PARENT_COL)
        if not parent:
            parent = bpy.data.collections.new(PARENT_COL)
            bpy.context.scene.collection.children.link(parent)

        gene_col = bpy.data.collections.new(col_name)
        parent.children.link(gene_col)

        bins = np.zeros(n_cells, dtype=np.int32)
        expressing = values > cutoff
        remapped = np.clip((values - r_min) / (r_max - r_min), 0.0, 1.0)
        bins[expressing] = np.clip(
            (remapped[expressing] * (N_BINS - 1)).astype(np.int32) + 1, 1, N_BINS
        )

        n_created = 0
        for b in range(N_BINS + 1):
            mask = bins == b
            count = int(mask.sum())
            if count == 0:
                continue
            if b == 0 and not show_zero:
                continue

            if b == 0:
                r, g, bl_c = 0.12, 0.12, 0.12
                emission = 0.1
            else:
                t = b / N_BINS
                r, g, bl_c = gene_color_ramp(gene, t, color_low=c_low, color_high=c_high)
                emission = 0.3 + 0.7 * t

            bin_label = f"{col_name}_b{b:02d}"
            mat = make_material(f"mat_{bin_label}", r, g, bl_c, emission)
            centers = _gene_coords[mask]
            mesh = build_cubes_mesh(f"mesh_{bin_label}", centers, size)
            obj = bpy.data.objects.new(bin_label, mesh)
            obj.data.materials.append(mat)
            gene_col.objects.link(obj)
            n_created += count

        layer = props.layers.add()
        cutoff_str = f" cut>{cutoff:.2f}" if cutoff > 0 else ""
        layer.layer_name = f"{gene} sz={size:.3f}{cutoff_str}"
        layer.collection_name = col_name
        layer.gene = gene
        layer.n_expressing = n_expr
        layer.n_total = n_cells
        layer.point_size = size
        layer.color = c_high
        layer.visible = True
        props.active_layer_index = len(props.layers) - 1

        self.report({'INFO'}, f"Painted {gene}: {n_expr} expressing")
        return {'FINISHED'}


class GENEPAINT_OT_remove_layer(Operator):
    bl_idname = "genepaint.remove_layer"
    bl_label = "Remove Layer"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.hox_gene_paint
        idx = props.active_layer_index
        if idx < 0 or idx >= len(props.layers):
            return {'CANCELLED'}
        layer = props.layers[idx]
        col = bpy.data.collections.get(layer.collection_name)
        if col:
            remove_collection_recursive(col)
        props.layers.remove(idx)
        if props.active_layer_index >= len(props.layers):
            props.active_layer_index = max(0, len(props.layers) - 1)
        return {'FINISHED'}


class GENEPAINT_OT_clear_all(Operator):
    bl_idname = "genepaint.clear_all"
    bl_label = "Clear All"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.hox_gene_paint
        for layer in props.layers:
            col = bpy.data.collections.get(layer.collection_name)
            if col:
                remove_collection_recursive(col)
        props.layers.clear()
        props.active_layer_index = 0
        parent = bpy.data.collections.get(PARENT_COL)
        if parent and len(parent.children) == 0 and len(parent.objects) == 0:
            bpy.data.collections.remove(parent)
        self.report({'INFO'}, "Cleared all gene layers")
        return {'FINISHED'}


class GENEPAINT_OT_toggle_layer(Operator):
    bl_idname = "genepaint.toggle_layer"
    bl_label = "Toggle Layer"
    layer_index: IntProperty()

    def execute(self, context):
        props = context.scene.hox_gene_paint
        if self.layer_index < 0 or self.layer_index >= len(props.layers):
            return {'CANCELLED'}
        layer = props.layers[self.layer_index]
        col = bpy.data.collections.get(layer.collection_name)
        if col:
            layer.visible = not layer.visible
            col.hide_viewport = not layer.visible
        return {'FINISHED'}


class GENEPAINT_UL_layers(UIList):
    bl_idname = "GENEPAINT_UL_layers"

    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            vis_icon = 'HIDE_OFF' if item.visible else 'HIDE_ON'
            op = row.operator("genepaint.toggle_layer", text="", icon=vis_icon, emboss=False)
            op.layer_index = index
            row.prop(item, "color", text="")
            row.label(text=f"{item.gene} ({item.n_expressing}/{item.n_total})")


class GENEPAINT_PT_panel(Panel):
    bl_label = "Gene Expression Painter"
    bl_idname = "GENEPAINT_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Hox Morphospace"
    bl_order = 4

    def draw(self, context):
        layout = self.layout
        props = context.scene.hox_gene_paint

        box = layout.box()
        if _gene_data is not None:
            box.label(text=f"Data: {len(_gene_data)} cells, {len(_gene_columns)} genes",
                      icon='CHECKMARK')
        else:
            box.label(text="Gene data not loaded", icon='ERROR')
            box.operator("genepaint.load", icon='FILE_REFRESH')
            return

        box = layout.box()
        box.label(text="Select Gene", icon='RNA')
        box.prop(props, "selected_gene", text="")
        gene = props.selected_gene
        if gene and gene != 'NONE' and gene in _gene_values:
            info = CURATED_GENES.get(gene, {})
            col = box.column(align=True)
            col.scale_y = 0.8
            if info:
                col.label(text=f"  {info.get('full', gene)}")
            n_pos = int((_gene_values[gene] > 0).sum())
            col.label(text=f"  {n_pos}/{len(_gene_data)} expressing")

        box = layout.box()
        box.label(text="Add Custom Gene", icon='VIEWZOOM')
        row = box.row(align=True)
        row.prop(props, "custom_gene_name", text="", icon='RNA')
        row.operator("genepaint.add_custom", text="Add", icon='IMPORT')

        box = layout.box()
        box.label(text="Settings", icon='PREFERENCES')
        row = box.row(align=True)
        row.label(text="Low:")
        row.prop(props, "color_low", text="")
        row.label(text="High:")
        row.prop(props, "color_high", text="")
        row = box.row(align=True)
        row.prop(props, "range_min", text="Range Min")
        row.prop(props, "range_max", text="Range Max")
        box.prop(props, "expr_cutoff")
        box.prop(props, "point_size")
        box.prop(props, "show_nonexpr")

        box = layout.box()
        row = box.row(align=True)
        row.scale_y = 1.5
        row.operator("genepaint.paint", icon='BRUSH_DATA', text="Paint Gene")

        box = layout.box()
        box.label(text=f"Painted Layers ({len(props.layers)})", icon='OUTLINER_COLLECTION')
        if len(props.layers) > 0:
            box.template_list(
                "GENEPAINT_UL_layers", "",
                props, "layers", props, "active_layer_index",
                rows=min(6, len(props.layers)),
            )
            row = box.row(align=True)
            row.operator("genepaint.remove_layer", icon='REMOVE', text="Remove")
            row.operator("genepaint.clear_all", icon='TRASH', text="Clear All")
        else:
            box.label(text="No layers yet.")

        box = layout.box()
        box.prop(props, "python_path", text="Python")
        box.scale_y = 0.75
        box.label(text="  System Python for custom genes (auto-detected)")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 4: SCENE BUILDER — Import morphospace from CSV files
# ═══════════════════════════════════════════════════════════════════════════════

import math
import bmesh

# Pair definitions (must match generate_morphospace_hoxaxes.py)
_PAIRS = [
    {'id':'P1', 'clusters':[4,11],  'color':'#0000CD', 'label':'P1 (C4 & C11)'},
    {'id':'P2', 'clusters':[5,0],   'color':'#CC0000', 'label':'P2 (C5 & C0)'},
    {'id':'P3', 'clusters':[26,18], 'color':'#006400', 'label':'P3 (C26 & C18)'},
    {'id':'P4', 'clusters':[1,36],  'color':'#C71585', 'label':'P4 (C1 & C36)'},
    {'id':'P5', 'clusters':[13,8],  'color':'#7B2D8E', 'label':'P5 (C13 & C8)'},
    {'id':'P6', 'clusters':[2,9],   'color':'#CC8400', 'label':'P6 (C2 & C9)'},
    {'id':'P7', 'clusters':[6,10],  'color':'#008B8B', 'label':'P7 (C6 & C10)'},
    {'id':'P8', 'clusters':[14,19], 'color':'#D2691E', 'label':'P8 (C14 & C19)'},
]
_TRIS = [
    {'id':'P9',  'clusters':[30,31,32], 'color':'#FF1493', 'label':'P9 (C20 3-way)'},
    {'id':'P10', 'clusters':[33,34,35], 'color':'#9ACD32', 'label':'P10 (C7 3-way)'},
]


def _hex_to_rgba(h):
    h = h.lstrip('#')
    return (int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255, 1.0)


def _make_scene_material(name, hex_color, alpha=1.0, emit=0.0):
    """Material with optional transparency for scene builder."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name=name)
    r, g, b, _ = _hex_to_rgba(hex_color)
    mat.diffuse_color = (r, g, b, alpha)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.7
    bsdf.inputs['Alpha'].default_value = alpha
    if emit > 0:
        bsdf.inputs['Emission Strength'].default_value = emit
        bsdf.inputs['Emission Color'].default_value = (r, g, b, 1.0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    if alpha < 1.0:
        try:
            mat.blend_method = 'BLEND'
            mat.shadow_method = 'NONE'
        except AttributeError:
            pass
    return mat


def _make_collection(name, parent=None):
    col = bpy.data.collections.new(name)
    if parent is None:
        bpy.context.scene.collection.children.link(col)
    else:
        parent.children.link(col)
    return col


def _link_to_collection(obj, collection):
    for col in obj.users_collection:
        col.objects.unlink(obj)
    collection.objects.link(obj)


def _create_curve(name, points, hex_color, width, collection):
    curve_data = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = width
    curve_data.bevel_resolution = 3
    spline = curve_data.splines.new('POLY')
    spline.points.add(len(points) - 1)
    for i, p in enumerate(points):
        spline.points[i].co = (p[0], p[1], p[2], 1)
    obj = bpy.data.objects.new(name, curve_data)
    mat = _make_scene_material(f"mat_{name}", hex_color, emit=0.3)
    obj.data.materials.append(mat)
    collection.objects.link(obj)
    return obj


def _create_plane_quad(name, center, normal, hex_color, alpha, size, collection):
    n = list(normal)
    nl = math.sqrt(sum(x*x for x in n))
    n = [x/nl for x in n]
    tmp = [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]
    e1 = [n[1]*tmp[2]-n[2]*tmp[1], n[2]*tmp[0]-n[0]*tmp[2], n[0]*tmp[1]-n[1]*tmp[0]]
    e1l = math.sqrt(sum(x*x for x in e1))
    e1 = [x/e1l for x in e1]
    e2 = [n[1]*e1[2]-n[2]*e1[1], n[2]*e1[0]-n[0]*e1[2], n[0]*e1[1]-n[1]*e1[0]]
    verts = []
    for s in [-size, size]:
        for t in [-size, size]:
            verts.append((center[0]+s*e1[0]+t*e2[0],
                          center[1]+s*e1[1]+t*e2[1],
                          center[2]+s*e1[2]+t*e2[2]))
    mesh = bpy.data.meshes.new(f"mesh_{name}")
    mesh.from_pydata(verts, [], [(0,1,3,2)])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    mat = _make_scene_material(f"mat_{name}", hex_color, alpha=alpha, emit=0.1)
    obj.data.materials.append(mat)
    collection.objects.link(obj)
    return obj


class HoxSceneBuilderProps(PropertyGroup):
    data_dir: StringProperty(
        name="Data Directory",
        default="",
        subtype='DIR_PATH',
        description="Directory containing blender_cells.csv, blender_polytope.csv, etc.",
    )
    cell_size: FloatProperty(
        name="Cell Size", default=0.008,
        min=0.002, max=0.03, step=1, precision=3,
    )
    edge_width: FloatProperty(
        name="Edge Width", default=0.005,
        min=0.001, max=0.02, step=1, precision=3,
    )
    vertex_size: FloatProperty(
        name="Vertex Size", default=0.02,
        min=0.005, max=0.05, step=1, precision=3,
    )
    clear_scene: BoolProperty(
        name="Clear Scene First", default=True,
        description="Remove all existing objects before building",
    )
    build_planes: BoolProperty(
        name="Build Coordinate Planes", default=True,
        description="Create H1=0, H2=0, H3=0 planes",
    )
    build_bbox: BoolProperty(
        name="Build Bounding Boxes", default=True,
    )


class HOX_OT_build_scene(Operator):
    bl_idname = "hox.build_scene"
    bl_label = "Build Morphospace"
    bl_description = "Import cells, polytope, lines, and set up the full Hox morphospace scene"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.hox_scene_builder
        data_dir = bpy.path.abspath(props.data_dir)

        if not data_dir or not os.path.isdir(data_dir):
            self.report({'ERROR'}, "Set a valid data directory first")
            return {'CANCELLED'}

        cells_path = os.path.join(data_dir, "blender_cells.csv")
        verts_path = os.path.join(data_dir, "blender_polytope.csv")
        edges_path = os.path.join(data_dir, "blender_polytope_edges.csv")
        lines_path = os.path.join(data_dir, "blender_lines.csv")

        if not os.path.exists(cells_path):
            self.report({'ERROR'}, f"blender_cells.csv not found in {data_dir}")
            return {'CANCELLED'}

        # ── Clear scene ─────────────────────────────────────────────────
        if props.clear_scene:
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()
            for block in bpy.data.meshes:
                if block.users == 0: bpy.data.meshes.remove(block)
            for block in bpy.data.materials:
                if block.users == 0: bpy.data.materials.remove(block)
            for block in bpy.data.curves:
                if block.users == 0: bpy.data.curves.remove(block)
            for col in list(bpy.data.collections):
                bpy.data.collections.remove(col)

        CELL_SIZE = props.cell_size
        EDGE_WIDTH = props.edge_width
        VERTEX_SIZE = props.vertex_size

        # ── 1. Import cells ─────────────────────────────────────────────
        cluster_cells = {}
        with open(cells_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cl = int(row['leiden'])
                pos = (float(row['H1']), float(row['H2']), float(row['H3']))
                color = row['color_hex']
                if cl not in cluster_cells:
                    cluster_cells[cl] = []
                cluster_cells[cl].append((pos, color))

        total_cells = sum(len(v) for v in cluster_cells.values())
        cells_root = _make_collection("Cells")

        # Base sphere template
        bpy.ops.mesh.primitive_ico_sphere_add(radius=CELL_SIZE, subdivisions=2)
        base_sphere = bpy.context.active_object
        base_sphere.name = "_cell_template"
        base_sphere.hide_set(True)
        base_sphere.hide_render = True
        _link_to_collection(base_sphere, cells_root)

        def add_cluster(cluster_id, cell_list, parent_col, display_name=None):
            if not cell_list:
                return
            name = display_name or f"C{cluster_id}"
            color_hex = cell_list[0][1]
            mesh = bpy.data.meshes.new(f"mesh_{name}")
            verts = [c[0] for c in cell_list]
            mesh.from_pydata(verts, [], [])
            mesh.update()
            obj = bpy.data.objects.new(name, mesh)
            parent_col.objects.link(obj)
            obj.instance_type = 'VERTS'
            child = base_sphere.copy()
            child.data = base_sphere.data.copy()
            child.name = f"_sphere_{name}"
            child.parent = obj
            child.hide_set(False)
            child.hide_render = False
            mat = _make_scene_material(f"mat_{name}", color_hex, emit=0.4)
            child.data.materials.clear()
            child.data.materials.append(mat)
            parent_col.objects.link(child)

        assigned = set()
        for p in _PAIRS:
            pair_col = _make_collection(f"{p['id']}_{p['label']}", parent=cells_root)
            for cl in p['clusters']:
                if cl in cluster_cells:
                    add_cluster(cl, cluster_cells[cl], pair_col, f"{p['id']}_C{cl}")
                    assigned.add(cl)

        for t in _TRIS:
            tri_col = _make_collection(f"{t['id']}_{t['label']}", parent=cells_root)
            for cl in t['clusters']:
                if cl in cluster_cells:
                    add_cluster(cl, cluster_cells[cl], tri_col, f"{t['id']}_C{cl}")
                    assigned.add(cl)

        unmatched = [cl for cl in cluster_cells if cl not in assigned]
        if unmatched:
            unm_col = _make_collection("Unmatched", parent=cells_root)
            for cl in sorted(unmatched):
                add_cluster(cl, cluster_cells[cl], unm_col)

        # ── 2. Polytope ─────────────────────────────────────────────────
        polytope_col = _make_collection("Polytope")
        poly_verts = []
        poly_labels = []

        if os.path.exists(verts_path):
            with open(verts_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    poly_verts.append((float(row['H1']), float(row['H2']), float(row['H3'])))
                    poly_labels.append(row['vertex_id'])

            for pos, label in zip(poly_verts, poly_labels):
                bpy.ops.mesh.primitive_ico_sphere_add(
                    radius=VERTEX_SIZE, location=pos, subdivisions=2)
                obj = bpy.context.active_object
                obj.name = f"polytope_{label}"
                mat = _make_scene_material("mat_polytope_vert", "#222222", emit=0.5)
                obj.data.materials.append(mat)
                _link_to_collection(obj, polytope_col)

                bpy.ops.object.text_add(
                    location=(pos[0]+0.03, pos[1]+0.03, pos[2]+0.03))
                txt = bpy.context.active_object
                txt.data.body = label
                txt.data.size = 0.05
                txt.name = f"label_{label}"
                mat_txt = _make_scene_material("mat_label", "#FFFFFF", emit=1.0)
                txt.data.materials.append(mat_txt)
                _link_to_collection(txt, polytope_col)

        if os.path.exists(edges_path):
            with open(edges_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    a, b = int(row['v_start']), int(row['v_end'])
                    if a < len(poly_verts) and b < len(poly_verts):
                        _create_curve(
                            f"edge_{poly_labels[a]}_{poly_labels[b]}",
                            [poly_verts[a], poly_verts[b]],
                            "#404040", EDGE_WIDTH, polytope_col)

        # Convex hull face mesh
        if poly_verts:
            pmesh = bpy.data.meshes.new("polytope_faces_mesh")
            pmesh.from_pydata(poly_verts, [], [])
            pmesh.update()
            pobj = bpy.data.objects.new("polytope_faces", pmesh)
            polytope_col.objects.link(pobj)
            bm = bmesh.new()
            bm.from_mesh(pmesh)
            bmesh.ops.convex_hull(bm, input=bm.verts)
            bm.to_mesh(pmesh)
            bm.free()
            pmesh.update()
            mat_face = _make_scene_material("mat_polytope_face", "#6496FF", alpha=0.2, emit=0.05)
            pobj.data.materials.append(mat_face)

        # ── 3. Coordinate planes ────────────────────────────────────────
        if props.build_planes:
            planes_col = _make_collection("Planes")
            for pname, pcenter, pnormal, pcolor, palpha in [
                ("H1=0 (Antp=Ubx)",       (0,0,0), (1,0,0), "#1E64DC", 0.25),
                ("H2=0 (abdA=AbdB)",       (0,0,0), (0,1,0), "#DC3232", 0.25),
                ("H3=0 (thorax=abdom)",    (0,0,0), (0,0,1), "#DCB400", 0.25),
            ]:
                _create_plane_quad(pname, pcenter, pnormal, pcolor, palpha,
                                   2.5, planes_col)

        # ── 4. Named lines ──────────────────────────────────────────────
        lines_col = _make_collection("Lines")
        if os.path.exists(lines_path):
            with open(lines_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    p1 = (float(row['x1']), float(row['y1']), float(row['z1']))
                    p2 = (float(row['x2']), float(row['y2']), float(row['z2']))
                    _create_curve(row['name'], [p1, p2], row['color_hex'],
                                  0.007, lines_col)

        # ── 5. Centers ──────────────────────────────────────────────────
        centers_col = _make_collection("Centers")
        for cname, cpos, ccolor in [
            ("Center_C",      (0, 0, 0),              "#FFFFFF"),
            ("Center_C_star", (-0.121, -0.014, 0.248), "#FFFF00"),
        ]:
            bpy.ops.mesh.primitive_ico_sphere_add(
                radius=0.025, location=cpos, subdivisions=3)
            obj = bpy.context.active_object
            obj.name = cname
            mat = _make_scene_material(f"mat_{cname}", ccolor, emit=0.8)
            obj.data.materials.append(mat)
            _link_to_collection(obj, centers_col)

        # ── 6. Bounding boxes ───────────────────────────────────────────
        if props.build_bbox:
            bbox_col = _make_collection("BoundingBoxes")
            for bname, mins, maxs, bcolor in [
                ("TheoBBox", (-1,-1,-2), (1,1,2), "#B40000"),
                ("DataBBox", (-0.87,-0.99,-1.36), (0.79,0.99,1.68), "#333333"),
            ]:
                corners = []
                for x in [mins[0], maxs[0]]:
                    for y in [mins[1], maxs[1]]:
                        for z in [mins[2], maxs[2]]:
                            corners.append((x, y, z))
                edges = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
                for ei, (a, b) in enumerate(edges):
                    _create_curve(f"{bname}_e{ei}", [corners[a], corners[b]],
                                  bcolor, 0.002, bbox_col)

        # ── 7. Lighting + Camera ────────────────────────────────────────
        light_col = _make_collection("Lighting")

        bpy.ops.object.light_add(type='SUN', location=(3, -3, 5))
        sun = bpy.context.active_object
        sun.name = "Sun"
        sun.data.energy = 2.0
        _link_to_collection(sun, light_col)

        bpy.ops.object.light_add(type='SUN', location=(-3, 3, 2))
        fill = bpy.context.active_object
        fill.name = "Fill"
        fill.data.energy = 1.0
        _link_to_collection(fill, light_col)

        bpy.ops.object.light_add(type='AREA', location=(0, 0, 4))
        area = bpy.context.active_object
        area.name = "TopArea"
        area.data.energy = 50
        area.data.size = 6
        _link_to_collection(area, light_col)

        bpy.ops.object.camera_add(location=(4, -3, 2.5))
        cam = bpy.context.active_object
        cam.name = "Camera"
        cam.rotation_euler = (1.1, 0, 0.8)
        bpy.context.scene.camera = cam
        _link_to_collection(cam, light_col)

        # Viewport to Material Preview
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'MATERIAL'
                        space.shading.use_scene_lights = True

        # EEVEE for transparency
        try:
            bpy.context.scene.render.engine = (
                'BLENDER_EEVEE_NEXT' if bpy.app.version >= (4, 0, 0)
                else 'BLENDER_EEVEE'
            )
        except Exception:
            pass

        self.report({'INFO'},
            f"Built morphospace: {total_cells} cells, "
            f"{len(poly_verts)} vertices, {len(assigned)} cluster pairs")
        return {'FINISHED'}


class HOX_PT_scene_builder_panel(Panel):
    bl_label = "Scene Builder"
    bl_idname = "HOX_PT_scene_builder_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Hox Morphospace"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        props = context.scene.hox_scene_builder

        box = layout.box()
        box.label(text="Data Source", icon='FILE_FOLDER')
        box.prop(props, "data_dir", text="")

        data_dir = bpy.path.abspath(props.data_dir)
        if data_dir and os.path.isdir(data_dir):
            cells_ok = os.path.exists(os.path.join(data_dir, "blender_cells.csv"))
            poly_ok = os.path.exists(os.path.join(data_dir, "blender_polytope.csv"))
            lines_ok = os.path.exists(os.path.join(data_dir, "blender_lines.csv"))

            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text=f"  Cells CSV: {'found' if cells_ok else 'MISSING'}",
                       icon='CHECKMARK' if cells_ok else 'ERROR')
            col.label(text=f"  Polytope CSV: {'found' if poly_ok else 'MISSING'}",
                       icon='CHECKMARK' if poly_ok else 'ERROR')
            col.label(text=f"  Lines CSV: {'found' if lines_ok else 'optional'}",
                       icon='CHECKMARK' if lines_ok else 'INFO')

        box = layout.box()
        box.label(text="Settings", icon='PREFERENCES')
        box.prop(props, "cell_size")
        box.prop(props, "edge_width")
        box.prop(props, "vertex_size")
        box.prop(props, "clear_scene")
        box.prop(props, "build_planes")
        box.prop(props, "build_bbox")

        box = layout.box()
        row = box.row(align=True)
        row.scale_y = 2.0
        row.operator("hox.build_scene", icon='SCENE_DATA', text="Build Morphospace")


# ═══════════════════════════════════════════════════════════════════════════════
#  REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

classes = (
    # Scene Builder
    HoxSceneBuilderProps,
    HOX_OT_build_scene,
    HOX_PT_scene_builder_panel,
    # Plane Creator
    HoxDiamondProperties,
    HOX_OT_generate_stripes,
    HOX_OT_clear_stripes,
    HOX_PT_diamond_panel,
    # Cell Layers
    HoxLayerItem,
    HoxLayerProperties,
    HOX_OT_load_metadata,
    HOX_OT_add_layer,
    HOX_OT_remove_layer,
    HOX_OT_clear_all_layers,
    HOX_OT_toggle_layer,
    HOX_OT_update_layer_color,
    HOX_UL_layer_list,
    HOX_PT_cell_layers_panel,
    # Gene Painter
    GenePaintLayerItem,
    HoxGenePaintProps,
    GENEPAINT_OT_load,
    GENEPAINT_OT_add_custom,
    GENEPAINT_OT_paint,
    GENEPAINT_OT_remove_layer,
    GENEPAINT_OT_clear_all,
    GENEPAINT_OT_toggle_layer,
    GENEPAINT_UL_layers,
    GENEPAINT_PT_panel,
)


def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except RuntimeError:
            bpy.utils.unregister_class(cls)
            bpy.utils.register_class(cls)

    bpy.types.Scene.hox_scene_builder = PointerProperty(type=HoxSceneBuilderProps)
    bpy.types.Scene.hox_diamond = PointerProperty(type=HoxDiamondProperties)
    bpy.types.Scene.hox_layers = PointerProperty(type=HoxLayerProperties)
    bpy.types.Scene.hox_gene_paint = PointerProperty(type=HoxGenePaintProps)


def unregister():
    del bpy.types.Scene.hox_gene_paint
    del bpy.types.Scene.hox_layers
    del bpy.types.Scene.hox_diamond
    del bpy.types.Scene.hox_scene_builder
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
    print("\n" + "=" * 55)
    print("  Hox Morphospace Toolkit v2.0 loaded!")
    print("=" * 55)
    print("3D Viewport > Sidebar (N) > Hox Morphospace tab")
    print("  1. Scene Builder")
    print("  2. Plane Creator")
    print("  3. Cell Layers")
    print("  4. Gene Expression Painter")
