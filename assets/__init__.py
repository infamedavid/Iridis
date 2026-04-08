bl_info = {
    "name": "Iridis: Image Processing for Blender",
    "author": "InfameDavid",
    "version": (0, 1, 1),
    "blender": (5, 0, 0),
    "location": "Image Editor > Sidebar > Iridis",
    "description": "Generate approximate derived texture maps from a single UV-baked image",
    "category": "Image",
}

import bpy

from .core.properties import IRIDIS_PG_Settings
from .operators.op_process import IRIDIS_OT_process
from .ui.panel_main import IRIDIS_PT_main_panel


classes = (
    IRIDIS_PG_Settings,
    IRIDIS_OT_process,
    IRIDIS_PT_main_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.iridis_settings = bpy.props.PointerProperty(
        type=IRIDIS_PG_Settings
    )


def unregister():
    if hasattr(bpy.types.Scene, "iridis_settings"):
        del bpy.types.Scene.iridis_settings

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)