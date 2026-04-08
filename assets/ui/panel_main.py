import bpy


class IRIDIS_PT_main_panel(bpy.types.Panel):
    bl_label = "Iridis"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Iridis"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.iridis_settings

        space = context.space_data
        image = getattr(space, "image", None)

        box = layout.box()
        box.label(text="Source")
        if image:
            box.label(text=f"Active Image: {image.name}", icon='IMAGE_DATA')
            box.label(text=f"{image.size[0]} x {image.size[1]}")
        else:
            box.label(text="No active image", icon='ERROR')

        box = layout.box()
        box.label(text="Processing")
        box.prop(settings, "preset")
        box.prop(settings, "output_dir")
        box.prop(settings, "base_name")
        box.prop(settings, "use_enhanced_relief_analysis")
        row = box.row(align=True)
        row.prop(settings, "overwrite_existing")
        row.prop(settings, "generate_aux_masks")
        box.prop(settings, "debug_print")

        box = layout.box()
        box.label(text="Maps")
        col = box.column(align=True)
        row = col.row(align=True)
        row.prop(settings, "generate_albedo")
        row.prop(settings, "generate_roughness")
        row = col.row(align=True)
        row.prop(settings, "generate_metallic")
        row.prop(settings, "generate_normal")
        row.prop(settings, "generate_height")

        box = layout.box()
        box.label(text="Albedo")
        box.prop(settings, "delight_strength")
        box.prop(settings, "highlight_suppression")
        box.prop(settings, "color_preservation")

        box = layout.box()
        box.label(text="Roughness")
        box.prop(settings, "roughness_bias")
        box.prop(settings, "microdetail_influence")
        box.prop(settings, "cavity_boost")

        box = layout.box()
        box.label(text="Metallic")
        box.prop(settings, "metallic_bias")
        box.prop(settings, "paint_color_rejection")
        box.prop(settings, "threshold_softness")

        box = layout.box()
        box.label(text="Normal")
        box.prop(settings, "normal_strength")
        box.prop(settings, "normal_mid_detail_influence")
        box.prop(settings, "normal_smoothing")
        box.prop(settings, "normal_format")

        box = layout.box()
        box.label(text="Height")
        box.prop(settings, "height_contrast")
        box.prop(settings, "height_smoothing")
        box.prop(settings, "macro_relief_weight")

        help_box = layout.box()
        help_box.label(text="Notes")
        help_box.label(text="Auxiliary Masks now exports cavity, dirt,")
        help_box.label(text="metal mask and edge wear.")

        layout.separator()
        layout.operator("iridis.process", icon='NODE_TEXTURE')
