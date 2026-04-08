import bpy


class IRIDIS_PG_Settings(bpy.types.PropertyGroup):
    preset: bpy.props.EnumProperty(
        name="Preset",
        description="Material context preset",
        items=[
            ('GENERIC', "Generic Neutral", ""),
            ('RAW_METAL', "Raw Metal", ""),
            ('CORRODED_METAL', "Corroded Metal", ""),
            ('PAINTED_METAL', "Painted Metal", ""),
            ('CONCRETE', "Concrete / Stone", ""),
            ('WOOD', "Wood", ""),
            ('PLASTIC', "Plastic", ""),
        ],
        default='GENERIC',
    )

    output_dir: bpy.props.StringProperty(
        name="Output Folder",
        description="Folder where generated maps will be saved",
        subtype='DIR_PATH',
        default="",
    )

    base_name: bpy.props.StringProperty(
        name="Base Name",
        description="Base name for generated files",
        default="iridis_surface",
    )

    overwrite_existing: bpy.props.BoolProperty(
        name="Overwrite Existing",
        description="Overwrite files with the same name",
        default=True,
    )

    generate_albedo: bpy.props.BoolProperty(
        name="Albedo",
        default=True,
    )

    generate_roughness: bpy.props.BoolProperty(
        name="Roughness",
        default=True,
    )

    generate_metallic: bpy.props.BoolProperty(
        name="Metallic",
        default=True,
    )

    generate_normal: bpy.props.BoolProperty(
        name="Normal",
        default=True,
    )

    generate_height: bpy.props.BoolProperty(
        name="Height",
        default=True,
    )

    generate_aux_masks: bpy.props.BoolProperty(
        name="Auxiliary Masks",
        description="Reserved for later stages",
        default=False,
    )

    use_enhanced_relief_analysis: bpy.props.BoolProperty(
        name="Use enhanced relief analysis",
        description="Improves normal and height generation. Processing may take longer and Blender may become temporarily unresponsive on some textures.",
        default=False,
    )

    delight_strength: bpy.props.FloatProperty(
        name="Delight Strength",
        description="Low-frequency lighting removal strength for albedo",
        min=0.0,
        max=1.0,
        default=0.5,
    )

    highlight_suppression: bpy.props.FloatProperty(
        name="Highlight Suppression",
        description="Highlight compression strength for albedo",
        min=0.0,
        max=1.0,
        default=0.5,
    )

    color_preservation: bpy.props.FloatProperty(
        name="Color Preservation",
        description="Preserve original color while flattening albedo",
        min=0.0,
        max=1.0,
        default=0.6,
    )

    roughness_bias: bpy.props.FloatProperty(
        name="Base Roughness Bias",
        description="Bias the base roughness",
        min=-1.0,
        max=1.0,
        default=0.0,
    )

    microdetail_influence: bpy.props.FloatProperty(
        name="Microdetail Influence",
        description="Influence of high-frequency detail",
        min=0.0,
        max=2.0,
        default=1.0,
    )

    cavity_boost: bpy.props.FloatProperty(
        name="Cavity Boost",
        description="Boost roughness in cavities/dark residue regions",
        min=0.0,
        max=2.0,
        default=1.0,
    )

    metallic_bias: bpy.props.FloatProperty(
        name="Metallic Bias",
        description="Bias the metallic estimation",
        min=-1.0,
        max=1.0,
        default=0.0,
    )

    paint_color_rejection: bpy.props.FloatProperty(
        name="Paint / Color Rejection",
        description="Reject colorful regions from metallic",
        min=0.0,
        max=1.0,
        default=0.5,
    )

    threshold_softness: bpy.props.FloatProperty(
        name="Threshold Softness",
        description="Softness of metallic decision threshold",
        min=0.0,
        max=1.0,
        default=0.5,
    )

    normal_strength: bpy.props.FloatProperty(
        name="Normal Strength",
        description="Strength of generated normal map",
        min=0.0,
        max=5.0,
        default=1.0,
    )

    normal_mid_detail_influence: bpy.props.FloatProperty(
        name="Mid Detail Influence",
        description="How much medium frequencies influence the normal",
        min=0.0,
        max=2.0,
        default=1.0,
    )

    normal_smoothing: bpy.props.FloatProperty(
        name="Smoothing",
        description="Smoothing applied before normal generation",
        min=0.0,
        max=1.0,
        default=0.5,
    )

    normal_format: bpy.props.EnumProperty(
        name="Normal Format",
        description="Normal map convention",
        items=[
            ('OPENGL', "OpenGL", ""),
            ('DIRECTX', "DirectX", ""),
        ],
        default='OPENGL',
    )

    height_contrast: bpy.props.FloatProperty(
        name="Height Contrast",
        description="Final height contrast",
        min=0.0,
        max=5.0,
        default=1.0,
    )

    height_smoothing: bpy.props.FloatProperty(
        name="Height Smoothing",
        description="Smoothing applied to the height map",
        min=0.0,
        max=1.0,
        default=0.5,
    )

    macro_relief_weight: bpy.props.FloatProperty(
        name="Macro Relief Weight",
        description="Weight of medium-scale relief in height generation",
        min=0.0,
        max=2.0,
        default=1.0,
    )

    debug_print: bpy.props.BoolProperty(
        name="Debug Print",
        description="Print debug info to Blender console",
        default=False,
    )