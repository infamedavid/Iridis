import bpy


class IRIDIS_PG_Settings(bpy.types.PropertyGroup):
    preset: bpy.props.EnumProperty(
        name="Preset",
        description=(
            "Sets the overall material profile used to guide map generation. "
            "Raise specificity by choosing a preset closer to your real surface when results feel generic or misclassified. "
            "Lower specificity by returning to Generic Neutral if a specialized preset pushes the look too far."
        ),
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
        description=(
            "Chooses where Iridis saves the generated texture maps. "
            "Raise organization by picking a dedicated folder when you want easier export and handoff. "
            "Lower folder sprawl by reusing your current project folder if files are getting scattered."
        ),
        subtype='DIR_PATH',
        default="",
    )

    base_name: bpy.props.StringProperty(
        name="Base Name",
        description=(
            "Sets the filename prefix shared by all exported maps. "
            "Raise clarity by using a more specific base name when managing multiple material versions. "
            "Lower name length by using a shorter base name if exported filenames become hard to scan."
        ),
        default="iridis_surface",
    )

    overwrite_existing: bpy.props.BoolProperty(
        name="Overwrite Existing",
        description=(
            "Controls whether exports replace older files with matching names. "
            "Raise it by turning this on when you are iterating and want one up-to-date set of maps. "
            "Lower it by turning this off when you need to keep earlier exports for comparison."
        ),
        default=True,
    )

    generate_albedo: bpy.props.BoolProperty(
        name="Albedo",
        description=(
            "Controls whether an Albedo map is generated on export. "
            "Raise it by turning this on when you need clean base color without lighting baked in. "
            "Lower it by turning this off when your workflow already has a usable base color map."
        ),
        default=True,
    )

    generate_roughness: bpy.props.BoolProperty(
        name="Roughness",
        description=(
            "Controls whether a Roughness map is generated on export. "
            "Raise it by turning this on when your material needs clear gloss-versus-matte variation. "
            "Lower it by turning this off when roughness is handled elsewhere in your shader setup."
        ),
        default=True,
    )

    generate_metallic: bpy.props.BoolProperty(
        name="Metallic",
        description=(
            "Controls whether a Metallic map is generated on export. "
            "Raise it by turning this on when you need metal/non-metal separation from the source image. "
            "Lower it by turning this off when the material is fully dielectric or uses a fixed metallic value."
        ),
        default=True,
    )

    generate_normal: bpy.props.BoolProperty(
        name="Normal",
        description=(
            "Controls whether a Normal map is generated on export. "
            "Raise it by turning this on when you want surface detail to read through lighting changes. "
            "Lower it by turning this off when your asset already has normals from sculpt or bake."
        ),
        default=True,
    )

    generate_height: bpy.props.BoolProperty(
        name="Height",
        description=(
            "Controls whether a Height map is generated on export. "
            "Raise it by turning this on when you need displacement or parallax depth from the texture. "
            "Lower it by turning this off when only normal detail is needed."
        ),
        default=True,
    )

    enable_heavier_relief: bpy.props.BoolProperty(
        name="Heavier Relief (Normal/Height)",
        description=(
            "Controls a stronger relief interpretation for Normal and Height generation. "
            "Raise it by turning this on if surface depth still looks too subtle or flat. "
            "Lower it by turning this off if relief starts looking exaggerated or crunchy."
        ),
        default=False,
    )

    enable_region_stabilization: bpy.props.BoolProperty(
        name="Region Stabilization",
        description=(
            "Improves stability for difficult materials such as painted metal, corrosion, labels, and noisy regions, "
            "but increases processing time."
        ),
        default=False,
    )

    delight_strength: bpy.props.FloatProperty(
        name="Delight Strength",
        description=(
            "Controls how strongly shading and lighting are removed from the Albedo result. "
            "Raise this if the albedo still looks like it has baked-in light or shadow gradients. "
            "Lower this if the albedo starts looking washed out or loses natural color depth."
        ),
        min=0.0,
        max=1.0,
        default=0.5,
    )

    highlight_suppression: bpy.props.FloatProperty(
        name="Highlight Suppression",
        description=(
            "Controls how much bright glare and specular hotspots are reduced in Albedo. "
            "Raise this if shiny spots or bright streaks still show up as painted color. "
            "Lower this if bright areas become dull and the texture loses believable contrast."
        ),
        min=0.0,
        max=1.0,
        default=0.5,
    )

    color_preservation: bpy.props.FloatProperty(
        name="Color Preservation",
        description=(
            "Controls how much original color richness is kept while flattening Albedo lighting. "
            "Raise this if the albedo looks too gray or muted after processing. "
            "Lower this if uneven source lighting is leaking back into the color map."
        ),
        min=0.0,
        max=1.0,
        default=0.6,
    )

    roughness_bias: bpy.props.FloatProperty(
        name="Base Roughness Bias",
        description=(
            "Controls the overall roughness level across the whole material. "
            "Raise this if the surface looks too glossy or wet in the renderer. "
            "Lower this if the surface looks too chalky or matte."
        ),
        min=-1.0,
        max=1.0,
        default=0.0,
    )

    microdetail_influence: bpy.props.FloatProperty(
        name="Microdetail Influence",
        description=(
            "Controls how much fine texture detail affects Roughness variation. "
            "Raise this if tiny scratches, grain, or pores are not reading in reflections. "
            "Lower this if roughness looks noisy or sparkly at small scale."
        ),
        min=0.0,
        max=2.0,
        default=1.0,
    )

    cavity_boost: bpy.props.FloatProperty(
        name="Cavity Boost",
        description=(
            "Controls how much darker recessed areas are pushed rougher than exposed areas. "
            "Raise this if crevices should look dustier, drier, or less reflective. "
            "Lower this if cracks and cavities become too flat, dark, or overemphasized."
        ),
        min=0.0,
        max=2.0,
        default=1.0,
    )

    metallic_bias: bpy.props.FloatProperty(
        name="Metallic Bias",
        description=(
            "Controls the overall tendency to classify pixels as metal. "
            "Raise this if true metal regions are not being detected strongly enough. "
            "Lower this if painted or non-metal areas are being tagged as metal too often."
        ),
        min=-1.0,
        max=1.0,
        default=0.0,
    )

    paint_color_rejection: bpy.props.FloatProperty(
        name="Paint / Color Rejection",
        description=(
            "Controls how strongly colorful painted areas are excluded from Metallic output. "
            "Raise this if paint layers are incorrectly showing up as metal. "
            "Lower this if real metal with color tint is being removed too aggressively."
        ),
        min=0.0,
        max=1.0,
        default=0.5,
    )

    threshold_softness: bpy.props.FloatProperty(
        name="Threshold Softness",
        description=(
            "Controls how gradual the transition is between metal and non-metal areas. "
            "Raise this if metallic edges look harsh, jagged, or posterized. "
            "Lower this if metal regions look too blurry or indecisive."
        ),
        min=0.0,
        max=1.0,
        default=0.5,
    )

    normal_strength: bpy.props.FloatProperty(
        name="Normal Strength",
        description=(
            "Controls the apparent depth strength of the generated Normal map. "
            "Raise this if surface relief still looks too flat under lighting. "
            "Lower this if normals look too sharp, bumpy, or noisy."
        ),
        min=0.0,
        max=5.0,
        default=1.0,
    )

    normal_mid_detail_influence: bpy.props.FloatProperty(
        name="Mid Detail Influence",
        description=(
            "Controls how much medium-scale forms contribute to the Normal map shape. "
            "Raise this if broader dents and forms are not showing clearly enough. "
            "Lower this if mid-size waviness starts overpowering fine detail."
        ),
        min=0.0,
        max=2.0,
        default=1.0,
    )

    normal_smoothing: bpy.props.FloatProperty(
        name="Smoothing",
        description=(
            "Controls pre-smoothing before the Normal map is built. "
            "Raise this if the normal map has grainy chatter or high-frequency noise. "
            "Lower this if useful detail is being blurred away."
        ),
        min=0.0,
        max=1.0,
        default=0.5,
    )

    normal_format: bpy.props.EnumProperty(
        name="Normal Format",
        description=(
            "Controls which tangent-space normal convention is exported. "
            "Raise compatibility by choosing the format your target engine expects when lighting looks inverted. "
            "Lower mismatch by switching to the other format if bumps appear as dents."
        ),
        items=[
            ('OPENGL', "OpenGL", ""),
            ('DIRECTX', "DirectX", ""),
        ],
        default='OPENGL',
    )

    height_contrast: bpy.props.FloatProperty(
        name="Height Contrast",
        description=(
            "Controls overall depth separation in the Height map. "
            "Raise this if height differences feel too weak or muddy. "
            "Lower this if displacement/parallax looks too extreme or clipped."
        ),
        min=0.0,
        max=5.0,
        default=1.0,
    )

    height_smoothing: bpy.props.FloatProperty(
        name="Height Smoothing",
        description=(
            "Controls smoothing of the final Height map surface. "
            "Raise this if the height map shows stair-stepping, grain, or chatter. "
            "Lower this if important shape definition gets overly softened."
        ),
        min=0.0,
        max=1.0,
        default=0.5,
    )

    macro_relief_weight: bpy.props.FloatProperty(
        name="Macro Relief Weight",
        description=(
            "Controls how strongly larger forms influence the Height map. "
            "Raise this if broad surface undulation is missing or too subtle. "
            "Lower this if large-scale swelling starts dominating the height result."
        ),
        min=0.0,
        max=2.0,
        default=1.0,
    )

    debug_print: bpy.props.BoolProperty(
        name="Debug Print",
        description=(
            "Controls whether Iridis prints extra processing messages to the Blender console. "
            "Raise it by turning this on when you need more visibility while troubleshooting a setup. "
            "Lower it by turning this off when you want a cleaner console during normal use."
        ),
        default=False,
    )

    protect_mask_image: bpy.props.PointerProperty(
        name="Protect Mask",
        description=(
            "Controls where processing is allowed across the image. "
            "Raise influence by painting areas whiter when you want full processing in those regions. "
            "Lower influence by painting areas darker when you want to preserve original source detail."
        ),
        type=bpy.types.Image,
    )

    roughness_control_mask_image: bpy.props.PointerProperty(
        name="Roughness Control Mask",
        description=(
            "Controls local strength of Roughness adjustments across the texture. "
            "Raise influence by painting areas whiter when you want roughness sliders to act more strongly there. "
            "Lower influence by painting areas darker when roughness changes are too aggressive locally."
        ),
        type=bpy.types.Image,
    )

    metallic_control_mask_image: bpy.props.PointerProperty(
        name="Metallic Control Mask",
        description=(
            "Controls local strength of Metallic adjustments across the texture. "
            "Raise influence by painting areas whiter when metallic detection needs a stronger push there. "
            "Lower influence by painting areas darker when metallic tagging is overreaching locally."
        ),
        type=bpy.types.Image,
    )

    relief_control_mask_image: bpy.props.PointerProperty(
        name="Relief Control Mask",
        description=(
            "Controls local strength of Normal and Height relief generation. "
            "Raise influence by painting areas whiter when depth detail needs to read more strongly there. "
            "Lower influence by painting areas darker when bumps or displacement are too intense locally."
        ),
        type=bpy.types.Image,
    )
