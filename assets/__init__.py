def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def resolve_effective_settings(settings, preset_profile: dict) -> dict:
    eff = {}

    # Albedo
    eff["albedo_delight_strength"] = clamp(
        preset_profile["albedo_delight_strength"] + (settings.delight_strength - 0.5)
    )
    eff["albedo_highlight_suppression"] = clamp(
        preset_profile["albedo_highlight_suppression"] + (settings.highlight_suppression - 0.5)
    )
    eff["albedo_color_protection"] = clamp(
        preset_profile["albedo_color_protection"] + (settings.color_preservation - 0.5)
    )
    eff["albedo_dirt_cleanup_bias"] = clamp(
        preset_profile["albedo_dirt_cleanup_bias"]
    )

    # Roughness
    eff["roughness_base"] = clamp(
        preset_profile["roughness_base"] + (settings.roughness_bias * 0.5)
    )
    eff["roughness_microdetail_weight"] = clamp(
        preset_profile["roughness_microdetail_weight"] * settings.microdetail_influence,
        0.0, 2.5,
    )
    eff["roughness_cavity_weight"] = clamp(
        preset_profile["roughness_cavity_weight"] * settings.cavity_boost,
        0.0, 2.5,
    )
    eff["roughness_highlight_response"] = clamp(
        preset_profile["roughness_highlight_response"]
    )
    eff["roughness_region_coherence"] = clamp(
        preset_profile["roughness_region_coherence"]
    )

    # Metallic
    eff["metallic_base_probability"] = clamp(
        preset_profile["metallic_base_probability"] + (settings.metallic_bias * 0.5)
    )
    eff["metallic_neutrality_weight"] = clamp(
        preset_profile["metallic_neutrality_weight"]
    )
    eff["metallic_color_rejection"] = clamp(
        preset_profile["metallic_color_rejection"] + (settings.paint_color_rejection - 0.5)
    )
    eff["metallic_region_coherence"] = clamp(
        preset_profile["metallic_region_coherence"]
    )
    eff["metallic_edge_exposure_weight"] = clamp(
        preset_profile["metallic_edge_exposure_weight"]
    )
    eff["metallic_threshold_softness"] = clamp(settings.threshold_softness)

    # Normal
    eff["normal_detail_weight"] = clamp(
        preset_profile["normal_detail_weight"] * settings.normal_strength,
        0.0, 5.0,
    )
    eff["normal_mid_weight"] = clamp(
        preset_profile["normal_mid_weight"] * settings.normal_mid_detail_influence,
        0.0, 3.0,
    )
    eff["normal_smoothing"] = clamp(
        (preset_profile["normal_smoothing"] + settings.normal_smoothing) * 0.5
    )
    eff["normal_format"] = settings.normal_format

    # Height
    eff["height_macro_weight"] = clamp(
        preset_profile["height_macro_weight"] * settings.macro_relief_weight,
        0.0, 3.0,
    )
    eff["height_detail_weight"] = clamp(
        preset_profile["height_detail_weight"] * settings.microdetail_influence,
        0.0, 3.0,
    )
    eff["height_smoothing"] = clamp(
        (preset_profile["height_smoothing"] + settings.height_smoothing) * 0.5
    )
    eff["height_contrast"] = max(0.0, settings.height_contrast)

    # Material expectations
    eff["dirt_expected"] = clamp(preset_profile["dirt_expected"])
    eff["corrosion_expected"] = clamp(preset_profile["corrosion_expected"])
    eff["painted_surface_expected"] = clamp(preset_profile["painted_surface_expected"])
    eff["exposed_edge_expected"] = clamp(preset_profile["exposed_edge_expected"])

    return eff
