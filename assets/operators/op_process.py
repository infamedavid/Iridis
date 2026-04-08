import bpy

from ..core.image_io import (
    get_active_image_from_image_editor,
    read_image_to_numpy,
    build_work_mask,
)
from ..core.output_writer import (
    ensure_output_dir,
    save_rgb_map,
    save_gray_map,
)
from ..core.preset_profiles import get_preset_profile
from ..core.settings_resolver import resolve_effective_settings
from ..core.buffers import build_common_buffers
from ..processing import (
    generate_albedo_map,
    generate_roughness_map,
    generate_metallic_map,
    generate_normal_map,
    generate_height_map,
    generate_cavity_mask,
    generate_dirt_mask,
    generate_metal_mask,
    generate_edge_wear_mask,
)


class IRIDIS_OT_process(bpy.types.Operator):
    bl_idname = "iridis.process"
    bl_label = "Process"
    bl_description = "Generate derived maps from the active image"

    def _debug(self, settings, text):
        if settings.debug_print:
            print(f"[Iridis] {text}")

    def execute(self, context):
        scene = context.scene
        settings = scene.iridis_settings

        try:
            image = get_active_image_from_image_editor(context)
            if image is None:
                self.report({'ERROR'}, "No active image found in the Image Editor.")
                return {'CANCELLED'}

            if not settings.output_dir:
                self.report({'ERROR'}, "Output folder is empty.")
                return {'CANCELLED'}

            if not settings.base_name.strip():
                self.report({'ERROR'}, "Base name is empty.")
                return {'CANCELLED'}

            if not any((
                settings.generate_albedo,
                settings.generate_roughness,
                settings.generate_metallic,
                settings.generate_normal,
                settings.generate_height,
                settings.generate_aux_masks,
            )):
                self.report({'ERROR'}, "No output maps are enabled.")
                return {'CANCELLED'}

            out_dir = bpy.path.abspath(settings.output_dir)
            ensure_output_dir(out_dir)

            self._debug(settings, f"Reading image: {image.name}")
            src = read_image_to_numpy(image)
            mask = build_work_mask(src["alpha"])

            self._debug(settings, f"Image size: {src['width']}x{src['height']}")
            self._debug(settings, f"Preset: {settings.preset}")
            preset_profile = get_preset_profile(settings.preset)
            eff = resolve_effective_settings(settings, preset_profile)

            self._debug(settings, "Building common buffers")
            common = build_common_buffers(
                src["rgb"],
                src["alpha"],
                mask,
                use_enhanced_relief_analysis=settings.use_enhanced_relief_analysis,
            )

            base = settings.base_name.strip()
            overwrite = settings.overwrite_existing
            alpha_out = common["work_mask"]
            saved = []

            metallic_map = None

            if settings.generate_albedo:
                self._debug(settings, "Generating albedo")
                albedo = generate_albedo_map(common, eff)
                path = save_rgb_map(albedo, out_dir, f"{base}_albedo.png", overwrite, alpha_out)
                saved.append(path)

            if settings.generate_roughness:
                self._debug(settings, "Generating roughness")
                rough = generate_roughness_map(common, eff)
                path = save_gray_map(rough, out_dir, f"{base}_roughness.png", overwrite, alpha_out)
                saved.append(path)

            if settings.generate_metallic or settings.generate_aux_masks:
                self._debug(settings, "Generating metallic")
                metallic_map = generate_metallic_map(common, eff)
                if settings.generate_metallic:
                    path = save_gray_map(metallic_map, out_dir, f"{base}_metallic.png", overwrite, alpha_out)
                    saved.append(path)

            if settings.generate_normal:
                self._debug(settings, "Generating normal")
                normal = generate_normal_map(common, eff)
                path = save_rgb_map(normal, out_dir, f"{base}_normal.png", overwrite, alpha_out)
                saved.append(path)

            if settings.generate_height:
                self._debug(settings, "Generating height")
                height = generate_height_map(common, eff)
                path = save_gray_map(height, out_dir, f"{base}_height.png", overwrite, alpha_out)
                saved.append(path)

            if settings.generate_aux_masks:
                self._debug(settings, "Generating auxiliary masks")
                cavity_mask = generate_cavity_mask(common)
                dirt_mask = generate_dirt_mask(common, eff)
                if metallic_map is None:
                    metallic_map = generate_metallic_map(common, eff)
                metal_mask = generate_metal_mask(metallic_map, common, eff)
                edge_wear = generate_edge_wear_mask(common, eff)

                saved.append(save_gray_map(cavity_mask, out_dir, f"{base}_cavity.png", overwrite, alpha_out))
                saved.append(save_gray_map(dirt_mask, out_dir, f"{base}_dirt.png", overwrite, alpha_out))
                saved.append(save_gray_map(metal_mask, out_dir, f"{base}_metalmask.png", overwrite, alpha_out))
                saved.append(save_gray_map(edge_wear, out_dir, f"{base}_edgewear.png", overwrite, alpha_out))

            self._debug(settings, f"Saved files: {saved}")
            self.report({'INFO'}, f"Iridis generated {len(saved)} file(s).")
            return {'FINISHED'}

        except Exception as exc:
            self.report({'ERROR'}, f"Iridis error: {exc}")
            print("[Iridis] ERROR:", exc)
            return {'CANCELLED'}
