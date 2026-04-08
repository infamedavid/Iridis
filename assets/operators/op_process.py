import bpy

from ..core.image_io import (
    get_active_image_from_image_editor,
    read_image_to_numpy,
    build_work_mask,
    read_mask_image_as_gray,
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
            )):
                self.report({'ERROR'}, "No output maps are enabled.")
                return {'CANCELLED'}

            out_dir = bpy.path.abspath(settings.output_dir)
            ensure_output_dir(out_dir)

            self._debug(settings, f"Reading image: {image.name}")
            src = read_image_to_numpy(image)
            mask = build_work_mask(src["alpha"])

            protect_mask = read_mask_image_as_gray(
                settings.protect_mask_image,
                src["width"],
                src["height"],
            )
            if protect_mask is not None:
                mask = (mask * protect_mask).astype(mask.dtype)

            roughness_control_mask = read_mask_image_as_gray(
                settings.roughness_control_mask_image,
                src["width"],
                src["height"],
            )
            metallic_control_mask = read_mask_image_as_gray(
                settings.metallic_control_mask_image,
                src["width"],
                src["height"],
            )
            relief_control_mask = read_mask_image_as_gray(
                settings.relief_control_mask_image,
                src["width"],
                src["height"],
            )

            self._debug(settings, f"Image size: {src['width']}x{src['height']}")
            self._debug(settings, f"Preset: {settings.preset}")
            preset_profile = get_preset_profile(settings.preset)
            eff = resolve_effective_settings(settings, preset_profile)
            eff["roughness_bias_slider"] = float(settings.roughness_bias)
            eff["metallic_bias_slider"] = float(settings.metallic_bias)

            self._debug(settings, "Building common buffers")
            common = build_common_buffers(
                src["rgb"],
                src["alpha"],
                mask,
                enable_heavier_relief=settings.enable_heavier_relief,
            )
            common["roughness_control_mask"] = roughness_control_mask
            common["metallic_control_mask"] = metallic_control_mask
            common["relief_control_mask"] = relief_control_mask

            base = settings.base_name.strip()
            overwrite = settings.overwrite_existing
            saved = []

            if settings.generate_albedo:
                self._debug(settings, "Generating albedo")
                albedo = generate_albedo_map(common, eff)
                path = save_rgb_map(albedo, out_dir, f"{base}_albedo.png", overwrite)
                saved.append(path)

            if settings.generate_roughness:
                self._debug(settings, "Generating roughness")
                rough = generate_roughness_map(common, eff)
                path = save_gray_map(rough, out_dir, f"{base}_roughness.png", overwrite)
                saved.append(path)

            if settings.generate_metallic:
                self._debug(settings, "Generating metallic")
                metallic_map = generate_metallic_map(common, eff)
                path = save_gray_map(metallic_map, out_dir, f"{base}_metallic.png", overwrite)
                saved.append(path)

            if settings.generate_normal:
                self._debug(settings, "Generating normal")
                normal = generate_normal_map(common, eff)
                path = save_rgb_map(normal, out_dir, f"{base}_normal.png", overwrite)
                saved.append(path)

            if settings.generate_height:
                self._debug(settings, "Generating height")
                height = generate_height_map(common, eff)
                path = save_gray_map(height, out_dir, f"{base}_height.png", overwrite)
                saved.append(path)

            self._debug(settings, f"Saved files: {saved}")
            self.report({'INFO'}, f"Iridis generated {len(saved)} file(s).")
            return {'FINISHED'}

        except Exception as exc:
            self.report({'ERROR'}, f"Iridis error: {exc}")
            print("[Iridis] ERROR:", exc)
            return {'CANCELLED'}
