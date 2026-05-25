# ComfyUI-PIH/pih_node.py
import os, sys, tempfile, uuid, subprocess, glob, shutil
from PIL import Image
import numpy as np
import torch
from .model import Model_Composite_PL, Model_Composite
import folder_paths as comfy_paths
import torch.nn.functional as F


# Comfy types
from typing import Any, Dict, Tuple

class PIH_Harmonize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Background Image": ("IMAGE",),      # [B,H,W,C] float [0,1], RGB
                "Foreground Image": ("IMAGE",),      # [B,H,W,C], RGB or RGBA
                "Strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}), # Strength of harmonization
                "PIH Processor": ("PIH_PROCESSOR",), # Model_Composite_PL instance
                "mask": ("MASK",),                   # [B,H,W] float [0,1] (optional, multiplies FG alpha)
            }
        }

    RETURN_TYPES = ("IMAGE",)                         # Composited, harmonized RGB
    FUNCTION = "run"
    CATEGORY = "Harmonization/PIH"

    def run(self, **kw):
        bg   = kw["Background Image"]      # [B,H,W,C]
        fg   = kw["Foreground Image"]      # [B,H,W,C] (RGB or RGBA)
        strength = kw["Strength"]          # float [0,1]
        model = kw["PIH Processor"]
        mask_in = kw.get("mask", None)     # [B,H,W] or None

        # device
        device = next(model.parameters()).device if any(True for _ in model.parameters()) else ("cuda" if torch.cuda.is_available() else "cpu")

        # to float32 on device
        bg = bg.to(device=device, dtype=torch.float32)
        fg = fg.to(device=device, dtype=torch.float32)
        if mask_in is not None:
            mask_in = mask_in.to(device=device, dtype=torch.float32)

        B, Hf, Wf, Cf = fg.shape
        Bb, Hb, Wb, Cb = bg.shape
        if B != Bb:
            raise ValueError("Batch size mismatch between Background and Foreground")

        # Resize BG to FG spatial size if needed
        if (Hb, Wb) != (Hf, Wf):
            bg = F.interpolate(bg.permute(0,3,1,2), size=(Hf,Wf), mode="bilinear", align_corners=False).permute(0,2,3,1)

        # Split FG into RGB (+ alpha if present)
        if Cf == 4:
            alpha_fg = fg[..., 3:4]       # [B,H,W,1], assumed straight alpha in [0,1]
            rgb_fg   = fg[..., :3]        # [B,H,W,3]
        else:
            alpha_fg = None
            rgb_fg   = fg

        # Effective mask: (provided mask) * (alpha if present) else (alpha) else ones
        if mask_in is not None:
            m = mask_in.unsqueeze(-1)     # [B,H,W,1]
            if alpha_fg is not None:
                m = m * alpha_fg
        else:
            m = alpha_fg if alpha_fg is not None else torch.ones((B,Hf,Wf,1), device=device)

        m = m.clamp(0,1)

        # Convert to BCHW
        bg_c = bg.permute(0,3,1,2).contiguous()        # [B,3,H,W]
        fg_c = rgb_fg.permute(0,3,1,2).contiguous()    # [B,3,H,W]
        mask_c = m.permute(0,3,1,2).contiguous()       # [B,1,H,W]

        # --- initial composite at full resolution (FG over BG using alpha) ---
        print(f"Shapes - Mask: {mask_c.shape}, Foreground: {fg_c.shape}, Background: {bg_c.shape}")
        comp_full = fg_c * mask_c + bg_c * (1 - mask_c)  # [B,3,H,W]

        # --- low-res pass (512x512) ---
        target = (512, 512)
        bg_low   = F.interpolate(bg_c,   size=target, mode="bilinear", align_corners=False)
        comp_low = F.interpolate(comp_full, size=target, mode="bilinear", align_corners=False)
        mask_low = F.interpolate(mask_c, size=target, mode="bilinear", align_corners=False)

        with torch.no_grad():
            _inter, _out, _scalars, _pl = model(bg_low, comp_low, mask_low)

        # --- high-res reapply: LUT then gain, still respecting alpha mask ---
        hr_intermediate = model.PL3D(model.pl_table, comp_full) * mask_c + (1 - mask_c) * bg_c
        gain_full = F.interpolate(model.gainmap, size=(Hf, Wf), mode="bilinear", align_corners=False)
        composite_c = (mask_c) * fg_c + (1 - mask_c) * bg_c
        out_full = hr_intermediate * gain_full * strength * mask_c + (1 - strength * mask_c) * composite_c  # [B,3,H,W]

        # Back to Comfy format [B,H,W,C]
        out_img = out_full.permute(0,2,3,1).clamp(0,1).contiguous()
        return (out_img,)



class PIHModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("PIH_PROCESSOR",)
    RETURN_NAMES = ("PIH Processor",)
    FUNCTION = "load_model"
    CATEGORY = "Adobe PIH"

    def load_model(self, checkpoint_folder="pretrained"):
        print("[PIH] PIHModelLoader.load_model() called")

        # device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # paths
        pih_folder = os.path.join(comfy_paths.models_dir, "PIH")
        ckpt_path = os.path.join(pih_folder, "ckpt_g39.pth")

        if not os.path.exists(ckpt_path):
            os.makedirs(pih_folder, exist_ok=True)
            print(f"[PIH] Checkpoint not found — downloading to {ckpt_path} ...")
            try:
                import gdown
                gdown.download(id="1seW8qSnaBOQ4_S9bQ4ThVOdeJGYJ-f74", output=ckpt_path, quiet=False)
            except Exception as e:
                raise RuntimeError(
                    f"[PIH] Auto-download failed: {e}\n"
                    f"Download manually from https://drive.google.com/file/d/1seW8qSnaBOQ4_S9bQ4ThVOdeJGYJ-f74/view "
                    f"and place it at: {ckpt_path}"
                )

        # build model
        model = Model_Composite_PL(
            dim=32,
            masking=True,
            brush=True,
            maskoffset=0.6,
            swap=True,
            Vit_bool=False,
            onlyupsample=True,
            aggupsample=True,
            light=False,
            Eff_bool=False,
        ).to(device)

        # load weights
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)  # tolerate raw SDs
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[PIH] load_state_dict: missing={missing}, unexpected={unexpected}")

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        print("[PIH] Model loaded and ready on", device)
        return (model,)



# Node registration
NODE_CLASS_MAPPINGS = {
    "PIH Harmonize": PIH_Harmonize,
    "PIHModelLoader": PIHModelLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PIH Harmonize": "PIH Harmonize",
    "PIHModelLoader": "PIH Model Loader",
}

