# ComfyUI-PIH

ComfyUI custom nodes for **Semi-supervised Parametric Real-world Image Harmonization** (PIH).

---

## Credits

This node wraps the original PIH research code by:

**Ke Wang, Michaël Gharbi, He Zhang, Zhihao Xia, Eli Shechtman**  
Adobe Research — CVPR 2023

[Project Page](http://people.eecs.berkeley.edu/~kewang/sprih/) | [Paper](https://arxiv.org/abs/2303.00157) | [Original Repo](https://github.com/adobe/PIH)

Please contact Ke (kewang@berkeley.edu) or Michaël (mgharbi@adobe.com) with questions about the underlying model.

If you use this in your research, please cite the original paper:

```bibtex
@article{wang2023semi,
  title={Semi-supervised Parametric Real-world Image Harmonization},
  author={Wang, Ke and Gharbi, Micha{\"e}l and Zhang, He and Xia, Zhihao and Shechtman, Eli},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}
```

---

PIH blends a composited foreground into a background by learning complex local appearance harmonization — matching color, tone, and shading — from unpaired real composites.

---

## Nodes

| Node | Description |
|------|-------------|
| **PIH Model Loader** | Loads the PIH checkpoint (auto-downloads on first use) |
| **PIH Harmonize** | Blends a foreground into a background with harmonization |

---

## Installation

### Via ComfyUI-Manager (recommended)
Search for **ComfyUI PIH** in the Custom Nodes section and click Install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/chrismyau/ComfyUI-PIH
cd ComfyUI-PIH
pip install -r requirements.txt
```

### Model Checkpoint
The checkpoint (~93M parameters, trained on Artist Retouched Dataset) **downloads automatically** the first time you run the **PIH Model Loader** node.

It is saved to `ComfyUI/models/PIH/ckpt_g39.pth`.

To download manually: [Google Drive link](https://drive.google.com/file/d/1seW8qSnaBOQ4_S9bQ4ThVOdeJGYJ-f74/view?usp=sharing) — place the file at `ComfyUI/models/PIH/ckpt_g39.pth`.

---

## Usage

1. Add **PIH Model Loader** → connect its `PIH Processor` output to **PIH Harmonize**
2. Connect your `Background Image` and `Foreground Image` (RGB or RGBA) to **PIH Harmonize**
3. Optionally connect a `mask` to control which region is harmonized
4. Adjust `Strength` (0–1) to blend between the original composite and the harmonized result

---

## License

Apache 2.0 — see [LICENSE](LICENSE). Original PIH model code © 2023 Adobe.
