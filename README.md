# SegFitTools

Small ComfyUI nodes for mask cleanup and mask-guided garment placement.

## Legal and Safety Notice

> [!WARNING]
> Read this section before using any node in this package.

SegFitTools is intended solely for lawful virtual try-on, garment fitting, and
apparel visualization. The reference workflow removes an existing garment only
to prepare a neutral, non-anatomical, mannequin-like fitting surface. It is not
designed or authorized to reconstruct hidden anatomy or to create realistic
nude or sexually explicit images.

Use this software only with images that:

- depict adults whose adult status is certain;
- you own or are legally authorized to process; and
- were obtained with the subject's explicit and informed consent for this use.

Do not use this software to:

- create, infer, alter, or distribute intimate or sexual imagery of any person
  without that person's explicit consent;
- process images of minors, or of anyone whose adult status is uncertain;
- harass, defame, impersonate, exploit, or violate the privacy, portrait,
  publicity, copyright, or other rights of any person;
- distribute, publicly display, sell, or otherwise provide unlawful obscene
  images or electronic records; or
- violate the law or the terms of any model, LoRA, platform, or service used
  with this package.

For use in Japan, users must pay particular attention to the following laws.
This list is not exhaustive:

- Article 175 of the
  [Penal Code](https://laws.e-gov.go.jp/law/140AC0000000045), concerning the
  distribution, public display, and certain forms of transmission or
  possession for distribution of obscene objects or electronic records;
- the
  [Act on Prevention of Damage Caused by Provision of Private Sexual Image Records](https://laws.e-gov.go.jp/law/426AC1000000126),
  including the provision or public display of identifiable private sexual
  images without the subject's consent; and
- the
  [Act on Regulation and Punishment of Acts Relating to Child Prostitution and Child Pornography, and the Protection of Children](https://laws.e-gov.go.jp/law/411AC0100000052).

Privacy, portrait rights, reputation, copyright, and other civil rights may
also apply independently of criminal law. Do not assume that an image is lawful
merely because it was generated or altered by AI, was produced locally, or has
not yet been published.

The supplied prompts attempt to produce a neutral fitting surface, but prompt
changes, alternative checkpoints, LoRAs, custom nodes, or downstream processing
can defeat that safeguard. The user is solely responsible for such changes and
for all resulting content. Intermediate images and masks may contain sensitive
personal information; store them securely and never publish or share them
without the subject's explicit consent.

Users are responsible for complying with all laws and regulations applicable
in their country or region. This notice is not legal advice. The software is
provided as-is, without warranty, to the maximum extent permitted by applicable
law.

## Installation

Copy `ComfyUI_SegFitTools` into `ComfyUI/custom_nodes`, then restart ComfyUI.
The nodes appear under `SegFitTools/Mask` and `SegFitTools/Fit`.

### Public and premium packages

The public and premium editions intentionally use the same folder name,
`ComfyUI_SegFitTools`. The public package does not contain `DensePose Cropper`,
`Underwear Detector`, their private implementations, or their optional registry.
All shared nodes, including `DensePose Cropper`, continue to use the normal base
registry in `__init__.py`.

To upgrade, stop ComfyUI and copy the complete premium package over the existing
public `ComfyUI_SegFitTools` folder, allowing files to be overwritten. Restarting
ComfyUI loads `premium_registry.py` and adds the two premium node mappings without
changing the node-pack name.

To return from premium to public, remove the existing `ComfyUI_SegFitTools`
folder first and then install the public package. Do not merely overwrite it with
the public package, because premium-only Python files left behind would still be
discovered and registered. Workflows containing premium nodes retain those node
references but report them as missing while only the public package is installed.

### SegFormer model

`SegFormer Mask Refiner` requires the
[`mattmdjaga/segformer_b2_clothes`](https://huggingface.co/mattmdjaga/segformer_b2_clothes)
model from Hugging Face. Clone the complete repository into ComfyUI's `models`
directory with the folder name `segformer_b2_clothes`:

```powershell
cd ComfyUI\models
git lfs install
git clone https://huggingface.co/mattmdjaga/segformer_b2_clothes segformer_b2_clothes
```

The resulting layout must include at least:

```text
ComfyUI/models/segformer_b2_clothes/
  config.json
  preprocessor_config.json
  model.safetensors
```

Do not rename or modify the files inside the model directory. The node loads
the model locally and does not download missing files while ComfyUI is running.

### Mannequin reference images

The mannequin replacement workflow uses two reference images prepared for
different bust ranges:

- `AtoC_mannequin.png`: reference mannequin for A-C cup bodies.
- `DtoE_mannequin.png`: reference mannequin for D-E cup bodies.

Both images use a canvas size of 840 x 1256 pixels. Place them under ComfyUI's
`input` directory; `input/SegFit_MANNEQUINS` is recommended so they appear in the
default `Load Image No Upload` model folder. The filenames are not hard-coded
by the crop or scale nodes, so another input subfolder may be used when needed.
Load the reference matching the intended bust range and generate an aligned
DensePose image from that same mannequin image.

## Nodes

### SegFormer Mask Refiner

Runs the local `models/segformer_b2_clothes` model directly on a person image,
extracts the fixed `Upper-clothes` (label 4) and `Pants` (label 6) classes, and
cleans both masks without per-label Boolean switches. Outputs are `tops_mask`,
`pants_mask`, a red/blue debug preview, and `fullbody_underwear_mask`. The last
output is a full-canvas union of the cleaned TOPS and PANTS masks for removing
the underwear already worn by the person or passing it to an inpaint workflow.

### Mask Cleanup

Applies the Refiner cleanup stage to any `MASK` produced by SegFormer, SAM3,
or a manual mask editor. It removes small isolated regions, fills small holes,
grows or shrinks the boundary, and feathers the result. Connect a manual MASK
switch before this node to replace a SegFormer mask with a SAM3 mask without
changing the downstream Fit Garment workflow.

```text
SegFormer Mask Refiner --\
                         Manual MASK Switch -> Mask Cleanup -> Fit Garment
SAM3 -------------------/
```

### Switch Mask

A manual two-input passthrough switch. Connect two `MASK` values to `mask_1`
and `mask_2`, then choose which one is sent unchanged through the single
`mask` output. It is intended for selecting a SegFormer mask normally and a
SAM3 replacement when manual inspection finds a defect.

### Image Auto Switching

An exclusive automatic IMAGE switch for workflows controlled by Fast Group
Muter. Set `input_count` from 2 to 8 and connect the alternative branches. The
only active IMAGE is passed through unchanged. The node raises an error when no
input is active or when two or more inputs are active, so a missed mute cannot
silently select the wrong branch.

### Mask Auto Switching

The MASK counterpart of `Image Auto Switching`. Set `input_count` from 2 to 8.
Exactly one active MASK is passed through unchanged; zero or multiple active
inputs raise an error. This is suitable for automatically choosing between a
SegFormer mask and a SAM3 mask under Fast Group Muter control.
This node has no route input and retains its original general-purpose behavior.

### Detected Mask Switching

Route-aware two-input switch for Body Line Detector fallback. Inputs are only
`A`, `B`, and `route`. Connect Restorer `combined_removal_mask` to `A`, Poly
Cutter `combined_mask` to `B`, and Restorer `route` to `route`. `RESTORER`
evaluates only A; `POLY_CUTTER` evaluates only B, so the unused Poly Cutter
editor is not started. Both mask inputs are lazy.

### Masked Area Fill

Replaces the white MASK area of an IMAGE with a solid RGB color. Black is the
default. Soft or feathered mask edges are blended with the source image, while
the unmasked image remains unchanged. This provides a deterministic first-stage
method for neutralizing the design of currently worn underwear before Fit
Garment. If a solid placeholder is not sufficient, use an inpaint workflow as
the higher-cost fallback.

### Masked Area Skin Fill

Fills a masked underwear area with synthetic skin texture or sampled skin tone,
preventing downstream generative diffusion/editing models (such as Qwen-Image-Edit
or Flux) from hallucinating unintended dark innerwear or layered undergarments.

Outputs are the skin-neutralized `image` and the dilated `mask`.

- **fill_method**:
  - `telea_inpaint`: Uses OpenCV's fast marching Telea algorithm to extend the
    surrounding skin texture smoothly into the masked cavity (recommended default).
  - `sampled_skin_color`: Samples pixel values from the outer ring boundary of
    the mask and fills the region with the median skin tone.
  - `sampled_skin_with_noise`: Applies slight Gaussian noise over the sampled
    median skin tone to give generative models micro-surface textures.
- **expand_mask**: Dilation radius (in pixels) applied to the input mask before
  filling, ensuring seams, lace borders, underwire ridges, and cast shadows are
  fully neutralized.
- **noise_intensity**: Standard deviation of the Gaussian noise applied when
  using `sampled_skin_with_noise`.

### Join Masked Area Gray

Converts source-image luminance to neutral grayscale shading inside a `MASK`.
The luminance and middle gray are blended equally, while pixels outside the
mask and any channels after RGB remain unchanged. Single masks are expanded
across an image batch, and masks with different dimensions are resized to the
image before compositing.

### Load Image No Upload

Loads an image already stored under the ComfyUI `input` directory without
showing an upload control. `folder_name` limits the image selector to that
folder and its descendants; the default is `SegFit_MODELS`, so the complete
input tree is no longer embedded in the node definition. Changing the folder
refreshes the image choices. The selected image is displayed inside the node.
The node returns `IMAGE`, the standard inverted-alpha `MASK`, the source base
`filename`, and `filename_stem` without its extension. It reads the existing
file directly and does not create a second input copy.

### Load Image No Upload LM

`LM` means `Low Profile Model`. The node provides the same inputs, preview, and
outputs as `Load Image No Upload`, but is intended for PCs without a high-end
GPU and a large amount of VRAM. Images whose longest side exceeds 2000 pixels
are downscaled on the CPU before a tensor is created or transferred to a
device. Aspect ratio is preserved, and smaller images are not enlarged.

The standard `Load Image No Upload` deliberately retains the original image
size for high-performance GPUs and cloud execution. It does not impose a size
guard. Workflow operators remain responsible for choosing the loader that
matches their execution environment.

### Save As Mask

Saves a `MASK` as an 8-bit grayscale PNG under the ComfyUI input directory so
it remains available to image loaders. The default folder is `SegFit_MASK`.
Connect `filename` from `Load Image No Upload` to `source_filename`; the saved
name retains that source stem and adds the free-form `identifier`, for example
`person_001__ESAM3.png`. `overwrite` maintains one current mask, `increment`
creates revisions such as `person_001__ESAM3__001.png`, and `error` prevents an
existing mask from being replaced. The saved mask is also displayed inside the
node, and its filename and full path are returned as strings.

### Save As Image

Saves an `IMAGE` as a PNG under the ComfyUI input directory. The default folder
is `SegFit_TEMP`. Connect `filename` from `Load Image No Upload` to
`source_filename`; the source stem is combined with the optional free-form
`identifier`, producing a stable name such as
`person_001__TEMP.png` or `person_001__prepared.png`.

`overwrite` replaces the same source and identifier, `increment` creates
numbered revisions, and `error` refuses to replace an existing image. This
makes the node suitable for temporary images produced between mask preparation
and Fit Garment processing. A batch receives explicit names such as
`person_001__TEMP__batch001.png`. RGB, RGBA, and single-channel images are
saved without forcing them to grayscale, and RGBA transparency is preserved.
The saved image is displayed inside the node, and its filename and full path
are returned as strings.

The temporary file is intentionally not deleted automatically. It remains
available for inspection, manual correction, or resuming work from a separate
workflow. With the default `overwrite` setting, repeated processing replaces
the same named file. Remove files from `SegFit_TEMP` only when they are no
longer useful.

Typical preparation flow:

```text
Load Image No Upload
  -> SegFormer Mask Refiner or EasySAM3
  -> Join Masked Area Gray / Masked Area Fill
  -> Image Composite Masked
  -> Save As Image
  -> Fit Garment workflow or later manual inspection
```

### Save As Tmp

Saves an `IMAGE` batch as an uncompressed NumPy array in a non-image `.tmp`
file under the ComfyUI `input` directory. The node completely suppresses
in-node UI previews and thumbnail generation. The default subfolder is
`SegFit_TEMP`.

Connect `filename_stem` from `Load Image No Upload` or its LM counterpart. The
fixed internal suffix produces a cache such as
`person_001__mannequin.tmp`. Reprocessing the same source replaces its current
cache. To allow synchronized cache purging and VRAM garbage collection after
the disk write completes, the input `IMAGE` tensor is passed through unchanged.

The `.tmp` content is not PNG or JPEG data. Renaming the extension does not make
it viewable in an ordinary image viewer, but the file is not encrypted. Users
remain responsible for protecting the ComfyUI input directory and controlling
access to intermediate files.

Inputs:

- `image`: source image tensor to save.
- `filename_stem`: original source stem used to name the matching cache.
- `output_folder`: target directory relative to the ComfyUI `input` root (default: `SegFit_TEMP`).

Outputs:

- `image`: unchanged passthrough `IMAGE` tensor for chaining into memory cleanup nodes.
- `saved_filename`: base name of the saved temporary file.
- `saved_path`: absolute file path of the saved temporary file.

Typical temporary caching & cleanup flow:

```text
VAE Decode
 -> Preview Image (optional UI verification)
 -> Save As Tmp
 -> Clear Cache (comfyui-easy-use)/ VRAM Cleanup (Eclipse)
 -> Downstream generation stage
```

### Load Tmp

Loads the NumPy cache created by `Save As Tmp` without publishing an in-node
preview. Connect the same original `filename_stem` and select the matching
input folder. The node applies the fixed `__mannequin.tmp` suffix internally,
validates the array type and `[B,H,W,C]` shape, and converts the stored `uint8`
batch back to a ComfyUI `IMAGE` in the `0.0-1.0` range.

Inputs:

- `filename_stem`: source stem originally supplied to `Save As Tmp`.
- `input_folder`: cache folder relative to the ComfyUI `input` root (default: `SegFit_TEMP`).

Outputs:

- `image`: restored ComfyUI `IMAGE` batch.
- `loaded_filename`: resolved cache filename.
- `loaded_path`: absolute path of the loaded cache.

`Load Tmp` is intentionally separate from `Load Image By Chain`. The latter
continues to load ordinary image files and does not interpret NumPy data.

### Load Image By Chain

Builds an image path from chainable `folder_name`, `filename`, and
`suffix_and_extension` STRING inputs. Relative folders are resolved under the
current ComfyUI input directory; absolute folders are also accepted. The suffix
is appended verbatim, so `.png` loads the same stem from another folder and
`_1.png` loads a name such as `person_001_1.png`. If `filename` already has an
extension, the suffix is ignored. An empty filename allows a full file path to
be supplied through `folder_name`. The node returns `IMAGE` and the standard
inverted-alpha `MASK`, preserves RGB values underneath transparent pixels
instead of compositing them onto black, and displays a temp PNG preview inside
the node after execution. Linked STRING values are resolved at execution time.
If the resulting file does not exist, the node reports
`[Load Image By Chain] Not Found` with the complete resolved path and stops the
workflow; it never substitutes a dummy image.
Set `preview` to `off` to skip creation and display of the temporary in-node
preview while keeping `IMAGE` and `MASK` outputs active.

#### Human parser limitations

Human parsers may produce incomplete masks for extremely pale garments,
particularly very light yellow, pink, and beige items whose color and brightness
are close to the wearer's skin. Printed areas with similarly low contrast may
also be omitted even when glossy fabric of the same general color is detected.
Small holes can be corrected with `Mask Cleanup`, but it cannot reconstruct a
large missing region or an absent outer boundary. Inspect the preview and switch
to the point-guided `ComfyUI-EasySAM3` branch, or correct the mask manually, when
the garment outline is incomplete.

#### Choosing a SAM3 helper

SegFormer can miss thin straps, lace edges, and other narrow garment details.
When its mask is incomplete, use a SAM3 helper as the replacement branch before
`Mask Auto Switching` and `Mask Cleanup`.

In testing for underwear extraction, the point-guided `ComfyUI-EasySAM3`
workflow produced more complete and accurate masks than the prompt-only
`ComfyUI-Easy-SAM3` workflow, especially for thin straps and for separating the
top and bottom garments. The prompt-only workflow is easier to operate, but its
mask may require closer inspection. Results can vary with the source image,
prompt, points, threshold, model version, and node settings; this is a practical
recommendation for this workflow rather than a general benchmark of either
project.

### DensePose Cropper

Creates the standard Body Line upper/lower crop geometry from a DensePose hint
without requiring HED, RMBG, Canny, or LineArt inputs. It is intended for a
lightweight workflow that only needs crop coordinates. The internal crop
margins use the same default values as `DensePose Cropper`, so its output can be
connected to nodes that accept the standard crop-data JSON.

Input:

- `densepose` (IMAGE): DensePose hint used to locate the crop regions.

Output:

- `crop_data` (STRING): Source canvas size, analysis values, and exclusive
  `upper_crop` / `lower_crop` coordinates.

### Mannequin Cropper

Uses only an aligned mannequin `IMAGE` and its DensePose `IMAGE`. HED, Canny,
LineArt, and RMBG inputs are not required. The node applies the same DensePose
geometry used by `DensePose Cropper`, then crops the original mannequin pixels
into upper and lower regions without saving files or resizing either region.

Inputs:

- `densepose_image` (IMAGE): DensePose generated from the mannequin image.
- `mannequin_image` (IMAGE): The aligned original mannequin image, normally
  `AtoC_mannequin.png` or `DtoE_mannequin.png` at 840 x 1256 pixels.
- `upper_bottom_margin` (FLOAT): Additional lower margin for the upper crop.
- `lower_side_margin` (FLOAT): Additional side margin for the lower crop.
- `upper_y_offset` (INT): Moves the located upper crop vertically from -5 to
  +5 pixels in 1-pixel steps. Default is 0; negative values move it downward
  and positive values move it upward.
- `lower_y_offset` (INT): Moves the located lower crop downward from 0 to -200
  pixels in 5-pixel steps. Default is 0.

Outputs:

- `upper_image` (IMAGE): Original mannequin pixels inside `upper_crop`.
- `lower_image` (IMAGE): Original mannequin pixels inside `lower_crop`.
- `crop_data` (STRING): Mannequin canvas size and its exclusive upper/lower
  crop coordinates.

The mannequin image and DensePose image must have identical batch, height, and
width. `crop_data` describes the mannequin source geometry and must be connected
to `Mannequin Body Scaler.mannequin_crop_data`. Vertical offsets are applied
after DensePose location, and the shifted coordinates are stored in
`crop_data`. At an image edge, the crop is clamped without changing its height.

### Mannequin Body Scaler

Fits the two images from `Mannequin Cropper` to the upper and lower crop regions
detected for the target woman. Each region uses aspect-preserving `cover`
scaling: the mannequin is enlarged until it fills the entire target rectangle,
then only the excess is cropped. Horizontal placement aligns the mannequin and
target DensePose torso centers instead of relying on asymmetric crop margins.
It therefore cannot become narrower or shorter than the removed underwear
region. This operation uses only the crop geometry derived from DensePose; it
does not inspect HED, Canny, LineArt, flank lines, waist lines, or crotch
contours.

Inputs:

- `upper_image` (IMAGE): `Mannequin Cropper.upper_image`.
- `lower_image` (IMAGE): `Mannequin Cropper.lower_image`.
- `target_crop_data` (STRING): `DensePose Cropper.crop_data` from the target
  woman.
- `mannequin_crop_data` (STRING): `Mannequin Cropper.crop_data` from the
  selected reference mannequin.
- `removed_image` (IMAGE, optional): Full-size RGBA target image produced by
  `Join Image With Alpha`. Its embedded alpha is used directly; no RMBG mask or
  separate composite mask is required.

Outputs:

- `upper_mannequin` (IMAGE): Upper mannequin region resized to the target upper
  crop dimensions.
- `lower_mannequin` (IMAGE): Lower mannequin region resized to the target lower
  crop dimensions.
- `full_mannequin` (IMAGE): Target-canvas-sized lower layer with the resized
  upper and lower regions placed at the target crop coordinates. Pixels outside
  those regions are black and are intended to remain hidden by the target image
  placed above this layer.
- `composite` (IMAGE): When `removed_image` is connected, its unchanged RGB is
  placed over `full_mannequin` using the embedded alpha channel. Opaque target
  pixels remain unchanged, while transparent bra and panties regions reveal the
  fitted mannequin. When the optional input is disconnected, this output is a
  copy of `full_mannequin`.

Typical connection:

```text
AtoC_mannequin.png or DtoE_mannequin.png -----> Mannequin Cropper.mannequin_image
Mannequin DensePose --------------------------> Mannequin Cropper.densepose_image

Mannequin Cropper.upper_image ----------------> Mannequin Body Scaler.upper_image
Mannequin Cropper.lower_image ----------------> Mannequin Body Scaler.lower_image
Mannequin Cropper.crop_data ------------------> Mannequin Body Scaler.mannequin_crop_data
Target DensePose Cropper.crop_data -----------> Mannequin Body Scaler.target_crop_data
Join Image With Alpha.IMAGE ------------------> Mannequin Body Scaler.removed_image

Mannequin Body Scaler.full_mannequin ----------> lower/background layer
Target image with removed-underwear area ------> upper/foreground layer

or, without a separate Image Composite Masked node:

Mannequin Body Scaler.composite ----------------> completed target image
```

### Body Crop and Save (DEBUG)

Temporary debug node for cropping the loaded original photo with Body Line
Cropper's JSON, without resizing or changing crop coordinates.

Inputs: `image` (IMAGE) from the Loader, `crop_data` (STRING) from DensePose
Body Cropper, and `filename_stem` (STRING) from the same Loader.
Outputs: `upper_image`, `lower_image` (IMAGE), and `upper_path`, `lower_path`
(STRING). Saving is internal; no separate Save node is needed.

Writes `<filename_stem>__upper.png` and `<filename_stem>__lower.png` under
ComfyUI's `input/SegFit_TEST`, with previews in the node. Repeated executions
overwrite those same debug files. A batch adds `__batch001`, `__batch002`, etc.
after each region suffix. Each JSON record must match the source image size;
batch counts and each region's crop sizes must also agree. Invalid geometry is
rejected before saving, rather than silently resizing or clamping the crop.

After debugging, remove the three `DebugBodyCropSave` lines marked
`Temporary debug registration` from `__init__.py`: the import, node class
mapping, and display name mapping. This unregisters only this debug node and
does not delete its code or saved images. R375 and Cropper remain unchanged.

### Underwear Mask Restorer

Restores crop-sized removal masks to the original image canvas described by
`crop_data`. The node creates a black `[B,H,W]` canvas, places the bra mask in
`upper_crop`, and places the panties mask in `lower_crop`. Mask dimensions must
match their recorded crop dimensions; the node never silently resizes a mask.

Inputs:

- `bra_mask` (MASK): Mask generated from `upper_image`.
- `panties_mask` (MASK): Mask generated from `lower_image`.
- `bra_option` (STRING): Upper Body Line Detector `Option` output.
- `panties_option` (STRING): Lower Body Line Detector `Option` output.
- `crop_data` (STRING): Direct link from Body Line Cropper.

Automatic restoration runs only when both options are `MASK_READY` and every
batch member contains a non-empty bra and panties mask. Otherwise all three
outputs return no value: execution is not blocked and no black mask is created.
Connect `combined_removal_mask` as A of Detected Mask Switching and Garment
Poly Cutter `combined_mask` as B. Only the selected line is
passed downstream.

Outputs:

- `bra_full_mask` (MASK): Bra mask restored to the source resolution.
- `panties_full_mask` (MASK): Panties mask restored to the source resolution.
- `combined_removal_mask` (MASK): Pixel-wise union of both restored masks.
- `route` (STRING): `RESTORER` when both masks are established; otherwise
  `POLY_CUTTER`. Connect this to Detected Mask Switching for lazy A/B selection.

#### Lingerie Boundary Enhancer

### Overview
A specialized preprocessing node designed to overcome low-contrast boundary 
detection failures in segmentation models (such as SAM, EasySAM, and SegFormer) 
when dealing with pale, white, beige, or nude-toned undergarments and swimwear 
that blend into surrounding skin tones.

Rather than relying on model fine-tuning or manual touch-ups, this node utilizes 
LAB color-space adaptive histogram equalization (CLAHE) alongside morphological 
gradient and black-hat filtering to expose microscopic chromatic discrepancies, 
fabric textures, seam ridges, and subtle drop shadows cast by garment edges 
onto the skin.

### Key Features

1. **LAB Chromatic & Luminance Dynamic Range Expansion**
   * Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) independently
     across the Luminance (L) and Chromaticity (A/B) channels, accentuating subtle 
     reflectance and tint discrepancies between fabric and bare skin.
2. **Micro-Shadow & Seam Ridge Extraction**
   * Combines morphological gradients (dilation minus erosion) with black-hat 
     transforms to isolate narrow indentation shadows and seam lines caused by 
     elastic bands and straps.
3. **Dual Guidance Outputs**
   * Generates both a fully enhanced RGB guidance image optimized for SAM boundary 
     adherence and an isolated shadow/edge map for inspection, thresholding, or 
     custom blending.

### Input / Output Specifications

## Inputs
* **`image`** (`IMAGE`): Input image tensor with shape `[B, H, W, C]`.
* **`clahe_clip_limit`** (`FLOAT`, Default: `3.0`, Range: `1.0` – `10.0`, Step: `0.5`):
  * Sets the threshold for contrast limiting. Higher values exaggerate faint tonal 
    boundaries between nude fabric and skin.
* **`shadow_boost`** (`FLOAT`, Default: `1.5`, Range: `0.0` – `5.0`, Step: `0.1`):
  * Amplification factor for micro-shadows and morphological dark-ridge detection.
* **`edge_sharpness`** (`FLOAT`, Default: `1.2`, Range: `0.0` – `3.0`, Step: `0.1`):
  * Blending intensity of the extracted edge map back onto the enhanced image to 
    guide prompt-less segmenters.

## Outputs
* **`enhanced_image`** (`IMAGE`): The contrast- and edge-boosted guidance image to 
                                  be fed into SAM or SegFormer.
* **`shadow_edge_map`** (`IMAGE`): Visualized micro-shadow and morphological 
                                   gradient map for debugging or auxiliary 
                                   conditioning.

## Recommended Workflow

[Raw Person Image] ──┬──> [Lingerie Boundary Enhancer]
                     │                 │
                     │                 ▼ (enhanced_image)
                     │        [EasySAM / SegFormer] ──> [Precise Mask]
                     │                                         │
                     ▼                                         ▼
              [Masked Area Skin Fill / Inpainting] <───────────┘
                     │
                     ▼
              [Downstream Fitting / Gen Pipeline]

### BaseCanvasPrepper

Prepares a clean canvas by replacing existing underwear with a neutral surface 
(flesh-colored tape/fabric) before fitting new garments. It optionally applies 
a preparation LoRA to the model/CLIP and appends trigger keywords to the positive 
prompt to suppress garment textures and lace patterns. Connect this node upstream 
of the base KSampler in the try-on pipeline.

Inputs:

- `model`: base diffusion model.
- `clip`: text encoder CLIP model.
- `base_positive_text`: original positive prompt.
- `prep_intensity`: LoRA patch strength (`0.0` - `2.0`, default: `0.8`).
- `enable_prep`: toggles LoRA patching and trigger injection (`True` / `False`).
- `extra_keywords`: optional keywords to append for boundary or surface conditioning.

Outputs:

- `model`: patched or passthrough base model.
- `clip`: patched or passthrough CLIP model.
- `conditioning`: positive conditioning with injected trigger keywords.
- `prompt_text`: final assembled prompt string.

LoRA Setup:

- Place the surface replacement LoRA (`surface_tape_preset.safetensors`) in `ComfyUI/models/loras/`.
- If the file is missing or `enable_prep` is set to `False`, LoRA patching is 
  automatically bypassed and the original Model/CLIP are passed through.

### Lingerie Mask Preset

Provides conservative cleanup settings based on lace, opacity, and lining.

### Split Lingerie Photo

Separates one full lingerie mask into bra and pants regions. The node finds the
empty vertical interval between the two garments, crops each garment to its
mask bounding box without resizing, and performs the equivalent of
`Join Image With Alpha` internally.

Inputs:

- `photo`: full source image. RGB and RGBA images are accepted.
- `full_mask`: full-canvas mask containing both the bra and pants.

Outputs:

- `bra_image`: tightly cropped RGBA bra image with transparent background.
- `pants_image`: tightly cropped RGBA pants image with transparent background.
- `bra_mask`: mask cropped to the same size as `bra_image`.
- `pants_mask`: mask cropped to the same size as `pants_image`.

The internal `0.5` threshold is used only to locate the separating gap. The
original soft mask values are preserved in both output masks. No scaling is
performed. `bra_image` and `pants_image` can be connected directly to
`Save As Image` and later loaded with their alpha masks for `Fit Garment Pro`.

### Fit Garment to Target Mask

Crops a garment using `source_mask`, measures `target_mask`, resizes the
garment, places it on `target_image`, and alpha-composites the result.

Inputs:

- `target_image`: try-on model or the previous composite.
- `source_image`: model wearing the source garment.
- `source_mask`: source garment selection mask; garment must be white.
- `target_mask`: garment region on the try-on model; region must be white.
- `fit_mode`: `width`, `height`, `contain`, or `stretch`.
- `anchor`: vertical placement relative to the target bounding box.
- `scale_adjust`: additional scale multiplier.
- `offset_x`, `offset_y`: final pixel offsets.
- `mask_threshold`, `mask_grow`, `mask_blur`: mask edge controls.

Outputs:

- `composite`: garment composited over the target.
- `garment_layer`: full-canvas RGB garment layer.
- `placed_mask`: full-canvas placement mask.
- `garment_crop`: fitted garment crop for reference-image workflows.
- `x`, `y`, `width`, `height`: final placement values.

Suggested starting values:

| Garment | fit_mode | anchor |
| --- | --- | --- |
| Bra / swim top | `width` | `center` |
| Panties / swim bottom | `width` | `top` |

Chain two fitter nodes to build a placement guide:

```text
try-on model
  -> Fit Garment to Target Mask (bra)
  -> Fit Garment to Target Mask (bottom)
  -> Flux2 / QIE placement guide
```

### Fit Garment Advanced

Adds garment-aware anchoring and a visual fitting guide while preserving the
original fitter for compact expert workflows.

- `bra` aligns the source and target at the underband center.
- `panties` aligns the source and target at the crotch center.
- `generic` aligns mask centers.
- `scale_x`, `scale_y`, `rotation`, and offsets provide manual correction.
- Red guides show the target mask; green guides show the fitted mask.

### Fit Review Gate

Connect `debug_preview` to `preview_image`, `composite` to `fitted_image`, and
`placed_mask` to `fitted_mask`. In `preview` mode the preview remains available
but the approved outputs block downstream execution. After inspection, select
`continue` and queue the workflow again to run QIE or another image model.

### Fit Garment Pro

Provides an interactive, engine-agnostic garment placement editor. Connect the
try-on model to `person_image` and the garment reference images directly to
`bra_image` and `panties_image`. Queue once in setup mode to load the images,
then drag the garments on the model, resize from a corner, rotate from the
circular handle, and select `RUN`. That run sends the confirmed fitting guide
and masks downstream to QIE, Z-Image, Flux, inpainting, or another image engine
without a review gate. Prompts and garment reference images stay in the engine
workflow so the generation backend can be replaced without changing Pro.

Optional source masks give the cleanest crop and remain useful for semi-automatic
workflows. They are not required target masks: if omitted, a conservative edge
background estimate creates the garment alpha. The editor JavaScript only
handles interaction and preview; Python performs the final image and mask
transforms.

An optional full-size `original_underwear_mask` marks all underwear already on
the person. `replace_mask` is the union of that removal area and the newly
placed bra/panties mask, suitable for downstream inpainting or protected final
compositing. The unexpanded masks are also returned so grow/blur can be tuned in
the generation workflow.

The compact outputs are `original_person_image`, `fitting_guidance`,
`placement_mask`, `original_underwear_mask`, and `replace_mask`. Garment source
images should branch directly from their Load Image nodes to the selected
generation engine when it accepts separate references.

### Garment Poly Cutter (SegFit)
Interactive polygon cropping node for separating bra and panties regions. It
can receive Body Line Detector results and keeps an established side as an
exact, non-editable raster layer while only the incomplete side is drawn by
polygon.

- **Inputs**:
  - `person_image` (IMAGE): Input model tensor `[B, H, W, C]`.
  - `poly_state` (STRING): Serialized JSON storing polygon vertex data for bra and panties.
  - `action` (COMBO): `"setup"` mode captures preview canvas; `"run"` executes clipping and downstream processing.
  - `bra_mask`, `panties_mask` (optional MASK): Crop-sized detector masks.
  - `bra_option`, `panties_option` (optional STRING): Corresponding detector
    `Option` outputs.
  - `crop_data` (optional STRING): Body Line Cropper geometry used to
    restore an established mask without resizing it.
- **Outputs**:
  - `person_image` (IMAGE): Untouched base person image.
  - `combined_mask` (MASK): Union mask of bra and panties areas.
  - `bra_crop` (IMAGE): Tight bounding-box image of the bra area.
  - `bra_mask` (MASK): Binary mask of the bra area.
  - `panties_crop` (IMAGE): Tight bounding-box image of the panties area.
  - `panties_mask` (MASK): Binary mask of the panties area.

An option equal to `MASK_READY` locks that side. Its original raster mask is
restored to the person canvas, overlaid in the editor, and its Bra or Panties
button is disabled. `POLY_CUTTER` leaves the side editable. When both masks are
locked, the manual editor is bypassed and the Cutter emits no mask because the
Restorer line owns the completed automatic result. Poly Cutter is not connected
to Restorer; both nodes receive their detector inputs directly. With either
side open, the Cutter emits its full-canvas `combined_mask`; that output enters
input B of Detected Mask Switching directly. A ready option paired with an empty,
malformed, or
geometry-mismatched mask raises an error instead of silently editing or
replacing it.

### HED Canny Overlay (TEST)

Displays Canny lines in color directly over an aligned HED image. Connect the
original HED and Canny `IMAGE` outputs; no Remove Black node, transparency,
mask, or image inversion is needed. Matching pixel coordinates are preserved:
the node does not resize or reposition either input. Choose `dark_green`,
`red`, or `blue`, and adjust `opacity` if the HED line should show through.
`canny_background` is `black` for normal white-on-black Canny output; choose
`white` only for black-on-white Canny. The single `hed_canny_overlay` `IMAGE`
output is a comparison preview, not a replacement HED input for R375.

### Archived Body Line Detector R37.2

Experimental geometry-based node for separating Body, Garment, and Connect
trace evidence and generating conservative upper- or lower-underwear removal
masks. The aligned RMBG person mask supplies the primary Body silhouette. Canny
supplies trace topology, while HED remains visible in the diagnostic composite.

R36 restores the complete six-output detector. It uses the validated R35.3 HED
center route as boundary evidence instead of treating the full gray HED band as
a traversable line. R37 preserves that upper-body result and adds a separate
lower-body face builder so bra-specific cup and strap reconstruction is never
applied to panties.

R37.1 historically used the former Cropper order: `Canny`, `LineArt`, `HED`,
then `RMBG`. The current Body Line Cropper order is documented above; archived
R37.1 wiring should not be copied into a current workflow.

R37.2 removes the manual upper/lower selector. The detector reads the existing
Body Line Cropper `crop_data` JSON and compares the input tensor dimensions with
`upper_crop` and `lower_crop`. A unique match selects the corresponding mask
builder. Missing and ambiguous matches raise an error instead of silently using
the upper-body path.

Body and Garment are multi-label layers. A bra side band, panty side seam, or
strap end may share pixels with the Body boundary without being removed from
the Garment evidence.

- **Inputs**:
  - `canny` (IMAGE): Canny crop used for trace topology.
  - `lineart` (IMAGE): Aligned LineArt crop used only beside existing HED/Canny
    evidence and for attached lace tips below the validated boundary.
  - `hed` (IMAGE): HED crop aligned with the other inputs.
  - `rmbg_mask` (MASK): Aligned RMBG person mask `[B, H, W]`.
  - `crop_data` (STRING): Existing Body Line Cropper JSON used to identify
    whether the aligned evidence is the upper or lower crop.
  - `border_margin` (INT): Crop-border evidence width.
  - `center_band` (INT): Center-origin Garment seed width.
  - `junction_radius` (INT): Multi-port junction expansion radius.
  - `min_straightness` (FLOAT): Minimum Body continuation collinearity.
  - `silhouette_margin` (INT): RMBG Body corridor expansion.
- **Outputs**:
  - `Full Debug` (IMAGE): Combined HED and trace-classification preview.
  - `Body` (IMAGE): Classified body-boundary trace.
  - `Garment` (IMAGE): Classified garment-boundary trace.
  - `Connect` (IMAGE): Saved Body/Garment collision zones.
  - `Unknown` (IMAGE): Unclassified Canny evidence.
  - `Removal Mask` (MASK): Filled outer-envelope mask for underwear removal.

The centerline helper preserves HED as three states: background, gray SUPPORT,
and dark CORE. R35.3 thins only CORE to a one-pixel route and never merges CORE
and SUPPORT into one binary region. Short isolated CORE specks are removed.

The R35.3 helper may connect two different CORE endpoints through SUPPORT only when their
directions continue each other, at least 88 percent of the short bridge remains
inside the SUPPORT corridor, and at least 70 percent is independently confirmed
by nearby Canny. A curved endpoint receives a stricter 92-percent SUPPORT and
95-percent Canny gate. After accepting the endpoint pair, the added route follows
Canny inside SUPPORT instead of drawing a straight chord. A one-pixel Canny
offset is allowed at moderate cost; SUPPORT-only pixels carry a high cost and
repair only missing detector pixels. Each endpoint can be used once. RMBG and
LineArt does not participate in HED center-route extraction.

The R36 output is a removal mask, not a product alpha matte. Every background
component enclosed by the detected garment envelope is filled regardless of its
size. Lace holes, transparent material, and spaces enclosed by center cords are
therefore removal area; background connected to the crop border remains open.
The finished mask grows two pixels into adjacent skin. A third pixel is added
only near gray HED SUPPORT, where the visible boundary is ambiguous. Pixel
margins are measured in the detector crop resolution.

The R37 lower path treats the waistband and two leg openings as an enclosed
garment face. A center seam or decoration may divide that interior into several
faces; adjacent faces within the same outer envelope are reunited before the
skin margin is added. Closed hand, finger, arm, and isolated lace loops are
rejected when they do not form the wide central lower-garment face or extend
below its apex. The lower path does not execute cup validation, strap filling,
strap-attachment reconstruction, side-band correction, or lower-lace extrusion.
If the outer outline remains open, it returns no lower mask instead of inventing
a large filled region.

The unchanged upper path first creates the bra-core candidate from plausible
upper and lower Garment boundaries. It then rechecks the candidate against the
classified Garment and Connect outline arrays. Body lines no longer extend the
fill. The left and right
cups are validated independently, repairing internal gaps near the side band and
lower boundary without filling the center opening below the gore. Body evidence
acts only as an outward overflow stop; it never creates mask area. Garment-only
detail supplies narrow straps and lace support. Expansion and feathering for final
underwear removal belong to a later processing stage.

At the two side-band collision zones, local HED CORE evidence may continue the
Body route while preserving shared Body/Garment labels. HED SUPPORT remains a
tolerance corridor and cannot become a route by itself. When paired side-band
contacts enter and leave the same flank boundary, R34 preserves the intervening
Body pixels as a shared Garment boundary. Paired strap contacts similarly close
the short shoulder segment forming the top of each strap.

R35 additionally rechecks the geometry created by those closures. Paired
HED/Canny strap edges reconstruct the narrow attachment face where transparent
lace caused one classified edge to disappear. Along each side band, a continuous
route is selected through competing Body and Garment branches, with the
torso-facing branch preferred at collisions. This permits missing garment pixels
to be added while independently removing mask pixels that crossed into skin.

LineArt never participates in general classification. Near a validated strap or
outer collision route, it may repair a one-pixel Canny offset only while inside
the HED corridor. Below the completed mask, attached horizontal or diagonal
components may restore lace tips; vertical arm/flank lines and detached
components are rejected. Each accepted lower component now forms a small local
face with its mask attachment instead of extending every occupied column
vertically. Small enclosed detector seams are closed after reconstruction.

### Archived Body Line Detector R38

R38 is registered as a separate experimental node and does not replace R37.2.
It reuses input normalization, DensePose crop-data inference, and binary
geometry, but changes the order of decisions:

1. Build a HED-centered line source with Canny as the preferred route and
   locally supported LineArt as a fallback.
2. Confirm crop-border-origin routes as Body. A route hidden between two
   collisions is restored only when both ends continue established Body paths;
   a waist or inner-thigh route may therefore stop at an underwear collision.
3. Add the RMBG silhouette and only the Body result from the thin raw-Canny
   classifier as independent Body confirmation. R37 Garment seeds are not used.
4. Remove confirmed Body, then treat the remaining connected lines as garment
   components rather than splitting every HED junction. A valid underwear
   network must contain both a side-to-center component and a Body-supported
   component. HED-supported endpoint gaps may join adjacent garment parts.
   For Upper only, grayscale HED supplies the preferred center route through a
   broad response. Canny is a lower-confidence fallback where fine lower lace
   interrupts HED; it can connect confirmed Garment endpoints but cannot claim
   an outline by itself. Longer reconstruction is limited to a known flank and
   cannot extend beyond the outer side of its endpoint pair.
5. Reuse Body only between paired confirmed garment contacts. After selecting
   a closed garment face, keep only lines contributing to its outer boundary;
   dangling branches return to Body when they reconnect there, otherwise to
   `Rejected`. For Upper, the torso-facing Body route remains Shared from the
   shoulder contact through the strap, so both sides of each strap can close.
6. Create a removal mask only after the underwear outline encloses a validated
   face. Narrow Upper strap faces are admitted only when an HED-supported
   outline encloses them and they attach near an established cup. An open or
   implausibly small outline returns an empty mask.

The inputs match R37.2. The diagnostic outputs are `Full Debug`,
`Body Confirmed`, `Garment Outline`, `Shared`, and `Rejected`, followed by the
`Removal Mask`. On the supplied lower crop, the confirmed outer face produces a
27,760-pixel removal mask; dangling hand and body-side branches are excluded
from the final Garment outline. On the supplied upper crop with the AnyLine
`lineart_realistic` input, HED-centered closure, lower-lace Canny support, and
the shoulder Shared routes close both cups and straps. The removal mask is
42,187 pixels. R37.2 remains available as the unchanged fallback.
