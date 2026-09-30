"""Text-to-image generation with local diffusion models.

Three backends are supported:

* SD-Turbo     (``stabilityai/sd-turbo``) — the original default. 1–2 step,
  512 × 512, ~1.7 GB. Good for quick throwaway demos.
* SDXL-Turbo   (``stabilityai/sdxl-turbo``) — recommended. Same ADD technique
  at SDXL scale: 512–1024 px, ~6.5 GB, no login required.
  Pass ``model="sdxl-turbo"``.
* FLUX.1-schnell (``black-forest-labs/FLUX.1-schnell``) — highest quality,
  1024 × 1024 native but defaulted to 512 here (1024 OOMs MPS on this Mac),
  ~12 GB. Requires a HuggingFace login (gated repo). Pass ``model="flux"``.

All three are latent diffusion models: they start from noise and iteratively
denoise toward an image that matches the prompt.

Attribution: any generated image used in the book must be followed by a
markdown caption cell citing the model — ``sauer2023adversarial`` for
SD-Turbo and SDXL-Turbo (both use ADD), ``flux2024schnell`` for FLUX —
and including the prompt used.
"""
import functools

import torch

DEFAULT_IMAGE_MODEL = "stabilityai/sd-turbo"
SDXL_TURBO_MODEL    = "stabilityai/sdxl-turbo"
FLUX_MODEL          = "black-forest-labs/FLUX.1-schnell"

_DEVICE = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available()
           else "cpu")


@functools.lru_cache(maxsize=2)
def _sd_pipeline(model: str):
    """Load and cache an SD / SDXL Turbo pipeline (downloaded once, reused after)."""
    from diffusers import AutoPipelineForText2Image
    dtype = torch.float32 if _DEVICE == "cpu" else torch.float16
    pipe = AutoPipelineForText2Image.from_pretrained(
        model, torch_dtype=dtype, variant="fp16"
    )
    return pipe.to(_DEVICE)


@functools.lru_cache(maxsize=1)
def _flux_pipeline():
    """Load and cache the FLUX.1-schnell pipeline (~12 GB, downloaded once)."""
    from diffusers import FluxPipeline
    dtype = torch.bfloat16 if _DEVICE != "cpu" else torch.float32
    pipe = FluxPipeline.from_pretrained(FLUX_MODEL, torch_dtype=dtype)
    if _DEVICE == "cpu":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(_DEVICE)
    return pipe


def generate_image(prompt: str, steps: int = None, seed: int = 0,
                   size: int = None, model: str = "sdxl-turbo"):
    """Generate an image from a text prompt; return a PIL.Image.

    Args:
        prompt: Text description of the desired image.
        steps:  Inference steps. Defaults: 2 (SD-Turbo), 4 (SDXL-Turbo / FLUX).
        seed:   Fixed seed for reproducibility.
        size:   Square output size in pixels.
                Defaults: 512 (SD-Turbo and FLUX), 1024 (SDXL-Turbo).
        model:  ``"sdxl-turbo"`` (default), ``"sd-turbo"``, or ``"flux"``.
    """
    gen = torch.Generator("cpu").manual_seed(seed)

    if model == "flux":
        if steps is None:
            steps = 4
        if size is None:
            size = 512  # FLUX at 1024 OOMs MPS on this Mac; 512 is the standard
        pipe = _flux_pipeline()
        return pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            height=size,
            width=size,
            generator=gen,
        ).images[0]

    hf_model = SDXL_TURBO_MODEL if model == "sdxl-turbo" else DEFAULT_IMAGE_MODEL
    if steps is None:
        steps = 4 if model == "sdxl-turbo" else 2
    if size is None:
        size = 1024 if model == "sdxl-turbo" else 512
    pipe = _sd_pipeline(hf_model)
    return pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=0.0,
        height=size,
        width=size,
        generator=gen,
    ).images[0]


def gallery(prompts, model="flux", size=512):
    """Generate several prompts and lay the images out in one left-to-right montage.

    Args:
        prompts: list of ``(prompt, seed)`` pairs. The seed is per-image so a
                 prompt that collapses to noise at one seed can be re-rolled.
        model, size: passed through to :func:`generate_image`.

    Renders a single matplotlib figure (one panel per prompt) and returns None,
    so the cell shows the montage and nothing else.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(prompts), figsize=(4 * len(prompts), 4))
    for ax, (prompt, seed) in zip(axes, prompts):
        ax.imshow(generate_image(prompt, model=model, seed=seed, size=size))
        ax.axis("off")
    plt.tight_layout(pad=0.4)
    plt.show()
