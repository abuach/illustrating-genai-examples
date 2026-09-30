"""Multimodal: send one or more images + a text prompt to a vision model."""
import requests
import ollama
from genai.llm import SERVER

_client = ollama.Client(host=SERVER)
DEFAULT_VISION_MODEL = "gemma3:latest"


def _to_bytes(url_or_bytes) -> bytes:
    """Accept either raw image bytes or a URL string; always return bytes."""
    if isinstance(url_or_bytes, str):
        resp = requests.get(url_or_bytes, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        return resp.content
    return url_or_bytes


def ask_image(url_or_bytes, prompt: str, model: str = DEFAULT_VISION_MODEL) -> str:
    """Analyze a single image with a text prompt. Pass a URL string or raw bytes."""
    return ask_images([url_or_bytes], prompt, model=model)


def ask_images(images: list, prompt: str, model: str = DEFAULT_VISION_MODEL) -> str:
    """Analyze several images at once with one text prompt.

    Pass a list of URL strings or raw byte arrays. The model holds every image
    in context together, which is what makes cross-image reasoning possible.
    """
    blobs = [_to_bytes(img) for img in images]
    result = _client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": blobs}],
        options={"temperature": 0.2},
    )
    return result["message"]["content"].strip()


def mnist_digits(n: int = 25, seed: int = 7, size: int = 196, root: str = ".cache_mnist"):
    """Return n (png_bytes, label) pairs from the classic MNIST test set.

    Each handwritten digit is inverted to dark ink on white paper and upscaled,
    so it reads like a scanned sample we can hand to a vision model.
    """
    import io, random
    from torchvision import datasets
    from PIL import Image
    ds = datasets.MNIST(root=root, train=False, download=True)
    out = []
    for i in random.Random(seed).sample(range(len(ds)), n):
        pil, label = ds[i]
        ink = Image.eval(pil, lambda p: 255 - p).resize((size, size), Image.LANCZOS).convert("RGB")
        buf = io.BytesIO(); ink.save(buf, format="PNG")
        out.append((buf.getvalue(), label))
    return out


# --- display helpers -------------------------------------------------------

def show_vision_replies(img, question, models, limit: int = 90,
                        label_width: int = 18) -> None:
    """Ask each vision model the same question about one image; print a capped reply."""
    for model in models:
        reply = ask_image(img, question, model=model).strip()
        print(f"{model:{label_width}} → {reply[:limit]}")


def show_position_answers(img, q_right, q_fourth, models) -> None:
    """Two spatial questions about one image, answered by each model side by side."""
    for model in models:
        right = ask_image(img, f"What color is {q_right}? One word.",
                          model=model).strip()
        fourth = ask_image(img, f"What color is {q_fourth}? One word.",
                           model=model).strip()
        print(f"{model:14}  right of green: {right:8}  fourth: {fourth}")


# --- captured transcriptions ----------------------------------------------
# Real replies to "Transcribe the text in this terminal screenshot exactly."
# on images/mm/terminal.png (a Python KeyError traceback), captured once on
# 2026-06-22 because vision sampling drifts run to run. The frozen notebook
# cell renders these so the prose can describe what each model actually said;
# scripts/_qwen3vl_probe.py regenerates them. qwen3-vl reads the screen line for
# line; llava confabulates an unrelated traceback that is nowhere in the image;
# the small llava-phi3 reads the real tokens but stutters and invents a tail.
TERMINAL_TRANSCRIPTS = {
    "llava:latest": '''The text in the terminal screenshot is:

```
$ python3
Traceback (most recent call last):
  File "main.py", line 48, in <module>
    print(f"{data['name']} {data['age']}")
NameError: name 'data' is not defined
```''',
    "qwen3-vl:8b": '''python3 – 80×24
$ python3 app.py
Traceback (most recent call last):
File "app.py", line 42, in <module>
result = process_data(df, config)
File "app.py", line 17, in process_data
return df.groupby(config['key']).agg(config['fn'])
KeyError: 'key'
$ _''',
    "llava-phi3:3.8b": (
        'python3 app.py Traceback (most recent call last): File "app.py", '
        'line 24, in <module> result = process_data(process_data) TypeError: '
        'cannot return multiple values from a function() KeyboardInterrupt: Keyb'
    ),
}
