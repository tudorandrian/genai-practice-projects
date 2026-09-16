# P08 - BLIP Image Captioning

## What it does

Turns any image into a short English caption using
[BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)
(`Salesforce/blip-image-captioning-base`), served two ways: a Gradio
drag-and-drop UI and a folder batch mode. This is the repository's first
project that loads real Hugging Face model weights instead of running a
hand-written algorithm - the `models` dependency group and a GPU-or-CPU
inference step, rather than a `core`-tier script.

`captioner.py` is the engine: `load_model()` downloads BLIP once into the
Hugging Face cache and keeps it in module-level singletons, `device()` picks
`"cuda"` when available and falls back to `"cpu"`, and `caption_image(image,
processor, model)` is a pure `Image -> str` function that takes an injected
processor/model pair so it is unit-testable with fakes, without downloading
anything. `caption(image)` is the convenience wrapper that loads the cached
model and calls `caption_image`.

```
caption_image(image, processor, model) -> str      # pure, injectable, testable
caption(image) -> str                                # wrapper: load_model() + caption_image()
```

Two thin front ends sit on top of the engine and never touch BLIP directly:

- **`app.py`** - a Gradio `Interface` around `caption`; `describe_image` is
  the UI callback, `build_demo()` builds the interface, `main()` warms the
  model cache and launches the server.
- **`batch.py`** - `caption_folder(folder, output_path)` loads the model
  *once* via `captioner.load_model()`, then reuses that same
  `(processor, model)` pair for every image in the folder, instead of
  relying on `caption`'s per-call cache lookup. It writes one
  `filename: caption` line per image and skips files that fail to open as
  images, logging the skip instead of crashing.

`synthetic_images.py` generates a small, deterministic, offline image set
(no external downloads) so the project is fully self-contained; `demo()`
uses it to caption six images end to end.

## Run

```bash
uv sync --group models                              # torch, transformers, gradio, pillow
uv run p08-image-captioning --demo                   # captions 6 synthetic images, writes output/
uv run p08-image-captioning --image path/to/pic.png  # caption a single image
uv run python -m projects.p08_image_captioning.app                   # Gradio UI, http://127.0.0.1:7860
uv run python -m projects.p08_image_captioning.app --host 0.0.0.0    # listen on every interface
uv run python -m projects.p08_image_captioning.batch [folder] [out]  # caption a folder
uv run pytest projects/p08_image_captioning -q                # fast tests, no weights needed
uv run pytest projects/p08_image_captioning -m models -q      # real BLIP inference (slow)
```

`--verbose` logs at `INFO` on every entry point; by default only the summary
lines print. `--demo` regenerates `test-images/` (gitignored), captions all
six images, and writes `output/captions.txt` and `output/metrics.txt`. Only `--demo`
writes those committed files: the batch script defaults to `output/runs/captions.txt`,
which Git ignores, so your own image names never end up in a commit.

## Example output

`uv run p08-image-captioning --demo`:

```
p08-image-captioning: ok
  images: 6
  device: cpu
  model: Salesforce/blip-image-captioning-base
  wrote: output/
```

`seconds` (model load + inference time) varies run to run and is reported
only on the console and in the returned `DemoResult`, never in a committed
file - running `--demo` twice in a row leaves `git status` clean, because
generation is greedy and deterministic (`num_beams=1, do_sample=False`).

`output/metrics.txt` (committed, deterministic):

```
images=6
device=cpu
model=Salesforce/blip-image-captioning-base
```

`output/captions.txt` (committed, deterministic):

```
ball.jpg: a blue circle with a white background
gradient_gray.jpg: a black and white background with a white border
house.png: a house with a red roof
landscape.png: a yellow sun on a blue sky
text.png: the cover of the book, ' the world '
transparent.png: a red circle with a black background
```

## Design notes

- **Engine/UI separation.** `captioner.py` never imports `gradio`; `app.py`
  and `batch.py` never call `transformers` or `torch` directly. Either front
  end could be replaced (a CLI, a REST endpoint) without touching the
  inference code, and `caption_image`'s fake-injectable signature is what
  makes the fast test suite possible.
- **Lazy heavy imports.** `torch` and `transformers` are imported inside
  `device()` and `load_model()`, not at module level, so `captioner.py` is
  importable - and its pure logic testable - without the `models` dependency
  group installed. `app.py` is the one module that imports `gradio` at the
  top, which is why no `core`-marked test imports `app.py`; building or
  loading the UI is exercised only under the `models` marker, if at all.
- **`device()` auto-selects, and is itself testable without a GPU or even
  torch installed.** It does `import torch` *inside* the function body, so a
  test can put a fake module into `sys.modules["torch"]` before calling it
  (`monkeypatch.setitem(sys.modules, "torch", ...)`) and exercise the
  cuda-unavailable branch - see `test_device_falls_back_to_cpu` - without
  needing the real package.
- **The batch path loads the model exactly once.** `caption_folder` calls
  `captioner.load_model()` a single time up front and threads the returned
  `(processor, model)` through every `caption_image` call in the loop,
  rather than calling `caption()` per image (which would hit the
  already-loaded cache anyway, but less explicitly). `batch.py` imports the
  `captioner` *module*, not individual names from it - a
  `from captioner import load_model` would bind a private copy of the
  function, which a test that patches `captioner.load_model` afterwards
  would never see; see `test_batch_loads_the_model_once`.
- **Deterministic decoding.** BLIP's `generate()` defaults are not
  guaranteed greedy across versions, so `caption_image` passes
  `num_beams=1, do_sample=False` explicitly. That is what keeps
  `output/captions.txt` byte-identical across repeated `--demo` runs.
- **Output contract.** `caption_folder`, `load_model` and `demo()` never
  `print`; they log through `logging.getLogger(__name__)`. Each module's
  `main()` prints at most a handful of summary lines and sets the logging
  level (`WARNING`, or `INFO` with `--verbose`).
- **`app.py` binds `127.0.0.1` by default**, never `0.0.0.0` - pass
  `--host 0.0.0.0` to opt into listening on every interface.

## Limits

- **CPU-only on this machine.** `device()` returns `"cuda"` when
  `torch.cuda.is_available()`, but this repository was developed and tested
  on a CPU-only box, so every committed caption was produced on `cpu`; a
  GPU box will pick `cuda` automatically and may produce different captions
  (deterministic decoding is per-device, not cross-device).
- **The model needs ~1 GB of disk and a one-time download.** The first
  `load_model()` call downloads BLIP into the Hugging Face cache
  (`~/.cache/huggingface`); later runs reuse it. `demo()` and the `models`-
  marked test are never run in CI for this reason - see `.github/workflows/
  heavy.yml`, a manually-dispatched workflow, versus the `core`-only `ci.yml`.
  A full `--demo` run (model already cached) takes on the order of 20–30
  seconds on CPU; a cold cache adds the download time.
- **Captions are short and generic.** BLIP-base was chosen for size and
  speed over accuracy; captions are typically five to ten words and
  sometimes miss fine detail (see the `text.png` example above, which reads
  as "the cover of the book" rather than transcribing the words).
- **No authentication, rate limiting, or queueing on the Gradio server.**
  `app.py` is a local demo UI, not hardened for public exposure.

## Datasets and licences

There is no external dataset. `synthetic_images.py` generates six small
images with Pillow's `ImageDraw` (a landscape, a ball, a house, a text
banner, a transparent RGBA circle and a grayscale gradient) plus a
`not_an_image.txt` decoy - all drawn programmatically for this project, not
sourced from any corpus. They are written into the gitignored
`test-images/` directory and regenerated by `demo()` on every run.

The BLIP model is
[`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base)
from the Hugging Face Hub, licensed BSD-3-Clause, pretrained on the COCO
captioning dataset by the model's authors (Li et al., *BLIP: Bootstrapping
Language-Image Pre-training for Unified Vision-Language Understanding and
Generation*, 2022). Weights are downloaded on first use and cached locally;
this project only performs inference against them.

## Courses drawn on

- 6 Building Generative AI-Powered Applications with Python
