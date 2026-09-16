# Third-party notices

The [MIT licence](LICENSE) covers what was written for this repository: the code, tests,
workflows and documentation, the invented lesson corpus in `projects/p12_study_hub/corpus/`,
the question fixtures in `projects/p02_question_bank/fixtures/`, the synthetic sample
`projects/p01_mini_etl/data/sample_raw.csv`, and the synthetic data the projects generate
when they run.

It does not cover the material below, which keeps its own terms. Each project README's
"Datasets and licences" section is the authoritative, per-file list.

## Datasets committed to this repository

| Files | Terms |
|---|---|
| `projects/p01_mini_etl/datasets/` (12 files) | Per file: CC0, CC BY 4.0, ODC-PDDL-1.0, BSD-3-Clause, public domain (NOAA), and, for five teaching datasets redistributed through seaborn-data and vega-datasets, no explicit licence upstream. See P01's README. |
| `projects/p03_regression/data/FuelConsumptionCo2.csv` | Open Government Licence – Canada. Contains information licensed under the Open Government Licence – Canada. |
| `projects/p03_regression/data/penguins.csv` | CC0 (Palmer Penguins, Gorman et al. 2014). |
| `projects/p03_regression/data/auto_mpg.csv`, `tips.csv` | No explicit licence upstream (redistributed through seaborn-data for teaching). |

"No explicit licence upstream" means exactly that: those files are not relicensed under MIT
and are not claimed to be in the public domain. Where a dataset was filtered or
re-serialised, the project README says how.

## Datasets fetched at run time

P03-P06 download further public datasets into a local cache on first use; they are never
committed. Their sources and licences are in each project's README.

## Model weights downloaded at run time

No model weights are stored in this repository. The projects download them on first use:

| Model | Used by | Licence |
|---|---|---|
| `Salesforce/blip-image-captioning-base` | P08 | BSD-3-Clause |
| `facebook/blenderbot-400M-distill` | P09 | Apache-2.0 |
| `openai/whisper-tiny.en` | P10 | MIT |
| `Qwen/Qwen2.5-0.5B-Instruct` (`local` provider) | P10 | Apache-2.0 |
| `qwen2.5:1.5b` through Ollama (`ollama` provider) | P10-P12 | Apache-2.0 |
| `sentence-transformers/all-MiniLM-L6-v2` | P11, P12 | Apache-2.0 |

## Python dependencies

Dependencies are installed from PyPI by `uv sync`; none is vendored into this repository.
Most are under permissive licences (MIT, BSD, Apache-2.0). Three are under copyleft licences
that apply to their own files only: `fpdf2` (LGPL-3.0), `pyttsx3` (MPL-2.0) and `certifi`
(MPL-2.0). Anyone who redistributes an environment or image built from this repository
(for example the Docker image) redistributes those packages under their own terms.
