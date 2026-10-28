# C389 CA1: People Counting Assistant + Twist 2 (Cyclists)

Counts people moving **left** vs **right** at a pedestrian crossing from a single still image, and (Twist 2) classifies each crosser as **pedestrian** or **cyclist**.

- Full spec: [PRD.md](PRD.md)
- Team briefing: [docs/CA1_Team_Briefing.md](docs/CA1_Team_Briefing.md)

## Hard rules (read before you commit)

- `fliplr=0.0` everywhere. No flip augmentation in Roboflow.
- Class order: `person_left=0`, `person_right=1`, `bicycle=2`.
- `ultralytics==8.4.120` (pinned).
- **Never** commit the Roboflow API key. Use Colab Secrets (`ROBOFLOW_API_KEY`) or `getpass`.
- **Never** commit photos, datasets, runs or `*.pt` weights. They stay on the module Drive / Roboflow.
- Clear notebook outputs before committing.
- The test set (`test_v1`) is used once, for the final comparison only.

## Layout

```
notebooks/   lecturer notebooks + our adapted copies (outputs cleared)
src/         config, data tooling, counting, training, evaluation
app/         Gradio service
scripts/     timing and helper scripts
tests/       pytest
reports/     generated metrics and plots (no raw photos)
docs/        team docs
```

## Setup

**Colab (T4):**

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/Morn-Thuta/c389-ca1.git
%cd c389-ca1
!pip install -r requirements.txt
import os; os.environ["PROJECT_ROOT"] = "/content/drive/MyDrive/C389"
```

**Local:**

```bash
pip install -r requirements.txt
export PROJECT_ROOT=/path/to/C389
```

## Data checks (run before every training run)

Run from the repo root, pointing at the downloaded Roboflow export (the folder with `data.yaml`):

```bash
python -m src.data.verify_export $PROJECT_ROOT/runs/L3/dataset --expect 2
python -m src.data.dedupe_check $PROJECT_ROOT/runs/L3/dataset
```

- `verify_export` checks class order, counts per split/class, empty labels, small boxes and bad IDs. `--strict` also fails on warnings.
- `dedupe_check` fails if any near-duplicate images sit in different splits.

## Pre-labelling (label every person and bike faster)

```bash
python -m src.data.prelabel ORIGINAL_EXPORT_DIR --out OUTSIDE_REPO_DIR --direction-weights best.pt
```

- **Boxes:** the stock `yolo26n.pt` draws a box for every person and bicycle.
- **Direction:** our baseline sets each person's left or right.
- **Existing human labels are kept.** Only missing people and bikes are added.
- **Output:** a 3-class YOLO dataset to upload to Roboflow, plus `prelabel_review.csv`, which lists what was added per image (busiest first, with `direction_unknown` flags).
- **Every image must still be checked by a person.**
- **Use the original photos, not the masked export.** Masked people are greyed out and can't be found.

Tests: `python -m pytest -q`

## Running the app (Gradio)

```bash
python -m app.app --share
```

- **Weights:** defaults to `$PROJECT_ROOT/artifacts/L3/best.pt`, the baseline saved by the L3 notebook. Override it with `--weights` or `MODEL_WEIGHTS`. Our Drive folder is `C389_2`, so on Colab set `PROJECT_ROOT=/content/drive/MyDrive/C389_2`.
- **Bicycles:** until `twist_3cls` is trained, they come from the stock `yolo26n.pt`, which downloads automatically. Once the weights include a `bicycle` class, that helper model is skipped. `--bike-weights none` turns it off.
- The reply keeps the L4 shape, with the counts from PRD §7.3. Every request logs one line with status, latency and counts.

Code: `src/counting.py` (linking and counts), `src/detector.py` (model wrapper), `app/service.py` (validate, count, reply, annotate) and `app/app.py` (the page).

## Workflow

- Branch off `main` for each task (`feat/verify-export`, `feat/counting`, ...), then open a PR.
- Milestones and acceptance criteria: PRD §11 and §12.
