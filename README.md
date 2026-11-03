# C389 CA1: People Counting Assistant + Twist 2 (Micromobility crossers)

Counts people moving **left** vs **right** at a pedestrian crossing from a single still image, and (Twist 2) counts **wheeled crossers** (a rider on a bike or scooter) moving left and right.

- Full spec: [PRD.md](PRD.md)
- Team briefing: [docs/CA1_Team_Briefing.md](docs/CA1_Team_Briefing.md)

## Hard rules (read before you commit)

- `fliplr=0.0` everywhere. No flip augmentation in Roboflow.
- Class order: `person_left=0`, `person_right=1`, `wheeled_left=2`, `wheeled_right=3` (the lecturer's Twist 2 classes).
- A rider and their bike or scooter share **one** `wheeled_*` box, in the direction they travel. Someone walking a bike is `person_*`. Parked bikes get no box.
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

## Pre-labelling (label every person and rider faster)

```bash
python -m src.data.prelabel ORIGINAL_EXPORT_DIR --out OUTSIDE_REPO_DIR --direction-weights best.pt
```

For large phone photos (e.g. 2560x1920) add `--imgsz 1280`: the stock model then finds far more small, far-away riders and people.

- **Boxes:** the stock `yolo26n.pt` draws a box for every person and bicycle. A person riding a bike becomes one `wheeled_*` box around both; bikes nobody rides are dropped.
- **Direction:** our baseline sets each person's left or right.
- **Existing human labels are kept.** Only missing people and riders are added.
- **Output:** a 4-class YOLO dataset to upload to Roboflow, plus `prelabel_review.csv`, which lists what was added per image (busiest first, with `direction_unknown` flags).
- **Every image must still be checked by a person.**
- **Use the original photos, not the masked export.** Masked people are greyed out and can't be found.
- **Existing labels are read by class name**, so Roboflow's alphabetical order is converted automatically.
- **`--split-log path/to/v2_cleaned/split_log.csv`** re-applies the v2 cleaning to the original v1 export. It moves images to the cleaned split and drops the 10 duplicates.

## After every Roboflow export: fix the class order

Roboflow lists classes alphabetically. Rewrite the export to `person_left=0, person_right=1, wheeled_left=2, wheeled_right=3` before checking or training:

```bash
python -m src.data.remap_classes EXPORT_DIR --out FIXED_DIR
```

### Older exports with a separate `bicycle` class

Versions labelled before the switch have a person box plus a bicycle box for each cyclist. Convert them instead of remapping:

```bash
python -m src.data.to_wheeled EXPORT_DIR --out WHEELED_DIR
```

- A person riding a bike becomes one `wheeled_*` box around both, in the person's direction (rule in `src/linking.py`).
- A bike nobody rides (parked, pushed) is removed. Everyone else stays `person_*`.
- `to_wheeled_review.csv` lists every image where a rider was merged or a bike removed. **Check those in Roboflow**: the rule was written for model detections, so it can get a hand-drawn box wrong.

## Then split 70/20/10 without leaking burst shots

Generating a version in Roboflow can re-split images at random, which mixes near-identical burst shots across train, valid and test. Copy the fixed export into a clean 70/20/10 split:

```bash
python -m src.data.resplit FIXED_DIR --out RESPLIT_DIR --ratio 70/20/10
```

- Near-identical photos (the same distance `dedupe_check` uses) form a **burst group**, and a group always stays in one split.
- Groups are shared out at 70/20/10 **within each session**, so every location and day appears in every split.
- Photos with wheeled crossers are shared out first, so riders also land 70/20/10 and valid and test always have some.
- `--seed` changes the shuffle; the same seed always gives the same split.
- `resplit_log.csv` in the output records each image's old split, new split and burst group.
- Then run `verify_export` and `dedupe_check` on `RESPLIT_DIR`.

To fix splits by hand instead, use `--split-log path/to/split_log.csv` and/or `--session DATE=SPLIT`.

Tests: `python -m pytest -q`

## Running the app (Gradio)

```bash
python -m app.app --share
```

- **Weights:** defaults to `$PROJECT_ROOT/artifacts/L3/best.pt`, the baseline saved by the L3 notebook. Override it with `--weights` or `MODEL_WEIGHTS`. Our Drive folder is `C389_2`, so on Colab set `PROJECT_ROOT=/content/drive/MyDrive/C389_2`.
- **Wheeled crossers:** counted when the weights include `wheeled_left` and `wheeled_right` (`twist_4cls`). With the 2-class baseline only people are counted.
- The reply keeps the L4 shape, with the counts from PRD §7.3. Every request logs one line with status, latency and counts.

Code: `src/counting.py` (counts), `src/linking.py` (rider + bike to one wheeled box, for data tools), `src/detector.py` (model wrapper), `app/service.py` (validate, count, reply, annotate) and `app/app.py` (the page).

## Workflow

- Branch off `main` for each task (`feat/verify-export`, `feat/counting`, ...), then open a PR.
- Milestones and acceptance criteria: PRD §11 and §12.
