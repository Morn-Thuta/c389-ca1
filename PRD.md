# PRD: People Counting Assistant + Twist 2 (Micromobility crossers)

C389 AI Solution Design & Implementation · CA1 · Team of 6
Deadline: **Lesson 8** (live demo + CA1 report, same day)

---

## 1. Context for Claude Code

- This is a polytechnic coursework project.
- The lecturer supplies Colab notebooks: `L3_training.ipynb`, `L4_deployment.ipynb`, `L5A_new_class.ipynb` and `L5B_tracking.ipynb`.
- We extend them rather than replace them. Keep their function names and the Drive folder layout so they still work.
- Individual worksheets and reflection journals are written by students, **not by AI**. This PRD covers code only.

## 2. Problem

- **Base task:** count how many people at a pedestrian crossing are moving **left** vs **right**, from a **single still image** taken by a fixed camera.
- **Twist 2 (micromobility crossers), with the lecturer's classes:** keep `person_left` / `person_right` and add `wheeled_left` / `wheeled_right`. A rider and their bike or scooter are **one** wheeled box, with the direction they travel.

**Users:** campus or traffic planners, who use the counts to judge flow imbalance and how many riders use the crossing.

## 3. Scope

**In scope**
- A YOLO26 detector with 4 classes: `person_left (0)`, `person_right (1)`, `wheeled_left (2)`, `wheeled_right (3)`
- Per-image counting by class and direction (the classes are the answer, so no linking step at inference)
- Evaluation: baseline vs twist model on the **same frozen test set**
- A Gradio web service with a structured JSON reply
- Tests for bad inputs, and cold/warm timing

**Out of scope**
- Identifying people (faces, re-identification)
- Tracking across frames, video streaming, counting across a sequence (a count is **per image**)
- Scooters, skateboards and other wheeled devices
- Alarms, alerts, and connections to other systems

## 4. Hard rules (never violate)

| Rule | Why |
|---|---|
| `fliplr=0.0` everywhere, and no flip augmentation in Roboflow | A flipped image reverses direction but keeps the old label |
| Class order: `person_left`=0, `person_right`=1, `wheeled_left`=2, `wheeled_right`=3 | Class IDs are positional, and the notebooks check the first two |
| Pin `ultralytics==8.4.120` | Results must be comparable across teams |
| Left/right = direction across the **image**, not the person's own left/right | Labelling convention |
| Never put the Roboflow API key in code or notebooks. Use Colab Secrets `ROBOFLOW_API_KEY` or `getpass` | Notebooks go to the team repo |
| Use the test set once, for the final comparison only. No tuning against it | Keeps the test result honest |
| Images of real people never leave the module drive or Roboflow, and are never committed to git | Privacy commitment from Lesson 1 |

## 5. Data

### 5.1 Current state
- **v2 cleaned dataset:** 191 images (132 train / 39 valid / 20 test), 536 person boxes (284 left / 252 right), 0 bicycle boxes.
- Images are stretched to 512×512. About 160 images have unlabelled people (listed in `REVIEW_LIST.csv`).
- **Known leak:** the old L3 run trained on leftover files in `MyDrive/C389/runs/L3/dataset`, so 577 images were counted instead of 191. That folder must be wiped before every download.

### 5.2 New data being collected
- About **120–150 pedestrian photos** from 3–4 new locations: groups, overlapping people, far-away people, varied light.
- About **100 rider photos**: 150–200+ `wheeled_*` boxes, about 50/50 left/right, mostly mixed with walkers.
- **Optional:** outside images (≤30% of the dataset, lecturer's limit) as extra training data only, never in valid or test. Each rider and vehicle become one `wheeled_*` box, and every person is labelled left/right.

### 5.3 Split policy
- Split **by capture session** (location + time), so one session never spans two splits.
- Test set = **our own photos only**: the current 20 test images plus about 15–20% of the new sessions. Freeze it as `test_v1` before any new training.

### 5.4 Data tooling to build (`src/data/`)
1. `verify_export.py`
   - Check that the `data.yaml` class order matches the expected order.
   - Count images and boxes per split and per class.
   - Flag empty label files and boxes smaller than 10% of image height.
   - Flag label IDs outside the class range.
2. `dedupe_check.py`
   - Use a perceptual hash (`imagehash.phash`) across all splits.
   - Report near-duplicates (Hamming distance ≤ 6) that appear in **different splits**.
   - Exit non-zero if any are found.
3. `session_split.py`
   - Assign images to train/valid/test by session.
   - Session = filename prefix or capture-date group, editable through `sessions.csv`.
   - Write a `split_log.csv`.
4. `merge_coco_bicycle.py` (optional)
   - Pull COCO images that contain `bicycle` and remap COCO classes to ours.
   - Write a CSV of the people that still need left/right labels, for review in Roboflow.
5. `make_review_sheet.py`
   - Draw labels on images in a contact-sheet grid, for quick human review.

## 6. Model

| Item | Value |
|---|---|
| Base weights | `yolo26n.pt` |
| imgsz | 640 (re-export from Roboflow with **Fit (black edges) 640**, not Stretch) |
| Epochs | 100 (raise if the step count is under about 1,500) |
| Batch | 16 |
| fliplr | 0.0 |
| seed | 0 |
| Runs | `baseline_2cls` (person_left, person_right) and `twist_4cls` (+ wheeled_left, wheeled_right) |

Outputs go to `MyDrive/C389/runs/<run>/`. The best weights are copied to `MyDrive/C389/artifacts/<run>/best.pt`, together with a `metrics.json`.

## 7. Counting (`src/counting.py`)

### 7.1 Inputs
Detections as `{class_name, confidence, box:[x1,y1,x2,y2]}`, the same shape as `to_detections()` in L3. Apply the confidence threshold first. The default is 0.40, and it must be justified with evidence.

### 7.2 Labelling rule for wheeled crossers
- A person **riding** a bike or scooter: one `wheeled_left` / `wheeled_right` box around rider and vehicle, in the direction they travel.
- A person **walking or carrying** a bike: also `wheeled_left` / `wheeled_right`, one box around person and bike (team decision, 3 Nov).
- A parked bike with nobody at it: no box.

Old exports labelled a cyclist as a person box plus a `bicycle` box. `src/data/to_wheeled.py` converts them with the linking rule in `src/linking.py`: the bike is the person's if it sits under their lower body (riding: bottom 60% of the person box covers ≥ 25% of the bike, x-centre within the bike's x-range ±10%) or beside it (walking: level with the lower body and within 25% of the person's width). The same rule merges stock person and bicycle boxes when pre-labelling. Every converted image is checked by a person.

### 7.3 Output
```json
{
  "status": 200,
  "counts": {
    "person_left": 2, "person_right": 1,
    "wheeled_left": 0, "wheeled_right": 1,
    "total_left": 2, "total_right": 2
  },
  "confidence_threshold": 0.40,
  "detections": [
    {"class_name": "wheeled_right", "confidence": 0.88, "box": [..], "id": 0, "direction": "right"},
    {"class_name": "person_left", "confidence": 0.91, "box": [..], "id": 1, "direction": "left"}
  ]
}
```
Every count key is always present, even at 0. An empty scene returns `200` with all zeros, not an error.

## 8. Evaluation (`src/eval/`)

1. **Detector metrics** on the frozen `test_v1`, at the same threshold for every model:
   - precision, recall, mAP50 and mAP50-95 **per class**
   - confusion matrix
2. **Comparison table:** old baseline (if reproducible) vs new `baseline_2cls` vs `twist_4cls`. Show that `person_left`/`person_right` did **not** get worse after adding the wheeled classes.
3. **Count accuracy:**
   - The team hand-counts each test image into `test_counts.csv` (`image, person_left, person_right, wheeled_left, wheeled_right`).
   - Report the mean absolute error per count and the % of images with an exact count.
4. **Person vs wheeled:** the confusion matrix rows for person ↔ wheeled mix-ups (a rider called a walker, or the reverse).
5. **Error gallery:** save the worst 10 images, with predictions and ground truth drawn on, sorted into missed crosser, wrong direction, person/wheeled mix-up, or false detection.
6. **Threshold sweep:** counts and F1 at 0.15, 0.25, 0.40, 0.60 and 0.80, on validation.

Outputs: `reports/metrics_*.json`, `reports/*.png`, `reports/error_gallery/`.

## 9. Deployment (`app/`)

Extend the L4 pattern. Keep these function names:
- `validate_image(file) -> (ok, image | error_reply)`
  - Refuse: nothing sent, an empty file, an unreadable or non-image file, an image over 4096 px or 15 MB.
  - Return `400` with a message saying what to send instead.
- `count_people(model, image, threshold) -> (counts, detections)`: uses `src/counting.py`.
- `build_reply(...) -> dict`: the JSON in section 7.3.
- `predict(image, threshold)`: the Gradio entry point. It returns the annotated image and the JSON.
- The model is loaded **once at startup**.
- Gradio inputs: image upload, threshold slider (0.05–0.95).
- Gradio outputs: annotated image (people red/blue by direction, wheeled crossers orange/purple with a thicker box) and the JSON reply.
- `demo.launch(share=True)`. Close it with `demo.close()`.
- A log line per request: time, status, counts, latency in ms.

**Tests** (`tests/test_service.py`, pytest):
- nothing sent, a text file, an empty file, a huge image, an empty scene (expects 200 with zeros), a normal image
- reply schema: every key is present

**Timing** (`scripts/timing.py`):
- Cold start in a fresh process (import, load the weights, answer once), with a breakdown.
- Warm: the mean of 5 requests.

## 10. Repo structure

```
c389-ca1/
├── README.md               # setup, how to run each step
├── PRD.md                  # this file
├── requirements.txt        # ultralytics==8.4.120, gradio, roboflow, imagehash, pandas, pytest
├── notebooks/              # lecturer notebooks + our adapted copies (outputs cleared)
├── src/
│   ├── config.py           # class names, thresholds, Drive paths
│   ├── data/               # verify_export, dedupe_check, session_split, merge_coco_bicycle, make_review_sheet
│   ├── counting.py
│   ├── train.py            # wraps model.train with the fixed config
│   └── eval/               # metrics, count accuracy, error gallery, threshold sweep
├── app/app.py              # Gradio service
├── scripts/timing.py
├── tests/
├── reports/                # generated metrics/plots (no raw photos)
└── .gitignore              # datasets/, runs/, *.pt, images, .env
```

- Everything must run on **Colab T4**, with `PROJECT_ROOT=/content/drive/MyDrive/C389`.
- Scripts must also run locally, with `PROJECT_ROOT` taken from an environment variable.

## 11. Milestones

| By | Deliverable |
|---|---|
| Now | Repo skeleton, `config.py`, `verify_export.py`, `dedupe_check.py`. Wipe and retrain the baseline on clean v2 |
| Lesson 4 | Gradio service with tests and timing (baseline 2-class) |
| Lesson 5 | Labelling rules agreed. New photos shot. Session split. `test_v1` frozen |
| Lesson 6 | `twist_4cls` trained. Service shows the person and wheeled counts |
| Lesson 7 | Full evaluation, error gallery, threshold sweep. Backup demo screen recording |
| Lesson 8 | Live demo (baseline first, then twist). Report submitted |

## 12. Acceptance criteria

- [ ] `verify_export.py` passes, and `dedupe_check.py` finds 0 cross-split near-duplicates
- [ ] `wheeled_left` + `wheeled_right` have at least 150 training boxes, within a 60/40 split
- [ ] Per-class test metrics are reported for all 4 classes on `test_v1`
- [ ] `person_left`/`person_right` mAP50 on `test_v1` doesn't drop more than 0.03 after adding the wheeled classes, or the drop is explained
- [ ] Count MAE per class is produced from `test_counts.csv`
- [ ] Service: all failure tests pass, an empty scene returns 200 with zeros, every count key is present
- [ ] Cold and warm timings are recorded
- [ ] No API key, raw photos or weights in git history

## 13. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Too few wheeled boxes | High | Shoot riders first, both directions. Outside data (≤30%, train only) with rider and vehicle in one box |
| Data leakage from burst shots | High | Session split + `dedupe_check.py` in CI |
| Unlabelled people in old data | Medium | Add them with the pre-labeller before training `twist_4cls` |
| Pedestrian next to a parked bike taken as wheeled | Medium | Review every converted image. Report it in the error gallery |
| Share link dies during the demo | Medium | Relaunch script ready. Backup screen recording |
| Colab GPU unavailable | Low | Pair up. Keep `best.pt` on Drive |

## 14. Open decisions (team must confirm)
1. ~~Someone pushing a bike~~: decided 3 Nov, walking or carrying a bike is `wheeled_*` (§7.2). Tell the lecturer.
2. People facing the camera or facing away: a third class `person_other`, or drop them?
3. Final confidence threshold, with its reason.
