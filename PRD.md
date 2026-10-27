# PRD: People Counting Assistant + Twist 2 (Cyclists)

C389 AI Solution Design & Implementation · CA1 · Team of 6
Deadline: **Lesson 8** (live demo + CA1 report, same day)

---

## 1. Context for Claude Code

- This is a polytechnic coursework project.
- The lecturer supplies Colab notebooks: `L3_training.ipynb`, `L4_deployment.ipynb`, `L5A_new_class.ipynb` and `L5B_tracking.ipynb`.
- We extend them rather than replace them. Keep their function names and the Drive folder layout so they still work.
- Individual worksheets, reflection journals and the **CA1 report** are written by students, **not by AI** (assessment brief §6: AI-written work is marked in the F band). This PRD covers code, and the evidence the code produces for the report.
- The report follows the lecturer's CA1 report template. Each section needs evidence from our own system, which section 8 and section 9 produce. Where each piece of evidence goes is listed in section 15.

## 2. Problem

- **Base task:** count how many people at a pedestrian crossing are moving **left** vs **right**, from a **single still image** taken by a fixed camera.
- **Twist 2 (bicycles only, confirmed by the teacher):** detect bicycles, and classify every crosser as **pedestrian** or **cyclist**, with direction.

**Scope decision (record in report §2.3):** the brief's T2 says "bicycles, skateboards or scooters" and classifies crossers as "pedestrian or wheeled". The teacher confirmed that our team does **bicycles only**. In our system **cyclist** means "wheeled" in the brief's sense, limited to bicycles. Skateboards and scooters were cut. The reason: we don't have enough data to label three extra vehicle types by Lesson 8. A rider on a scooter or skateboard is counted as a **pedestrian**, and this is listed as a known limit (report §3.4).

**Users:** campus or traffic planners, who use the counts to judge flow imbalance and how many cyclists use the crossing.

## 3. Scope

**In scope**
- A YOLO26 detector with 3 classes: `person_left (0)`, `person_right (1)`, `bicycle (2)`
- Per-image counting, plus logic that links each person to a bike
- Evaluation: baseline vs twist model on the **same frozen test set**
- A Gradio web service with a structured JSON reply
- Tests for bad inputs, and cold/warm timing

**Out of scope**
- Identifying people (faces, re-identification)
- Tracking across frames, video streaming, counting across a sequence (a count is **per image**)
- Scooters, skateboards and other wheeled devices (scope decision, see section 2)
- Alarms, alerts, and connections to other systems

## 4. Hard rules (never violate)

| Rule | Why |
|---|---|
| `fliplr=0.0` everywhere, and no flip augmentation in Roboflow | A flipped image reverses direction but keeps the old label |
| Class order: `person_left`=0, `person_right`=1, `bicycle`=2 | Class IDs are positional, and the notebooks check the first two |
| Pin `ultralytics==8.4.120` | Results must be comparable across teams |
| Left/right = direction across the **image**, not the person's own left/right | Labelling convention |
| Never put the Roboflow API key in code or notebooks. Use Colab Secrets `ROBOFLOW_API_KEY` or `getpass` | Notebooks go to the team repo |
| Use the test set once, for the final comparison only. No tuning against it | Keeps the test result honest |
| Images of real people never leave the module drive or Roboflow, and are never committed to git | Privacy commitment from Lesson 1 |
| Notebooks are committed with **all outputs cleared**. `scripts/check_notebooks.py` must pass before every push | Notebook outputs can embed photos of people, and the brief requires notebooks in the repo |
| At the end of the module, delete every photo, dataset, run and Roboflow project (see section 11) | Assessment brief §6, data and privacy |

## 5. Data

### 5.1 Current state
- **v2 cleaned dataset:** 191 images (132 train / 39 valid / 20 test), 536 person boxes (284 left / 252 right), 0 bicycle boxes.
- Images are stretched to 512×512. About 160 images have unlabelled people (listed in `REVIEW_LIST.csv`).
- **Known leak:** the old L3 run trained on leftover files in `MyDrive/C389/runs/L3/dataset`, so 577 images were counted instead of 191. That folder must be wiped before every download.

### 5.2 New data being collected
- About **120–150 pedestrian photos** from 3–4 new locations: groups, overlapping people, far-away people, varied light.
- About **100 cyclist photos**: 150–200+ bicycle boxes, about 50/50 left/right, mostly mixed with walkers.
- **Optional:** COCO `bicycle` images, as extra training data only, never in test. Every person in them must be relabelled left/right.

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
| Runs | `baseline_2cls` (person_left, person_right) and `twist_3cls` (+ bicycle) |

Outputs go to `MyDrive/C389/runs/<run>/`. The best weights are copied to `MyDrive/C389/artifacts/<run>/best.pt`, together with a `metrics.json`.

## 7. Counting and linking people to bikes (`src/counting.py`)

### 7.1 Inputs
Detections as `{class_name, confidence, box:[x1,y1,x2,y2]}`, the same shape as `to_detections()` in L3. Apply the confidence threshold first. The default is 0.40, and it must be justified with evidence.

### 7.2 Linking rule (cyclist vs pedestrian)
For each person P and bicycle B:
- `lower` = the bottom 60% of P's box
- `overlap = area(intersection(lower, B)) / area(B)`
- `x_ok` = the horizontal centre of P lies within B's x-range, expanded by 10% of B's width
- B is a candidate for P if `overlap ≥ 0.25` **and** `x_ok`

Then match greedily, highest overlap first. **Each bike links to at most one person, and each person to at most one bike.**
- A linked person is a **cyclist**. An unlinked person is a **pedestrian**.
- Direction comes from the person's class.
- A bike with no linked person is an **unridden bike**. It's counted but makes nobody a cyclist.

All thresholds are constants at the top of the file. A small grid search on the **validation** set picks them, never the test set.

**Open decision:** someone pushing a bike. Default is **pedestrian**, which the rule above already gives because the bike sits beside the person, not under them. The team must confirm this.

### 7.3 Output
```json
{
  "status": 200,
  "message": "ok",
  "threshold": 0.40,
  "counts": {
    "pedestrian_left": 2, "pedestrian_right": 1,
    "cyclist_left": 0, "cyclist_right": 1,
    "bicycle_total": 2, "bicycle_unridden": 1,
    "person_left": 2, "person_right": 2
  },
  "detections": [
    {"class_name": "person_right", "confidence": 0.91, "box": [..], "role": "cyclist", "linked_bicycle": 3},
    {"class_name": "bicycle", "confidence": 0.88, "box": [..], "id": 3}
  ]
}
```
Every count key is always present, even at 0. An empty scene returns `200` with all zeros, not an error.

## 8. Evaluation (`src/eval/`)

1. **Detector metrics** on the frozen `test_v1`, at the same threshold for every model:
   - precision, recall, mAP50 and mAP50-95 **per class**
   - confusion matrix
2. **Comparison table:** old baseline (if reproducible) vs new `baseline_2cls` vs `twist_3cls`. Show that `person_left`/`person_right` did **not** get worse after adding `bicycle`.
3. **Count accuracy:**
   - The team hand-counts each test image into `test_counts.csv` (`image, pedestrian_left, pedestrian_right, cyclist_left, cyclist_right`).
   - Report the mean absolute error per count and the % of images with an exact count.
4. **Pedestrian vs cyclist classification:** a confusion table over matched ground-truth people.
5. **Error gallery:** save the worst 10 images with predictions and ground truth drawn on, and write `reports/error_gallery.csv` (`image, failure_type, source, note`).
   - `failure_type` is one of: missed person, false person, wrong direction, missed bike, false bike, wrong role.
   - `source` is **detection** or **logic**, as report §3.3 requires. Use this rule:

   | failure_type | source |
   |---|---|
   | missed person, false person, wrong direction, missed bike, false bike | detection |
   | wrong role, where the person and the bike were both detected correctly | **logic** (the linking rule decided wrongly) |
   | wrong role, caused by a missed or false bike | detection |

6. **Threshold sweep:** counts and F1 at 0.15, 0.25, 0.40, 0.60 and 0.80, on validation.
7. **Logic-only check:** run `counting.py` on the **ground-truth boxes** of each test image instead of the model's predictions, then compare with `test_counts.csv`.
   - Any error that remains here comes from the linking rule, not the detector.
   - Report two count MAEs side by side: rule on ground-truth boxes, and rule on predictions. The gap between them is the detector's share of the error.

Outputs: `reports/metrics_*.json`, `reports/*.png`, `reports/error_gallery/`, `reports/error_gallery.csv`.

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
- Gradio outputs: annotated image (pedestrians in one colour, cyclists in another, bike–rider links drawn as lines) and the JSON reply.
- `demo.launch(share=True)`. Close it with `demo.close()`.
- A log line per request: time, status, counts, latency in ms.

**Tests** (`tests/test_service.py`, pytest):
- nothing sent, a text file, an empty file, a huge image, an empty scene (expects 200 with zeros), a normal image
- reply schema: every key is present
- The report §4.2 table has three rows. Fill each one from these tests and keep a screenshot of the reply for each:
  - **A valid image:** a normal crossing photo with at least one cyclist.
  - **A failure case:** an image it gets wrong, taken from the error gallery.
  - **An input it should not handle:** a text file or an oversized image (returns 400), or a scooter rider (a known limit, counted as a pedestrian).

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
├── scripts/
│   ├── timing.py
│   └── check_notebooks.py  # fails if any notebook has outputs or a hard-coded API key
├── tests/
├── reports/                # generated metrics/plots (no raw photos)
├── docs/                   # team docs
└── .gitignore              # datasets/, runs/, *.pt, images, .env
```

- **Notebooks:** commit the lecturer notebooks and our adapted copies (the brief requires them in the repo).
  - Clear every output first (Colab: *Edit → Clear all outputs*), because outputs can embed photos of people.
  - `scripts/check_notebooks.py` exits 1 if any `.ipynb` has outputs, or has a string that looks like a Roboflow key passed to `Roboflow(api_key=...)`.

- Everything must run on **Colab T4**, with `PROJECT_ROOT=/content/drive/MyDrive/C389`.
- Scripts must also run locally, with `PROJECT_ROOT` taken from an environment variable.

## 11. Milestones

| By | Deliverable |
|---|---|
| Now | Repo skeleton, `config.py`, `verify_export.py`, `dedupe_check.py`. Wipe and retrain the baseline on clean v2 |
| Lesson 4 | Gradio service with tests and timing (baseline 2-class). L3/L4 notebooks pushed with outputs cleared, plus `check_notebooks.py` |
| Lesson 5 | Labelling rules agreed. Section 14's open decisions settled and written down **before** capture (brief §5). New photos shot. Session split. `test_v1` frozen |
| Lesson 6 | `twist_3cls` trained. `counting.py` with linking. Service shows the new counts |
| Lesson 7 | Full evaluation, error gallery with detection/logic tags, logic-only check, threshold sweep. Backup demo screen recording. Demo rehearsed, with every member able to explain every part |
| Lesson 8 | Live demo (baseline first, then twist). Report submitted on SA3.0 by 23:59 with the repo link on page 1, and the lecturer added as a repo collaborator so the link opens |
| End of module | Delete every photo, dataset, run and weight file on Drive and local devices, and the Roboflow project (brief §6). Confirm in the team chat |

## 12. Acceptance criteria

- [ ] `verify_export.py` passes, and `dedupe_check.py` finds 0 cross-split near-duplicates
- [ ] `bicycle` has at least 150 training boxes, and left/right are within a 60/40 split
- [ ] Per-class test metrics are reported for all 3 classes on `test_v1`
- [ ] `person_left`/`person_right` mAP50 on `test_v1` doesn't drop more than 0.03 after adding `bicycle`, or the drop is explained
- [ ] Count MAE and the pedestrian-vs-cyclist confusion table are produced from `test_counts.csv`
- [ ] Service: all failure tests pass, an empty scene returns 200 with zeros, every count key is present
- [ ] Cold and warm timings are recorded
- [ ] Every error gallery entry is tagged detection or logic, and the logic-only count MAE is reported
- [ ] Notebooks are in the repo, and `check_notebooks.py` passes
- [ ] No API key, raw photos or weights in git history
- [ ] The lecturer can open the repo link

## 13. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Too few bicycle boxes | High | Shoot cyclists first. Add COCO bicycles to training only |
| Data leakage from burst shots | High | Session split, plus `dedupe_check.py` before every training run |
| Unlabelled people in old data | Medium | Fix them from `REVIEW_LIST.csv` before training `twist_3cls` |
| Linking errors when a walker overlaps a bike | Medium | Lower-body plus x-centre rule. Shoot this case on purpose. Report it in the error gallery |
| Share link dies during the demo | Medium | Relaunch script ready. Backup screen recording |
| Colab GPU unavailable | Low | Pair up. Keep `best.pt` on Drive |

## 14. Open decisions (team must confirm)
1. Someone pushing a bike: pedestrian (default) or cyclist?
2. People facing the camera or facing away: a third class `person_other`, or drop them?
3. Final confidence threshold, with its reason.

## 15. Report evidence map

The team writes the report. The code's job is to produce the evidence each section asks for.

| Report section (weight) | Evidence | Produced by |
|---|---|---|
| §1.1–1.2 Twist and term definitions (10%) | Section 2 scope decision, section 7.2 linking rule, section 14 decisions | This PRD, plus team decisions |
| §1.3 Technique options | New class vs tracking: tracking is out of scope because counts are per image | Team |
| §2.1 Pipeline diagram (15%) | capture → Roboflow → `verify_export` / `dedupe_check` → train → detect → `counting.py` link → JSON + annotated image, with the twist parts marked | Team draws it |
| §2.2 Data added | Image and box counts per class and split | `verify_export.py --json`, `split_log.csv` |
| §2.3 Problems and decisions | The L3 dataset leak (section 5.1), the bicycles-only scope cut (section 2), Stretch → Fit re-export (section 6) | This PRD, git history |
| §3.1 Test description (15%) | `test_v1` frozen before training, split by session, hand-counted | `split_log.csv`, `test_counts.csv` |
| §3.2 Results | Per-class metrics, comparison table, count MAE, pedestrian-vs-cyclist confusion table | Section 8, items 1–4 and 7 |
| §3.3 Failures | Worst images, each tagged detection or logic | `reports/error_gallery/`, `error_gallery.csv` |
| §3.4 Limits | Test set size, locations and lighting not covered, scooters counted as pedestrians | `verify_export` counts, section 2 |
| §4.1 Service (10%) | Reply format, screenshot of a real answer | Section 7.3, `app/app.py` |
| §4.2 Tests | Valid / failure / should-not-handle rows | `tests/test_service.py`, `scripts/timing.py` |
| §5 Contributions | Who built what | Git history: everyone commits under their own account |
