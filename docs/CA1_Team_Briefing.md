# CA1 Team Briefing: People Counting Assistant + Twist 2 (Bicycles)

## 1. What CA1 is

- **Worth 15%** of C389.
- **Graded on:** a **live 15-minute demo** (demo plus Q&A, and **every one of us** must be able to answer questions) and the **CA1 report** on PoliteMall.
- **Both happen in Lesson 8**, and the report is **due the same day**.
- **What we build:**
  1. **Baseline (Lessons 1–4):** a YOLO model that looks at one photo and counts `person_left` and `person_right`.
  2. **Our twist (Lessons 5–8):** Twist 2, Micromobility crossers.

## 2. Our twist: Twist 2

> Detect people crossing with **bicycles**, and classify each crosser as **pedestrian or cyclist**.
>
> **Scope confirmed by our teacher: bicycles only.** No scooters or skateboards.

**How it works**

- We **add new classes** and retrain. We **don't need tracking**, because everything happens within one image.
- We add **one** new class, **after** the existing two:
  `person_left (0), person_right (1), bicycle (2)`
- **Linking a person to a bike:** if a person's box overlaps a bicycle box (the bike sits under their lower body), that person is a **cyclist**. Otherwise they're a **pedestrian**.
- **Output per image:**
  - `pedestrian_left`, `pedestrian_right`
  - `cyclist_left`, `cyclist_right`
  - total bicycles

## 3. Lesson-by-lesson recap

| Lesson | What happens | Status |
|---|---|---|
| 1 | Define the problem and scope, left/right rules, pick a twist | Done |
| 2 | Take photos and label them in Roboflow (2 classes, **no flip**) | Done (needs more data) |
| 3 | Train YOLO26n, read per-class metrics, pick a threshold | Trained, **needs a rerun** (see section 4) |
| 4 | Deployment: Gradio page, test bad inputs, cold/warm timing, cloud vs edge | Next |
| 5 | Lab A: add a new class. Lab B: tracking. Make the CA1 plan | Plan due |
| 6 | Clinic: build the twist. Compare the old and new models on the same test set | Bring work in progress |
| 7 | Self-paced e-learning, **no class support** | Work on our own |
| 8 | **Live demo + CA1 report due** | Deadline |

## 4. Where we are now

**Our dataset (v2 cleaned):**
- 191 images and 536 people, about balanced (284 left / 252 right).
- **Only 4 bicycle labels so far** (in the old v1 export). This is our biggest gap.
- Not much variety: 44% of the images are the same crossing with 1–2 people, from only about 3–4 locations.
- About 160 images still have unlabelled people in them (listed in `REVIEW_LIST.csv`).

**Our L3 model:**
- The training settings and curves are fine. Left and right were learned about equally well, and the two directions are rarely mixed up.
- **But the run mixed in old data.** Old Roboflow downloads were still in `MyDrive/C389/runs/L3/dataset`, so it trained on 577 images instead of 191. Some of the same photos ended up in both train and test, so **the test mAP of 0.919 is inflated.**
- **Fix:** delete `MyDrive/C389/runs/L3/dataset`, rerun the notebook, and check that the image counts print as **132 / 39 / 20**. Use the new numbers in the sheets.

## 5. What we need to do

### A. Data (most urgent)
1. **Agree on labelling rules first and write them down:**
   - Someone **pushing** a bike: pedestrian or cyclist?
   - A parked bike with no rider: label it, but it must not make anyone a cyclist.
   - People facing the camera or facing away: give them a third class, or drop those images?
   - People cut off at the frame edge: label them if at least ⅓ of the body is visible.
2. **Shoot bicycle photos.** Aim for **150–200+ bicycle boxes** (about 100 photos), half riding left and half riding right. Mix cyclists and walkers in the same frames.
   - Optional extra: COCO `bicycle` images as additional training data only. Every person in them must get a left/right label, and none of them go in the test set.
3. **Shoot the tricky cases on purpose:** pushing a bike, a parked bike, a rider overlapping a walker, and someone cut off at the frame edge.
4. **Fix the unlabelled people** in `REVIEW_LIST.csv`.
5. **Export:** YOLOv8 or YOLO26 format, **Fit (black edges) 640** (not Stretch), **no flip**.

### B. Model
6. **Retrain the baseline** on clean data. These are the numbers we report.
7. **Retrain with the `bicycle` class.** Check:
   - per-class mAP for `bicycle`
   - that `person_left` and `person_right` didn't get worse
8. **Compare the baseline and the twist model** on the **same test set** with the **same threshold**.

### C. Counting logic and deployment
9. **Write the person-to-bike matching code** that decides pedestrian vs cyclist.
10. **Update the L4 Gradio page** so it shows the new counts. Every class must appear in the reply, even at 0.
11. **Test the service:** bad inputs, an empty scene, and cold vs warm timing.

### D. Evaluation (this is what gets marked)
12. **Hand-count a set of test images** and build a pedestrian-vs-cyclist confusion table.
13. **Error analysis:** what fails, why, and what we would fix. Use real examples.
14. **List the limitations honestly.**

### E. Demo and report
15. **Demo order:** baseline first, then the twist, live.
16. **Prepare:**
   - Run the demo once before our slot.
   - Load the weights in advance.
   - **Record a backup screen video.**
   - Agree who speaks and who drives.
   - Keep slides minimal.
17. **Report:** design decisions, data, metrics, error analysis, limitations, privacy.
18. **Push everything to the team repo** (notebooks, but **never the Roboflow API key**).

## 6. Proposed task split (open to change)

| Member | Main task |
|---|---|
| Ethan | Shoot and label bicycles (locations 1–3) |
| Rohith | Shoot and label bicycles (locations 4–6) |
| Yusuf | Shoot the tricky cases, add COCO bicycle images (optional) |
| Chofon | Fix unlabelled people (REVIEW_LIST) and check label consistency |
| CY | Gradio deployment and testing (L4), demo backup recording |
| Thuta | Retrain models, matching code, evaluation, report compilation |

Everyone helps with shooting, and **everyone must understand the whole pipeline** for Q&A.

## 7. How to shoot photos (summary, see the 6 guide images)

- **Phone on a tripod or resting on something**, on the pavement, **6–12 m back**, at head height (~1.5 m), **landscape**, 1× lens. **Never stand on the road.**
- **Side-on to the crossing** so people move **across** the frame. 45° is OK for some shots. **No head-on shots.**
- **No burst mode.** One shot per crossing, at least 1 second apart. Max about 15–20 photos per setup, then change something.
- **Variety:**
  - **5–6+ locations**
  - morning, noon and evening
  - sunny, cloudy and wet weather
  - different people and clothes
  - near, mid and far distances
  - solo shots and groups
- **Balance left and right** for cyclists and walkers at every location.
- Avoid motion blur (good light, tap to focus). No zoom, filters or portrait mode.
- **Privacy:** teammates or volunteers only, no close-ups of strangers' faces, keep photos only on our own drives, and delete everything after the module.

## 8. Rules that must not be broken

- **Never use horizontal flip** (in Roboflow or in training: `fliplr=0.0`). It reverses the person's direction but keeps the old label.
- Class order: `person_left` (0), `person_right` (1), and new classes **appended** after them.
- Leave the test set alone. Use it once, for the final comparison.
- Never commit the Roboflow API key to the repo.
- Individual sheets must be written **in our own words**, not copied from AI.

## 9. Next steps (this week)

1. Agree the labelling rules in our group chat. **Owner: everyone.**
2. Book bicycle shooting sessions: who, where, when, and whose bikes. **Owner: Ethan + Rohith.**
3. Rerun L3 on clean data. **Owner: Thuta.**
4. Run L4 deployment. **Owner: CY + everyone.**
5. Bring the CA1 plan (tasks, owners, dates, top 3 risks) to Lesson 5/6.
