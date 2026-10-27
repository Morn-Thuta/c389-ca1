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

## Workflow

- Branch off `main` for each task (`feat/verify-export`, `feat/counting`, ...), then open a PR.
- Milestones and acceptance criteria: PRD §11 and §12.
