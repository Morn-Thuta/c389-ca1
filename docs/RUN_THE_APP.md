# Running the People Counting Assistant

The app is a Gradio page. It runs the model on your own computer (or in Colab) and can give you a
temporary public link to share with the team.

You need two things:

| What | Where |
|---|---|
| The code | This repo, branch `feat/wheeled-classes` (or the share zip) |
| The model weights `final_4cls_ig_v4_m.pt` (44 MB) | The team's private Drive folder or the share zip. Never in git. |

## Option A: your laptop (recommended for the demo)

Use Python 3.11 or 3.12. From the folder that holds `app/` and `src/`:

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install ultralytics==8.4.120 gradio
python -m app.app --weights weights\final_4cls_ig_v4_m.pt
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics==8.4.120 gradio
python -m app.app --weights weights/final_4cls_ig_v4_m.pt
```

Open http://127.0.0.1:7860. The first install takes a few minutes (it downloads PyTorch).

**To share a link,** add `--share` to the last command. A `https://….gradio.live` address appears in the
terminal after a few seconds. It works while the command keeps running (keep the laptop awake) and lasts
up to a week. Press Ctrl+C to stop it.

If `--weights` points somewhere else (for example the repo's `../local_data/weights/`), change the path.

## Option B: Google Colab

1. Open `notebooks/CA1_deployment.ipynb` in Colab and pick **Runtime → Change runtime type → T4 GPU**.
2. Add a GitHub token as the Colab secret `GITHUB_TOKEN` (read-only Contents access to `c389-ca1`).
3. Put `final_4cls_ig_v4_m.pt` in your Drive at `MyDrive/C389_demo/`, or upload it when the notebook asks.
4. Run the cells top to bottom. Section 3 prints the public link. Section 6 closes it.

## Using the page

- **Input:** upload a crossing photo (JPEG or PNG, up to 15 MB) and set the confidence threshold.
- **Output:** the photo with its boxes. People: red = moving left, blue = moving right. Wheeled
  (rider + bike or scooter, thick box): orange = left, purple = right.
- **Counts:** `person_left`, `person_right`, `wheeled_left`, `wheeled_right`, the totals, and the JSON reply.
- **Crossing area:** click the crossing's corners on the output photo. From the third corner, only people
  whose feet are inside the green area are counted; the rest turn grey (`outside_roi`). *Undo corner* and
  *Clear area* edit it. Wait a second after uploading before clicking.
- **Refusals:** a non-photo, empty or oversized file is refused with its status (415, 400, 413) and a reason.

## Troubleshooting

| Problem | Fix |
|---|---|
| `weights not found` | Check the `--weights` path. |
| `No module named app` | Run the command from the folder that contains `app/` and `src/`. |
| Port 7860 in use | Another copy is running; close it, or set `GRADIO_SERVER_PORT=7861`. |
| Slow (several seconds a photo) | Normal on a laptop without an NVIDIA GPU. Colab's T4 is faster. |

## Privacy

- Send the link only to the team. Anyone holding it can use the page while it runs.
- Photos people upload are processed in memory and not saved.
- Keep the weights and the share zip in the private Drive folder, never in git or public places, and
  delete them at the end of the module with the rest of the project data.
