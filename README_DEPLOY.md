# Fake News Detection — Streamlit deployment

## What is in this folder

| File | What it is |
|---|---|
| `app.py` | The Streamlit web app (FR-06, FR-10) |
| `model.json` | The real trained model: TF-IDF (5,000 words) + Linear SVM, best of 3 models by F1 (0.9903). Only 146 KB. |
| `metrics.json` | Dataset stats, all 3 models compared, confusion matrix, robustness + error analysis |
| `requirements.txt` | Python packages Streamlit Cloud needs (`streamlit`) |
| `visualizations/` | The training charts shown in the app's report section |
| `../train_model.py` | The full training script (FR-01 to FR-09) — retrain with `python train_model.py` |
| `../README.md` | How the model works, its limits, how to retrain |

No API keys, no secrets, no database needed. The model runs fully offline
from `model.json` — the app never sends your pasted news anywhere.

## How to get it on your computer

1. Download the project zip from Muse and unzip it on your computer.
2. Open a terminal in the `streamlit` folder and run:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Your browser opens at `http://localhost:8501`. Paste a headline + article,
   click **Check this news**.

## How to deploy it on Streamlit Community Cloud (free)

1. Create a free GitHub account (if you don't have one) and create a new
   public repository, e.g. `fake-news-detection`.
2. Upload the contents of this `streamlit` folder to that repo
   (`app.py`, `model.json`, `metrics.json`, `requirements.txt`, `visualizations/`).
   You can do this in the browser: GitHub → Add file → Upload files.
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click **Create app** / **New app**, pick your repository, branch (`main`)
   and main file path (`app.py`), then **Deploy**.
5. After a minute or two you get a public link like
   `https://your-app-name.streamlit.app` — share it with anyone.

Notes:
- The repo must be public for the free plan.
- If you update the model later, re-upload `model.json` and reboot the app
  from the Streamlit Cloud dashboard.
- The app shows "Potentially Fake / Potentially Genuine" — predictions are
  based on learned textual patterns, not a verification of facts.

## Retraining

The model was trained on the Kaggle Fake and Real News Dataset
(https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
To retrain, put `Fake.csv` and `True.csv` in `../data/` and run
`python train_model.py` — it rewrites `model.json`, `metrics.json` and the
charts. Copy the new `model.json` into this folder and redeploy.
