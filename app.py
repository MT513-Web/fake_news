"""Fake News Detection — Streamlit app (FR-06, FR-10).

Uses the real trained model exported by train_model.py:
  model.json — TF-IDF (5,000 words) + Linear SVM, best model by F1 (0.9903)
  metrics.json — dataset stats, all three models, confusion matrix, robustness

Run locally:
  pip install -r requirements.txt
  streamlit run app.py

Deploy on Streamlit Community Cloud:
  1. Put this folder in a GitHub repo (app.py, model.json, metrics.json,
     requirements.txt, visualizations/).
  2. Go to share.streamlit.io -> New app -> pick the repo, branch and app.py.
  3. Deploy. No secrets or API keys are needed.
"""
import json
import math
import os
import re
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    with open(BASE_DIR / "model.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_metrics():
    path = BASE_DIR / "metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


MODEL = load_model()
METRICS = load_metrics()
VOCAB = MODEL["vocabulary"]
IDF = MODEL["idf"]
COEF = MODEL["coefficients"]
INTERCEPT = MODEL["intercept"]
STOP_WORDS = set(MODEL["stop_words"])
VOCAB_INDEX = {w: i for i, w in enumerate(VOCAB)}


def clean_text(s):
    """Same cleaning as train_model.py (FR-02)."""
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -500 else 0.0


def predict(headline, text):
    """TF-IDF + linear SVM, exactly as in train_model.py's export test.

    Returns (label, probability_fake, score, top_words) or None for empty input.
    Top words are the tokens with the largest absolute contribution to the score.
    """
    combined = (clean_text(headline) + " " + clean_text(text)).strip()
    if not combined:
        return None
    tokens = re.findall(r"\b\w\w+\b", combined)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    counts = {}
    for t in tokens:
        if t in VOCAB_INDEX:
            counts[VOCAB_INDEX[t]] = counts.get(VOCAB_INDEX[t], 0) + 1
    if not counts:
        return ("Potentially Genuine", 0.5, 0.0, [], "No known words found — the model has nothing to judge on.")
    vec = {i: c * IDF[i] for i, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    score = float(INTERCEPT)
    contributions = {}
    for i, v in vec.items():
        contrib = (v / norm) * float(COEF[i])
        score += contrib
        contributions[VOCAB[i]] = round(contrib, 4)
    prob = sigmoid(score)
    label = "Potentially Fake" if prob >= 0.5 else "Potentially Genuine"
    top = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    warning = None
    if len(tokens) < 20:
        warning = "Short text — a headline alone is much less reliable than a full article."
    return (label, prob, score, top, warning)


# --------------------------------------------------------------------- UI
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")
st.title("📰 Fake News Detection System")
st.write(
    "Paste an English news headline and article. The trained model (TF-IDF + Support Vector Machine, "
    "best of three models by F1-score) predicts **Potentially Fake** or **Potentially Genuine** "
    "based on learned textual patterns — this is a prediction, not a verification of the article's facts. "
    "Always check a trusted source."
)

if METRICS:
    best = METRICS["best_model_by_f1"]
    st.info(
        f"Model: {best['model']} — Test accuracy **{best['accuracy']:.2%}**, "
        f"F1-score **{best['f1']:.4f}** on 7,766 test articles. "
        f"Trained on the Kaggle Fake and Real News Dataset (44,898 articles, 38,826 after cleaning)."
    )

EXAMPLES = {
    "genuine": {
        "headline": "As U.S. budget fight looms, Republicans flip their fiscal script",
        "text": (
            "WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, "
            "who voted this month for a huge expansion of the national debt to pay for tax cuts, called "
            "himself a fiscal conservative on Sunday and urged budget restraint in 2018."
        ),
    },
    "fake": {
        "headline": "Donald Trump Sends Out Embarrassing New Year's Eve Message; This is Disturbing",
        "text": (
            "Donald Trump just could not wish all Americans a Happy New Year and leave it at that. "
            "Instead, he had to give a shout out to his enemies, haters and the very dishonest fake news media."
        ),
    },
}


def load_example(kind):
    st.session_state["headline"] = EXAMPLES[kind]["headline"]
    st.session_state["text"] = EXAMPLES[kind]["text"]


ex1, ex2 = st.columns(2)
with ex1:
    st.button("Load example: genuine-style article", on_click=load_example, args=("genuine",))
with ex2:
    st.button("Load example: fake-style article", on_click=load_example, args=("fake",))

headline = st.text_input("Headline (optional)", key="headline", placeholder="Paste the news headline here")
text = st.text_area(
    "Article text",
    key="text",
    height=180,
    placeholder="Paste the full news article here...",
)

if st.button("🔍 Check this news", type="primary"):
    if not (headline or "").strip() and not (text or "").strip():
        st.warning("⚠️ Please paste a headline or article text first — there is nothing to classify yet.")
    else:
        result = predict(headline, text)
        if result is None:
            st.warning("⚠️ Please paste a headline or article text first — there is nothing to classify yet.")
        else:
            label, prob, score, top_words, warning = result
            if label == "Potentially Fake":
                st.error(f"### 🚨 {label}")
            else:
                st.success(f"### ✅ {label}")
            st.metric("Probability it is fake", f"{prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(f"Model score: {score:.2f} (probability shown is sigmoid(score), for display only)")
            if warning:
                st.warning(f"⚠️ {warning}")
            if top_words:
                st.subheader("Words that pushed the decision most")
                st.write(
                    "Positive values push toward **Fake**, negative values toward **Genuine**."
                )
                st.table([{"word": w, "contribution": c} for w, c in top_words])
            st.caption(
                "This prediction is based on learned textual patterns only. "
                "It does not independently verify the article's factual accuracy."
            )

st.divider()
st.header("📊 Model report (FR-05, FR-07, FR-09)")

if METRICS:
    rows = METRICS["comparison_table"]
    st.subheader("Model comparison — same test set")
    st.table([
        {
            "Model": r["model"],
            "Accuracy": f"{r['accuracy']:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall": f"{r['recall']:.4f}",
            "F1-score": f"{r['f1']:.4f}",
        }
        for r in rows
    ])
    st.write(f"**Best model by F1-score: {METRICS['best_model_by_f1']['model']}**")

    viz = BASE_DIR / "visualizations"
    for img, caption in [
        ("distribution.png", "Fake vs genuine articles (after cleaning)"),
        ("model_comparison.png", "Model comparison on the test set"),
        ("confusion_matrix_best.png", "Confusion matrix — best model (test set)"),
        ("accuracy_by_subject.png", "Accuracy by subject — best model"),
        ("robustness.png", "Robustness — trivial changes flip rate"),
    ]:
        path = viz / img
        if path.exists():
            st.image(str(path), caption=caption)

    st.subheader("Robustness (FR-08)")
    rob = METRICS["robustness_FR08"]
    st.write(
        f"On 200 test articles, changing capitalisation flipped **{rob['uppercase_flip_rate']:.1%}** of predictions, "
        f"extra spaces **{rob['extra_spaces_flip_rate']:.1%}**, punctuation changes **{rob['punctuation_change_flip_rate']:.1%}**."
    )

    st.subheader("Error analysis (FR-07)")
    err = METRICS["error_analysis_FR07"]
    st.write(
        f"The best model made **{err['total_errors_best_model']} errors out of {err['errors_out_of_test']}** "
        f"test articles ({err['error_rate']:.2%} error rate). {err['note']}"
    )
    for ex in err["examples"]:
        st.write(f"- *{ex['title']}* — true: {ex['true_label']}, predicted: {ex['predicted_label']}")

    st.subheader("Limits")
    for limit in METRICS["limits"]:
        st.write(f"- {limit}")
else:
    st.warning("metrics.json not found — the report section is unavailable, but classification still works.")
