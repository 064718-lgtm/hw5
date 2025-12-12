import math
import re
import statistics
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Basic stopword list to support the heuristic fallback.
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}

AI_LABEL_TOKENS = ("ai", "gpt", "fake", "machine", "generated", "synthetic", "bot")
HUMAN_LABEL_TOKENS = ("human", "real", "organic", "authentic")


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@st.cache_resource(show_spinner=False)
def load_detector():
    """
    Load a lightweight transformer-based detector if transformers are available.
    Falls back to heuristics when the model cannot be loaded (e.g., offline).
    """
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - dependency check
        return None, f"transformers not available: {exc}"

    try:
        classifier = pipeline(
            task="text-classification",
            model="Hello-SimpleAI/chatgpt-detector-roberta",
            device=-1,
        )
        return classifier, None
    except Exception as exc:  # pragma: no cover - model fetch may fail offline
        return None, f"model could not be loaded: {exc}"


def ai_score_from_transformer(text: str, detector) -> Tuple[Optional[float], List[Dict]]:
    outputs = detector(
        text,
        truncation=True,
        max_length=512,
        return_all_scores=True,
    )
    if not outputs:
        return None, []

    scores = outputs[0]
    ai_score = None
    human_score = None

    for row in scores:
        label = row.get("label", "").lower()
        if any(token in label for token in AI_LABEL_TOKENS):
            ai_score = row["score"]
        if any(token in label for token in HUMAN_LABEL_TOKENS):
            human_score = row["score"]

    if ai_score is None and human_score is not None:
        ai_score = 1.0 - human_score

    if ai_score is None and len(scores) == 2:
        ai_score = scores[1]["score"]

    if ai_score is None:
        ai_score = scores[0]["score"]

    return clamp(ai_score), scores


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def heuristic_features(text: str) -> Dict[str, float]:
    tokens = tokenize(text)
    num_tokens = len(tokens)
    unique_ratio = len(set(tokens)) / num_tokens if num_tokens else 0.0
    stopword_hits = sum(1 for token in tokens if token in STOPWORDS)
    stop_ratio = stopword_hits / num_tokens if num_tokens else 0.0

    sentences = [segment.strip() for segment in re.split(r"[.!?]+", text) if segment.strip()]
    sentence_lengths = [len(tokenize(sentence)) for sentence in sentences] or [0]
    avg_len = statistics.mean(sentence_lengths)
    length_std = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0

    return {
        "tokens": num_tokens,
        "unique_ratio": unique_ratio,
        "stop_ratio": stop_ratio,
        "avg_sentence_len": avg_len,
        "sentence_len_std": length_std,
    }


def heuristic_score(text: str) -> Tuple[float, Dict[str, float]]:
    features = heuristic_features(text)
    if features["tokens"] == 0:
        return 0.5, features

    repetition_score = clamp((0.55 - features["unique_ratio"]) / 0.55, 0.0, 1.0)
    smoothness_score = clamp((12.0 - features["sentence_len_std"]) / 12.0, 0.0, 1.0)
    medium_length_score = clamp(1.0 - abs(features["avg_sentence_len"] - 18.0) / 18.0, 0.0, 1.0)
    stopword_balance = 1.0 - clamp(abs(features["stop_ratio"] - 0.48) / 0.48, 0.0, 1.0)

    weighted_score = (
        0.4 * repetition_score
        + 0.25 * smoothness_score
        + 0.2 * medium_length_score
        + 0.15 * stopword_balance
    )
    return clamp(weighted_score, 0.0, 1.0), features


def analyze(text: str, prefer_model: bool = True) -> Dict:
    detector, load_error = load_detector()
    meta: Dict[str, object] = {"detector_loaded": detector is not None, "detector_error": load_error}

    if prefer_model and detector is not None:
        try:
            ai_prob, raw_scores = ai_score_from_transformer(text, detector)
            meta["raw_scores"] = raw_scores
            return {
                "ai_prob": ai_prob,
                "human_prob": 1.0 - ai_prob,
                "method": "transformer",
                "meta": meta,
            }
        except Exception as exc:  # pragma: no cover - runtime safety
            meta["detector_error"] = f"inference failed, fallback to heuristics: {exc}"

    ai_prob, features = heuristic_score(text)
    meta["features"] = features
    return {
        "ai_prob": ai_prob,
        "human_prob": 1.0 - ai_prob,
        "method": "heuristic",
        "meta": meta,
    }


st.set_page_config(page_title="AI / Human Detector", page_icon="🛰️", layout="wide")
st.title("AI / Human 文章偵測器")
st.caption("輸入文本後立即估計 AI vs Human 的比例。預設使用 transformer 模型，若無法下載則改用啟發式特徵。")

with st.sidebar:
    st.subheader("設定")
    prefer_model = st.checkbox("Prefer transformer detector", value=True)
    sample_ai = st.checkbox("Use a short AI-like sample")
    st.markdown(
        """
        - 安裝需求：`pip install -r requirements.txt`
        - 啟動：`streamlit run app.py`
        """
    )

default_text = (
    "Artificial intelligence systems can produce fluent and well-structured prose. "
    "This detector estimates how likely the passage was generated by a language model "
    "compared with a human author."
)

if "input_text" not in st.session_state:
    st.session_state["input_text"] = default_text

if sample_ai:
    st.session_state["input_text"] = (
        "As an AI language model, I can provide guidance, structure, and coherent paragraphs "
        "that align with the requested topic while maintaining a neutral tone throughout the response."
    )

text = st.text_area(
    "輸入或貼上文本（英文與中英混排皆可）",
    value=st.session_state["input_text"],
    height=260,
)

if text.strip():
    with st.spinner("Analyzing..."):
        result = analyze(text.strip(), prefer_model=prefer_model)

    ai_pct = result["ai_prob"] * 100
    human_pct = result["human_prob"] * 100
    col1, col2 = st.columns(2)
    col1.metric("AI 機率", f"{ai_pct:.1f}%")
    col2.metric("Human 機率", f"{human_pct:.1f}%")

    st.bar_chart(
        pd.DataFrame({"probability": [ai_pct, human_pct]}, index=["AI", "Human"])
    )

    if result["method"] == "transformer":
        st.success("使用 Transformer 模型判斷 (Hello-SimpleAI/chatgpt-detector-roberta)")
        with st.expander("模型分數", expanded=False):
            raw_scores = result["meta"].get("raw_scores", [])
            for row in raw_scores:
                st.write(f"{row['label']}: {row['score']:.3f}")
    else:
        st.warning("使用啟發式特徵判斷（未載入模型）")
        with st.expander("特徵摘要", expanded=False):
            feats = result["meta"].get("features", {})
            st.write(
                f"Tokens: {feats.get('tokens', 0)}, "
                f"Unique ratio: {feats.get('unique_ratio', 0.0):.3f}, "
                f"Stopword ratio: {feats.get('stop_ratio', 0.0):.3f}"
            )
            st.write(
                f"Avg sentence length: {feats.get('avg_sentence_len', 0.0):.2f}, "
                f"Sentence length std: {feats.get('sentence_len_std', 0.0):.2f}"
            )

    if result["meta"].get("detector_error"):
        st.info(result["meta"]["detector_error"])
