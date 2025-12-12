import re
import statistics
from typing import Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"
AI_LABEL_TOKENS = ("ai", "gpt", "fake", "machine", "generated", "synthetic", "bot")
HUMAN_LABEL_TOKENS = ("human", "real", "organic", "authentic")

# Basic stopword list for heuristic scoring.
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


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@st.cache_resource(show_spinner=False)
def load_detector():
    """Load the pretrained detector; cache for reuse. Falls back gracefully if unavailable."""
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - dependency guard
        return None, f"transformers unavailable: {exc}"

    try:
        detector = pipeline(
            task="text-classification",
            model=MODEL_NAME,
            device=-1,
            truncation=True,
        )
        return detector, None
    except Exception as exc:  # pragma: no cover - download/runtime guard
        return None, f"model load failed: {exc}"


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def token_weights(text: str) -> pd.DataFrame:
    tokens = tokenize(text)
    total = len(tokens)
    data = []
    if total == 0:
        return pd.DataFrame(columns=["token", "count", "weight"])

    counts: Dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1

    for tok, count in counts.items():
        data.append({"token": tok, "count": count, "weight": count / total})

    df = pd.DataFrame(data).sort_values(by=["weight", "token"], ascending=[False, True])
    return df


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

    # AI-like texts tend to have balanced vocabulary diversity, smoother sentence lengths,
    # and moderate stopword ratios. Bias the heuristic toward those patterns.
    unique_balance = 1.0 - clamp(abs(features["unique_ratio"] - 0.62) / 0.62, 0.0, 1.0)
    smoothness_score = clamp((12.0 - features["sentence_len_std"]) / 12.0, 0.0, 1.0)
    medium_length_score = clamp(1.0 - abs(features["avg_sentence_len"] - 18.0) / 18.0, 0.0, 1.0)
    stopword_balance = 1.0 - clamp(abs(features["stop_ratio"] - 0.46) / 0.46, 0.0, 1.0)

    weighted_score = (
        0.3 * unique_balance
        + 0.25 * smoothness_score
        + 0.25 * stopword_balance
        + 0.2 * medium_length_score
    )
    return clamp(weighted_score, 0.0, 1.0), features


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


def analyze(text: str, prefer_model: bool = True) -> Dict:
    detector, load_error = load_detector()
    meta: Dict[str, object] = {"detector_loaded": detector is not None, "detector_error": load_error}

    if prefer_model and detector is not None:
        try:
            ai_prob, raw_scores = ai_score_from_transformer(text, detector)
            meta["raw_scores"] = raw_scores
            meta["features"] = heuristic_features(text)
            return {
                "ai_prob": ai_prob,
                "human_prob": 1.0 - ai_prob,
                "method": "transformer",
                "meta": meta,
            }
        except Exception as exc:  # pragma: no cover - runtime guard
            meta["detector_error"] = f"inference failed: {exc}"

    ai_prob, features = heuristic_score(text)
    meta["features"] = features
    return {
        "ai_prob": ai_prob,
        "human_prob": 1.0 - ai_prob,
        "method": "heuristic",
        "meta": meta,
    }


st.set_page_config(page_title="AI / Human 文章偵測器", page_icon="🛰️", layout="wide")
st.title("AI / Human 文章偵測器")
st.caption("輸入文本後按下分析，估計 AI vs Human 的比例。預設使用預訓練模型，無法載入時改用啟發式。")

with st.sidebar:
    st.subheader("設定")
    sample_choice = st.selectbox(
        "範例文本",
        ["-- 不使用範例 --", "AI-like sample", "Human-like sample"],
        index=0,
    )

default_text = (
    "Artificial intelligence systems can produce fluent and well-structured prose. "
    "This detector estimates how likely the passage was generated by a language model "
    "compared with a human author."
)

if "input_text" not in st.session_state:
    st.session_state["input_text"] = default_text

prev_choice = st.session_state.get("sample_choice", "-- 不使用範例 --")
if sample_choice != prev_choice:
    if sample_choice == "AI-like sample":
        st.session_state["input_text"] = (
            "As an AI language model, I will outline the key points, provide structured explanations, "
            "and maintain a neutral, helpful tone. I do not possess personal opinions or emotions, "
            "but I can generate coherent paragraphs that summarize the requested topic with clear transitions "
            "and balanced phrasing throughout the response."
        )
    elif sample_choice == "Human-like sample":
        st.session_state["input_text"] = (
            "I remember walking past the old bookstore on my way home and pausing to read the hand-written sign "
            "in the window. It wasn't polished, but the crooked letters felt sincere, and that small detail "
            "stuck with me more than the novel I eventually bought."
        )
    else:
        st.session_state["input_text"] = default_text
st.session_state["sample_choice"] = sample_choice

with st.form("detector_form", clear_on_submit=False):
    text = st.text_area(
        "輸入或貼上文本（英文與中英混排皆可）",
        key="input_text",
        height=260,
    )
    submitted = st.form_submit_button("分析 Analyze", type="primary")

if submitted and text.strip():
    with st.spinner("Analyzing..."):
        result = analyze(text.strip(), prefer_model=True)

    # Persist results for interactive controls.
    st.session_state["analysis"] = result
    st.session_state["analysis_text"] = text.strip()
    token_df = token_weights(text)
    st.session_state["token_df"] = token_df
    st.session_state["top_n_slider"] = min(15, max(1, len(token_df))) if len(token_df) else 1
elif submitted:
    st.warning("請輸入文本再進行分析。")

analysis = st.session_state.get("analysis")
token_df = st.session_state.get("token_df", pd.DataFrame())

if analysis:
    ai_pct = analysis["ai_prob"] * 100
    human_pct = analysis["human_prob"] * 100

    mode_label = "Transformer 模型" if analysis["method"] == "transformer" else "啟發式特徵"
    st.caption(f"偵測模式：{mode_label}")

    col1, col2 = st.columns(2)
    col1.metric("AI 機率", f"{ai_pct:.1f}%")
    col2.metric("Human 機率", f"{human_pct:.1f}%")

    st.bar_chart(
        pd.DataFrame({"probability": [ai_pct, human_pct]}, index=["AI", "Human"])
    )

    with st.expander("特徵摘要", expanded=False):
        feats = analysis.get("meta", {}).get("features", {})
        if feats:
            st.write(
                f"Tokens: {feats.get('tokens', 0)}, "
                f"Unique ratio: {feats.get('unique_ratio', 0.0):.3f}, "
                f"Stopword ratio: {feats.get('stop_ratio', 0.0):.3f}"
            )
            st.write(
                f"Avg sentence length: {feats.get('avg_sentence_len', 0.0):.2f}, "
                f"Sentence length std: {feats.get('sentence_len_std', 0.0):.2f}"
            )
        if analysis["method"] == "transformer":
            raw_scores = analysis.get("meta", {}).get("raw_scores", [])
            for row in raw_scores:
                st.write(f"{row['label']}: {row['score']:.3f}")

    st.subheader("Token 權重視覺化")
    if not token_df.empty:
        max_tokens = max(1, min(50, len(token_df)))
        default_n = min(st.session_state.get("top_n_slider", 15), max_tokens)
        top_n = st.slider(
            "顯示前 N 個權重較高的 token",
            min_value=1,
            max_value=max_tokens,
            value=default_n,
            key="top_n_slider",
        )
        top_tokens = token_df.head(top_n)
        chart = (
            alt.Chart(top_tokens)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("weight:Q", title="權重 (頻率佔比)", axis=alt.Axis(format="~%")),
                y=alt.Y("token:N", sort="-x", title="Token"),
                color=alt.value("#4C78A8"),
                tooltip=["token", "count", alt.Tooltip("weight:Q", format=".3f")],
            )
            .properties(height=400)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption("權重 = token 出現次數 / 總 token 數，便於觀察高頻詞對判斷的影響。")
    else:
        st.info("尚無可視覺化的 token（請輸入文本並按分析）。")

    detector_error = analysis.get("meta", {}).get("detector_error")
    if detector_error and analysis["method"] != "transformer":
        st.caption(f"系統訊息：{detector_error}（已改用啟發式計分）")
