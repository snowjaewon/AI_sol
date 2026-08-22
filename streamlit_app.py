"""AI Insight Engine — Streamlit app.

Notebook 02(Part 2)의 분석 로직을 배포 가능한 형태로 재구성한 것.
- 분석 코어(read_and_clean_csv / build_analysis 등)는 Streamlit에 의존하지 않는다.
- UI 레이어는 파일 하단 render_* 함수와 main()에만 있다.
"""

import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ENCODINGS = ("utf-8-sig", "utf-8", "cp949")


# ============================================================
# 한국어 토큰 정규화
# ============================================================
SUFFIXES = sorted(
    [
        "으로는", "에서는", "에게서", "이라는", "라고는", "다라고",
        "에서", "에게", "으로", "이나", "까지", "부터", "라고", "다고",
        "다는", "이라", "와의", "과의", "들이", "들을", "들은", "들의",
        "마다",
        "을", "를", "이", "가", "은", "는", "에", "의", "도", "만",
        "와", "과", "로", "나", "다",
        # 단일 문자 "고"는 넣지 않는다. 어미로 쓰이기도 하지만
        # "채용공고" -> "채용공"처럼 명사를 잘라먹는다.
        # "-다고" / "-라고" 2글자 규칙이 어미 쪽은 이미 처리한다.
    ],
    key=len,
    reverse=True,
)

KOREAN_STOPWORDS = {
    # 도메인 공통어
    "청년", "지역", "광주", "전남", "정보", "경우", "부분", "요즘",
    "실제로", "개인적으로", "있으면",
    # 서술 상투구
    "생각합니다", "좋겠습니다", "어렵다", "어렵습니다", "필요하다", "필요합니다",
    # 설문 템플릿 상투구
    "입장", "점이", "가장", "주변", "많이", "말하는데", "준비하면서",
    "이용해보니", "생활하면서", "지내다", "합니다", "느낍니다", "느끼는",
    "불편합니다", "의견입니다", "개선됐으면",
}


def normalize_token(token: str) -> str:
    """어절 끝의 조사/어미를 반복적으로 떼어낸다. 어간이 2글자 미만이 되면 멈춘다."""
    changed = True
    while changed:
        changed = False
        for suffix in SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                token = token[: -len(suffix)]
                changed = True
                break
    return token


def korean_tokenizer(text: str) -> list:
    tokens = (normalize_token(t) for t in re.findall(r"[가-힣]{2,}", text))
    return [t for t in tokens if len(t) >= 2]


# ============================================================
# 분석 코어 (UI 비의존)
# ============================================================
def read_and_clean_csv(file_path) -> pd.DataFrame:
    """경로 또는 file-like 객체를 받아 검증·정제된 DataFrame을 돌려준다."""
    if hasattr(file_path, "read"):
        raw = file_path.read()
    else:
        with open(file_path, "rb") as fh:
            raw = fh.read()

    if not raw:
        raise ValueError("빈 파일입니다. CSV 내용을 확인하세요.")

    df = None
    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError(f"인코딩을 읽지 못했습니다. 시도한 인코딩: {', '.join(ENCODINGS)}")

    if "text" not in df.columns:
        raise ValueError(f"'text' 컬럼이 필요합니다. 현재 컬럼: {list(df.columns)}")

    df = df.copy()
    df["text"] = df["text"].astype("string").str.strip()
    df = df.dropna(subset=["text"])
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)

    if df.empty:
        raise ValueError("정제 후 남은 문장이 없습니다.")
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    return df


def get_cluster_keywords(df, top_n=6, extra_stopwords=None) -> pd.DataFrame:
    """클러스터를 하나의 문서로 합쳐 TF-IDF 상위 키워드를 뽑는다."""
    stop = set(KOREAN_STOPWORDS) | set(extra_stopwords or [])
    stop = sorted({normalize_token(w) for w in stop})

    docs = df.groupby("cluster")["text"].apply(" ".join)
    vectorizer = TfidfVectorizer(
        tokenizer=korean_tokenizer,
        token_pattern=None,
        lowercase=False,
        stop_words=stop,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(docs.values)
    vocab = np.array(vectorizer.get_feature_names_out())

    rows = []
    for i, cluster_id in enumerate(docs.index):
        scores = matrix[i].toarray().ravel()
        top = vocab[scores.argsort()[::-1][:top_n]]
        rows.append({"cluster": cluster_id, "keywords": ", ".join(top)})
    return pd.DataFrame(rows)


def get_representative_comments(df, embeddings, kmeans, top_n=3) -> pd.DataFrame:
    """각 클러스터 중심에 코사인 유사도가 가장 높은 실제 문장을 뽑는다."""
    rows = []
    for cluster_id in sorted(df["cluster"].unique()):
        idx = np.where(df["cluster"].values == cluster_id)[0]
        center = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
        sims = cosine_similarity(embeddings[idx], center).ravel()
        for rank, pos in enumerate(sims.argsort()[::-1][:top_n], start=1):
            rows.append(
                {
                    "cluster": cluster_id,
                    "rank": rank,
                    "similarity_to_center": round(float(sims[pos]), 4),
                    "text": df["text"].iloc[idx[pos]],
                }
            )
    return pd.DataFrame(rows)


def build_topic_summary(df, keywords_df, rep_df) -> pd.DataFrame:
    counts = df.groupby("cluster").size().reset_index(name="n_comments")
    top1 = (
        rep_df[rep_df["rank"] == 1][["cluster", "text"]]
        .rename(columns={"text": "representative"})
    )
    summary = (
        counts.merge(keywords_df, on="cluster", how="left")
        .merge(top1, on="cluster", how="left")
    )
    summary["share_%"] = (summary["n_comments"] / len(df) * 100).round(1)
    return (
        summary[["cluster", "n_comments", "share_%", "keywords", "representative"]]
        .sort_values("n_comments", ascending=False)
        .reset_index(drop=True)
    )


def _wrap(text, width=28):
    return "<br>".join(text[i : i + width] for i in range(0, len(text), width))


def build_topic_map(df, coords, keywords_df):
    """PCA 2D 좌표로 Plotly 산점도를 만든다. topic을 문자열로 넘겨 이산 색상을 강제한다."""
    kw_map = dict(zip(keywords_df["cluster"], keywords_df["keywords"]))
    plot_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "topic": [
                f"{c}: " + ", ".join(kw_map.get(c, "").split(", ")[:3])
                for c in df["cluster"]
            ],
            "hover": df["text"].apply(_wrap).values,
        }
    )
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="topic",
        custom_data=["hover"],
        opacity=0.75,
        category_orders={"topic": sorted(plot_df["topic"].unique())},
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        marker=dict(size=8, line=dict(width=0.5, color="white")),
    )
    fig.update_layout(height=560, legend_title_text="Topic", margin=dict(t=30))
    return fig


def semantic_search(query, df, embeddings, model, top_k=5, min_score=None) -> pd.DataFrame:
    """자연어 질의와 가장 가까운 의견 Top-K를 돌려준다.

    Notebook Mission 8과 완전히 같은 시그니처라 노트북/Streamlit/Gradio에서
    같은 함수를 그대로 재사용할 수 있다.
    """
    columns = ["rank", "score", "cluster", "text"]
    if not query or not query.strip():
        return pd.DataFrame(columns=columns)

    q_emb = model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q_emb, embeddings).ravel()

    order = sims.argsort()[::-1][:top_k]
    if min_score is not None:
        order = [o for o in order if sims[o] >= min_score]
    if len(order) == 0:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(
        {
            "rank": range(1, len(order) + 1),
            "score": np.round(sims[order], 4),
            "cluster": df["cluster"].iloc[order].values,
            "text": df["text"].iloc[order].values,
        }
    )


def evaluate_k_range(embeddings, k_min=2, k_max=10) -> pd.DataFrame:
    """k를 바꿔가며 실루엣 점수를 계산한다.

    실루엣은 -1~1이며 클수록 군집 내부가 조밀하고 군집 간에는 떨어져 있다는 뜻이다.
    문장 임베딩에서는 0.03~0.10 정도가 흔하므로, 절대값이 아니라 k끼리의
    상대 비교로만 읽어야 한다.
    """
    k_max = min(k_max, len(embeddings) - 1)
    rows = []
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=SEED, n_init="auto").fit_predict(embeddings)
        rows.append({"k": k, "silhouette": round(float(silhouette_score(embeddings, labels)), 4)})
    return pd.DataFrame(rows)


def build_analysis(file_path, n_clusters=7, model=None) -> dict:
    """CSV → 정제 → 임베딩 → 클러스터링 → 키워드 → 대표의견 → 토픽맵까지 한 번에."""
    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    df = read_and_clean_csv(file_path)

    n_texts = len(df)
    if n_clusters < 2:
        raise ValueError("클러스터 수는 2 이상이어야 합니다.")
    if n_clusters > n_texts:
        raise ValueError(f"클러스터 수({n_clusters})가 문장 수({n_texts})보다 많습니다.")

    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init="auto")
    df["cluster"] = kmeans.fit_predict(embeddings)

    keywords_df = get_cluster_keywords(df)
    rep_df = get_representative_comments(df, embeddings, kmeans)
    topic_summary = build_topic_summary(df, keywords_df, rep_df)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    df["x"], df["y"] = coords[:, 0], coords[:, 1]

    return {
        "df": df,
        "embeddings": embeddings,
        "kmeans": kmeans,
        "model": model,
        "keywords_df": keywords_df,
        "rep_df": rep_df,
        "topic_summary": topic_summary,
        "topic_map": build_topic_map(df, coords, keywords_df),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "n_texts": n_texts,
        "n_clusters": n_clusters,
    }


# ============================================================
# UI 레이어
# ============================================================
@st.cache_resource(show_spinner="임베딩 모델을 불러오는 중…")
def load_model():
    return SentenceTransformer(MODEL_NAME)


def render_sidebar():
    with st.sidebar:
        st.header("1. 데이터")
        uploaded = st.file_uploader("의견 CSV 업로드", type="csv")
        st.caption("`text` 컬럼이 반드시 있어야 합니다.")

        st.header("2. 설정")
        n_clusters = st.slider("주제 개수 (k)", 2, 15, 7)

        analyze = st.button("Analyze", type="primary", use_container_width=True)
    return uploaded, n_clusters, analyze


def render_results(state):
    left, right = st.columns(2)
    left.metric("분석된 의견 수", f"{state['n_texts']:,}")
    right.metric("주제 수 (k)", state["n_clusters"])

    st.subheader("Topic Summary")
    st.dataframe(state["topic_summary"], use_container_width=True, hide_index=True)

    st.download_button(
        "분석 결과 CSV 다운로드",
        data=state["df"][["id", "text", "cluster"]]
        .to_csv(index=False)
        .encode("utf-8-sig"),
        file_name="ai_insight_result.csv",
        mime="text/csv",
    )

    st.subheader("Topic Map")
    st.caption(
        f"PCA 2D · 원본 분산의 {state['explained_variance'] * 100:.1f}%만 반영됩니다. "
        "겹쳐 보이는 것이 곧 클러스터링 실패를 뜻하지는 않습니다."
    )
    st.plotly_chart(state["topic_map"], use_container_width=True)


def render_k_advisor(state):
    st.subheader("적정 주제 수(k) 진단")
    st.caption(
        "k를 2~10으로 바꿔가며 실루엣 점수를 계산합니다. "
        "지금 분석에 쓴 임베딩을 재사용하므로 다시 인코딩하지 않습니다."
    )

    if st.button("k = 2~10 비교하기"):
        with st.spinner("k별로 클러스터링을 다시 계산하는 중…"):
            st.session_state["k_scores"] = evaluate_k_range(state["embeddings"])

    scores = st.session_state.get("k_scores")
    if scores is None:
        return

    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    col_table, col_chart = st.columns([1, 2])
    col_table.dataframe(scores, use_container_width=True, hide_index=True)
    col_chart.line_chart(scores.set_index("k"), y="silhouette")

    st.info(
        f"실루엣이 가장 높은 값은 **k = {best_k}** 입니다. "
        f"현재 분석에 사용한 값은 k = {state['n_clusters']} 입니다."
    )
    st.caption(
        "실루엣은 군집이 기하학적으로 얼마나 잘 갈렸는지만 봅니다. 작은 k일수록 "
        "높게 나오는 경향이 있어, 점수가 가장 높은 k가 해석하기 가장 좋은 k라는 뜻은 "
        "아닙니다. 최종 판단은 Topic Summary의 키워드와 대표 의견을 읽고 하세요."
    )


def render_search(state):
    st.subheader("Semantic Search")
    col_q, col_k, col_t = st.columns([4, 1, 1])
    query = col_q.text_input("검색어", placeholder="예: 취업 지원")
    top_k = col_k.number_input("Top-K", min_value=1, max_value=50, value=5)
    min_score = col_t.number_input(
        "최소 유사도", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        help="0이면 필터를 끕니다. 관련 없는 결과가 섞이면 0.3~0.4부터 올려보세요.",
    )

    if not query:
        return

    results = semantic_search(
        query,
        state["df"],
        state["embeddings"],
        state["model"],
        top_k=int(top_k),
        min_score=min_score if min_score > 0 else None,
    )
    if results.empty:
        st.info("기준을 넘는 결과가 없습니다. 최소 유사도를 낮추거나 검색어를 바꿔보세요.")
        return

    st.dataframe(results, use_container_width=True, hide_index=True)
    st.caption(
        "score는 이 데이터 안에서의 상대적 유사도입니다. "
        "관련 문장이 없어도 Top-K는 항상 채워집니다."
    )


def main():
    st.set_page_config(page_title="AI Insight Engine", page_icon="🧭", layout="wide")
    st.title("AI Insight Engine")
    st.caption("자유 응답 의견을 주제별로 묶고 검색합니다.")

    uploaded, n_clusters, analyze = render_sidebar()

    if analyze:
        if uploaded is None:
            st.warning("먼저 CSV 파일을 업로드하세요.")
        else:
            try:
                with st.spinner("분석 중… (최초 실행은 모델 다운로드로 시간이 걸립니다)"):
                    st.session_state["analysis"] = build_analysis(
                        uploaded, n_clusters=n_clusters, model=load_model()
                    )
                # 새 분석이므로 이전 k 진단 결과는 버린다.
                st.session_state.pop("k_scores", None)
            except ValueError as exc:
                st.error(str(exc))

    state = st.session_state.get("analysis")
    if state is None:
        st.info("왼쪽에서 CSV를 업로드하고 **Analyze**를 누르세요.")
        return

    render_results(state)
    st.divider()
    render_k_advisor(state)
    st.divider()
    render_search(state)


if __name__ == "__main__":
    main()
