"""
ETF Ranker — Streamlit 앱
국내 ETF의 구성종목 forward 컨센서스를 가중평균하여 랭킹한다.
"""

import numpy as np
import pandas as pd
import streamlit as st

from score_etf import score_etfs, load_consensus, load_constituents, DATA_DIR

st.set_page_config(
    page_title="ETF Forward 랭커",
    page_icon="📊",
    layout="wide",
)

st.title("📊 ETF Forward 랭커")
st.caption(
    "국내 주식형 ETF의 구성종목 forward 컨센서스(영업이익 성장률, PER)를 "
    "비중 가중평균하여 순위를 매깁니다."
)


@st.cache_data(ttl=3600, show_spinner="스코어 계산 중...")
def get_scores() -> pd.DataFrame:
    return score_etfs()


@st.cache_data(ttl=3600)
def get_constituents() -> pd.DataFrame:
    return load_constituents()


@st.cache_data(ttl=3600)
def get_consensus() -> pd.DataFrame:
    return load_consensus()


# ── 사이드바 필터 ──────────────────────────────────────────
st.sidebar.header("필터")

min_coverage = st.sidebar.slider(
    "최소 컨센서스 커버리지 (%)",
    min_value=0,
    max_value=100,
    value=70,
    step=5,
    help="비중 기준으로 컨센서스가 존재하는 종목 비율",
)

sort_by = st.sidebar.selectbox(
    "정렬 기준",
    options=[
        "composite_score",
        "op_growth_2026",
        "op_growth_2027",
        "fwd_per",
        "coverage",
    ],
    format_func=lambda x: {
        "composite_score": "복합 점수",
        "op_growth_2026": "2026E 영업이익 성장률",
        "op_growth_2027": "2027E 영업이익 성장률",
        "fwd_per": "Forward PER (낮을수록 상위)",
        "coverage": "컨센서스 커버리지",
    }.get(x, x),
)

top_n = st.sidebar.slider("상위 N개 표시", 10, 100, 30, 5)

# ── 메인 테이블 ────────────────────────────────────────────
scores = get_scores()

if scores.empty:
    st.error("데이터가 없습니다. 크롤링을 먼저 실행해주세요 (python crawl_etf.py)")
    st.stop()

filtered = scores[scores["coverage"] >= min_coverage].copy()

ascending = sort_by == "fwd_per"
filtered = filtered.sort_values(sort_by, ascending=ascending, na_position="last")
filtered = filtered.head(top_n).reset_index(drop=True)
filtered.index += 1
filtered.index.name = "순위"

st.subheader(f"ETF 랭킹 (커버리지 ≥ {min_coverage}%)")

col_config = {
    "etf_code": st.column_config.TextColumn("종목코드", width="small"),
    "etf_name": st.column_config.TextColumn("ETF명", width="medium"),
    "n_stocks": st.column_config.NumberColumn("구성종목수", format="%d"),
    "n_covered": st.column_config.NumberColumn("컨센서스 종목수", format="%d"),
    "coverage": st.column_config.ProgressColumn(
        "커버리지(%)", min_value=0, max_value=100, format="%.1f%%"
    ),
    "op_growth_2026": st.column_config.NumberColumn(
        "2026E 영업이익 성장률(%)", format="%.1f%%"
    ),
    "op_growth_2027": st.column_config.NumberColumn(
        "2027E 영업이익 성장률(%)", format="%.1f%%"
    ),
    "fwd_per": st.column_config.NumberColumn("Fwd PER", format="%.1f"),
    "composite_score": st.column_config.NumberColumn("복합점수", format="%.1f"),
}

st.dataframe(filtered, column_config=col_config, use_container_width=True)

# ── ETF 상세 (구성종목 드릴다운) ────────────────────────────
st.divider()
st.subheader("ETF 구성종목 상세")

etf_options = filtered[["etf_code", "etf_name"]].apply(
    lambda r: f"{r['etf_name']} ({r['etf_code']})", axis=1
).tolist()

if etf_options:
    selected = st.selectbox("ETF 선택", etf_options)
    sel_code = selected.split("(")[-1].rstrip(")")

    const_df = get_constituents()
    cons_df = get_consensus()

    etf_const = const_df[const_df["etf_code"] == sel_code].copy()

    lookup_cols = {"종목명": "stock_name"}
    metric_cols = {}
    for c in ["영업이익_성장률_2026", "영업이익_성장률_2027", "PER"]:
        if c in cons_df.columns:
            metric_cols[c] = c
    renamed = cons_df.rename(columns=lookup_cols)

    detail = etf_const.merge(
        renamed[["stock_name"] + list(metric_cols.values())],
        on="stock_name",
        how="left",
    )

    detail = detail.sort_values("weight", ascending=False)

    has_data = detail["영업이익_성장률_2026"].notna() if "영업이익_성장률_2026" in detail.columns else pd.Series(False, index=detail.index)

    def highlight_coverage(row):
        if "영업이익_성장률_2026" in row.index and pd.isna(row.get("영업이익_성장률_2026")):
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(
        detail.style.apply(highlight_coverage, axis=1),
        use_container_width=True,
        height=400,
    )

    covered_w = detail.loc[has_data, "weight"].sum()
    total_w = detail["weight"].sum()
    st.metric(
        "컨센서스 커버리지",
        f"{covered_w / total_w * 100:.1f}%" if total_w > 0 else "N/A",
    )

    # 비중 상위 10 차트
    top10 = detail.head(10)
    chart_data = top10.set_index("stock_name")["weight"]
    st.bar_chart(chart_data, horizontal=True)

# ── 데이터 갱신일 ──────────────────────────────────────────
st.sidebar.divider()
const_path = DATA_DIR / "etf_constituents.csv"
if const_path.exists():
    import os, datetime
    mtime = os.path.getmtime(const_path)
    dt = datetime.datetime.fromtimestamp(mtime)
    st.sidebar.caption(f"데이터 갱신: {dt:%Y-%m-%d %H:%M}")
else:
    st.sidebar.caption("데이터 없음 — crawl_etf.py 실행 필요")
