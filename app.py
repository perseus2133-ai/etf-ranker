"""
ETF 퀀트 터미널 — Streamlit 대시보드
국내 주식형 ETF의 forward 컨센서스를 비중 가중평균하여 랭킹.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from score_etf import (
    DATA_DIR,
    get_etf_constituents_with_consensus,
    load_consensus,
    score_etfs,
)
from indicators import analyze, obv_verdict, rsi_verdict

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="ETF 퀀트 터미널",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 다크 퀀트 카드 CSS (ai2 스타일 차용)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', 'Pretendard', sans-serif;
    background-color: #3E4A59 !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] > div:first-child { background-color: #1A1C24 !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}
div[data-baseweb="select"] > div { background-color: #1A1C24 !important; color: #FFF !important; }
div[data-baseweb="select"] span { color: #FFF !important; }
div[data-baseweb="popover"] ul { background-color: #1A1C24 !important; }
div[data-baseweb="popover"] li { color: #FFF !important; }

/* Hero header */
.hero-header {
    border-bottom: 1px solid #4C566A;
    padding-bottom: 16px; margin-bottom: 24px;
    text-align: left;
}
.hero-header p { color: #A0AEC0; font-size: 1.0rem; margin: 0; font-family: 'JetBrains Mono', monospace; }

@keyframes rainbow-text {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.rainbow-title {
    font-weight: 900; font-size: 2.4rem;
    background: linear-gradient(to right, #62efff, #ffb3fd, #ffeead, #62efff);
    background-size: 400% 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: rainbow-text 5s ease infinite;
    display: flex; align-items: center; gap: 8px;
    margin: 0 0 6px 0;
}
.rainbow-score {
    background: linear-gradient(to right, #62efff, #ffb3fd, #ffeead, #62efff);
    background-size: 400% 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: rainbow-text 5s ease infinite;
}

/* Quant card */
.quant-card {
    background: linear-gradient(135deg, #3F4C60 0%, #313B4D 100%);
    color: #E2E8F0;
    border: 1px solid #4A5568;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 16px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.30), 0 2px 6px rgba(0,0,0,0.20);
    transition: all 0.3s ease;
}
.quant-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 32px rgba(0,0,0,0.40), 0 3px 8px rgba(0,0,0,0.25);
    border-color: #62efff;
}

.qc-rank {
    background: rgba(98, 239, 255, 0.12); color: #62EFFF;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
    font-weight: 700; padding: 3px 9px; border-radius: 5px;
    border: 1px solid rgba(98, 239, 255, 0.25);
}
.qc-name { color: #FFF; font-size: 1.18rem; font-weight: 700; }
.qc-code { color: #94A3B8; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; margin-left: 6px; }

.qc-naver-link {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(98,239,255,0.08);
    color: #62EFFF !important;
    border: 1px solid rgba(98,239,255,0.35);
    padding: 6px 12px; border-radius: 6px;
    font-size: 0.78rem; font-weight: 700; text-decoration: none !important;
    transition: all 0.2s ease;
}
.qc-naver-link:hover {
    background: rgba(98,239,255,0.18);
    transform: translateY(-1px);
    box-shadow: 0 0 12px rgba(98,239,255,0.25);
}

.qc-stat-label { color: #94A3B8; font-size: 0.72rem; font-weight: 600; }
.qc-stat-val { color: #FFF; font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.0rem; }

.qc-pill {
    background: rgba(17,24,39,0.55); border: 1px solid #4A5568; border-radius: 9px;
    padding: 9px 14px; min-width: 78px; text-align: center;
}
.qc-pill .lbl { color: #94A3B8; font-size: 0.7rem; font-weight: 600; }
.qc-pill .val { color: #FFF; font-family: 'JetBrains Mono', monospace; font-weight: 800; font-size: 1.05rem; margin-top: 2px; }
.qc-pill.hi { border-color: #62EFFF; box-shadow: 0 0 10px rgba(98,239,255,0.18) inset; }
.qc-pill.hi .lbl, .qc-pill.hi .val { color: #62EFFF; }

.qc-tech {
    background: rgba(17,24,39,0.45); border: 1px solid #4A5568; border-radius: 10px;
    padding: 10px 14px; display: flex; gap: 18px; align-items: center; flex-wrap: wrap;
    margin-top: 10px;
}
.qc-tech .item { display: flex; flex-direction: column; gap: 2px; }
.qc-tech .item .k { color: #94A3B8; font-size: 0.7rem; font-weight: 600; }
.qc-tech .item .v { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.95rem; }

/* Forecast table inside expander */
.forecast-table {
    width: 100%; border-collapse: collapse; margin: 8px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
}
.forecast-table th {
    background: rgba(98,239,255,0.10);
    color: #62EFFF; padding: 8px 10px; text-align: right;
    border-bottom: 1px solid #4A5568;
    font-weight: 700;
}
.forecast-table th:first-child { text-align: left; }
.forecast-table td {
    padding: 8px 10px; text-align: right;
    border-bottom: 1px solid rgba(74,85,104,0.4);
    color: #E2E8F0;
}
.forecast-table td:first-child { text-align: left; color: #FFF; font-weight: 600; }
.forecast-table td.subtle { color: #94A3B8; }
.forecast-table tr.section-row td {
    background: rgba(17,24,39,0.4);
    color: #94A3B8; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.growth-pos { color: #34D399; font-weight: 700; }
.growth-neg { color: #FB7185; font-weight: 700; }
.growth-mega { color: #FCD34D; font-weight: 800; }

/* Streamlit overrides */
div[data-testid="stExpander"] details {
    background: rgba(26,28,36,0.65) !important;
    border: 1px solid #4A5568 !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
}
div[data-testid="stExpander"] summary {
    padding: 16px 20px !important;
    color: #E2E8F0 !important;
}
div[data-testid="stExpander"] summary:hover {
    background: rgba(98,239,255,0.05) !important;
}
.stProgress > div > div { background: linear-gradient(90deg, #62EFFF, #1D3557) !important; }

button[data-baseweb="tab"] {
    background: linear-gradient(135deg, #2D3139, #3E4A59) !important;
    border: 1px solid #4C566A !important;
    border-radius: 10px !important;
    padding: 8px 18px !important;
    margin-right: 5px;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1D3557, #457B9D) !important;
    border-color: #62efff !important;
}
button[data-baseweb="tab"] > div p { color: #A0AEC0 !important; font-weight: 700 !important; }
button[data-baseweb="tab"][aria-selected="true"] > div p { color: #FFF !important; }

#MainMenu { visibility: hidden; } footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 데이터 로드 (캐시)
# ============================================================
@st.cache_data(ttl=3600, show_spinner="ETF 스코어 계산 중...")
def get_scores() -> pd.DataFrame:
    return score_etfs()


@st.cache_data(ttl=3600)
def get_consensus() -> pd.DataFrame:
    return load_consensus()


@st.cache_data(ttl=3600)
def get_etf_detail(etf_code: str, top_n: int = 10) -> pd.DataFrame:
    return get_etf_constituents_with_consensus(etf_code, top_n)


@st.cache_data(ttl=1800, show_spinner=False)
def get_indicators(stock_code: str) -> dict:
    try:
        return analyze(stock_code, days=90)
    except Exception:
        return {"prices": [], "volumes": [], "rsi": None, "obv_trend": "",
                "support": None, "resistance": None, "last_price": None}


# ============================================================
# 헬퍼: 포맷터
# ============================================================
def fmt_growth(v: float | None) -> str:
    if v is None or pd.isna(v):
        return '<span class="subtle">—</span>'
    cls = "growth-mega" if v >= 100 else ("growth-pos" if v >= 0 else "growth-neg")
    return f'<span class="{cls}">{v:+.1f}%</span>'


def fmt_money_eok(v) -> str:
    """억원 단위 (이미 억원 단위로 들어옴)."""
    if v is None or pd.isna(v):
        return '<span class="subtle">—</span>'
    if abs(v) >= 10000:
        return f"{v/10000:,.2f}조"
    return f"{v:,.0f}억"


def fmt_pct(v) -> str:
    if v is None or pd.isna(v):
        return '<span class="subtle">—</span>'
    return f"{v:.1f}%"


def fmt_num(v, digits=1) -> str:
    if v is None or pd.isna(v):
        return '<span class="subtle">—</span>'
    return f"{v:.{digits}f}"


def naver_etf_url(code: str) -> str:
    return f"https://finance.naver.com/item/main.naver?code={code}"


# ============================================================
# 차트 SVG (가격 + RSI 미니 차트)
# ============================================================
def build_price_chart_svg(prices: list[int], support: int | None, resistance: int | None,
                          width: int = 600, height: int = 180) -> str:
    """가격 차트 (최신순 입력 → 좌→우 시간순으로 표시)."""
    if not prices or len(prices) < 5:
        return ('<div style="height:180px;display:flex;align-items:center;justify-content:center;'
                'color:#94A3B8;font-family:JetBrains Mono;">차트 데이터 없음</div>')

    p = list(reversed(prices))  # 과거→현재
    pad_l, pad_r, pad_t, pad_b = 50, 14, 14, 24
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b

    vmin = min(p + ([support] if support else []))
    vmax = max(p + ([resistance] if resistance else []))
    span = vmax - vmin if vmax != vmin else 1
    pad = span * 0.08
    vmin -= pad; vmax += pad
    span = vmax - vmin

    n = len(p)
    def x_at(i): return pad_l + (inner_w * i / max(1, n - 1))
    def y_at(v): return pad_t + inner_h - ((v - vmin) / span) * inner_h

    # 그리드 + 지지/저항 라인
    grid = ''
    for frac in (0.0, 0.5, 1.0):
        gy = pad_t + inner_h * frac
        grid += (f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width-pad_r}" y2="{gy:.1f}" '
                 f'stroke="#4A5568" stroke-width="0.5" stroke-dasharray="2,3" opacity="0.5"/>')
    if resistance:
        ry = y_at(resistance)
        grid += (f'<line x1="{pad_l}" y1="{ry:.1f}" x2="{width-pad_r}" y2="{ry:.1f}" '
                 f'stroke="#FB7185" stroke-width="1" stroke-dasharray="4,3" opacity="0.7"/>')
        grid += (f'<text x="{width-pad_r}" y="{ry-3:.1f}" fill="#FB7185" font-size="9" '
                 f'text-anchor="end" font-family="JetBrains Mono">저항 {resistance:,}</text>')
    if support:
        sy = y_at(support)
        grid += (f'<line x1="{pad_l}" y1="{sy:.1f}" x2="{width-pad_r}" y2="{sy:.1f}" '
                 f'stroke="#34D399" stroke-width="1" stroke-dasharray="4,3" opacity="0.7"/>')
        grid += (f'<text x="{width-pad_r}" y="{sy+11:.1f}" fill="#34D399" font-size="9" '
                 f'text-anchor="end" font-family="JetBrains Mono">지지 {support:,}</text>')

    # 가격 라인 + 영역
    pts = [(x_at(i), y_at(v)) for i, v in enumerate(p)]
    d = ' '.join(f'{"M" if i==0 else "L"} {x:.1f} {y:.1f}' for i, (x, y) in enumerate(pts))
    fill_d = (d + f' L {pts[-1][0]:.1f} {pad_t+inner_h:.1f} '
              f'L {pts[0][0]:.1f} {pad_t+inner_h:.1f} Z')

    # Y축 라벨
    y_labels = (
        f'<text x="{pad_l-6}" y="{pad_t+4:.1f}" fill="#94A3B8" font-size="9" '
        f'text-anchor="end" font-family="JetBrains Mono">{vmax:,.0f}</text>'
        f'<text x="{pad_l-6}" y="{pad_t+inner_h+3:.1f}" fill="#94A3B8" font-size="9" '
        f'text-anchor="end" font-family="JetBrains Mono">{vmin:,.0f}</text>'
    )
    # X축 라벨
    x_labels = (
        f'<text x="{pad_l}" y="{height-6}" fill="#94A3B8" font-size="9" '
        f'text-anchor="start" font-family="JetBrains Mono">-{n}d</text>'
        f'<text x="{width-pad_r}" y="{height-6}" fill="#94A3B8" font-size="9" '
        f'text-anchor="end" font-family="JetBrains Mono">최근</text>'
    )

    return (
        f'<svg width="100%" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<defs><linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="#62EFFF" stop-opacity="0.4"/>'
        f'<stop offset="100%" stop-color="#62EFFF" stop-opacity="0"/>'
        f'</linearGradient></defs>'
        f'{grid}'
        f'<path d="{fill_d}" fill="url(#priceGrad)"/>'
        f'<path d="{d}" fill="none" stroke="#62EFFF" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'{y_labels}{x_labels}'
        f'</svg>'
    )


# ============================================================
# 헤더
# ============================================================
st.markdown("""
<div class="hero-header">
    <div class="rainbow-title">
        <svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" stroke-width="2.5"
             fill="none" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>
        ETF 퀀트 터미널
    </div>
    <p>국내 주식형 ETF · Forward 컨센서스 가중평균 랭킹 · 2026E~2028E</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 사이드바
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ 스크리닝 설정")
    st.markdown("---")

    st.markdown("### 📊 필터")
    min_coverage = st.slider("최소 컨센서스 커버리지 (%)", 0, 100, 70, 5,
                             help="비중 기준으로 컨센서스가 존재하는 종목 비율")

    sort_by = st.selectbox(
        "정렬 기준",
        options=[
            "composite_score",
            "op_growth_2026", "op_growth_2027", "op_growth_2028",
            "rev_growth_2026", "rev_growth_2027", "rev_growth_2028",
            "fwd_per", "coverage",
        ],
        format_func=lambda x: {
            "composite_score": "🏆 복합 점수",
            "op_growth_2026": "📈 2026E 영업이익 성장률",
            "op_growth_2027": "📈 2027E 영업이익 성장률",
            "op_growth_2028": "📈 2028E 영업이익 성장률",
            "rev_growth_2026": "💰 2026E 매출 성장률",
            "rev_growth_2027": "💰 2027E 매출 성장률",
            "rev_growth_2028": "💰 2028E 매출 성장률",
            "fwd_per": "💸 Forward PER (낮은순)",
            "coverage": "🎯 컨센서스 커버리지",
        }[x],
    )

    top_n = st.slider("상위 N개 표시", 10, 100, 30, 5)

    st.markdown("---")
    keyword = st.text_input("🔎 ETF명 검색", "", placeholder="예: 반도체, AI, 2차전지...")

    st.markdown("---")
    st.markdown("### 📦 데이터")
    const_path = DATA_DIR / "etf_constituents.csv"
    if const_path.exists():
        mtime = os.path.getmtime(const_path)
        dt = datetime.datetime.fromtimestamp(mtime)
        st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#94A3B8;'>"
                    f"갱신: {dt:%Y-%m-%d %H:%M}</div>", unsafe_allow_html=True)


# ============================================================
# 데이터 가져오기 + 필터
# ============================================================
scores = get_scores()

if scores.empty:
    st.error("데이터가 없습니다. `python crawl_etf.py` 를 실행해주세요.")
    st.stop()

filtered = scores[scores["coverage"] >= min_coverage].copy()
if keyword:
    filtered = filtered[filtered["etf_name"].str.contains(keyword, case=False, na=False)]

ascending = sort_by == "fwd_per"
filtered = filtered.sort_values(sort_by, ascending=ascending, na_position="last")
filtered = filtered.head(top_n).reset_index(drop=True)


# ============================================================
# 상단 요약 통계
# ============================================================
total_etfs = len(scores)
filtered_n = len(filtered)
avg_op26 = filtered["op_growth_2026"].mean()
avg_op27 = filtered["op_growth_2027"].mean()
avg_op28 = filtered["op_growth_2028"].mean()

col1, col2, col3, col4, col5 = st.columns(5)
for col, label, val in [
    (col1, "전체 ETF", f"{total_etfs:,}"),
    (col2, "필터 결과", f"{filtered_n:,}"),
    (col3, "2026E 평균 OP↑", f"{avg_op26:.1f}%" if pd.notna(avg_op26) else "—"),
    (col4, "2027E 평균 OP↑", f"{avg_op27:.1f}%" if pd.notna(avg_op27) else "—"),
    (col5, "2028E 평균 OP↑", f"{avg_op28:.1f}%" if pd.notna(avg_op28) else "—"),
]:
    col.markdown(
        f'<div style="background:rgba(17,24,39,0.45);border:1px solid #4A5568;'
        f'border-radius:10px;padding:14px;text-align:center;">'
        f'<div style="color:#94A3B8;font-size:0.75rem;margin-bottom:4px;">{label}</div>'
        f'<div style="color:#62EFFF;font-family:JetBrains Mono;font-weight:800;font-size:1.4rem;">{val}</div>'
        f'</div>', unsafe_allow_html=True
    )

st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)


# ============================================================
# ETF 카드 렌더링
# ============================================================
def render_etf_card(row, rank: int):
    code = row["etf_code"]
    name = row["etf_name"]
    nurl = naver_etf_url(code)

    # 헤더
    header_html = (
        f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;'
        f'border-bottom:1px solid #4A5568;padding-bottom:10px;margin-bottom:12px;">'
        f'<span class="qc-rank">#{rank}</span>'
        f'<span class="qc-name">{name}</span>'
        f'<span class="qc-code">{code}</span>'
        f'<div style="flex:1;"></div>'
        f'<a href="{nurl}" target="_blank" class="qc-naver-link">'
        f'<svg viewBox="0 0 24 24" width="13" height="13" stroke="currentColor" stroke-width="2" '
        f'fill="none" stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>'
        f'<polyline points="15 3 21 3 21 9"></polyline>'
        f'<line x1="10" y1="14" x2="21" y2="3"></line></svg>네이버 증권</a>'
        f'</div>'
    )

    # 통계
    stats_html = (
        f'<div style="display:flex;gap:22px;flex-wrap:wrap;margin-bottom:10px;align-items:flex-end;">'
        f'<div><div class="qc-stat-label">구성종목</div>'
        f'<div class="qc-stat-val">{row["n_stocks"]}개</div></div>'
        f'<div><div class="qc-stat-label">컨센서스 종목</div>'
        f'<div class="qc-stat-val">{row["n_covered"]}개</div></div>'
        f'<div><div class="qc-stat-label">커버리지</div>'
        f'<div class="qc-stat-val" style="color:#62EFFF;">{row["coverage"]:.1f}%</div></div>'
        f'<div style="flex:1;"></div>'
        f'<div style="text-align:center;padding:8px 16px;background:rgba(17,24,39,0.5);'
        f'border:1px solid #62EFFF;border-radius:8px;">'
        f'<div class="qc-stat-label">복합점수</div>'
        f'<div class="rainbow-score" style="font-family:JetBrains Mono;font-weight:900;font-size:1.4rem;">'
        f'{row["composite_score"]:.1f}</div></div>'
        f'</div>'
    )

    # Pills (성장률 + PER)
    def pill(label, val, hi=False, suffix="%"):
        cls = "qc-pill hi" if hi else "qc-pill"
        if val is None or pd.isna(val):
            return f'<div class="{cls}"><div class="lbl">{label}</div><div class="val">—</div></div>'
        v_str = f"{val:.1f}{suffix}"
        return f'<div class="{cls}"><div class="lbl">{label}</div><div class="val">{v_str}</div></div>'

    pills_html = (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:6px;">'
        + pill("2026 OP↑", row.get("op_growth_2026"), hi=True)
        + pill("2027 OP↑", row.get("op_growth_2027"), hi=True)
        + pill("2028 OP↑", row.get("op_growth_2028"), hi=True)
        + pill("2026 매출↑", row.get("rev_growth_2026"))
        + pill("2027 매출↑", row.get("rev_growth_2027"))
        + pill("2028 매출↑", row.get("rev_growth_2028"))
        + pill("Fwd PER", row.get("fwd_per"), suffix="")
        + pill("Fwd ROE", row.get("fwd_roe"))
        + f'</div>'
    )

    st.markdown(
        f'<div class="quant-card">{header_html}{stats_html}{pills_html}</div>',
        unsafe_allow_html=True,
    )


def render_etf_detail(etf_code: str, etf_name: str):
    """확장된 상세: 포트폴리오 TOP10 + 매출/영업이익 forecast + 차트 + RSI/OBV."""
    detail = get_etf_detail(etf_code, top_n=10)

    # ── 차트 + 기술 지표 ─────────────────────────────────
    with st.spinner("차트/지표 로딩 중..."):
        ind = get_indicators(etf_code)

    rsi_text, rsi_color = rsi_verdict(ind.get("rsi"))
    obv_text, obv_color = obv_verdict(ind.get("obv_trend", ""))
    last_price = ind.get("last_price")
    support = ind.get("support")
    resistance = ind.get("resistance")

    chart_svg = build_price_chart_svg(ind.get("prices", []), support, resistance)

    tech_html = (
        f'<div class="qc-tech">'
        f'<div class="item"><div class="k">현재가</div>'
        f'<div class="v" style="color:#62EFFF;">{last_price:,}원</div></div>' if last_price else
        f'<div class="qc-tech"><div class="item"><div class="k">현재가</div><div class="v">—</div></div>'
    )
    tech_html += (
        f'<div class="item"><div class="k">RSI(14)</div>'
        f'<div class="v" style="color:{rsi_color};">{rsi_text}</div></div>'
        f'<div class="item"><div class="k">OBV 추세</div>'
        f'<div class="v" style="color:{obv_color};">{obv_text}</div></div>'
        f'<div class="item"><div class="k">60일 저항</div>'
        f'<div class="v" style="color:#FB7185;">{resistance:,}원</div></div>' if resistance else
        f'<div class="item"><div class="k">60일 저항</div><div class="v">—</div></div>'
    )
    tech_html += (
        f'<div class="item"><div class="k">60일 지지</div>'
        f'<div class="v" style="color:#34D399;">{support:,}원</div></div>' if support else
        f'<div class="item"><div class="k">60일 지지</div><div class="v">—</div></div>'
    )
    tech_html += '</div>'

    st.markdown(f"### 📈 {etf_name} · 가격 추이 (90일)")
    st.markdown(
        f'<div style="background:rgba(17,24,39,0.45);border:1px solid #4A5568;'
        f'border-radius:12px;padding:16px;margin-bottom:8px;">{chart_svg}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(tech_html, unsafe_allow_html=True)

    # ── 포트폴리오 TOP10 + 매출/영업이익 forecast 테이블 ─────
    st.markdown(f"### 📋 포트폴리오 TOP {len(detail)} · Forward 컨센서스")

    if detail.empty:
        st.warning("구성종목 데이터가 없습니다.")
        return

    rows_html = ""
    for _, r in detail.iterrows():
        sname = r.get("stock_name", "—")
        scode = r.get("stock_code")
        weight = r.get("weight", 0)

        if pd.notna(scode) and scode:
            scode_str = str(scode).zfill(6) if str(scode).isdigit() else str(scode)
            stock_link = f'<a href="https://finance.naver.com/item/main.naver?code={scode_str}" target="_blank" style="color:#62EFFF;text-decoration:none;">{sname} <span style="color:#94A3B8;font-size:0.78rem;font-family:JetBrains Mono;">{scode_str}</span> ↗</a>'
        else:
            stock_link = sname

        # 매출
        rev25 = r.get("rev_2025"); rev26 = r.get("rev_2026")
        rev27 = r.get("rev_2027"); rev28 = r.get("rev_2028")
        # 영업이익
        op25 = r.get("op_2025"); op26 = r.get("op_2026")
        op27 = r.get("op_2027"); op28 = r.get("op_2028")
        # 성장률
        rg26 = r.get("rev_growth_2026"); rg27 = r.get("rev_growth_2027"); rg28 = r.get("rev_growth_2028")
        og26 = r.get("op_growth_2026"); og27 = r.get("op_growth_2027"); og28 = r.get("op_growth_2028")

        rows_html += (
            f'<tr class="section-row"><td colspan="5">{stock_link} · 비중 {weight:.2f}%</td></tr>'
            f'<tr><td>매출액</td>'
            f'<td>{fmt_money_eok(rev25)}</td>'
            f'<td>{fmt_money_eok(rev26)} <span class="subtle" style="font-size:0.78rem;">({fmt_growth(rg26)})</span></td>'
            f'<td>{fmt_money_eok(rev27)} <span class="subtle" style="font-size:0.78rem;">({fmt_growth(rg27)})</span></td>'
            f'<td>{fmt_money_eok(rev28)} <span class="subtle" style="font-size:0.78rem;">({fmt_growth(rg28)})</span></td>'
            f'</tr>'
            f'<tr><td>영업이익</td>'
            f'<td>{fmt_money_eok(op25)}</td>'
            f'<td>{fmt_money_eok(op26)} <span class="subtle" style="font-size:0.78rem;">({fmt_growth(og26)})</span></td>'
            f'<td>{fmt_money_eok(op27)} <span class="subtle" style="font-size:0.78rem;">({fmt_growth(og27)})</span></td>'
            f'<td>{fmt_money_eok(op28)} <span class="subtle" style="font-size:0.78rem;">({fmt_growth(og28)})</span></td>'
            f'</tr>'
        )

    table_html = (
        '<table class="forecast-table">'
        '<thead><tr>'
        '<th>종목 / 항목</th><th>2025A/E</th><th>2026E</th><th>2027E</th><th>2028E</th>'
        '</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


# ============================================================
# ETF 리스트 (카드 + 클릭 시 expander)
# ============================================================
if filtered.empty:
    st.warning("조건에 맞는 ETF가 없습니다. 필터를 완화해주세요.")
else:
    for idx, row in filtered.iterrows():
        rank = idx + 1
        # 카드를 expander로 감싸 클릭 시 확장
        with st.expander(
            f"#{rank}  ·  {row['etf_name']}  ({row['etf_code']})  ·  복합점수 {row['composite_score']:.1f}",
            expanded=False,
        ):
            # 카드 (헤더+stats+pills)
            render_etf_card(row, rank)
            # 상세 (차트 + 포트폴리오 forecast)
            render_etf_detail(row["etf_code"], row["etf_name"])

st.markdown(
    '<div style="margin-top:32px;padding:16px;color:#94A3B8;font-size:0.78rem;text-align:center;'
    'border-top:1px solid #4A5568;font-family:JetBrains Mono;">'
    'ETF Quant Terminal · Naver Finance + ai2 consensus · '
    '<a href="https://github.com/perseus2133-ai/etf-ranker" style="color:#62EFFF;">GitHub</a>'
    '</div>', unsafe_allow_html=True
)
