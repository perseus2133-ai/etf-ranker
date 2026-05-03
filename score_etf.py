"""
ETF 스코어링 엔진
- etf_constituents.csv  (구성종목/비중)
- consensus_data.csv (ai2 레포에서 다운로드)
를 조인하여 ETF별 가중평균 점수를 계산한다.

메트릭:
  1) 2026E 영업이익 성장률 (가중평균)
  2) 2027E 영업이익 성장률 (가중평균)
  3) 2028E 영업이익 성장률 (가중평균)
  4) Forward PER (가중평균)
  5) 컨센서스 커버리지 (비중 기준 %)
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

CONSENSUS_URL = (
    "https://raw.githubusercontent.com/perseus2133-ai/ai2/main/data/consensus_data.csv"
)


def load_consensus(local_path: Path | None = None, prefer_remote: bool = False) -> pd.DataFrame:
    """ai2 레포의 consensus_data.csv를 로드한다."""
    cached = DATA_DIR / "forward_consensus.csv"
    if not prefer_remote and local_path and Path(local_path).exists():
        df = pd.read_csv(local_path, encoding="utf-8-sig")
    elif not prefer_remote and cached.exists():
        df = pd.read_csv(cached, encoding="utf-8-sig")
    else:
        df = pd.read_csv(CONSENSUS_URL, encoding="utf-8-sig")
        df.to_csv(cached, index=False, encoding="utf-8-sig")
    # 종목코드는 6자리 zero-pad 문자열로
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
    return df


def load_constituents() -> pd.DataFrame:
    path = DATA_DIR / "etf_constituents.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"etf_code": str})
    df["etf_code"] = df["etf_code"].astype(str).str.zfill(6)
    return df


def build_consensus_lookup(cons_df: pd.DataFrame) -> pd.DataFrame:
    """컨센서스 DF에서 스코어링/표시에 필요한 컬럼만 추출."""
    rename_map = {
        "종목코드": "stock_code",
        "종목명": "stock_name",
        "매출액_2025": "rev_2025",
        "매출액_2026": "rev_2026",
        "매출액_2027": "rev_2027",
        "매출액_2028": "rev_2028",
        "영업이익_2025": "op_2025",
        "영업이익_2026": "op_2026",
        "영업이익_2027": "op_2027",
        "영업이익_2028": "op_2028",
        "매출액_성장률_2026": "rev_growth_2026",
        "매출액_성장률_2027": "rev_growth_2027",
        "매출액_성장률_2028": "rev_growth_2028",
        "영업이익_성장률_2026": "op_growth_2026",
        "영업이익_성장률_2027": "op_growth_2027",
        "영업이익_성장률_2028": "op_growth_2028",
        "PER": "per",
        "PBR": "pbr",
        "ROE": "roe",
        "RSI": "rsi",
        "OBV_trend": "obv_trend",
        "현재가": "price",
        "시가총액": "market_cap_stock",
        "시장": "market",
    }
    available = {k: v for k, v in rename_map.items() if k in cons_df.columns}
    df = cons_df.rename(columns=available)
    return df[list(available.values())]


def score_etfs(min_coverage: float = 0.0) -> pd.DataFrame:
    """ETF별 가중평균 점수를 계산하고 랭킹한다.
    min_coverage 미만 ETF도 포함하지만 표시 시 필터링.
    """
    const_df = load_constituents()
    cons_df = load_consensus()
    lookup = build_consensus_lookup(cons_df)

    merged = const_df.merge(lookup, on="stock_name", how="left")

    results = []
    for etf_code, grp in merged.groupby("etf_code"):
        etf_name = grp["etf_name"].iloc[0]
        total_weight = grp["weight"].sum()

        has_consensus = grp["op_growth_2026"].notna()
        covered_weight = grp.loc[has_consensus, "weight"].sum()
        coverage = covered_weight / total_weight if total_weight > 0 else 0

        covered = grp[has_consensus].copy()
        if len(covered) == 0:
            continue

        w = covered["weight"].values
        w_norm = w / w.sum()

        def wavg(col):
            if col not in covered.columns:
                return np.nan
            vals = covered[col].values
            mask = np.isfinite(vals)
            if mask.sum() == 0:
                return np.nan
            return np.average(vals[mask], weights=w_norm[mask])

        op_g26 = wavg("op_growth_2026")
        op_g27 = wavg("op_growth_2027")
        op_g28 = wavg("op_growth_2028")
        rev_g26 = wavg("rev_growth_2026")
        rev_g27 = wavg("rev_growth_2027")
        rev_g28 = wavg("rev_growth_2028")
        w_per = wavg("per")
        w_pbr = wavg("pbr")
        w_roe = wavg("roe")

        results.append({
            "etf_code": etf_code,
            "etf_name": etf_name,
            "n_stocks": len(grp),
            "n_covered": int(has_consensus.sum()),
            "coverage": round(coverage * 100, 1),
            "rev_growth_2026": round(rev_g26, 2) if np.isfinite(rev_g26) else None,
            "rev_growth_2027": round(rev_g27, 2) if np.isfinite(rev_g27) else None,
            "rev_growth_2028": round(rev_g28, 2) if np.isfinite(rev_g28) else None,
            "op_growth_2026": round(op_g26, 2) if np.isfinite(op_g26) else None,
            "op_growth_2027": round(op_g27, 2) if np.isfinite(op_g27) else None,
            "op_growth_2028": round(op_g28, 2) if np.isfinite(op_g28) else None,
            "fwd_per": round(w_per, 2) if np.isfinite(w_per) else None,
            "fwd_pbr": round(w_pbr, 2) if np.isfinite(w_pbr) else None,
            "fwd_roe": round(w_roe, 2) if np.isfinite(w_roe) else None,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # ── 복합 점수 ──────────────────────────────────────────
    # 영업이익 성장률 (2026/2027/2028) 평균 + PER 역수 가중
    def norm(s, ascending=True):
        s = s.fillna(s.median() if not s.dropna().empty else 0)
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series(50.0, index=s.index)
        return (s - lo) / (hi - lo) * 100 if ascending else (hi - s) / (hi - lo) * 100

    df["score_op26"] = norm(df["op_growth_2026"])
    df["score_op27"] = norm(df["op_growth_2027"])
    df["score_op28"] = norm(df["op_growth_2028"])
    df["score_per"] = norm(df["fwd_per"], ascending=False)

    # 2026: 30%, 2027: 25%, 2028: 25%, PER: 20%
    df["composite_score"] = (
        df["score_op26"] * 0.30
        + df["score_op27"] * 0.25
        + df["score_op28"] * 0.25
        + df["score_per"] * 0.20
    ).round(2)

    df = df.drop(columns=["score_op26", "score_op27", "score_op28", "score_per"])
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    return df


def get_etf_constituents_with_consensus(etf_code: str, top_n: int | None = 10) -> pd.DataFrame:
    """특정 ETF의 구성종목 + 컨센서스 데이터를 비중 내림차순으로 반환."""
    const_df = load_constituents()
    cons_df = load_consensus()
    lookup = build_consensus_lookup(cons_df)

    etf_const = const_df[const_df["etf_code"] == etf_code].copy()
    detail = etf_const.merge(lookup, on="stock_name", how="left")
    detail = detail.sort_values("weight", ascending=False).reset_index(drop=True)
    if top_n:
        detail = detail.head(top_n)
    return detail


def main():
    df = score_etfs()
    out = DATA_DIR / "etf_scores.csv"
    df.to_csv(out, encoding="utf-8-sig")
    print(f"저장 완료: {out} ({len(df)} ETFs)")
    print(df.head(20).to_string())


if __name__ == "__main__":
    main()
