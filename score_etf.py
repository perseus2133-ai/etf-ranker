"""
ETF 스코어링 엔진
- etf_constituents.csv  (구성종목/비중)
- forward_consensus.csv (ai2 레포에서 다운로드)
를 조인하여 ETF별 가중평균 점수를 계산한다.

메트릭:
  1) 2026E 영업이익 성장률 (가중평균)
  2) 2027E 영업이익 성장률 (가중평균)
  3) Forward PER (가중평균의 역수로 랭킹)
  4) 컨센서스 커버리지 (비중 기준 %)
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

CONSENSUS_URL = (
    "https://raw.githubusercontent.com/perseus2133-ai/ai2/main/data/consensus_data.csv"
)


def load_consensus(local_path: Path | None = None) -> pd.DataFrame:
    """ai2 레포의 forward_consensus.csv를 로드한다."""
    if local_path and local_path.exists():
        df = pd.read_csv(local_path, encoding="utf-8-sig")
    else:
        df = pd.read_csv(CONSENSUS_URL, encoding="utf-8-sig")
        out = DATA_DIR / "forward_consensus.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


def load_constituents() -> pd.DataFrame:
    path = DATA_DIR / "etf_constituents.csv"
    return pd.read_csv(path, encoding="utf-8-sig")


def build_consensus_lookup(cons_df: pd.DataFrame) -> pd.DataFrame:
    """컨센서스 DF에서 스코어링에 필요한 컬럼만 추출."""
    cols = {
        "종목명": "stock_name",
        "영업이익_성장률_2026": "op_growth_2026",
        "영업이익_성장률_2027": "op_growth_2027",
        "PER": "per",
        "시가총액": "market_cap_stock",
    }
    available = {k: v for k, v in cols.items() if k in cons_df.columns}
    df = cons_df.rename(columns=available)
    keep = [v for v in available.values() if v in df.columns]
    return df[keep]


def score_etfs(
    min_coverage: float = 0.70,
) -> pd.DataFrame:
    """ETF별 가중평균 점수를 계산하고 랭킹한다."""
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
            vals = covered[col].values
            mask = np.isfinite(vals)
            if mask.sum() == 0:
                return np.nan
            return np.average(vals[mask], weights=w_norm[mask])

        op_g26 = wavg("op_growth_2026")
        op_g27 = wavg("op_growth_2027") if "op_growth_2027" in covered.columns else np.nan
        w_per = wavg("per") if "per" in covered.columns else np.nan

        n_stocks = len(grp)
        n_covered = int(has_consensus.sum())

        results.append({
            "etf_code": etf_code,
            "etf_name": etf_name,
            "n_stocks": n_stocks,
            "n_covered": n_covered,
            "coverage": round(coverage * 100, 1),
            "op_growth_2026": round(op_g26, 2) if np.isfinite(op_g26) else None,
            "op_growth_2027": round(op_g27, 2) if op_g27 is not None and np.isfinite(op_g27) else None,
            "fwd_per": round(w_per, 2) if w_per is not None and np.isfinite(w_per) else None,
        })

    result_df = pd.DataFrame(results)

    # 복합 점수: 2026 영업이익 성장률 40% + 2027 영업이익 성장률 30% + PER 역수 30%
    df = result_df.copy()
    if len(df) == 0:
        return df

    # 정규화 (min-max → 0~100)
    def norm(series, ascending=True):
        s = series.copy()
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series(50.0, index=s.index)
        if ascending:
            return (s - lo) / (hi - lo) * 100
        return (hi - s) / (hi - lo) * 100

    df["score_op26"] = norm(df["op_growth_2026"].fillna(0))
    df["score_op27"] = norm(df["op_growth_2027"].fillna(0))
    df["score_per"] = norm(df["fwd_per"].fillna(df["fwd_per"].max()), ascending=False)

    df["composite_score"] = (
        df["score_op26"] * 0.40
        + df["score_op27"] * 0.30
        + df["score_per"] * 0.30
    ).round(2)

    df = df.drop(columns=["score_op26", "score_op27", "score_per"])
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"

    return df


def main():
    df = score_etfs()
    out = DATA_DIR / "etf_scores.csv"
    df.to_csv(out, encoding="utf-8-sig")
    print(f"저장 완료: {out} ({len(df)} ETFs)")
    print(df.head(20).to_string())


if __name__ == "__main__":
    main()
