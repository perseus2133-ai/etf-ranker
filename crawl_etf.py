"""
ETF 구성종목 크롤러
- 네이버금융 API에서 전체 ETF 목록을 가져온 뒤,
  wisereport에서 각 ETF의 CU당 구성종목/비중을 크롤링한다.
- 결과를 data/etf_constituents.csv 에 저장한다.
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
}

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

ETF_LIST_URL = "https://finance.naver.com/api/sise/etfItemList.nhn"
WISEREPORT_URL = (
    "https://navercomp.wisereport.co.kr/v2/ETF/index.aspx?cmp_cd={code}"
)

# 해외 ETF 키워드 — 국내 주식 구성종목이 없으므로 제외
FOREIGN_KEYWORDS = [
    "미국", "나스닥", "S&P", "글로벌", "중국", "일본", "유럽",
    "선진국", "신흥국", "인도", "베트남", "MSCI", "다우", "항셍",
    "해외", "달러", "엔화", "위안", "미국채", "국채", "채권",
    "금", "은", "원유", "WTI", "천연가스", "구리", "원자재",
    "리츠", "REIT", "TDF", "MMF", "단기자금", "머니마켓",
    "레버리지", "인버스", "2X", "곱버스",
]


def fetch_etf_list() -> pd.DataFrame:
    """네이버 API에서 전체 ETF 리스트를 가져온다."""
    r = requests.get(ETF_LIST_URL, headers=HEADERS, timeout=15)
    r.raise_for_status()
    items = r.json()["result"]["etfItemList"]
    df = pd.DataFrame(items)
    df = df.rename(columns={
        "itemcode": "etf_code",
        "itemname": "etf_name",
        "marketSum": "market_cap",
    })
    return df[["etf_code", "etf_name", "market_cap"]]


def is_domestic_equity_etf(name: str) -> bool:
    """해외/채권/원자재/레버리지/인버스 ETF를 걸러낸다."""
    for kw in FOREIGN_KEYWORDS:
        if kw in name:
            return False
    return True


def fetch_constituents(etf_code: str) -> list[dict]:
    """wisereport에서 CU당 구성종목 JSON을 파싱한다."""
    url = WISEREPORT_URL.format(code=etf_code)
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    html = r.content.decode("utf-8", errors="replace")

    m = re.search(r"var CU_data\s*=\s*({.*?});", html, re.DOTALL)
    if not m:
        return []

    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []

    rows = []
    for item in data.get("grid_data", []):
        name = item.get("STK_NM_KOR", "").strip()
        weight = item.get("ETF_WEIGHT")
        if name and weight and weight > 0:
            rows.append({"stock_name": name, "weight": weight})
    return rows


def crawl_all(min_market_cap: int = 500) -> pd.DataFrame:
    """
    전체 국내 주식형 ETF의 구성종목을 크롤링한다.
    min_market_cap: 시총 하한 (억원) — 너무 소형은 제외.
    """
    log.info("ETF 목록 조회 중...")
    etf_df = fetch_etf_list()
    log.info("전체 ETF 수: %d", len(etf_df))

    etf_df = etf_df[etf_df["etf_name"].apply(is_domestic_equity_etf)]
    etf_df = etf_df[etf_df["market_cap"] >= min_market_cap]
    log.info("필터 후 국내 주식형 ETF 수: %d", len(etf_df))

    all_rows = []
    for i, row in enumerate(etf_df.itertuples(), 1):
        code, name = row.etf_code, row.etf_name
        try:
            constituents = fetch_constituents(code)
            for c in constituents:
                all_rows.append({
                    "etf_code": code,
                    "etf_name": name,
                    "stock_name": c["stock_name"],
                    "weight": c["weight"],
                })
            log.info("[%d/%d] %s (%s) - %d stocks",
                     i, len(etf_df), name, code, len(constituents))
        except Exception as e:
            log.warning("[%d/%d] %s (%s) 실패: %s", i, len(etf_df), name, code, e)

        if i % 20 == 0:
            time.sleep(1)

    df = pd.DataFrame(all_rows)
    return df


def main():
    df = crawl_all()
    out = DATA_DIR / "etf_constituents.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info("저장 완료: %s (%d rows)", out, len(df))


if __name__ == "__main__":
    main()
