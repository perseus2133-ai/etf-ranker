"""
종목/ETF 기술 지표 계산 모듈
- 네이버 JSON API로 5~10년치 일봉 OHLCV를 한 번에 받아온 뒤,
  일봉/주봉/월봉으로 리샘플링하고 RSI(14) / OBV trend / 지지·저항 계산.
"""

from __future__ import annotations

import datetime
import json
import re
from typing import Any, Literal

import pandas as pd
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# 네이버 JSON 시세 API (한 번에 수년치 받아옴)
SISE_JSON_URL = (
    "https://api.finance.naver.com/siseJson.naver"
    "?symbol={code}&requestType=1&startTime={start}&endTime={end}&timeframe=day"
)

Interval = Literal["D", "W", "M"]


def fetch_daily_ohlcv(stock_code: str, years: int = 11) -> pd.DataFrame:
    """네이버 JSON API에서 (date, close, volume) DataFrame을 최신순으로 반환.
    years: 거슬러 받아올 연도 수 (월봉 10년치 표시 위해 기본 11년)."""
    end = datetime.date.today()
    start = end.replace(year=end.year - years)
    url = SISE_JSON_URL.format(
        code=stock_code,
        start=start.strftime("%Y%m%d"),
        end=end.strftime("%Y%m%d"),
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
    except Exception:
        return pd.DataFrame()

    text = r.text.strip()
    # API가 작은따옴표 + 줄바꿈 포함 JSON-like를 반환 → JSON 호환으로 변환
    text = text.replace("'", '"')
    text = re.sub(r"\s+", " ", text)
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        return pd.DataFrame()

    if not rows or len(rows) < 2:
        return pd.DataFrame()

    # 첫 행은 헤더: ["날짜","시가","고가","저가","종가","거래량","외국인소진율"]
    header = rows[0]
    data_rows = rows[1:]
    if len(header) < 6:
        return pd.DataFrame()

    records = []
    for row in data_rows:
        if len(row) < 6:
            continue
        try:
            date = pd.to_datetime(str(row[0]), format="%Y%m%d")
            close = int(row[4])
            volume = int(row[5])
        except (ValueError, IndexError, TypeError):
            continue
        records.append({"date": date, "close": close, "volume": volume})

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.drop_duplicates(subset="date").sort_values("date", ascending=False).reset_index(drop=True)
    return df


def resample_ohlcv(daily_df: pd.DataFrame, interval: Interval) -> pd.DataFrame:
    """일봉을 주봉/월봉으로 리샘플링. 입력은 최신순, 출력도 최신순."""
    if daily_df.empty or interval == "D":
        return daily_df.copy()

    # 시간순 정렬 후 리샘플
    df = daily_df.sort_values("date").set_index("date")
    rule = "W-FRI" if interval == "W" else "ME"
    agg = df.resample(rule).agg({"close": "last", "volume": "sum"}).dropna(subset=["close"])
    agg["volume"] = agg["volume"].astype(int)
    agg["close"] = agg["close"].astype(int)
    agg = agg.reset_index().sort_values("date", ascending=False).reset_index(drop=True)
    return agg


def calc_rsi(closes: list[int], period: int = 14) -> float | None:
    """Wilder's RSI. closes는 최신순(현재→과거)."""
    if len(closes) < period + 1:
        return None
    p = list(reversed(closes))  # 과거→현재
    gains, losses = [], []
    for i in range(1, len(p)):
        d = p[i] - p[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    if len(gains) < period:
        return None
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 1)


def calc_obv_trend(closes: list[int], volumes: list[int], lookback: int = 10) -> str:
    """OBV 누적 후 최근 N봉 추세 (up/down/flat)."""
    if len(closes) < lookback + 1:
        return ""
    p = list(reversed(closes))
    v = list(reversed(volumes))
    obv = [0]
    for i in range(1, len(p)):
        if p[i] > p[i - 1]:
            obv.append(obv[-1] + v[i])
        elif p[i] < p[i - 1]:
            obv.append(obv[-1] - v[i])
        else:
            obv.append(obv[-1])
    n = min(lookback, len(obv))
    diff = obv[-1] - obv[-n]
    if diff > 0:
        return "up"
    if diff < 0:
        return "down"
    return "flat"


def support_resistance(closes: list[int], lookback: int = 60) -> tuple[int | None, int | None]:
    if not closes:
        return None, None
    recent = closes[:lookback]
    return max(recent), min(recent)


# 인터벌별 기본 표시 봉 수 / 지지·저항 룩백 — 네이버 증권 기본 차트와 유사
DEFAULTS: dict[str, dict[str, int]] = {
    "D": {"max_bars": 130, "support_lookback": 60},   # 일봉: ~6개월, 60일 지지/저항
    "W": {"max_bars": 156, "support_lookback": 52},   # 주봉: ~3년, 1년 지지/저항
    "M": {"max_bars": 120, "support_lookback": 60},   # 월봉: ~10년, 5년 지지/저항
}

# 인터벌별 데이터 수집 연도 (10년 월봉을 위해 충분히)
FETCH_YEARS: dict[str, int] = {"D": 1, "W": 4, "M": 11}


def analyze(stock_code: str, interval: Interval = "D",
            max_bars: int | None = None, years: int | None = None) -> dict[str, Any]:
    """일봉/주봉/월봉 가격 시리즈 + RSI + OBV trend + 지지·저항을 한 번에 계산."""
    cfg = DEFAULTS.get(interval, DEFAULTS["D"])
    if max_bars is None:
        max_bars = cfg["max_bars"]
    if years is None:
        years = FETCH_YEARS.get(interval, 11)

    daily = fetch_daily_ohlcv(stock_code, years=years)
    out = {
        "interval": interval,
        "dates": [], "prices": [], "volumes": [],
        "rsi": None, "obv_trend": "",
        "support": None, "resistance": None,
        "last_price": None, "bars_count": 0,
    }
    if daily.empty:
        return out

    bars = resample_ohlcv(daily, interval).head(max_bars)
    closes = bars["close"].tolist()
    vols = bars["volume"].tolist()
    dates = bars["date"].tolist()

    out["dates"] = dates
    out["prices"] = closes
    out["volumes"] = vols
    out["bars_count"] = len(closes)
    out["rsi"] = calc_rsi(closes)
    out["obv_trend"] = calc_obv_trend(closes, vols)
    res, sup = support_resistance(closes, lookback=min(cfg["support_lookback"], len(closes)))
    out["resistance"] = res
    out["support"] = sup
    out["last_price"] = closes[0] if closes else None
    return out


def rsi_verdict(rsi: float | None) -> tuple[str, str]:
    if rsi is None:
        return "데이터 없음", "#94A3B8"
    if rsi >= 70:
        return f"과매수 (RSI {rsi:.0f})", "#FB7185"
    if rsi <= 30:
        return f"과매도 (RSI {rsi:.0f})", "#60A5FA"
    if rsi >= 55:
        return f"강세 (RSI {rsi:.0f})", "#34D399"
    if rsi <= 45:
        return f"약세 (RSI {rsi:.0f})", "#FBBF24"
    return f"중립 (RSI {rsi:.0f})", "#CBD5E0"


def obv_verdict(trend: str) -> tuple[str, str]:
    return {
        "up": ("OBV 상승 (자금 유입)", "#34D399"),
        "down": ("OBV 하락 (자금 유출)", "#FB7185"),
        "flat": ("OBV 횡보", "#94A3B8"),
        "": ("OBV 데이터 없음", "#94A3B8"),
    }.get(trend, ("OBV 알 수 없음", "#94A3B8"))


if __name__ == "__main__":
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "069500"
    for iv in ["D", "W", "M"]:
        res = analyze(code, interval=iv)
        print(f"\n[{iv}] bars={res['bars_count']}  last={res['last_price']:,}  "
              f"RSI={res['rsi']}  OBV={res['obv_trend']}  "
              f"S={res['support']:,}  R={res['resistance']:,}")
