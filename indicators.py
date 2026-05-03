"""
종목/ETF 기술 지표 계산 모듈
- 네이버 일간 시세를 받아 RSI(14), OBV trend, 60일 가격 차트 등을 계산.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

DAILY_URL = "https://finance.naver.com/item/sise_day.naver?code={code}&page={page}"


def fetch_daily_prices(stock_code: str, days: int = 90) -> tuple[list[int], list[int]]:
    """네이버 일별 시세에서 (종가 리스트, 거래량 리스트)를 최신순으로 반환."""
    prices, volumes = [], []
    pages_needed = (days // 10) + 2
    for page in range(1, pages_needed + 1):
        try:
            r = requests.get(
                DAILY_URL.format(code=stock_code, page=page),
                headers=HEADERS, timeout=10
            )
            r.encoding = "euc-kr"
            html = r.text
        except Exception:
            break

        # 테이블 행에서 종가/거래량 추출
        # <td class="num"><span class="tah ...">CLOSE</span></td>
        # 행마다 6개 td: 날짜, 종가, 전일비, 시가, 고가, 저가, 거래량
        rows = re.findall(r"<tr[^>]*onmouseover[^>]*>(.*?)</tr>", html, re.DOTALL)
        if not rows:
            # 다른 셀렉터 시도: 날짜 행 패턴
            rows = re.findall(
                r'<td align="center"><span class="tah[^>]*>(\d{4}\.\d{2}\.\d{2})</span></td>'
                r'.*?<td class="num"><span[^>]*>([\d,]+)</span></td>'
                r'(?:.*?<td class="num">.*?</td>){4}'
                r'.*?<td class="num"><span[^>]*>([\d,]+)</span></td>',
                html, re.DOTALL
            )
            for _, close, vol in rows:
                try:
                    prices.append(int(close.replace(",", "")))
                    volumes.append(int(vol.replace(",", "")))
                except ValueError:
                    continue
            if len(prices) >= days:
                break
            continue

        for row in rows:
            nums = re.findall(r'<span class="tah[^>]*>([\d,\.]+)</span>', row)
            if len(nums) < 6:
                continue
            try:
                close = int(nums[1].replace(",", ""))
                vol = int(nums[5].replace(",", ""))
                prices.append(close)
                volumes.append(vol)
            except (ValueError, IndexError):
                continue
        if len(prices) >= days:
            break

    return prices[:days], volumes[:days]


def calc_rsi(prices: list[int], period: int = 14) -> float | None:
    """Wilder's RSI(14). prices는 최신순(현재→과거)."""
    if len(prices) < period + 1:
        return None
    p = list(reversed(prices))  # 과거→현재
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


def calc_obv_trend(prices: list[int], volumes: list[int]) -> str:
    """OBV 누적 후 최근 10일 추세 (up/down/flat)."""
    if len(prices) < 11:
        return ""
    p = list(reversed(prices))
    v = list(reversed(volumes))
    obv = [0]
    for i in range(1, len(p)):
        if p[i] > p[i - 1]:
            obv.append(obv[-1] + v[i])
        elif p[i] < p[i - 1]:
            obv.append(obv[-1] - v[i])
        else:
            obv.append(obv[-1])
    n = min(10, len(obv))
    diff = obv[-1] - obv[-n]
    if diff > 0:
        return "up"
    if diff < 0:
        return "down"
    return "flat"


def support_resistance(prices: list[int], lookback: int = 60) -> tuple[int | None, int | None]:
    if not prices:
        return None, None
    recent = prices[:lookback]
    return max(recent), min(recent)


def analyze(stock_code: str, days: int = 90) -> dict[str, Any]:
    """한 번에 RSI/OBV/지지·저항/가격 시리즈를 모두 계산."""
    prices, volumes = fetch_daily_prices(stock_code, days)
    out = {
        "prices": prices,
        "volumes": volumes,
        "rsi": calc_rsi(prices),
        "obv_trend": calc_obv_trend(prices, volumes),
    }
    res, sup = support_resistance(prices)
    out["resistance"] = res
    out["support"] = sup
    out["last_price"] = prices[0] if prices else None
    return out


def rsi_verdict(rsi: float | None) -> tuple[str, str]:
    """RSI에 대한 한 줄 코멘트와 색."""
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
    res = analyze(code)
    print(f"Code: {code}")
    print(f"Last price: {res['last_price']}, RSI: {res['rsi']}, OBV: {res['obv_trend']}")
    print(f"Support: {res['support']}, Resistance: {res['resistance']}")
    print(f"Prices fetched: {len(res['prices'])}")
