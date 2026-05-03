# ETF Forward 랭커

국내 주식형 ETF의 구성종목 forward 컨센서스(영업이익 성장률, PER)를 비중 가중평균하여 순위를 매기는 Streamlit 앱.

## 데이터 소스

- **ETF 구성종목/비중**: 네이버금융 + wisereport
- **Forward 컨센서스**: [perseus2133-ai/ai2](https://github.com/perseus2133-ai/ai2) 레포의 `data/consensus_data.csv`

## 스코어링 로직

ETF별 복합 점수 = 구성종목 비중 × 종목 forward 메트릭의 가중평균
- 2026E 영업이익 성장률 (40%)
- 2027E 영업이익 성장률 (30%)
- Forward PER 역수 (30%)

컨센서스 커버리지 70% 미만 ETF는 기본 필터에서 제외.

## 실행

```bash
pip install -r requirements.txt

# 크롤링 + 스코어 계산
python crawl_etf.py
python score_etf.py

# 앱 실행
streamlit run app.py
```

## 자동 갱신

`.github/workflows/daily_crawl.yml`이 매주 평일 16:00 KST에 크롤링하고 결과를 commit/push.

## 배포

Streamlit Cloud에 연결:
1. https://share.streamlit.io 접속
2. 본 레포 연결, main 브랜치, `app.py` 지정
3. Python 3.12 권장
