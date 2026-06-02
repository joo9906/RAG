"""MCP server for real-time stock data using yfinance.

Run standalone:
    python mcp_server/stock_mcp_server.py
"""

import json
from typing import Any

import yfinance as yf
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Financial Stock Server")


def _safe_float(value: Any, precision: int = 2) -> float | str:
    try:
        return round(float(value), precision)
    except (TypeError, ValueError):
        return "N/A"


@mcp.tool()
def get_stock_price(ticker: str) -> str:
    """주식 티커 심볼로 현재가, 시가, 고가, 저가, 거래량을 조회합니다.
    예: AAPL (애플), 005930.KS (삼성전자), TSLA (테슬라)
    """
    stock = yf.Ticker(ticker.upper())
    hist = stock.history(period="1d")

    if hist.empty:
        return f"'{ticker}' 티커의 가격 데이터를 찾을 수 없습니다. 티커 심볼을 확인해 주세요."

    latest = hist.iloc[-1]
    info = stock.info

    currency = info.get("currency", "USD")
    name = info.get("longName") or info.get("shortName") or ticker

    result = {
        "ticker": ticker.upper(),
        "name": name,
        "currency": currency,
        "current_price": _safe_float(latest["Close"]),
        "open": _safe_float(latest["Open"]),
        "high": _safe_float(latest["High"]),
        "low": _safe_float(latest["Low"]),
        "volume": int(latest["Volume"]),
        "prev_close": _safe_float(info.get("previousClose")),
    }

    # 등락률 계산
    if result["prev_close"] != "N/A" and result["prev_close"] != 0:
        change = result["current_price"] - result["prev_close"]
        change_pct = (change / result["prev_close"]) * 100
        result["change"] = _safe_float(change)
        result["change_pct"] = _safe_float(change_pct)
    else:
        result["change"] = "N/A"
        result["change_pct"] = "N/A"

    lines = [
        f"📈 {result['name']} ({result['ticker']})",
        f"현재가: {result['current_price']:,} {currency}",
        f"시가: {result['open']:,} | 고가: {result['high']:,} | 저가: {result['low']:,}",
        f"거래량: {result['volume']:,}",
    ]
    if result["change"] != "N/A":
        arrow = "▲" if result["change"] >= 0 else "▼"
        lines.append(
            f"전일대비: {arrow} {abs(result['change']):,} ({result['change_pct']:+.2f}%)"
        )
    return "\n".join(lines)


@mcp.tool()
def get_stock_info(ticker: str) -> str:
    """주식의 기업 개요, 시가총액, PER, 배당수익률 등 기본 정보를 조회합니다."""
    stock = yf.Ticker(ticker.upper())
    info = stock.info

    if not info or "symbol" not in info:
        return f"'{ticker}' 정보를 가져올 수 없습니다."

    currency = info.get("currency", "USD")

    def fmt_large(n: Any) -> str:
        try:
            n = float(n)
            if n >= 1e12:
                return f"{n/1e12:.2f}조"
            if n >= 1e8:
                return f"{n/1e8:.2f}억"
            return f"{n:,.0f}"
        except (TypeError, ValueError):
            return "N/A"

    fields = {
        "종목명": info.get("longName") or info.get("shortName", ticker),
        "거래소": info.get("exchange", "N/A"),
        "섹터": info.get("sector", "N/A"),
        "산업": info.get("industry", "N/A"),
        "시가총액": fmt_large(info.get("marketCap")),
        "52주 최고": _safe_float(info.get("fiftyTwoWeekHigh")),
        "52주 최저": _safe_float(info.get("fiftyTwoWeekLow")),
        "PER": _safe_float(info.get("trailingPE")),
        "PBR": _safe_float(info.get("priceToBook")),
        "배당수익률": (
            f"{round(info['dividendYield']*100, 2)}%"
            if info.get("dividendYield")
            else "없음"
        ),
        "EPS": _safe_float(info.get("trailingEps")),
        "ROE": (
            f"{round(info['returnOnEquity']*100, 2)}%"
            if info.get("returnOnEquity")
            else "N/A"
        ),
    }

    lines = [f"🏢 기업 정보: {fields['종목명']} ({ticker.upper()})"]
    lines += [f"  {k}: {v}" for k, v in fields.items() if k != "종목명"]

    summary = info.get("longBusinessSummary", "")
    if summary:
        lines.append(f"\n📝 기업 개요: {summary[:300]}...")

    return "\n".join(lines)


@mcp.tool()
def get_stock_history(ticker: str, period: str = "1mo") -> str:
    """주식의 과거 가격 이력을 조회합니다.

    Args:
        ticker: 주식 티커 심볼
        period: 조회 기간. 가능한 값: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y
    """
    valid_periods = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"}
    if period not in valid_periods:
        return f"유효하지 않은 기간입니다. 다음 중 선택하세요: {', '.join(sorted(valid_periods))}"

    stock = yf.Ticker(ticker.upper())
    hist = stock.history(period=period)

    if hist.empty:
        return f"'{ticker}' 이력 데이터를 찾을 수 없습니다."

    start_price = hist["Close"].iloc[0]
    end_price = hist["Close"].iloc[-1]
    change_pct = ((end_price - start_price) / start_price) * 100

    high = hist["High"].max()
    low = hist["Low"].min()
    avg_vol = hist["Volume"].mean()

    arrow = "▲" if change_pct >= 0 else "▼"
    lines = [
        f"📊 {ticker.upper()} — {period} 이력",
        f"시작가: {start_price:,.2f} → 종료가: {end_price:,.2f}",
        f"수익률: {arrow} {change_pct:+.2f}%",
        f"기간 내 최고가: {high:,.2f} | 최저가: {low:,.2f}",
        f"평균 거래량: {avg_vol:,.0f}",
    ]

    # 최근 5일 데이터
    lines.append("\n최근 5거래일:")
    for date, row in hist.tail(5).iterrows():
        lines.append(
            f"  {date.strftime('%Y-%m-%d')}: {row['Close']:,.2f} (거래량 {int(row['Volume']):,})"
        )

    return "\n".join(lines)


@mcp.tool()
def compare_stocks(tickers: str) -> str:
    """여러 주식을 비교합니다. 쉼표로 구분된 티커 심볼을 입력하세요.
    예: AAPL,MSFT,GOOGL
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(ticker_list) < 2:
        return "비교를 위해 쉼표로 구분된 2개 이상의 티커가 필요합니다."
    if len(ticker_list) > 5:
        return "최대 5개 종목까지 비교 가능합니다."

    rows = []
    for ticker in ticker_list:
        try:
            info = yf.Ticker(ticker).info
            rows.append(
                {
                    "ticker": ticker,
                    "name": (info.get("shortName") or ticker)[:20],
                    "price": _safe_float(info.get("currentPrice") or info.get("regularMarketPrice")),
                    "change_pct": _safe_float(
                        info.get("regularMarketChangePercent", 0), 2
                    ),
                    "market_cap": info.get("marketCap", 0),
                    "per": _safe_float(info.get("trailingPE")),
                    "dividend": (
                        f"{round(info['dividendYield']*100,2)}%"
                        if info.get("dividendYield")
                        else "-"
                    ),
                }
            )
        except Exception:
            rows.append({"ticker": ticker, "name": "오류", "price": "N/A", "change_pct": "N/A", "market_cap": 0, "per": "N/A", "dividend": "N/A"})

    lines = ["📊 종목 비교", f"{'티커':<10} {'종목명':<22} {'현재가':>10} {'등락%':>8} {'PER':>6} {'배당':>6}"]
    lines.append("-" * 70)
    for r in rows:
        lines.append(
            f"{r['ticker']:<10} {r['name']:<22} {str(r['price']):>10} {str(r['change_pct']):>8} {str(r['per']):>6} {r['dividend']:>6}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
