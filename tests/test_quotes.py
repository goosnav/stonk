import pandas as pd

from specforge.broker.robinhood_mcp import RobinhoodMCPBroker
from specforge.quotes import QuoteService


def test_robinhood_quotes_are_chunked_at_fifty():
    broker = object.__new__(RobinhoodMCPBroker)
    calls = []

    def call(_tool, args):
        calls.append(list(args["symbols"]))
        return {"results": [{"symbol": symbol, "last_trade_price": "10.5"}
                            for symbol in args["symbols"]]}

    broker._call = call
    symbols = [f"S{i}" for i in range(121)]
    result = broker._get_quotes_uncached(symbols)
    assert [len(batch) for batch in calls] == [50, 50, 21]
    assert len(result) == len(symbols)


def test_yfinance_fallback_is_one_batch(cfg, monkeypatch):
    import yfinance as yf
    calls = []

    def download(symbols, **kwargs):
        calls.append(symbols)
        columns = pd.MultiIndex.from_product([symbols, ["Close"]])
        return pd.DataFrame([[9.0] * len(symbols), [10.0] * len(symbols)],
                            columns=columns)

    monkeypatch.setattr(yf, "download", download)
    service = QuoteService(cfg)
    result = service._yfinance(["AAA", "BBB"])
    assert len(calls) == 1
    assert result["AAA"]["price"] == 10.0
    assert result["AAA"]["change_pct"] == 0.1111
