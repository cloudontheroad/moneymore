"""Read-only replacement contract check, including bounded historical recovery."""

import json
from datetime import datetime
from pathlib import Path

from moneymore.data.alphafeed_provider import SHANGHAI, AlphaFeedProvider


def main():
    root = Path(__file__).resolve().parents[1]
    provider = AlphaFeedProvider(root)
    symbols = [
        "600036.SH",
        "600900.SH",
        "600519.SH",
        "000001.SZ",
        "300750.SZ",
        "688981.SH",
        "510300.SH",
    ]
    quotes = provider.intraday_quotes(symbols)
    assert set(quotes.symbol) == set(symbols)
    assert quotes.vwap.gt(0).all()
    assert quotes.bid1.gt(0).all() and quotes.ask1.ge(quotes.bid1).all()
    now = datetime.now(SHANGHAI)
    day = now.strftime("%Y-%m-%d")
    bars = provider.minute_history(["600036.SH"], f"{day} 09:00", f"{day} 16:00")
    assert not bars.empty
    assert not bars.duplicated(["symbol", "timestamp"]).any()
    assert (bars.high + 1e-8 >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars.low - 1e-8 <= bars[["open", "close"]].min(axis=1)).all()
    payload = {
        "observed_at": now.isoformat(),
        "mode": "READ_ONLY",
        "production_switched": False,
        "minute_rows": len(bars),
        "quotes": quotes.to_dict("records"),
        "live_session_acceptance": "NOT_PROVEN_BY_AFTER_HOURS_TEST",
    }
    path = root / "state/private/alphafeed-adapter-acceptance.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "quotes": len(quotes),
                "minutes": len(bars),
                "ready": int(quotes.execution_ready.sum()),
                "report": str(path),
            }
        )
    )


if __name__ == "__main__":
    main()
