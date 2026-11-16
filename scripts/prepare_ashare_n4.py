"""Bounded official historical data, only isolated N4 cache; no credentials in output."""
import json
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from moneymore.data.tushare_provider import TushareProvider
from moneymore.research.ashare_mainline import sha256
from prepare_four_strategy_2026 import save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'state/ashare-mainline-20261112/n4-data-v1'


def main():
    calendar = pd.DatetimeIndex(pd.to_datetime((ROOT / 'state/mature-multifactor/p1-native-20261103-a/provider/calendars/day.txt').read_text().splitlines())).strftime('%Y%m%d')
    days = [d for d in calendar if '20250630' <= d <= '20251231']
    load_dotenv(ROOT / '.env')
    client = TushareProvider()._client
    rows = []
    for day in days:
        for endpoint, param, nonempty in [('daily_basic', 'trade_date', True), ('stk_limit', 'trade_date', True), ('suspend_d', 'trade_date', False), ('dividend', 'record_date', False)]:
            dest = OUT / 'official' / f'{endpoint}-{day}.parquet'
            dest.parent.mkdir(parents=True, exist_ok=True)
            original = ROOT / 'state/four-strategy-2026-jan-sep-v1/official' / dest.name
            if not dest.exists() and original.exists():
                dest.write_bytes(original.read_bytes())
            status = 'CACHED'
            if not dest.exists():
                for attempt in range(2):
                    try:
                        result = getattr(client, endpoint)(**{param: day})
                        if result is None or (nonempty and len(result) < 1000):
                            raise ValueError('EMPTY_OR_TRUNCATED')
                        if not result.empty and not result[param].astype(str).eq(day).all():
                            raise ValueError('WRONG_DATE')
                        temporary = dest.with_suffix('.tmp')
                        result.to_parquet(temporary, index=False)
                        temporary.replace(dest)
                        status = 'FETCHED'
                        break
                    except Exception as error:
                        if attempt:
                            save(OUT / 'progress.json', {'status': 'FAILED', 'day': day, 'endpoint': endpoint, 'error_type': type(error).__name__, 'completed': len(rows)})
                            raise RuntimeError('OFFICIAL_FETCH_FAILED_ERROR_BODY_SUPPRESSED') from None
                        time.sleep(3)
                    finally:
                        time.sleep(1.1)
            result = pd.read_parquet(dest)
            if (nonempty and len(result) < 1000) or (not result.empty and not result[param].astype(str).eq(day).all()):
                raise ValueError('INVALID_CACHED_OFFICIAL')
            rows.append({'file': dest.name, 'rows': len(result), 'status': status, 'sha256': sha256(dest)})
            save(OUT / 'progress.json', {'status': 'RUNNING', 'completed': len(rows), 'total': len(days)*4, 'day': day})
        print(json.dumps({'day': day, 'completed': len(rows)}), flush=True)
    save(OUT / 'results.json', {'status': 'COMPLETE', 'days': days, 'files': rows})
    save(OUT / 'progress.json', {'status': 'COMPLETE', 'completed': len(rows), 'total': len(rows)})


if __name__ == '__main__':
    main()
