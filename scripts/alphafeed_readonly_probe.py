"""Bounded, opt-in AlphaFeed acceptance probe. Never submits orders or changes providers."""
from __future__ import annotations

import json
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.runtime' / 'alphafeed-sdk'))


def main():
    from dotenv import dotenv_values
    from alphafeed import AlphaFeed

    key = dotenv_values(ROOT / 'state/private/alphafeed.env').get('ALPHAFEED_API_KEY')
    if not key:
        raise SystemExit('Missing private AlphaFeed key')
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--output', default='alphafeed-probe.json')
    args = parser.parse_args()
    if Path(args.output).name != args.output:
        raise SystemExit('Output must be a filename')
    report = {'observed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'mode': 'READ_ONLY', 'results': {}}
    client = AlphaFeed(api_key=key, timeout=15, max_retries=0,
                       cache_dir=str(ROOT / 'state/private/alphafeed-cache'))
    symbols = ['600036.SH', '600900.SH', '600519.SH', '000001.SZ', '300750.SZ',
               '688981.SH', '510300.SH']
    calls = [
        ('quotes', lambda: client.quotes.get(symbols=symbols)),
        ('depth', lambda: client.depth.batch(symbols, max_workers=1)),
        ('instruments', lambda: client.instruments.get(symbols)),
        ('daily', lambda: client.klines.get('600036.SH', period='1d', count=5,
                                           adjust='none', to_dataframe=True)),
        ('intraday', lambda: client.klines.intraday('600036.SH', to_dataframe=True)),
        ('minute_history', lambda: client.klines.get('600036.SH', period='1m', count=300,
                                                    adjust='none', to_dataframe=True)),
    ]
    try:
        for name, call in calls:
            if args.only and name not in args.only:
                continue
            started = time.monotonic()
            try:
                value = call()
                if hasattr(value, 'to_dict'):
                    value = value.to_dict('records')
                report['results'][name] = {'ok': True, 'elapsed_seconds': round(time.monotonic()-started, 3),
                                           'data': value}
                print(name, 'OK', 'rows=', len(value), flush=True)
            except Exception as exc:
                detail = str(exc).replace(key, '[REDACTED]')
                report['results'][name] = {'ok': False, 'type': type(exc).__name__, 'error': detail}
                print(name, type(exc).__name__, detail[:500], flush=True)
                if '401' in detail or '429' in detail:
                    break
            time.sleep(2)
    finally:
        client.close()
    output = ROOT / 'state/private' / args.output
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('Saved read-only report:', output)


if __name__ == '__main__':
    main()
