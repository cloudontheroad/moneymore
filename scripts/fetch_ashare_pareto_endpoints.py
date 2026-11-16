"""Bounded official endpoint supplementation, 2 requests/security, no live writes."""
import json
import time
import pandas as pd
from dotenv import load_dotenv
from filelock import FileLock
from audit_ashare_pareto_inputs import ROOT, OUT
from moneymore.data.tushare_provider import TushareProvider


def main():
    missing = pd.read_parquet(OUT / 'label-repair-v1/unresolved.parquet')
    symbols = sorted(missing.symbol.unique())
    if len(symbols) > 240:
        raise ValueError('ENDPOINT_REQUEST_BUDGET_EXCEEDED')
    folder = OUT / 'endpoint-cache'
    folder.mkdir(exist_ok=True)
    load_dotenv(ROOT / '.env', override=False)
    client = TushareProvider()._client
    completed = []; failures = []
    with FileLock(str(folder / 'request.lock'), timeout=1):
        for symbol in symbols:
            path = folder / (symbol + '.parquet')
            if path.exists():
                completed.append(symbol)
                continue
            scope = missing.loc[missing.symbol.eq(symbol)]
            start = min(scope.entry_date.min(), scope.label_end.min())
            end = max(scope.entry_date.max(), scope.label_end.max())
            try:
                daily = client.daily(ts_code=symbol, start_date=start, end_date=end)
                time.sleep(1.2)
                adj = client.adj_factor(ts_code=symbol, start_date=start, end_date=end)
                time.sleep(1.2)
                cols = ['ts_code', 'trade_date', 'open', 'close', 'adj_factor']
                if daily is None or adj is None or daily.empty or adj.empty:
                    frame = pd.DataFrame(columns=cols)
                else:
                    frame = daily.merge(adj[['ts_code', 'trade_date', 'adj_factor']], on=['ts_code', 'trade_date'], validate='one_to_one')
                    if not frame.ts_code.eq(symbol).all():
                        raise ValueError('UNEXPECTED_SECURITY')
                frame.to_parquet(path, index=False)
                (folder / (symbol + '.json')).write_text(json.dumps({'source': 'Tushare daily + adj_factor', 'symbol': symbol, 'start': start, 'end': end, 'rows': len(frame), 'empty_not_zero_return': frame.empty}), encoding='utf-8')
                completed.append(symbol)
            except Exception as error:
                failures.append({'symbol': symbol, 'error_type': type(error).__name__})
                # No error bodies or credentials in logs, no runaway retry loops.
                if len(failures) >= 3:
                    break
            progress = {'total_symbols': len(symbols), 'completed': len(completed), 'failures': failures, 'last': symbol, 'max_requests': 480, 'production_modified': False}
            (folder / 'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
            if len(completed) % 10 == 0:
                print(progress, flush=True)
        (folder / 'result.json').write_text(json.dumps({'total': len(symbols), 'completed': len(completed), 'failures': failures, 'complete': len(completed)==len(symbols)}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
