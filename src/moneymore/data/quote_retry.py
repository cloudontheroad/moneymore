"""Bounded execution-only quality retries; never soften a quote's protection."""
import time
import pandas as pd


class RetryingQuotes:
    def __init__(self, provider, *, sleep=time.sleep, attempts=3, now=time.time):
        self.provider = provider
        self.name = provider.name
        self.sleep = sleep
        self.attempts = min(3, max(1, attempts))
        self.last_attempts = 0
        self.now = now

    def _fresh(self, row, max_age):
        return all(0 <= self.now()*1000 - float(row[k]) <= max_age*1000
                   for k in ("timestamp", "depth_timestamp") if k in row)

    def intraday_quotes(self, symbols):
        return self._quotes(symbols, 30)

    def fresh_intraday_quotes(self, symbols):
        return self._quotes(symbols, 10)

    def _quotes(self, symbols, max_age):
        snapshots = {}
        pending = sorted(set(symbols))
        for index in range(self.attempts):
            self.last_attempts = index + 1
            # Previously valid rows may have aged during the backoff itself.
            pending = [s for s in sorted(set(symbols)) if s not in snapshots
                       or not snapshots[s].get('execution_ready') or not self._fresh(snapshots[s], max_age)]
            # Transport/auth/rate-limit exceptions retain the underlying
            # provider's cooldown. Do not hammer them with an outer retry.
            frame = self.provider.intraday_quotes(pending)
            if frame is not None and not frame.empty:
                snapshots.update({r['symbol']: r for r in frame.to_dict('records') if r['symbol'] in symbols})
            pending = [s for s in sorted(set(symbols)) if s not in snapshots
                       or not snapshots[s].get('execution_ready') or not self._fresh(snapshots[s], max_age)]
            if not pending:
                return pd.DataFrame([snapshots[s] for s in sorted(set(symbols))])
            if index + 1 < self.attempts:
                self.sleep(2 * (index + 1))  # exceed the provider's 2s cache TTL
        for row in snapshots.values():
            if not self._fresh(row, max_age):
                row['execution_ready'] = False
                row['quality_reasons'] = str(row.get('quality_reasons','')) + ',RETRY_QUOTE_EXPIRED'
        return pd.DataFrame([snapshots[s] for s in sorted(set(symbols)) if s in snapshots])

    def __getattr__(self, name):
        return getattr(self.provider, name)
