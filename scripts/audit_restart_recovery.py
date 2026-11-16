"""Read-only ledger/service proof; writes only a recovery evidence report."""
import json
import sqlite3
from pathlib import Path
from dataclasses import asdict
from urllib.request import urlopen
from moneymore.execution.paper import PaperBroker
from moneymore.challenger_diagnostics import write_diagnostic

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];db=root/'state/paper_orders.sqlite3'
    with sqlite3.connect('file:'+db.as_posix()+'?mode=ro',uri=True) as c:
        integrity=c.execute('PRAGMA quick_check').fetchone()[0]
        fills=[dict(account_id=a,count=n) for a,n in c.execute('SELECT account_id,COUNT(*) FROM fills WHERE trade_date=? GROUP BY account_id',('20261111',))]
        pending=[dict(account_id=a,count=n) for a,n in c.execute("SELECT account_id,COUNT(*) FROM orders WHERE status='PENDING' AND signal_date=? GROUP BY account_id",('20261110',))]
        accounts=[r[0] for r in c.execute('SELECT account_id FROM accounts')]
        duplicate=c.execute('SELECT COUNT(*) FROM (SELECT idempotency_key FROM fills GROUP BY idempotency_key HAVING COUNT(*)>1)').fetchone()[0]
    broker=PaperBroker(db);mismatches=[a for a in accounts if not broker.reconcile(a).matched]
    # The September 8 comparison fork inherited positions, not a second set of
    # fills. Prove its opening basis from the actual parent's earlier fills.
    inherited=None
    branch='multi_sector_intraday_shadow'
    if branch in mismatches:
        with sqlite3.connect('file:'+db.as_posix()+'?mode=ro',uri=True) as c:
            prefix=c.execute('SELECT symbol,side,quantity,price,fee FROM fills WHERE account_id=? AND trade_date<?',('multi_sector_shadow','20260908')).fetchall()
            own=c.execute('SELECT symbol,side,quantity,price,fee FROM fills WHERE account_id=?',(branch,)).fetchall()
            shares={};seed_cash=0.
            for symbol,side,q,p,fee in prefix:
                sign=1 if side=='BUY' else -1
                seed_cash-=sign*q*p+fee;shares[symbol]=shares.get(symbol,0)+sign*q
            seed_quantity=sum(shares.values())
            for symbol,side,q,p,fee in own:
                shares[symbol]=shares.get(symbol,0)+(q if side=='BUY' else -q)
            for symbol,q in c.execute("SELECT symbol,share_quantity FROM corporate_action_ledger WHERE account_id=? AND action_type='STOCK_DIVIDEND'",(branch,)):
                shares[symbol]=shares.get(symbol,0)+q
            actual=dict(c.execute('SELECT symbol,quantity FROM positions WHERE account_id=? AND quantity>0',(branch,)))
        rec=broker.reconcile(branch)
        proven=(abs(rec.cash_difference-seed_cash)<.01 and rec.quantity_difference==seed_quantity
            and {s:q for s,q in shares.items() if q}==actual and rec.missing_fill_count==0 and rec.orphan_fill_count==0)
        inherited=dict(account_id=branch,source='multi_sector_shadow',branch_date='20260908',
            prefix_fill_count=len(prefix),inherited_cash_flow=seed_cash,inherited_quantity=seed_quantity,
            matched_including_inherited_basis=proven,no_fills_written=True)
        if proven:mismatches.remove(branch)
    with urlopen('http://127.0.0.1:8788/api/health',timeout=20) as r:health=json.load(r)
    payload=dict(health=health,integrity=integrity,today_fills=fills,pending_prior_close=pending,
        duplicate_fill_keys=duplicate,reconciliation_mismatches=mismatches,inherited_basis_audit=inherited,
        baseline_risk=broker.risk_state('multi_sector_shadow'),
        unresolved=['AlphaFeed quality and post-solve risk guards for A/B pending live-session retry', '1027 existing turnover bound'],
        real_orders_enabled=False)
    write_diagnostic(root/'state/restart-recovery-20261111/final-audit.json',payload)
    print(json.dumps(payload),flush=True)
