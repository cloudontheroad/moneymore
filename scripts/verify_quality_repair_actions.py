"""Official action reconciliation and idempotent private ledger continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from moneymore.execution.paper import PaperBroker
from moneymore.research.replay_actions import official_replay_actions


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_ACTION_VERIFICATION')
    root=Path(__file__).resolve().parents[1];evidence=root/'state/style-balance/quality-repair-evidence-20261103-a'
    manifest=json.loads((evidence/'manifest.json').read_text())
    if manifest['status']!='READY':raise ValueError('OFFICIAL_EVIDENCE_INCOMPLETE')
    for item in manifest['requests']:
        if hashlib.sha256((evidence/item['file']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('OFFICIAL_EVIDENCE_CHANGED')
    args.output.mkdir(parents=True);matrix=root/'state/style-balance/quality-integer-matrix-20261103-a'
    obligations=json.loads((root/'state/style-balance/quality-drift-obligations-20261103-a/results.json').read_text())
    source_prices=pd.read_parquet(root/'state/style-balance/real-archive-group-decomposition-20261102-b/source-price-history.parquet')
    differences=[]
    for day in ['20260928','20260929','20260930']:
        cached=source_prices.loc[source_prices.trade_date.eq(day)].set_index('ts_code').adj_factor
        official=pd.read_parquet(evidence/f'adj_factor-{day}.parquet').set_index('ts_code').adj_factor.reindex(cached.index)
        for symbol in cached.index:
            if pd.isna(official[symbol]) or not np.isclose(cached[symbol],official[symbol],atol=1e-8,rtol=0):
                differences.append({'date':day,'symbol':symbol,'cached':float(cached[symbol]),'official':float(official[symbol]) if pd.notna(official[symbol]) else None})
    pd.DataFrame(differences).to_parquet(args.output/'adjustment-disagreements.parquet',index=False)
    cases=[]
    for row in obligations['obligations']:
        if row['variant']!='quality_buffered':continue
        key=(row['signal'],row['probe']);day=row['close_date'];next_day=row['next_open'] or manifest['next_trade_date']
        if any(x['signal']==key[0] and x['probe']==key[1] for x in cases):continue
        source=matrix/f'{key[0]}-{key[1]}'/'quality_buffered'/'paper.sqlite3'
        digest=hashlib.sha256(source.read_bytes()).hexdigest();folder=args.output/f'{key[0]}-{key[1]}';folder.mkdir();destination=folder/'paper.sqlite3'
        with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as original,sqlite3.connect(destination) as target:
            original.backup(target)
            held=dict(original.execute("SELECT symbol,quantity FROM positions WHERE account_id='isolated' AND quantity>0").fetchall())
            before=float(original.execute("SELECT cash FROM accounts WHERE account_id='isolated'").fetchone()[0])
        raw=pd.read_parquet(evidence/f'dividend-ex-{next_day}.parquet')
        relevant=raw.loc[raw.ts_code.isin(held)&raw.div_proc.eq('实施')]
        actions=official_replay_actions(relevant)
        if not actions.empty and (not actions.record_date.eq(pd.Timestamp(day)).all() or (actions.known_date>pd.Timestamp(day)).any()):
            raise ValueError('ACTION_NOT_KNOWN_OR_ENTITLEMENT_REQUIRES_EARLIER_HOLDING')
        broker=PaperBroker(destination);expected_cash=0.;expected=dict(held)
        for event in actions.to_dict('records'):
            qty=held[event['symbol']]
            if pd.notna(event['cash_pay_date']) and event['cash_pay_date']<=pd.Timestamp(next_day):expected_cash+=qty*event['cash_per_share']
            if pd.notna(event['stock_list_date']) and event['stock_list_date']<=pd.Timestamp(next_day):expected[event['symbol']]+=int(qty*event['stock_ratio'])
        for repeat in range(2):
            for event in actions.to_dict('records'):
                broker.register_corporate_entitlement('isolated',event['symbol'],event['action_key'],day,event['cash_per_share'],event['stock_ratio'],
                    event['cash_pay_date'].strftime('%Y%m%d') if pd.notna(event['cash_pay_date']) else None,
                    event['stock_list_date'].strftime('%Y%m%d') if pd.notna(event['stock_list_date']) else None)
            broker.settle_corporate_actions('isolated',next_day)
            with sqlite3.connect(destination) as con:
                after=float(con.execute("SELECT cash FROM accounts WHERE account_id='isolated'").fetchone()[0])
                actual=dict(con.execute("SELECT symbol,quantity FROM positions WHERE account_id='isolated' AND quantity>0").fetchall())
            if abs(after-before-expected_cash)>.01 or actual!=expected:raise ValueError('ACTION_IDEMPOTENCY_OR_LEDGER_MISMATCH')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise ValueError('SOURCE_LEDGER_CHANGED')
        actions.to_parquet(folder/'official-actions.parquet',index=False)
        cases.append({'signal':key[0],'probe':key[1],'close_date':day,'next_trade_date':next_day,'source_sha256':digest,
            'implemented_actions':len(actions),'cash_dividend_settled':expected_cash,'cash_error':abs(after-before-expected_cash),
            'repeated_settlement_verified':True,'repair_orders_executed':False})
    result={'cases':cases,'adjustment_disagreement_rows':len(differences),'adjustment_disagreement_dates':{d:sum(r['date']==d for r in differences) for d in ['20260928','20260929','20260930']},
        'status':'ACTION_ACCOUNTING_VERIFIED_RISK_INPUT_REVALIDATION_REQUIRED','orders_enabled':False,
        'limitations':['latest_official_snapshot_not_original_vintage_proof','no_strategy_returns_rewritten','no_risk_repair_trade_claim']}
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result))


if __name__=='__main__':main()
