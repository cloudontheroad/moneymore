"""Independent read-only daily ledger, fills, T+1, fees and terminal exit checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import yaml
from moneymore.research.replay_attribution import reconstruct_cash,reconstruct_positions
from moneymore.research.order_chunks import market_order_cap


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--retain-terminal-holdings',action='store_true');args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_COST_AUDIT')
    protocol=yaml.safe_load((args.study/'protocol.yaml').read_text());report=json.loads((args.study/'results.json').read_text());root=Path(__file__).resolve().parents[1]
    if args.retain_terminal_holdings and report.get('terminal_policy')!='mark_to_market_no_forced_liquidation':raise ValueError('EXPLICIT_ONGOING_REPLAY_POLICY_REQUIRED')
    binding=json.loads((args.study/'episode-binding.json').read_text()) if (args.study/'episode-binding.json').exists() else None
    evidence=Path(binding['evidence']) if binding else root/'state/style-balance/s3-cost-evidence-fold0-20261103-a'
    checked=Path(binding['audit']) if binding else root/'state/style-balance/s3-cost-evidence-audit-fold0-20261103-a'
    if binding and (hashlib.sha256((evidence/'manifest.json').read_bytes()).hexdigest()!=binding['manifest_sha256'] or hashlib.sha256((checked/'results.json').read_bytes()).hexdigest()!=binding['audit_sha256']):raise ValueError('EPISODE_EVIDENCE_BINDING_CHANGED')
    actions=pd.read_parquet(checked/'official-actions.parquet').set_index('action_key')
    slip=protocol['slippage_bps_each_side']/10000
    market=pd.read_parquet(args.study/'market-history.parquet').set_index(['trade_date','ts_code']);rows=[]
    for case in report['cases']:
        folder=args.study/case['account'];database=folder/'paper.sqlite3';digest=hashlib.sha256(database.read_bytes()).hexdigest()
        equity=pd.read_parquet(folder/'equity.parquet');dates=equity.date.astype(str).tolist()
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as con:
            fills=pd.read_sql_query('SELECT * FROM fills',con);ledger=pd.read_sql_query('SELECT * FROM corporate_action_ledger',con)
            rights=pd.read_sql_query('SELECT * FROM corporate_entitlements',con)
            final=dict(con.execute("SELECT symbol,quantity FROM positions WHERE account_id='isolated' AND quantity>0").fetchall())
            cash=float(con.execute("SELECT cash FROM accounts WHERE account_id='isolated'").fetchone()[0])
        rebuilt_cash=reconstruct_cash(fills,ledger,dates,protocol['capital_each_episode']);positions=reconstruct_positions(fills,ledger,dates)
        previous={};cash_error=0.;fees_error=0.;slippage_cost=0.;equity_error=0.;independent_cash=float(protocol['capital_each_episode'])
        prior_market=market.reset_index();prior_market=prior_market.loc[prior_market.trade_date.astype(str).lt(dates[0])].sort_values('trade_date')
        last_marks=prior_market.groupby('ts_code').tail(1).set_index('ts_code').close.to_dict()
        for day in dates:
            sellable=dict(previous)
            day_ledger=ledger.loc[ledger.trade_date.astype(str).eq(day)]
            independent_cash+=float(day_ledger.loc[day_ledger.action_type.eq('CASH_DIVIDEND'),'cash_amount'].sum())
            for action in ledger.loc[ledger.trade_date.astype(str).eq(day)&ledger.action_type.eq('STOCK_DIVIDEND')].itertuples():sellable[action.symbol]=sellable.get(action.symbol,0)+action.share_quantity
            independent_positions=dict(sellable)
            today=fills.loc[fills.trade_date.astype(str).eq(day)]
            for f in today.itertuples():
                if f.quantity>market_order_cap(f.symbol):raise ValueError('SINGLE_ORDER_CAP_FAILED')
                if f.side=='BUY' and (f.quantity%100 or (f.symbol.startswith(('688','689')) and f.quantity<200)):raise ValueError('BUY_LOT_RULE_FAILED')
                if f.side=='SELL':
                    if f.quantity>sellable.get(f.symbol,0):raise ValueError('T1_RULE_FAILED')
                    sellable[f.symbol]-=f.quantity
                ref=float(market.loc[(day,f.symbol),'open']);direction=1 if f.side=='BUY' else -1
                if abs(f.price-ref*(1+direction*slip))>1e-8:raise ValueError('SLIPPAGE_MISMATCH')
                slippage_cost+=abs(f.price-ref)*f.quantity
                notional=f.price*f.quantity;expected=max(protocol['minimum_commission'],notional*protocol['commission_rate'])+notional*protocol['transfer_fee_rate']+(notional*protocol['stamp_duty_rate'] if f.side=='SELL' else 0)
                fees_error=max(fees_error,abs(expected-f.fee))
                independent_cash+=(-1 if f.side=='BUY' else 1)*f.price*f.quantity-f.fee
                independent_positions[f.symbol]=independent_positions.get(f.symbol,0)+(1 if f.side=='BUY' else -1)*f.quantity
                gate=pd.read_parquet(evidence/f'stk_limit-{day}.parquet').set_index('ts_code')
                if not gate.loc[f.symbol,'down_limit']-1e-8<=f.price<=gate.loc[f.symbol,'up_limit']+1e-8:raise ValueError('PRICE_LIMIT_FAILED')
            independent_positions={s:q for s,q in independent_positions.items() if q>0}
            if independent_positions!=positions[day]:raise ValueError('INDEPENDENT_DAILY_POSITION_MISMATCH')
            stored=equity.loc[equity.date.eq(day)].iloc[0];cash_error=max(cash_error,abs(rebuilt_cash[day]-stored.cash),abs(independent_cash-stored.cash))
            today_marks=market.loc[day].close.dropna().to_dict()
            for s in set(independent_positions)-set(today_marks):
                if s not in last_marks:raise ValueError('INDEPENDENT_VALUATION_MISSING')
                value=last_marks[s]
                for a in actions.loc[actions.symbol.eq(s)&actions.ex_date.eq(pd.Timestamp(day))].itertuples():value=(value-a.cash_per_share)/(1+a.stock_ratio)
                today_marks[s]=value
            receivable=0.;stock_right_value=0.
            for entitlement in rights.itertuples():
                action=actions.loc[entitlement.action_key]
                if pd.Timestamp(day)<action.ex_date:continue
                settled=ledger.loc[ledger.action_key.eq(entitlement.action_key)&ledger.trade_date.astype(str).le(day)]
                if entitlement.cash_per_share>0 and not settled.action_type.eq('CASH_DIVIDEND').any():receivable+=entitlement.entitled_quantity*entitlement.cash_per_share
                if entitlement.stock_ratio>0 and not settled.action_type.eq('STOCK_DIVIDEND').any():
                    if entitlement.symbol not in today_marks:raise ValueError('RIGHT_VALUATION_MISSING')
                    stock_right_value+=int(entitlement.entitled_quantity*entitlement.stock_ratio)*today_marks[entitlement.symbol]
            rebuilt_equity=independent_cash+sum(q*today_marks[s] for s,q in independent_positions.items())+receivable+stock_right_value
            equity_error=max(equity_error,abs(rebuilt_equity-stored.equity),abs(receivable-stored.receivable_cash),abs(stock_right_value-stored.unsettled_stock_value))
            last_marks.update(today_marks);previous=positions[day]
        if cash_error>.01 or equity_error>.01 or abs(cash-rebuilt_cash[dates[-1]])>.01 or fees_error>1e-6 or final!=positions[dates[-1]]:
            raise ValueError('INDEPENDENT_LEDGER_FAILED:'+json.dumps({'account':case['account'],'cash_error':cash_error,'equity_error':equity_error,'fee_error':fees_error,'positions_match':final==positions[dates[-1]]}))
        terminal=dates[-1];gate=pd.read_parquet(evidence/f'stk_limit-{terminal}.parquet').set_index('ts_code')
        suspend=pd.read_parquet(evidence/f'suspend_d-{terminal}.parquet')
        terminal_halts=set(suspend.loc[suspend.suspend_type.eq('S')&(suspend.suspend_timing.isna()|suspend.suspend_timing.eq('')),'ts_code']) if not suspend.empty else set()
        for s,q in ({} if args.retain_terminal_holdings else final).items():
            price=market.loc[(terminal,s),'open'] if (terminal,s) in market.index else np.nan
            if s not in terminal_halts and np.isfinite(price) and price*(1-slip)>gate.down_limit.get(s,np.nan)+1e-8:raise ValueError('TRADABLE_TERMINAL_POSITION_NOT_EXITED')
        if hashlib.sha256(database.read_bytes()).hexdigest()!=digest:raise ValueError('LEDGER_CHANGED_DURING_AUDIT')
        complete=case['passed_rebalances']==case['planned_rebalances'] and (args.retain_terminal_holdings or not final) and case['unsettled_stock_value']==0
        rows.append({'account':case['account'],'fills':len(fills),'cash_error':cash_error,'equity_and_rights_error':equity_error,'fee_error':fees_error,'slippage_currency':slippage_cost,
            'net_return':case['net_return'],'planned_schedule_fully_executed':complete,'remaining_positions':final,
            'database_sha256':digest,'not_eligible_for_winner_selection':not complete})
    args.output.write_text(json.dumps({'status':'PASS_LEDGER_AUDIT_NOT_STRATEGY_ACCEPTANCE','cases':rows,'orders_enabled':False,
        'source_result_sha256':hashlib.sha256((args.study/'results.json').read_bytes()).hexdigest(),
        'limitations':['blocked_schedules_not_equal_to_optimal_strategy_returns','latest_vendor_snapshots_not_original_vintage_proof']},indent=2,allow_nan=False))
    print(json.dumps(rows))


if __name__=='__main__':main()
