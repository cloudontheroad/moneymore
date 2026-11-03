"""Continue four actual private checkpoints under repaired risk inputs, no live orders."""
import argparse
from pathlib import Path
import hashlib
import json
import sqlite3
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from qlib.model.riskmodel import StructuredCovEstimator
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.data.tushare_provider import TushareProvider
from moneymore.challenger_diagnostics import trading_dates
from moneymore.execution.paper import PaperBroker,ExecutionBar
from moneymore.research.frozen_risk_scale import CORE
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.style_replay_inputs import optimize_daily,prospective_capture
from moneymore.research.quality_contract import quality_upper_only_bounds
from moneymore.research.integer_execution import plan_integer_execution,paper_intents_from_integer_plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_REPAIR')
    root=Path(__file__).resolve().parents[1];version=root/'state/style-balance/quality-risk-input-v2-20261103-a'
    calibration=root/'state/style-balance/quality-calibration-v2-20261103-a'
    checkpoints=root/'state/style-balance/quality-repair-actions-20261103-a'
    evidence=root/'state/style-balance/quality-repair-evidence-20261103-a';gates=root/'state/style-balance/buffer-integer-quotes-20261102-a'
    parent=json.loads((checkpoints/'results.json').read_text());envelope=json.loads((calibration/'observed-error-envelope.json').read_text())
    price_path=version/'source-price-history.parquet';info=json.loads((version/'results.json').read_text())
    if hashlib.sha256(price_path.read_bytes()).hexdigest()!=info['source_price_history_sha256']:raise ValueError('REPAIRED_PRICE_VERSION_CHANGED')
    bars=pd.read_parquet(price_path);prices=bars.pivot(index='trade_date',columns='ts_code',values='total_price')
    liquidity=pd.read_parquet(calibration/'registered-liquidity.parquet').set_index(['date','symbol']).turnover_mean_20
    calendar=trading_dates(ParquetStore(root/'data'),'20261008');args.output.mkdir(parents=True)
    # Exactly two additional official quote reads; close columns are NOT
    # exposed to either optimizer or execution decisions below.
    load_dotenv(root/'.env');client=TushareProvider()._client;quotes={};quote_hashes={}
    for day in ['20260929','20260930']:
        data=client.daily(trade_date=day)
        if data is None or data.empty or not data.trade_date.eq(day).all():raise ValueError('OFFICIAL_REPAIR_QUOTES_UNAVAILABLE')
        file=args.output/f'official-daily-{day}.parquet';data.to_parquet(file,index=False)
        quote_hashes[day]=hashlib.sha256(file.read_bytes()).hexdigest();quotes[day]=data.set_index('ts_code');time.sleep(1.1)
    quotes['20261008']=pd.read_parquet(evidence/'daily-20261008.parquet').set_index('ts_code')
    cfg=BacktestConfig(initial_cash=10_000_000,commission_rate=.00015,minimum_commission=5,stamp_duty_rate=.0005,transfer_fee_rate=.00001,slippage_bps=10,lot_size=100,max_position_weight=.15,max_gross_exposure=1.,max_drawdown=.2)
    protocol={'scope':'four_actual_checkpoint_continuations_not_a_combined_account','capital_reset':False,'input_version':str(version),
        'calibration':str(calibration),'new_quote_reads':2,'turnover_l1_limit':.20,'seed_seconds':10,'target_seconds':20,
        'orders_enabled':False,'limits':'unchanged_quality_upper_only_v1_with_recalibrated_full_errors'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
    for checkpoint in parent['cases']:
        day,next_day=checkpoint['close_date'],checkpoint['next_trade_date'];key=f"{checkpoint['signal']}-{checkpoint['probe']}"
        source=checkpoints/key/'paper.sqlite3';digest=hashlib.sha256(source.read_bytes()).hexdigest();folder=args.output/key;folder.mkdir();db=folder/'paper.sqlite3'
        with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as original,sqlite3.connect(db) as target:original.backup(target)
        broker=PaperBroker(db);quote=quotes[next_day]
        broker.settle_available_positions('isolated',next_day)
        opening=broker.account_snapshot(quote.open.to_dict(),'isolated')
        held={x['symbol']:int(x['quantity']) for x in opening['positions'] if x['quantity']>0}
        available={x['symbol']:int(x['available_quantity']) for x in opening['positions']}
        raw=pd.read_parquet(version/f'observations-{day}.parquet')
        raw=raw.drop(columns=['turnover_rate']).assign(turnover_mean_20=liquidity.loc[day].reindex(raw.index))
        c=uncertain_controls(raw,CORE)[0]
        for field in raw.select_dtypes(include='number').columns:
            if field not in CORE:c['unknown_'+field]=raw[field].replace([np.inf,-np.inf],np.nan).isna().astype(float)
        if set(held)-set(c.index):raise ValueError('ACTUAL_HELD_RISK_MISSING_NOT_POOL_EXIT')
        wb=np.exp(raw.log_size);wb/=wb.sum();limits=pd.Series({f:.01 if f.startswith('unknown_') else .03 if f.startswith(('industry_','board_')) else .1 for f in c.columns})
        low,high=quality_upper_only_bounds(c,wb,limits,envelope);centered=c-(wb @ c)
        directory=root/('state/qlib-challenger-shadow' if checkpoint['probe']=='archived_main_gru' else 'state/qlib-candidate-shadow/20260812')
        archive_path=directory/f'{day}.json';archive=json.loads(archive_path.read_text(encoding='utf-8'))
        if archive['trade_date']!=day or not prospective_capture(day,archive['created_at'],next_day):raise ValueError('REPAIR_SIGNAL_NOT_PROSPECTIVE')
        score=pd.DataFrame(archive['scores']).set_index('instrument').score
        days=[d for d in calendar if d<=day][-121:];returns=prices.reindex(index=days,columns=raw.index).pct_change(fill_method=None).iloc[1:]
        if not np.isfinite(returns).all().all():raise ValueError('REPAIR_RISK_HISTORY_MISSING')
        np.random.seed(17);F,cov,specific=StructuredCovEstimator(num_factors=10,factor_model='pca',scale_return=False).predict(returns,is_price=False,return_decomposed_components=True)
        prior_close=bars.loc[bars.trade_date.eq(day)].set_index('ts_code').close
        held_values=pd.Series(held)*prior_close.reindex(held)
        gate_source=evidence if next_day=='20261008' else gates
        gate=pd.read_parquet(gate_source/f'stk_limit-{next_day}.parquet').set_index('ts_code')
        susp=pd.read_parquet(gate_source/f'suspend_d-{next_day}.parquet');halted=set(susp.loc[susp.suspend_type.eq('S'),'ts_code'])
        buy=(quote.open*1.001<gate.up_limit.reindex(quote.index)-1e-8)&~quote.index.isin(halted)
        sell=(quote.open*.999>gate.down_limit.reindex(quote.index)+1e-8)&~quote.index.isin(halted)
        row={'checkpoint':key,'signal_date':day,'execution_date':next_day,'initial_cash':opening['cash'],'initial_equity':opening['equity'],
            'source_ledger_sha256':digest,'signal_sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),'fills':0,'repaired':False}
        try:
            target=optimize_daily(score,raw,c,F,cov,specific,wb,limits,held_values,research_quality_envelope=envelope)
            plan=plan_integer_execution(target,quote.open,held,available,float(opening['cash']),float(opening['equity']),cfg,c,limits,
                can_buy=buy.to_dict(),can_sell=sell.to_dict(),benchmark_weights=wb,style_controls_prestandardized=True,research_quality_envelope=envelope,
                time_limit=10,research_staged_warm_start=True,research_target_time_limit=20,solver_threads=2,mip_relative_gap=.01)
        except ValueError as error:plan={'status':'BLOCKED','reason':str(error),'orders':[],'orders_enabled':False}
        row.update(status=plan['status'],reason=plan.get('reason'));(folder/'plan.json').write_text(json.dumps(plan,indent=2,allow_nan=False))
        if plan['status']=='PASS_REFERENCE_QUOTES_ONLY':
            broker.cancel_pending('isolated','ISOLATED_REPAIR_REPLAN',signal_date=None)
            for intent in paper_intents_from_integer_plan(plan,'quality_repair_v2',day):broker.submit(intent,'isolated')
            for order in plan['orders']:
                symbol=order['symbol'];price=float(quote.loc[symbol,'open'])
                broker.execute_pending(ExecutionBar(symbol,next_day,price,price,bool(buy[symbol]),bool(sell[symbol])),cfg,'isolated')
            actual=broker.account_snapshot(quote.open.to_dict(),'isolated');q={x['symbol']:int(x['quantity']) for x in actual['positions'] if x['quantity']>0}
            if q!={s:v for s,v in plan['target_quantities'].items() if v>0} or abs(actual['cash']-plan['estimated_cash'])>.01:raise ValueError('ACTUAL_REPAIR_LEDGER_MISMATCH')
            values=pd.Series(q)*quote.open.reindex(q);active=(values/values.sum()) @ centered.reindex(q)
            excess=max(float((active-high).max()),float((low-active).max()))
            if not np.isfinite(active).all() or excess>1e-5:raise ValueError('ACTUAL_REPAIR_RISK_BOUNDARY_FAILED')
            with sqlite3.connect(db) as con:count=con.execute('SELECT COUNT(*) FROM fills WHERE trade_date=?',(next_day,)).fetchone()[0]
            row.update(fills=count,repaired=True,cash_error=abs(actual['cash']-plan['estimated_cash']),maximum_style_excess=excess,turnover_l1=plan['turnover_l1'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise ValueError('ORIGINAL_CHECKPOINT_CHANGED')
        rows.append(row);(args.output/'progress.json').write_text(json.dumps(rows,indent=2,allow_nan=False));print(json.dumps(row),flush=True)
    (args.output/'results.json').write_text(json.dumps({'cases':rows,'repaired':sum(r['repaired'] for r in rows),'adopted':False,'orders_enabled':False,
        'quote_hashes':quote_hashes,'limitations':['four_independent_continuations_not_continuous_equity','open_repair_does_not_guarantee_close_risk','latest_official_reconstructed_inputs_not_original_vintage']},indent=2,allow_nan=False))


if __name__=='__main__':main()
