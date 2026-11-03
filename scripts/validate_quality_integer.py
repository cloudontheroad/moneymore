"""First fixed real-score paired integer case, isolated PaperBroker only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from moneymore.config import BacktestConfig
from moneymore.execution_replay import run_execution_replay
from moneymore.research.integer_execution import plan_integer_execution
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.frozen_risk_scale import CORE
from moneymore.research.input_versions import require_input_version


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    transitions={'20260922':'20260923','20260923':'20260924','20260924':'20260928','20260928':'20260929','20260929':'20260930'}
    parser.add_argument('--signal',choices=list(transitions),default='20260922')
    parser.add_argument('--probe',choices=['archived_main_gru','archived_candidate_20260812'],default='archived_main_gru')
    parser.add_argument('--allocations',type=Path);parser.add_argument('--calibration',type=Path);parser.add_argument('--inputs',type=Path)
    parser.add_argument('--capital',type=float);args=parser.parse_args()
    capital_config=Path(__file__).resolve().parents[1]/'configs/style_quality_execution_v1.yaml'
    explicit_capital=args.capital is not None
    if args.capital is None:args.capital=float(yaml.safe_load(capital_config.read_text(encoding='utf-8'))['research_initial_capital'])
    if not np.isfinite(args.capital) or args.capital<=0:raise ValueError('INVALID_RESEARCH_CAPITAL')
    if args.output.exists():raise FileExistsError('IMMUTABLE_RESEARCH')
    root=Path(__file__).resolve().parents[1];source=args.allocations or root/'state/style-balance/quality-score-heldout-20261103-c'
    parent=args.calibration or root/'state/style-balance/real-archive-group-decomposition-20261102-b';gates=root/'state/style-balance/buffer-integer-quotes-20261102-a'
    report=json.loads((source/'results.json').read_text())
    require_input_version(report,args.inputs)
    calibration_report=json.loads((parent/'results.json').read_text())
    require_input_version(calibration_report,args.inputs)
    for p,digest in report['input_hashes'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest:raise ValueError('SOURCE_CHANGED')
    manifest=json.loads((gates/'manifest.json').read_text())
    if manifest['status']!='READY':raise ValueError('OFFICIAL_GATES_UNAVAILABLE')
    for item in manifest['requests']:
        if hashlib.sha256((gates/item['file']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('GATE_CHANGED')
    envelope=json.loads((parent/'observed-error-envelope.json').read_text())
    if hashlib.sha256((parent/'observed-error-envelope.json').read_bytes()).hexdigest()!=report['envelope_sha256']:raise ValueError('ENVELOPE_CHANGED')
    signal,date,probe=args.signal,transitions[args.signal],args.probe
    liquidity=pd.read_parquet(parent/'registered-liquidity.parquet').set_index(['date','symbol']).turnover_mean_20
    def inputs(day):
        raw=pd.read_parquet((args.inputs or root/'state/style-balance/preperiod-drift-202609-b')/f'observations-{day}.parquet')
        raw=raw.drop(columns=['turnover_rate']).assign(turnover_mean_20=liquidity.loc[day].reindex(raw.index))
        c=uncertain_controls(raw,CORE)[0]
        for f in raw.select_dtypes(include='number').columns:
            if f not in CORE:c['unknown_'+f]=raw[f].replace([np.inf,-np.inf],np.nan).isna().astype(float)
        wb=np.exp(raw.log_size);return c,wb/wb.sum()
    c,wb=inputs(signal);new,new_wb=inputs(date);limits=pd.Series({f:.01 if f.startswith('unknown_') else .03 if f.startswith(('industry_','board_')) else .1 for f in c.columns})
    weights=pd.read_parquet(source/f'weights-{signal}-{probe}.parquet')
    bars=pd.read_parquet(parent/'source-price-history.parquet');bars=bars.loc[bars.trade_date.isin([signal,date]) & bars.ts_code.isin(c.index),['trade_date','ts_code','open','close']]
    gate=pd.read_parquet(gates/f'stk_limit-{date}.parquet').set_index('ts_code')
    susp=pd.read_parquet(gates/f'suspend_d-{date}.parquet');halted=set(susp.loc[susp.suspend_type.eq('S'),'ts_code'])
    cfg=BacktestConfig(initial_cash=args.capital,commission_rate=.00015,minimum_commission=5,stamp_duty_rate=.0005,transfer_fee_rate=.00001,slippage_bps=10,lot_size=100,max_position_weight=.15,max_gross_exposure=1.,max_drawdown=.2)
    bars['can_buy']=[r.ts_code not in halted and r.open*1.001<gate.up_limit.get(r.ts_code,np.nan)-1e-8 for r in bars.itertuples()]
    bars['can_sell']=[r.ts_code not in halted and r.open*.999>gate.down_limit.get(r.ts_code,np.nan)+1e-8 for r in bars.itertuples()]
    bars=bars.rename(columns={'trade_date':'date','ts_code':'symbol'});bars['date']=pd.to_datetime(bars.date)
    args.output.mkdir(parents=True);rows=[];empty=pd.DataFrame(columns=['date','symbol','target_weight'])
    protocol={'signal':signal,'trade_date':date,'probe':probe,'selection':'fixed_predeclared_heldout_case_not_return_selected',
        'input_version':str(args.inputs) if args.inputs else None,'calibration':str(parent),'allocations':str(source),
        'capital':args.capital,'capital_policy':'new_quality_default_10m_legacy_accounts_unchanged','explicit_capital_override':explicit_capital,
        'capital_config_sha256':hashlib.sha256(capital_config.read_bytes()).hexdigest(),'slippage_bps':10,'contract':'quality_upper_only_v1','orders_enabled':False,
        'feasibility_seconds':10,'target_seconds':20,'mip_relative_gap':.01,'scope':'one_session_integer_cash_entry_not_continuous_strategy'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for variant in ['quality_unbuffered','quality_buffered']:
        folder=args.output/variant;folder.mkdir();holder={}
        margins={f:{'upper':0.,'lower':0.} for f in c.columns} if variant=='quality_unbuffered' else envelope
        def opening(day,snapshot,quotes):
            if day!=date:return {'status':'NO_PRIOR_SIGNAL','orders':[]}
            plan=plan_integer_execution(weights[variant],quotes.open,{}, {},snapshot['cash'],snapshot['equity'],cfg,c,limits,
                can_buy=quotes.can_buy.to_dict(),can_sell=quotes.can_sell.to_dict(),benchmark_weights=wb,style_controls_prestandardized=True,
                research_quality_envelope=margins,time_limit=10,research_staged_warm_start=True,research_target_time_limit=20,solver_threads=2,mip_relative_gap=.01)
            plan['signal_date']=signal
            if plan['status']=='PASS_REFERENCE_QUOTES_ONLY':
                plan['execution_style_proof']={'standardized':(c-(wb @ c)).to_dict('index'),'limits':limits.to_dict(),
                    'quality_contract':{'controls':c.to_dict('index'),'benchmark_weights':wb.to_dict(),'envelope':margins}}
            holder.update(plan);return plan
        result=run_execution_replay(bars=bars,targets=empty,config=cfg,database=folder/'paper.sqlite3',strategy_id='quality_integer_diagnostic',account_id='isolated',
            closing_target_provider=lambda day,snapshot:empty.copy(),opening_plan_provider=opening)
        row={'variant':variant,'plan_status':holder['status'],'reason':holder.get('reason'),'fills':len(result.fills),'reconciled':result.reconciliation['matched'],
            'fees':result.summary['transaction_cost'],'close_economic_breaches':None,'close_quality_breaches':None,'plan':holder}
        if holder['status']=='PASS_REFERENCE_QUOTES_ONLY':
            if not result.reconciliation['matched']:raise ValueError('RECONCILIATION_FAILED')
            actual=result.summary['opening_plan_audits'][-1]['execution_check']['actual_quantities']
            q=pd.Series({s:v for s,v in actual.items() if v>0},dtype=float)
            if set(q.index)-set(new.index):raise ValueError('NEXT_RISK_UNOBSERVABLE')
            marks=bars.loc[bars.date.eq(pd.Timestamp(date))].set_index('symbol');value=q*marks.close.reindex(q.index)
            exposure=(value/value.sum()) @ (new-(new_wb @ new)).reindex(index=q.index,columns=c.columns)
            unknown=exposure.index.str.startswith('unknown_')
            row.update(close_economic_breaches=int((exposure.loc[~unknown].abs()-limits.loc[~unknown]>1e-6).sum()),
                close_quality_breaches=int((exposure.loc[unknown]-limits.loc[unknown]>1e-6).sum()),gross=holder['estimated_gross'],cash=holder['estimated_cash'])
            for fill in result.fills:
                if not gate.loc[fill['symbol'],'down_limit']<=fill['price']<=gate.loc[fill['symbol'],'up_limit']:raise ValueError('FILL_OUTSIDE_GATE')
        rows.append(row)
        (args.output/'progress.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({k:v for k,v in row.items() if k!='plan'}),flush=True)
    (args.output/'results.json').write_text(json.dumps({'cases':rows,'adopted':False,'orders_enabled':False},indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
