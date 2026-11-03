"""Independent arithmetic/risk audit of the four solved frozen execution inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from moneymore.research.integer_case import decode_case
from moneymore.research.quality_contract import quality_upper_only_bounds


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_INTEGER_AUDIT')
    report=json.loads((args.study/'results.json').read_text());rows=[]
    for row in report['cases']:
        account,stem=row['case'].split('-integer-case-');file=args.capture/account/f'integer-case-{stem}.json'
        if hashlib.sha256(file.read_bytes()).hexdigest()!=row['input_sha256']:raise ValueError('INTEGER_INPUT_CHANGED')
        inputs,kw=decode_case(json.loads(file.read_text()));target,prices,held,available,cash,equity,cfg,c,limits=inputs
        plan=json.loads((args.study/f"{row['case']}.json").read_text())
        if plan['status']!='PASS_REFERENCE_QUOTES_ONLY':raise ValueError('INTEGER_CASE_NOT_SOLVED')
        diagnostic=plan['solver_diagnostics']
        if diagnostic['solver_status']!='optimal' or diagnostic['mip_gap']>.010001:raise ValueError('OPTIMALITY_CONTRACT_NOT_MET')
        q=pd.Series(plan['target_quantities'],dtype=float);initial=pd.Series(held,dtype=float);fee_total=0.;expected_cash=cash
        orders={o['symbol']:o for o in plan['orders']}
        for symbol in sorted(set(q.index)|set(initial.index)):
            before=int(initial.get(symbol,0));after=int(q.get(symbol,0));diff=after-before
            if after<0 or after!=q.get(symbol,0):raise ValueError('INVALID_FINAL_QUANTITY')
            if not diff:
                if symbol in orders:raise ValueError('SPURIOUS_ORDER')
                continue
            side='BUY' if diff>0 else 'SELL';count=abs(diff);order=orders.pop(symbol)
            if order['side']!=side or order['quantity']!=count:raise ValueError('ORDER_QUANTITY_MISMATCH')
            if side=='SELL' and count>available.get(symbol,0):raise ValueError('T1_VIOLATION')
            if not kw['can_buy' if side=='BUY' else 'can_sell'].get(symbol,False):raise ValueError('MARKET_GATE_VIOLATION')
            if side=='BUY' and (count%100 or count<(200 if symbol.startswith(('688','689')) else 100)):raise ValueError('BUY_LOT_VIOLATION')
            if side=='SELL' and count%100 and after!=0:raise ValueError('ODD_LOT_SELL_VIOLATION')
            price=prices[symbol]*(1+cfg.slippage_bps/10000*(1 if side=='BUY' else -1));notional=price*count
            fee=max(cfg.minimum_commission,cfg.commission_rate*notional)+cfg.transfer_fee_rate*notional+(cfg.stamp_duty_rate*notional if side=='SELL' else 0.)
            if abs(price-order['estimated_execution_price'])>1e-8 or abs(fee-order['estimated_fee'])>1e-6:raise ValueError('COST_MISMATCH')
            expected_cash+=(-notional-fee if side=='BUY' else notional-fee);fee_total+=fee
        if orders or expected_cash<-.01 or abs(expected_cash-plan['estimated_cash'])>.01:raise ValueError('CASH_OR_EXTRA_ORDER_MISMATCH')
        values=q*prices.reindex(q.index);invested=float(values.sum());wb=pd.Series(kw['benchmark_weights']).reindex(c.index,fill_value=0.)
        active=(values.reindex(c.index,fill_value=0.)/invested-wb)@c
        lo,hi=quality_upper_only_bounds(c,wb,limits,kw['research_quality_envelope'])
        if not np.isfinite(active).all() or max((active-hi).max(),(lo-active).max())>1e-5:raise ValueError('STYLE_VIOLATION')
        if values.max()>cfg.max_position_weight*equity+.01 or invested>min(target.sum(),cfg.max_gross_exposure)*equity+.01:raise ValueError('CAPITAL_LIMIT_VIOLATION')
        old_values=initial*prices.reindex(initial.index);old_weight=(old_values/old_values.sum()).reindex(c.index,fill_value=0.)
        turnover=float(abs(values.reindex(c.index,fill_value=0.)/invested-old_weight).sum())
        if turnover>.20+1e-5:raise ValueError('TURNOVER_VIOLATION')
        rows.append({'case':row['case'],'orders':len(plan['orders']),'fee':fee_total,'turnover':turnover,'cash_error':abs(expected_cash-plan['estimated_cash']),
                     'mip_gap':diagnostic['mip_gap'],'input_sha256':row['input_sha256']})
    if len(rows)!=4:raise ValueError('FOUR_CASES_REQUIRED')
    result={'status':'FOUR_FIXED_INTEGER_CASES_INDEPENDENTLY_AUDITED','cases':rows,'orders_enabled':False,'not_full_path_replay':True}
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
