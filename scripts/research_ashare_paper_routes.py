"""Bounded three-route experiment; writes ONLY its isolated research directory."""
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import time

import numpy as np
import pandas as pd

from moneymore.config import BacktestConfig
from moneymore.costs import execution_price, transaction_fee
from moneymore.execution.paper import PaperBroker, ExecutionBar
from moneymore.execution.risk import OrderIntent, RiskResult
from moneymore.models import Side
from moneymore.research.mature_multifactor import topk_dropout_targets
from moneymore.research.paper_routes import earnings_yield, buffered_targets, monthly_inverse_variance
from moneymore.research.period_statistics import period_statistics
from moneymore.research.quote_permissions import quote_permissions
from moneymore.research.replay_actions import official_replay_actions, economic_snapshot
from prepare_four_strategy_2026 import ROOT, OUT as STUDY, save
from cost_study_quotes import ensure_quotes

OUT = ROOT/'state/ashare-paper-routes-20261111/controlled-drop5-v2'
KINDS = ['ridge_reference','ep','ridge_buffer','ep_buffer','ridge_vol','ep_buffer_vol','ridge_static']


def load_inputs():
    source = ROOT/'state/mature-multifactor/p1-native-20261103-a'
    bars = pd.read_parquet(source/'bars-input.parquet')
    extras = list((ROOT/'state/cost-aware-ab-v2/quote-cache').glob('*.parquet'))
    extras += list((STUDY/'official').glob('held-exit-*.parquet'))
    bars = pd.concat([bars, *[pd.read_parquet(p) for p in extras]], ignore_index=True)
    bars.trade_date = bars.trade_date.astype(str)
    bars = bars.drop_duplicates(['trade_date','ts_code'], keep='first')
    member = pd.read_parquet(source/'membership.parquet')
    member.datetime = pd.to_datetime(member.datetime)
    membership = {d.strftime('%Y%m%d'):set(g.instrument) for d,g in member.groupby('datetime')}
    prices = bars.assign(adjusted=bars.close*bars.adj_factor).pivot(index='trade_date',columns='ts_code',values='adjusted').sort_index()
    prices.index = pd.to_datetime(prices.index)
    mask = member.assign(present=True).pivot(index='datetime',columns='instrument',values='present').reindex(index=prices.index,columns=prices.columns).fillna(False).astype(bool)
    returns = prices.pct_change(fill_method=None).where(mask.shift(1,fill_value=False))
    market = returns.mean(axis=1)
    # Exclude dates without historical membership; never fill an unknown market return with zero.
    market = market.loc[returns.count(axis=1).ge(500)]
    sessions = sorted(d for d in bars.trade_date.unique() if '20260101'<=d<='20260930')
    calendar = sorted(d for d in bars.trade_date.unique() if d<='20260930')
    signals = {d:calendar[calendar.index(d)-1] for d in sessions}
    scale, c = monthly_inverse_variance(market,list(signals.values()))
    predictions = pd.read_parquet(STUDY/'model/predictions.parquet').score
    predictions = {d.strftime('%Y%m%d'):g.droplevel('datetime') for d,g in predictions.groupby(level='datetime')}
    official = {}
    for d in sorted(set(sessions)|set(signals.values())):
        official[d] = {}
        for key in ['daily_basic','stk_limit','suspend_d','dividend']:
            path = STUDY/'official'/f'{key}-{d}.parquet'
            if not path.exists(): raise ValueError('MISSING_INPUT:'+str(path))
            official[d][key] = pd.read_parquet(path)
    instruments = pd.read_parquet(ROOT/'data/processed/instruments.parquet').drop_duplicates('ts_code').set_index('ts_code')
    return bars,membership,sessions,signals,scale,c,predictions,official,instruments


def execute(broker, kind, day, signal, desired, opened, buy, sell, config):
    snap = broker.account_snapshot(opened,'isolated')
    held = {p['symbol']:int(p['quantity']) for p in snap['positions'] if p['quantity']>0}
    available = {p['symbol']:int(p['available_quantity']) for p in snap['positions']}
    cash = snap['cash']; rows=[]; blocked=[]
    requests = [(s,int(desired.get(s,0))-held.get(s,0)) for s in sorted(set(desired)|set(held))]
    requests.sort(key=lambda item:(item[1]>0,item[0]))
    for symbol,delta in requests:
        if not delta:continue
        side = Side.BUY if delta>0 else Side.SELL
        allowed = buy.get(symbol,False) if delta>0 else sell.get(symbol,False)
        if not allowed or symbol not in opened:
            blocked.append({'symbol':symbol,'reason':'PRICE_OR_LIMIT_OR_SUSPENSION','side':side.value});continue
        price = execution_price(opened[symbol],side,config.slippage_bps)
        q = abs(delta)
        if side == Side.SELL:
            q = min(q,available.get(symbol,0))
            if q < held.get(symbol,0): q = q//100*100
        else:
            q = q//100*100
            cap = max(0., config.max_position_weight*snap['equity']-held.get(symbol,0)*opened[symbol])
            q = min(q,int(cap/price)//100*100)
            q = min(q,max(0,int((cash-config.minimum_commission)/(price*(1+config.commission_rate+config.transfer_fee_rate))))//100*100)
            if symbol.startswith('688') and held.get(symbol,0)==0 and q<200:q=0
            while q>0 and q*price+transaction_fee(q*price,side,config)>cash+1e-8:q-=100
        if q<=0:
            blocked.append({'symbol':symbol,'reason':'CASH_LOT_OR_AVAILABILITY','side':side.value});continue
        fee = transaction_fee(q*price,side,config)
        intent = OrderIntent(f'{kind}:{day}:{symbol}:{side.value}',kind,symbol,side,q,signal,'ISOLATED_PAPER_ROUTE')
        assert broker.submit(RiskResult(True,intent,None),'isolated')=='PENDING'
        result = broker.execute_pending(ExecutionBar(symbol,day,opened[symbol],opened[symbol],True,True),config,'isolated')
        if not result or any(r.get('outcome')!='FILLED' for r in result):
            raise ValueError('UNEXPECTED_EXECUTION:'+str(result))
        cash += q*price-fee if side==Side.SELL else -q*price-fee
        rows.append({'date':day,'symbol':symbol,'side':side.value,'quantity':q,'price':price,'fee':fee,'slippage':q*abs(price-opened[symbol]),'notional':q*price})
    assert abs(broker.account_snapshot(opened,'isolated')['cash']-cash)<.01
    return rows,blocked


def run(bps, smoke=False, score_overrides=None, selection_intervals=None, target_builder=None, execution_planner=None, config_overrides=None):
    started=time.monotonic()
    output=OUT/('smoke-v2' if smoke else f'bps{bps}')
    if (output/'results.json').exists():
        print('ALREADY_COMPLETED',output);return
    bars,membership,sessions,signals,scale,c,predictions,official,instruments=load_inputs()
    if smoke:sessions=sessions[:3]
    quotes={d:g.set_index('ts_code') for d,g in bars.groupby('trade_date')}
    static=float(np.mean([scale[signals[d]] for d in sessions]))
    config=BacktestConfig.model_validate({**BacktestConfig.from_yaml(ROOT/'configs/default.yaml').model_dump(),'initial_cash':10_000_000,'slippage_bps':bps})
    if config_overrides:
        config=BacktestConfig.model_validate({**config.model_dump(),**config_overrides})
    save(output/'protocol.json',{'kinds':KINDS,'window':[sessions[0],sessions[-1]],'capital':10_000_000,'slippage_bps':bps,'vol_c_pre2026':c,'static_ex_post_mean_target':static,'real_orders':False,'plan':'docs/research/ASHARE_PAPER_ROUTES_20261111.md','price_timeline':'previous_close_share_plan_next_open_cash_only_revalidation','not_exact_paper_replication':True})
    books={};histories={k:[] for k in KINDS};fills={k:[] for k in KINDS};blocks={k:[] for k in KINDS};holdings=[];coverage=[]
    checkpoint=output/'checkpoint.json';completed=0
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_bytes());completed=saved['completed'];histories=saved['histories'];fills=saved['fills'];blocks=saved['blocks'];holdings=saved['holdings'];coverage=saved['coverage']
    for kind in KINDS:
        path=output/kind/'paper.sqlite3';path.parent.mkdir(parents=True,exist_ok=True)
        backup=path.with_name('resume.sqlite3')
        if checkpoint.exists():
            with sqlite3.connect(backup) as src,sqlite3.connect(path) as dst:src.backup(dst)
        elif path.exists():raise ValueError('PARTIAL_RUN_WITHOUT_CHECKPOINT:'+str(path))
        books[kind]=PaperBroker(path);books[kind].initialize_account(config.initial_cash,'isolated')
    if not checkpoint.exists():
        for broker in books.values():
            with sqlite3.connect(broker.database) as src,sqlite3.connect(broker.database.with_name('resume.sqlite3')) as dst:src.backup(dst)
        save(checkpoint,{'completed':0,'histories':histories,'fills':fills,'blocks':blocks,'holdings':holdings,'coverage':coverage})
    last=quotes[signals[sessions[0]]].close.dropna().to_dict();raw_actions=pd.DataFrame()
    for i,day in enumerate(sessions):
        signal=signals[day];members=membership[signal]
        if len(members)!=1000:raise ValueError('INCOMPLETE_MEMBERSHIP:'+signal)
        div=official[day]['dividend']
        if not div.empty:raw_actions=pd.concat([raw_actions,div],ignore_index=True).drop_duplicates()
        opened=quotes[day].open.dropna().to_dict();closed=quotes[day].close.dropna().to_dict()
        actions=official_replay_actions(raw_actions.loc[raw_actions.ts_code.isin(set().union(*membership.values()))] if not raw_actions.empty else raw_actions)
        susp=official[day]['suspend_d']
        halted=set(susp.loc[susp.suspend_type.eq('S')&(susp.suspend_timing.isna()|susp.suspend_timing.eq('')),'ts_code']) if not susp.empty else set()
        for s in halted-set(opened):
            if s in last:
                mark=last[s]
                for event in actions.loc[actions.symbol.eq(s)&actions.ex_date.eq(pd.Timestamp(day))].itertuples():mark=(mark-event.cash_per_share)/(1+event.stock_ratio)
                opened[s]=closed[s]=mark
        if i<completed:last.update(closed);continue
        ep=earnings_yield(official[signal]['daily_basic'],members)
        ridge=predictions[signal].reindex(sorted(members)).dropna()
        coverage.append({'date':day,'ep_scored':len(ep),'ridge_scored':len(ridge),'vol_target':scale[signal]})
        buy,sell=quote_permissions(opened,halted,official[day]['stk_limit'].set_index('ts_code'),bps)
        shared={}
        for kind,broker in books.items():
            night=economic_snapshot(broker,broker.account_snapshot(last,'isolated'),last,'isolated',signal,actions)
            nh={p['symbol']:int(p['quantity']) for p in night['positions'] if p['quantity']>0}
            nw=pd.Series({s:(q+night['unsettled_stock_rights'].get(s,0))*last[s]/night['equity'] for s,q in nh.items()},dtype=float)
            broker.settle_corporate_actions('isolated',day);broker.settle_available_positions('isolated',day)
            snap=economic_snapshot(broker,broker.account_snapshot(opened,'isolated'),opened,'isolated',day,actions)
            held={p['symbol']:int(p['quantity']) for p in snap['positions'] if p['quantity']>0}
            missing=set(held)-set(opened)|set(held)-set(closed)
            if missing:
                bars=ensure_quotes(ROOT,bars,missing,day);quotes={d:g.set_index('ts_code') for d,g in bars.groupby('trade_date')}
                opened.update(quotes[day].open.dropna().to_dict());closed.update(quotes[day].close.dropna().to_dict())
                if set(held)-set(opened)|set(held)-set(closed):raise ValueError('UNKNOWN_HELD_PRICE:'+day)
                buy,sell=quote_permissions(opened,halted,official[day]['stk_limit'].set_index('ts_code'),bps)
            scores=ep if kind.startswith('ep') else ridge
            if score_overrides is not None and kind in score_overrides:
                scores=score_overrides[kind].get(signal,pd.Series(dtype=float)).reindex(sorted(members)).dropna()
            target_scale=scale[signal] if kind.endswith('_vol') else static if kind=='ridge_static' else 1.
            interval=(selection_intervals or {}).get(kind,1)
            scheduled=i%interval==0 or not nh
            custom=target_builder(kind,signal,day,members,nh,nw,halted) if target_builder else None
            if custom is not None:
                targets,frozen,reason=custom
                if reason:blocks[kind].append({'date':day,'reason':reason,'symbol':None})
            elif not scheduled or len(scores)<50 or scores.std()<1e-12:
                # Mandatory universe exits are not optional rebalance signals.
                frozen=(set(nh)&members)|(set(nh)&halted)
                targets=nw.reindex(sorted(frozen)).dropna().to_dict()
                shared[kind]=targets
                if len(scores)<50 or scores.std()<1e-12:
                    blocks[kind].append({'date':day,'reason':'SIGNAL_MISSING_OR_DEGENERATE','symbol':None})
            elif kind in ('ridge_vol','ridge_static','ep_buffer_vol'):
                targets={s:w*target_scale for s,w in shared['ep_buffer' if kind=='ep_buffer_vol' else 'ridge_reference'].items()}
                frozen=set()
            else:
                # Missing member data is not an exit. Freeze its actual existing position.
                frozen=(set(nh)&members)-set(scores.index)
                frozen |= set(nh)&halted
                locked=nw.reindex(sorted(frozen)).dropna()
                free=max(0.,1-float(locked.sum()))
                tradable_scores=scores.drop(index=list(frozen),errors='ignore')
                active=nw.reindex(sorted((set(nh)&members&set(scores.index))-frozen)).dropna()
                if free<=0:targets=locked.to_dict()
                else:
                    selector=buffered_targets if kind.endswith('_buffer') else topk_dropout_targets
                    targets={s:w*free for s,w in selector(tradable_scores,active/free,topk=max(1,50-len(locked))).items()}
                    targets.update(locked.to_dict())
                shared[kind]=targets
            desired={}
            for s,w in targets.items():
                if s not in last or last[s]<=0:continue
                if target_scale==1 and s in nh and abs(w-nw.get(s,0))<1e-10:
                    desired[s]=held.get(s,0);continue
                q=w*night['equity']/last[s]
                for event in actions.loc[actions.symbol.eq(s)&actions.ex_date.eq(pd.Timestamp(day))].itertuples():q*=1+event.stock_ratio
                q=max(0,q-snap['unsettled_stock_rights'].get(s,0))
                desired[s]=int(q/100)*100
            for s in frozen:desired[s]=held.get(s,0)
            if execution_planner:
                desired,reason=execution_planner(kind=kind,index=i,signal=signal,day=day,members=members,night=night,snapshot=snap,night_holdings=nh,held=held,last=last,opened=opened,buy=buy,sell=sell,config=config,actions=actions,halted=halted)
                if reason:blocks[kind].append({'date':day,'reason':reason,'symbol':None})
            trades,blocked=execute(broker,kind,day,signal,desired,opened,buy,sell,config)
            if execution_planner:
                actual={p['symbol']:int(p['quantity']) for p in broker.account_snapshot(opened,'isolated')['positions'] if p['quantity']>0}
                assert actual=={s:int(q) for s,q in desired.items() if q>0}, 'PRECHECK_AND_EXECUTION_QUANTITIES_DIVERGED'
            fills[kind].extend(trades);blocks[kind].extend([{'date':day,**r} for r in blocked])
            owned={p['symbol'] for p in broker.account_snapshot(closed,'isolated')['positions'] if p['quantity']>0}
            for event in actions.loc[actions.symbol.isin(owned)&actions.record_date.eq(pd.Timestamp(day))].itertuples():
                broker.register_corporate_entitlement(account_id='isolated',symbol=event.symbol,action_key=event.action_key,record_date=day,cash_per_share=event.cash_per_share,stock_ratio=event.stock_ratio,cash_pay_date=None if pd.isna(event.cash_pay_date) else event.cash_pay_date.strftime('%Y%m%d'),stock_list_date=None if pd.isna(event.stock_list_date) else event.stock_list_date.strftime('%Y%m%d'))
            final=economic_snapshot(broker,broker.account_snapshot(closed,'isolated'),closed,'isolated',day,actions)
            assert broker.reconcile('isolated').matched
            assert final['cash']>=-.01
            histories[kind].append({'date':day,'equity':final['equity'],'cash':final['cash'],'exposure':final['market_value']/final['equity'],'target_scale':target_scale,'fills':len(trades),'fees':sum(r['fee'] for r in trades),'slippage':sum(r['slippage'] for r in trades)})
            for pos in final['positions']:
                if pos['quantity']>0:holdings.append({'date':day,'kind':kind,'symbol':pos['symbol'],'weight':pos['quantity']*closed[pos['symbol']]/final['equity']})
        last.update(closed)
        for kind,broker in books.items():
            with sqlite3.connect(broker.database) as src,sqlite3.connect(broker.database.with_name('resume.sqlite3')) as dst:src.backup(dst)
        save(checkpoint,{'completed':i+1,'histories':histories,'fills':fills,'blocks':blocks,'holdings':holdings,'coverage':coverage})
        save(output/'progress.json',{'completed':i+1,'total':len(sessions),'date':day,'elapsed_seconds':round(time.monotonic()-started)})
        if i%10==0:print(json.dumps({'bps':bps,'completed':i+1,'date':day}),flush=True)
    results=[];hf=pd.DataFrame(holdings,columns=['date','kind','symbol','weight'])
    for kind in KINDS:
        frame=pd.DataFrame(histories[kind]);ff=pd.DataFrame(fills[kind]);h=hf.loc[hf.kind.eq(kind)].copy()
        h['industry']=h.symbol.map(instruments.industry) if 'industry' in instruments else 'unavailable'
        h['growth_board']=h.symbol.str.startswith(('30','68'))
        groups=(h.groupby('industry').weight.sum()/len(sessions)).sort_values(ascending=False)
        quarters={}
        prev=config.initial_cash
        for q,g in frame.groupby(pd.to_datetime(frame.date).dt.to_period('Q')):
            quarters[str(q)]=period_statistics(g.equity,prev);prev=float(g.equity.iloc[-1])
        row={'strategy':kind,**period_statistics(frame.equity,config.initial_cash),'mean_exposure':float(frame.exposure.mean()),'fills':len(ff),'fees':float(ff.fee.sum()) if len(ff) else 0.,'slippage':float(ff.slippage.sum()) if len(ff) else 0.,'blocked_orders':len(blocks[kind]),'trading_days':int(frame.fills.gt(0).sum()),'growth_board_mean_weight':float(h.loc[h.growth_board].weight.sum()/len(sessions)),'industry_mean_weights':groups.to_dict(),'quarters':quarters}
        results.append(row);frame.to_parquet(output/kind/'equity.parquet',index=False);ff.to_parquet(output/kind/'fills.parquet',index=False)
        save(output/kind/'blocks.json',blocks[kind])
    hf.to_parquet(output/'holdings.parquet',index=False)
    save(output/'coverage.json',coverage)
    save(output/'results.json',{'status':'COMPLETE_RETROSPECTIVE_MECHANISM_STUDY','smoke':smoke,'cases':results,'all_daily_reconciliations_pass':True,'production_modified':False,'elapsed_seconds':time.monotonic()-started})
    print(json.dumps([{k:r[k] for k in ['strategy','net_return','sharpe','maximum_drawdown','fills']} for r in results]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bps',type=int,default=10,choices=[10,20]);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    run(args.bps,args.smoke)
