"""Close the bounded N4 experiment including post-result explanatory controls."""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.research.ashare_mainline import sha256
from moneymore.research.period_statistics import period_statistics
from prepare_four_strategy_2026 import save
from report_ashare_mainline_replay import audit_ledger
from report_ashare_n4 import attribution
from report_ashare_rank_preserving import daily_returns

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'state/ashare-mainline-20261112'
RUNS = BASE / 'n4-matrix-v2'
OUT = RUNS / 'completion-v2'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    evidence = [RUNS/'final/results.json', RUNS/'fixed10-allocation-execution20/results.json']
    primary = json.loads(evidence[0].read_bytes())
    fixed = json.loads(evidence[1].read_bytes())
    if primary['status'] != 'N4_COMPLETE_ISOLATED_RESEARCH' or len(primary['cases']) != 22:
        raise ValueError('INCOMPLETE_PRIMARY')
    if fixed['status'] != 'FIXED_POLICY_COST_DIAGNOSIS_COMPLETE' or not fixed['ledger']['matched']:
        raise ValueError('INCOMPLETE_FIXED_COST_CONTROL')
    rows = list(primary['cases'])
    fixed_row = {'period': 'fixed_cost_control', 'case': 'hrp_positive_fixed10', 'bps': 20,
                 **fixed['stats'], 'ledger': fixed['ledger'], 'attribution': fixed['attribution'],
                 'fees': fixed['fees'], 'slippage': fixed['slippage'], 'fills': fixed['fills']}
    rows.append(fixed_row)
    panel = pd.read_parquet(BASE/'panel-v2/panel.parquet', columns=['date', 'symbol', 'industry'])
    for bps in [10, 20]:
        allocation_path = RUNS/f'allocation-cost{bps}.json'
        allocation = json.loads(allocation_path.read_bytes())
        evidence.append(allocation_path)
        if allocation['status'] != 'COMPLETE' or len(allocation['records']) != 30:
            raise ValueError('INCOMPLETE_NATIVE_ALLOCATION')
        for record in allocation['records']:
            if not record['return_end'] <= record['signal_date'] < record['execution_date']:
                raise ValueError('FUTURE_ALLOCATION')
        run = RUNS/f'style-controls-v2-cost{bps}'
        result_path = run/'results.json'
        result = json.loads(result_path.read_bytes())
        evidence.append(result_path)
        if (result['status'] != 'SIGN_LOCKED_STYLE_CONTROLS_COMPLETE'
                or result['runner_sha256'] != sha256(ROOT/'scripts/replay_ashare_n4_style_controls.py')
                or result['config_sha256'] != sha256(ROOT/'configs/ashare_n4_style_controls_v1.json')):
            raise ValueError('INCOMPLETE_OR_CHANGED_STYLE_CONTROL')
        folder = run/f'bps{bps}'
        holding_all = pd.read_parquet(folder/'holdings.parquet')
        core = pd.read_parquet(RUNS/f'evaluation-cost{bps}/bps{bps}/doubleensemble_full--rank_cap/equity.parquet')
        for case in ['factor_momentum--rank_cap', 'factor_reversal--rank_cap']:
            efile, ffile = folder/case/'equity.parquet', folder/case/'fills.parquet'
            equity, fills = pd.read_parquet(efile), pd.read_parquet(ffile)
            evidence.extend([efile, ffile])
            pd.testing.assert_series_equal(equity.date.reset_index(drop=True), core.date.reset_index(drop=True))
            stats = period_statistics(equity.equity, 10_000_000)
            recorded = next(r for r in result['cases'] if r['strategy'] == case)
            for field in ['net_return', 'sharpe', 'maximum_drawdown', 'annualized_volatility']:
                if not np.isclose(stats[field], recorded[field], atol=1e-10):
                    raise ValueError('STYLE_METRIC_MISMATCH')
            holdings = holding_all.loc[holding_all.kind.eq(case)]
            decisions = [json.loads(p.read_bytes()) for p in sorted((run/'decisions'/case).glob('*.json'))]
            if len(decisions) != 181:
                raise ValueError('INCOMPLETE_STYLE_DECISIONS')
            for d in decisions:
                if not d['signal_date'] < d['execution_date']:
                    raise ValueError('STYLE_LOOKAHEAD')
                if d['status'] == 'OPEN_PLAN_VERIFIED':
                    post = d['open_postcheck']
                    if (not post['passed'] or post['maximum_stock_weight'] > .05001
                            or any(w > post['industry_limits'][s]+1e-5 for s, w in post['industry_weights'].items())):
                        raise ValueError('STYLE_POST_EXECUTION_LIMIT')
            rows.append({'period': 'sign_locked_control', 'case': case, 'bps': bps, **stats,
                         'mean_exposure': float(equity.exposure.mean()), 'fills': len(fills),
                         'fees': float(fills.fee.sum()), 'slippage': float(fills.slippage.sum()),
                         'growth_board_mean_weight': float(holdings.loc[holdings.symbol.str.startswith(('30','68'))].weight.sum()/181),
                         'correlation_with_core': float(np.corrcoef(daily_returns(equity, 10_000_000), daily_returns(core, 10_000_000))[0, 1]),
                         'decision_reasons': dict(Counter(d['reason'] for d in decisions if d.get('reason'))),
                         'ledger': audit_ledger(folder/case/'paper.sqlite3', fills),
                         'attribution': attribution(folder, case, equity, fills, holdings, panel),
                         'quarters': recorded['quarters'], 'post_result_explanatory_not_preregistered': True})
    if len(rows) != 27 or not all(r['ledger']['matched'] for r in rows):
        raise ValueError('INCOMPLETE_ACCOUNT_AUDIT')
    max_residual = max(r['attribution']['max_daily_unattributed_rights_or_receivable_residual'] for r in rows)
    if max_residual > 1e-5:
        raise ValueError('UNEXPLAINED_PNL')
    modules = ['n4_allocation', 'n4_execution', 'rank_preserving', 'rank_preserving_execution',
               'rank_preserving_report', 'rank_cadence_execution', 'rank_cadence_report',
               'mainline_execution', 'mainline_n3_final', 'ashare_mainline', 'period_statistics']
    command = [sys.executable, '-m', 'pytest', *[f'tests/test_{m}.py' for m in modules],
               '-q', '-p', 'no:cacheprovider', '--basetemp', str(OUT/'pytest-temp'),
               '--junitxml', str(OUT/'tests.xml')]
    tested = subprocess.run(command, cwd=ROOT, capture_output=True, encoding='utf-8', errors='replace')
    (OUT/'tests.log').write_text(tested.stdout+'\n'+tested.stderr, encoding='utf-8')
    if tested.returncode:
        raise ValueError('TESTS_FAILED')
    suites = ET.parse(OUT/'tests.xml').getroot().findall('testsuite')
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ['tests', 'failures', 'errors', 'skipped']}
    if counts['failures'] or counts['errors'] or counts['tests'] < 223:
        raise ValueError('TEST_REPORT_INCOMPLETE')
    evidence.extend([OUT/'tests.xml', OUT/'tests.log', ROOT/'configs/ashare_n4_v1.json',
                     ROOT/'configs/ashare_n4_execution_scope_v1.json', ROOT/'configs/ashare_n4_style_controls_v1.json'])
    result = {'status': 'N4_COMPLETE_WITH_EXPLANATORY_CONTROLS', 'account_count': len(rows),
              'cases': rows, 'max_daily_pnl_residual_yuan': max_residual, 'tests': counts,
              'evidence_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in evidence},
              'stale_allocation_progress_recovered': 'Original worker import failure retained; both completed native allocations and six blended accounts independently verified.',
              'failed_style_v1_preserved': 'Missing ROC5 previously dropped; v2 explicitly uses neutral 1.0, restarted from initial capital, no stitched NAV.',
              'production_modified': False, 'historical_period_previously_examined': True,
              'cross_security_merger_support': 'NOT_IMPLEMENTED; disclosed prior-known execution exclusion applies uniformly'}
    save(OUT/'results.json', result)
    print(json.dumps({'status': result['status'], 'accounts': len(rows), 'tests': counts, 'max_daily_pnl_residual_yuan': max_residual}), flush=True)


if __name__ == '__main__':
    main()
