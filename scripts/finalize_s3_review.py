"""Close this frozen research batch; never select by returns or start an account."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

from moneymore.research.s3_review import review_decision, read_public_review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish-view', action='store_true')
    parser.add_argument('--ui', type=Path)
    parser.add_argument('--close-workflow', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / 'state/style-balance'
    if args.output.exists():
        raise FileExistsError('IMMUTABLE_RESEARCH_REVIEW')
    hashes = {}

    def read(path):
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError('EVIDENCE_OUTSIDE_PROJECT')
        body = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(body).hexdigest()
        return json.loads(body)

    matrix = read(base / 's3-cost-matrix-20261103-a/cost-matrix-complete.json')
    counts, fills = Counter(), 0
    if len(matrix['studies']) != 8:
        raise ValueError('EIGHT_COMPLETED_STUDIES_REQUIRED')
    for study in matrix['studies']:
        folder = Path(study['path'])
        ledger = read(folder / 'independent-audit.json')
        read(folder / 'results.json')
        original = read(folder / 'execution-summary.json')
        reviewed = folder / 'execution-summary-reviewed.json'
        execution = read(reviewed) if reviewed.exists() else original
        if ledger['status'] != 'PASS_LEDGER_AUDIT_NOT_STRATEGY_ACCEPTANCE' or len(ledger['cases']) != 6:
            raise ValueError('INDEPENDENT_SIX_ACCOUNT_LEDGER_AUDIT_REQUIRED')
        if execution['counts'] != study['counts'] or execution['fills'] != study['fills']:
            raise ValueError('MATRIX_EXECUTION_BINDING_CHANGED')
        if set(execution['counts']) - {'executed', 'verified_constraint_no_trade', 'data_policy_no_trade_not_model_success'}:
            raise ValueError('UNRESOLVED_EXECUTION_OUTCOME')
        counts.update(execution['counts'])
        fills += execution['fills']
    attributed = read(base / 's3-cost-attribution-reviewed-20261103-a/results.json')
    residual = read(base / 's3-residual-audit-20261103-a/results.json')
    named = read(base / 's3-named-history-20261103-a/results.json')
    risk = read(base / 's3-daily-risk-audit-20261103-a/results.json')
    reference = read(base / 's3-reference-contract-20261103-a/results.json')
    external = read(base / 's3-external-index-20261103-a/results.json')
    qa = read(args.qa / 'results.json')
    ui = read(args.ui / 'results.json') if args.ui else None
    if args.close_workflow and (not args.publish_view or ui is None or ui['status'] != 'LOCAL_API_AND_BROWSER_VERIFIED'):
        raise ValueError('WORKFLOW_CLOSE_REQUIRES_ACTUAL_SERVING_ACCEPTANCE')
    if qa['status'] != 'DELIVERY_MECHANISMS_VERIFIED_NOT_MODEL_PUBLICATION' or any(c['status'] != 'PASS' for c in qa['checks']):
        raise ValueError('DELIVERY_VERIFICATION_REQUIRED')
    cases = attributed['cases']
    identities = {(c['fold'], c['slippage_bps'], c['account']) for c in cases}
    if len(cases) != 48 or len(identities) != 48:
        raise ValueError('COMPLETE_UNIQUE_COST_MATRIX_REQUIRED')
    for case in cases:
        for size in ('3', '5', '10'):
            attribution = case['attribution'][size]
            if not attribution['accounting_complete'] or abs(attribution['linked']['reconciliation_error']) > 1e-8:
                raise ValueError('RETURN_ACCOUNTING_NOT_RECONCILED')
    comparison = []
    labels = {'raw': '原始信号', 'contemporaneous_style_component': '风格分量', 'contemporaneous_score_residual': '残差信号'}
    for account in sorted({c['account'] for c in cases}):
        group = [c for c in cases if c['account'] == account]
        high = [c for c in group if c['slippage_bps'] == 10]
        low = [c for c in group if c['slippage_bps'] == 2]
        if len(high) != 4 or len(low) != 4:
            raise ValueError('FOUR_WINDOWS_PER_COST_REQUIRED')
        boards = [d['dual_board_stock_sleeve_weight'] for c in high for d in c['board_exposure'][:-1]
                  if d['held_names'] > 0 and d['dual_board_stock_sleeve_weight'] is not None]
        comparison.append({
            'account': account,
            'name': ('LightGBM' if group[0]['model'] == 'lightgbm' else 'Ridge') + ' · ' + labels[group[0]['variant']],
            'return_10bps': mean(c['net_return'] for c in high),
            'return_2bps': mean(c['net_return'] for c in low),
            'pure_cost_delta': mean(c['pure_cost_counterfactual']['cash_saved'] / 10_000_000 for c in high),
            'board_weight': mean(boards) if boards else None,
            'board_measured_nonterminal_held_days': len(boards),
            'matched_windows': str(sum(c['attribution']['5']['complete'] for c in high)) + '/4（10bps）',
            'matched_capital_coverage': mean(c['attribution']['5']['matched_capital_coverage_mean'] for c in high),
        })
    intervals = {s['model']: s['metrics']['score_residual_vs_raw_return'] for s in residual['summary']}
    independent_positive = [model for model, interval in intervals.items() if interval['lower'] > 0]
    # A positive interval would still require an economic-increment assessment.
    # This finite batch cannot turn a mere return ranking into a PASS decision.
    prediction_status = 'FAIL' if not independent_positive else 'INCOMPLETE'
    reports = {
        'prediction': {'status': prediction_status,
            'reason': '两模型残差信号的区块置信区间均跨0，未证明稳定独立选股增量；不按短期收益或低含科率选赢家。'
                      if not independent_positive else '独立预测区间为正仍须验证成本后增量，不能仅按收益排名发布。',
            'independent_ic_intervals': intervals, 'independently_positive_models': independent_positive},
        'risk': {'status': 'INCOMPLETE',
            'reason': '已补命名风险历史；流通市值参照仅53/84日完整，仍有持仓风险缺测与收盘漂移，发布级连续保护未获验证。',
            'reference_complete_dates': reference['complete_dates'], 'reference_required_dates': reference['required_dates'],
            'named_factor_sessions': named['measured_factor_sessions'],
            'asof_covariance_sessions': sum(r['status'] == 'MEASURED' for r in named['covariance_audits']),
            'covariance_warmup_unavailable': sum(r['status'] != 'MEASURED' for r in named['covariance_audits']),
            'normal_close_risk_counts': risk['normal_counts'],
            'drift_is_not_a_retroactive_fill_violation': True, 'new_named_covariance_deployed': False},
        'execution': {'status': 'PASS',
            'reason': f'8次独立账本审计通过，{fills:,}笔模拟成交；184次调仓执行，8次保护性不交易均有解释，无未解决求解失败。',
            'scope': 'offline_paper_execution_only_not_broker_or_live_account_acceptance',
            'counts': dict(counts), 'fills': fills, 'unresolved_solver_failures': 0},
        'runtime': {'status': 'NOT_TRIGGERED',
            'reason': '研究交付回归通过；A/B未通过，不创建新账户、不启动新模型周任务或替换旧策略。',
            'delivery_checks': qa['checks'], 'pytest': qa['pytest'], 'new_account_created': False},
        'forward_observation': {'status': 'NOT_TRIGGERED',
            'reason': '无合格新模型，未启动新的前瞻观察；历史回放不计入20/60日评审。',
            'mature_prospective_days': 0, 'backfilled_days_counted': False},
    }
    decision = review_decision(reports)
    if decision['observation_activation_authorized'] or decision['real_orders_enabled']:
        raise ValueError('THIS_BATCH_HAS_NO_AUTHORIZED_NEW_MODEL')
    packet = {
        'schema': 'S3_REVIEW_V1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'frozen_S3_research_batch_only_not_existing_1029_V2',
        'input_hashes': hashes, 'reports': reports, 'decision': decision,
        'public_summary': {
            'title': '本批研究暂无合格新模型',
            'reason': '成本回放和对账已完成，但独立预测与发布级风险证据未过门禁。维持现有账户；不修改历史、不强行开新候选。',
            'time_range': '2024—2025 · 四个独立21交易日窗口',
            'completed_studies': 8, 'account_cases': 48, 'fills': fills, 'comparison': comparison,
        },
        'accounting': {
            'complete_bridges': sum(c['attribution']['5']['accounting_complete'] for c in cases),
            'full_exact_peer_windows': sum(c['attribution']['5']['complete'] for c in cases),
            'mean_matched_capital_coverage': mean(c['attribution']['5']['matched_capital_coverage_mean'] for c in cases),
            'partial_bridge_is_not_full_style_attribution_or_causal_alpha': True,
            'unmatched_stock_proxy_pnl_separate': True,
            'maximum_reconciliation_error': max(abs(c['attribution']['5']['linked']['reconciliation_error']) for c in cases),
        },
        'external_price_index_reference': external,
        'real_orders_enabled': False, 'legacy_accounts_or_history_modified': False,
        'new_candidate_created': False, 'research_batch_closed': True,
        'serving_acceptance': ui,
    }
    args.output.mkdir(parents=True)
    serialized = json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False)
    (args.output / 'results.json').write_text(serialized, encoding='utf-8')
    if args.publish_view:
        view = base / 's3-release-review.json'
        temporary = view.with_suffix('.json.tmp')
        temporary.write_text(serialized, encoding='utf-8')
        checked = read_public_review(temporary)
        if ui and hashlib.sha256(json.dumps(checked, sort_keys=True).encode()).hexdigest() != ui['public_payload_sha256']:
            raise ValueError('PUBLIC_RESULT_DIFFERS_FROM_BROWSER_ACCEPTANCE')
        os.replace(temporary, view)
        public = read_public_review(view)
        if public is None or public['status'] == 'UNAVAILABLE':
            raise ValueError('PUBLIC_REVIEW_INTEGRITY_NOT_VERIFIED')
    if args.close_workflow:
        progress_path = base / 's3-cost-matrix-20261103-a/progress.json'
        progress = json.loads(progress_path.read_bytes())
        progress.update(status='RESEARCH_REVIEW_COMPLETE_NO_ELIGIBLE_NEW_MODEL', current=None,
                        final_review_path=str((args.output / 'results.json').resolve()),
                        final_review_sha256=hashlib.sha256((args.output / 'results.json').read_bytes()).hexdigest(),
                        local_service_and_browser_verified=True, real_orders_enabled=False)
        temporary_progress = progress_path.with_suffix('.json.tmp')
        temporary_progress.write_text(json.dumps(progress, indent=2), encoding='utf-8')
        os.replace(temporary_progress, progress_path)
    print(json.dumps({'status': decision['status'], 'fills': fills, 'pytest': qa['pytest'],
                      'new_account_created': False, 'real_orders_enabled': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
