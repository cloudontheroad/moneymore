"""Pin mutable reference tables before the fixed six-cell final study."""
import hashlib,json,shutil
from pathlib import Path
from prepare_four_strategy_2026 import ROOT,OUT as STUDY,save
OUT=ROOT/'state/cost-aware-ab-v2'
TABLES=['instruments','trade_calendar','daily_basic','fina_indicator_research_v2','quarterly_revenue_supplement_v2','baseline_v2_industry_membership']
def freeze():
    folder=OUT/'frozen-data/processed';folder.mkdir(parents=True,exist_ok=True)
    for name in TABLES:
        source=ROOT/'data/processed'/(name+'.parquet');target=folder/source.name
        if source.exists() and not target.exists():shutil.copy2(source,target)
    paths=[*sorted(folder.glob('*.parquet')),STUDY/'model/predictions.parquet',OUT/'t2/historical.json',
        ROOT/'state/mature-multifactor/p1-native-20261103-a/bars-input.parquet',ROOT/'state/mature-multifactor/p1-native-20261103-a/membership.parquet',
        *sorted((ROOT/'src/moneymore/research').glob('cost_*.py'))]
    manifest={'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'reference_worker_compatibility':'legacy reference logic and immutable source tables unchanged; non-reference preflight outputs archived',
        'tables':TABLES,'real_orders_enabled':False}
    path=OUT/'t5-frozen-inputs.json'
    if path.exists() and json.loads(path.read_bytes())!=manifest:raise ValueError('FROZEN_COMPONENT_INPUTS_CHANGED')
    save(path,manifest)
if __name__=='__main__':freeze()
