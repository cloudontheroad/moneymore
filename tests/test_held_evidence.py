import hashlib
import json
import pandas as pd
import pytest
from moneymore.research.held_evidence import read_held_basic


def fixture(folder, symbol='A', day='20240722'):
    file=folder/'basic.parquet'
    pd.DataFrame([{'ts_code':symbol,'trade_date':day,'total_mv':100.,'pb':1.,'pe_ttm':10.,'dv_ttm':2.}]).to_parquet(file)
    manifest={'status':'READY','requests':[{'endpoint':'daily_basic','file':file.name,
        'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'params':{'ts_code':'A','trade_date':'20240722'}}]}
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return file


def test_read_exact_dated_basic(tmp_path):
    fixture(tmp_path)
    assert read_held_basic(tmp_path).total_mv.iloc[0]==100.


def test_mutated_evidence_blocks(tmp_path):
    file=fixture(tmp_path);file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='HELD_EVIDENCE_CHANGED'):read_held_basic(tmp_path)


@pytest.mark.parametrize('symbol,day',[('B','20240722'),('A','20240723')])
def test_wrong_identity_blocks_even_with_matching_hash(tmp_path,symbol,day):
    fixture(tmp_path,symbol,day)
    with pytest.raises(ValueError,match='IDENTITY_MISMATCH'):read_held_basic(tmp_path)
