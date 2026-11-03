import pytest
from moneymore.research.input_versions import require_input_version


@pytest.mark.parametrize('report,inputs',[({},'v2'),({'input_version':'v2'},None),({'input_version':'v1'},'v2')])
def test_missing_or_mixed_versions_rejected(report,inputs):
    with pytest.raises(ValueError,match='VERSION_MISMATCH'):require_input_version(report,inputs)


def test_same_version_and_legacy_defaults_allowed(tmp_path):
    require_input_version({},None)
    require_input_version({'input_version':str(tmp_path/'v2')},tmp_path/'v2')
