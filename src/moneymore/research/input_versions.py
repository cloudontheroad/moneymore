"""Version identity guard; file content hashes are checked by each caller."""
from pathlib import Path


def require_input_version(report, inputs):
    expected=report.get('input_version')
    if (expected is None)!=(inputs is None):
        raise ValueError('RESEARCH_INPUT_VERSION_MISMATCH')
    if expected is not None and Path(expected).resolve()!=Path(inputs).resolve():
        raise ValueError('RESEARCH_INPUT_VERSION_MISMATCH')
