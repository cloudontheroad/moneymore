import hashlib
import json
from pathlib import Path
import runpy
import sys

import pytest


def closed_workflow(tmp_path):
    evidence = tmp_path / 'evidence.json'
    evidence.write_text('{}')
    review = tmp_path / 'review.json'
    review.write_text(json.dumps({'input_hashes': {str(evidence): hashlib.sha256(evidence.read_bytes()).hexdigest()}}))
    progress = {'status': 'RESEARCH_REVIEW_COMPLETE_NO_ELIGIBLE_NEW_MODEL', 'steps': [], 'studies': [],
                'final_review_path': str(review), 'final_review_sha256': hashlib.sha256(review.read_bytes()).hexdigest()}
    (tmp_path / 'progress.json').write_text(json.dumps(progress))
    return evidence


def invoke(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run_s3_cost_matrix.py', '--output', str(tmp_path), '--first-study', str(tmp_path), '--resume', '--complete-analysis'])
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/run_s3_cost_matrix.py'), run_name='__main__')


def test_closed_review_resume_does_not_reopen_or_recompute(tmp_path, monkeypatch, capsys):
    closed_workflow(tmp_path)
    before = (tmp_path / 'progress.json').read_bytes()
    invoke(tmp_path, monkeypatch)
    assert 'ALREADY_CLOSED_NO_RECOMPUTE' in capsys.readouterr().out
    assert before == (tmp_path / 'progress.json').read_bytes()


def test_closed_review_resume_rejects_changed_evidence(tmp_path, monkeypatch):
    evidence = closed_workflow(tmp_path)
    evidence.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='FINAL_REVIEW_EVIDENCE_CHANGED'):
        invoke(tmp_path, monkeypatch)
