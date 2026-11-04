import importlib.util
import json
from pathlib import Path

def test_atomic_research_progress_write_recovers_from_brief_windows_reader_lock(tmp_path,monkeypatch):
    source=Path('scripts/prepare_four_strategy_2026.py')
    spec=importlib.util.spec_from_file_location('research_save_test',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    real=Path.replace;attempts=[]
    def transient(self,target):
        attempts.append(str(self))
        if len(attempts)<3:raise PermissionError('simulated Windows reader lock')
        return real(self,target)
    monkeypatch.setattr(Path,'replace',transient);monkeypatch.setattr(module.time,'sleep',lambda *a:None)
    target=tmp_path/'progress.json';module.save(target,{'completed':67})
    assert json.loads(target.read_bytes())=={'completed':67}
    assert len(attempts)==3 and attempts[0]==attempts[1]
