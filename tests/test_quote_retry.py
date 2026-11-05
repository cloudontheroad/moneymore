import pandas as pd
import pytest
from moneymore.data.quote_retry import RetryingQuotes


class Provider:
    name = "fixture"
    def __init__(self, successes):
        self.successes = iter(successes)
        self.calls = 0
    def intraday_quotes(self, symbols):
        self.calls += 1
        ready = next(self.successes)
        if isinstance(ready, Exception):
            raise ready
        return pd.DataFrame({"symbol": symbols, "execution_ready": [ready]*len(symbols)})


def test_quality_retry_recovers_without_altering_flags():
    p = Provider([False, False, True]); waits = []
    result = RetryingQuotes(p, sleep=waits.append).intraday_quotes(["x"])
    assert result.execution_ready.all() and p.calls == 3 and waits == [2, 4]


def test_persistent_bad_quality_stays_blocked_and_is_bounded():
    p = Provider([False]*5)
    result = RetryingQuotes(p, sleep=lambda _: None).intraday_quotes(["x"])
    assert not result.execution_ready.any() and p.calls == 3


def test_rate_limit_or_auth_failure_not_retried():
    p = Provider([RuntimeError("provider cooling down")])
    with pytest.raises(RuntimeError):
        RetryingQuotes(p, sleep=lambda _: None).intraday_quotes(["x"])
    assert p.calls == 1


def test_retries_only_unready_symbols_and_preserves_timestamps():
    class Partial:
        name = 'partial'
        def __init__(self): self.calls=[]
        def intraday_quotes(self, symbols):
            self.calls.append(symbols)
            return pd.DataFrame([{'symbol':s,'execution_ready':s=='a' or len(self.calls)>1,
                'timestamp':100000,'depth_timestamp':100000} for s in symbols])
    p=Partial()
    result=RetryingQuotes(p, sleep=lambda _:None, now=lambda:101).intraday_quotes(['a','b'])
    assert p.calls==[['a','b'],['b']]
    assert result.execution_ready.all() and result.timestamp.tolist()==[100000,100000]


def test_retained_quote_must_not_expire_during_retry():
    class Old:
        name='old'
        def intraday_quotes(self, symbols):
            return pd.DataFrame([{'symbol':s,'execution_ready':True,'timestamp':1000,'depth_timestamp':1000} for s in symbols])
    result=RetryingQuotes(Old(),sleep=lambda _:None,now=lambda:40).intraday_quotes(['a'])
    assert not result.execution_ready.any()


def test_post_solver_retry_uses_existing_stricter_ten_second_gate():
    class Advancing:
        name='advancing'
        def __init__(self): self.calls=0
        def intraday_quotes(self, symbols):
            self.calls+=1
            stamp=5000 if self.calls==1 else 19000
            return pd.DataFrame([{'symbol':s,'execution_ready':True,'timestamp':stamp,'depth_timestamp':stamp} for s in symbols])
    p=Advancing()
    result=RetryingQuotes(p,sleep=lambda _:None,now=lambda:20).fresh_intraday_quotes(['a'])
    assert p.calls==2 and result.timestamp.iloc[0]==19000


def test_backoff_expired_rows_are_requested_again():
    clock=[20.]
    class Aging:
        name='aging'
        def __init__(self): self.calls=[]
        def intraday_quotes(self,symbols):
            self.calls.append(symbols)
            return pd.DataFrame([{'symbol':s,'execution_ready':s=='a' or len(self.calls)>1,
                'timestamp':11000 if len(self.calls)==1 else 22000,'depth_timestamp':11000 if len(self.calls)==1 else 22000} for s in symbols])
    p=Aging()
    result=RetryingQuotes(p,sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds),now=lambda:clock[0]).fresh_intraday_quotes(['a','b'])
    assert p.calls==[['a','b'],['a','b']] and result.execution_ready.all()
