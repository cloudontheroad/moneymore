"""User-directed paper candidate archival; preserves holdings, fills and NAV."""
import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import sqlite3

from moneymore.candidate_lifecycle import lifecycle_rows
from moneymore.qlib_candidate_observer import candidate_catalog
from moneymore.execution.paper import PaperBroker


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tags',nargs='+',required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];catalog={row['candidate_tag']:row for row in candidate_catalog(root)}
    tags=list(dict.fromkeys(args.tags))
    if set(tags)-set(catalog):raise ValueError('UNKNOWN_CANDIDATE')
    now=datetime.now(UTC).isoformat();day=datetime.now().strftime('%Y%m%d');database=root/'state/candidate-lifecycle.sqlite3'
    backup=root/'state/candidate-archive-backups'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'.sqlite3');backup.parent.mkdir(parents=True,exist_ok=True)
    previous=lifecycle_rows(root)
    with sqlite3.connect(database,timeout=30) as connection,sqlite3.connect(backup) as saved:
        connection.backup(saved)
        connection.execute('CREATE TABLE IF NOT EXISTS manual_archive_events (candidate_tag TEXT, archived_at TEXT, previous_payload TEXT, new_payload TEXT, PRIMARY KEY(candidate_tag,archived_at))')
        for tag in tags:
            state={**previous.get(tag,{}),'candidate_tag':tag,'status':'REJECTED','manual_archive':True,
                'review_date':day,'archived_at':now,'reason':'用户主动终止观察并归档；非统计评审结论，不再新增交易',
                'promotion_automatic':False,'history_preserved':True,'position_policy':'FREEZE_LEDGER_NO_SYNTHETIC_LIQUIDATION'}
            payload=json.dumps(state,ensure_ascii=False,allow_nan=False)
            connection.execute('INSERT INTO manual_archive_events VALUES (?,?,?,?)',(tag,now,json.dumps(previous.get(tag),ensure_ascii=False),payload))
            connection.execute('INSERT INTO candidate_state VALUES (?,?) ON CONFLICT(candidate_tag) DO UPDATE SET payload=excluded.payload',(tag,payload))
    broker=PaperBroker(root/'state/paper_orders.sqlite3')
    for tag in tags:
        cancelled=broker.cancel_pending(catalog[tag]['account_id'],'USER_ARCHIVED_CANDIDATE')
        print(json.dumps({'tag':tag,'archived':True,'cancelled_pending':cancelled,'positions_and_history_preserved':True}))


if __name__=='__main__':main()
