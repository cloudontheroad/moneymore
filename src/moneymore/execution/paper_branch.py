"""Atomic, auditable paper-account forks; never fabricates a new execution."""
from datetime import datetime, UTC
import json
import sqlite3


def annotate_inherited_fills(broker,account,rows=None):
    rows=broker.fills(account) if rows is None else rows
    with sqlite3.connect(broker.database) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_inherited_fills'").fetchone():return rows
        sources=dict(db.execute('SELECT fill_id,source_account FROM paper_inherited_fills WHERE account_id=?',(account,)))
    if not sources:return rows
    return [{**row,'inherited':row['id'] in sources,'source_account_id':sources.get(row['id'])} for row in rows]


def fork_paper_account(broker, source, target, branch_date):
    if source == target or not target.startswith('qlib_candidate_'):
        raise ValueError('INVALID_PAPER_BRANCH')
    if not broker.reconcile(source).matched:
        raise ValueError('SOURCE_RECONCILIATION_FAILED')
    with sqlite3.connect(broker.database, timeout=10) as db:
        db.row_factory=sqlite3.Row
        db.execute('BEGIN IMMEDIATE')
        db.execute('CREATE TABLE IF NOT EXISTS paper_account_branches (account_id TEXT PRIMARY KEY, source_account TEXT NOT NULL, branch_date TEXT NOT NULL, created_at TEXT NOT NULL, receipt TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS paper_inherited_fills (account_id TEXT NOT NULL, fill_id INTEGER NOT NULL, source_account TEXT NOT NULL, source_fill_id INTEGER NOT NULL, PRIMARY KEY(account_id, fill_id))')
        prior=db.execute('SELECT * FROM paper_account_branches WHERE account_id=?',(target,)).fetchone()
        if prior:
            if prior['source_account']!=source or prior['branch_date']!=branch_date:raise ValueError('BRANCH_IDENTITY_CHANGED')
            return json.loads(prior['receipt'])
        if db.execute('SELECT 1 FROM accounts WHERE account_id=?',(target,)).fetchone():raise ValueError('BRANCH_TARGET_ALREADY_EXISTS')
        if db.execute("SELECT 1 FROM orders WHERE account_id=? AND status='PENDING'",(source,)).fetchone():raise ValueError('SOURCE_HAS_PENDING_ORDERS')
        def rows(table):return [dict(r) for r in db.execute(f'SELECT * FROM {table} WHERE account_id=?',(source,))]
        def insert(table,row):
            columns=list(row)
            return db.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",[row[c] for c in columns])
        account=rows('accounts')
        if len(account)!=1:raise ValueError('SOURCE_ACCOUNT_MISSING')
        for table in ['accounts','positions','corporate_entitlements','corporate_action_ledger','account_risk_states']:
            for row in rows(table):
                row['account_id']=target
                row.pop('id',None)
                insert(table,row)
        inherited=0
        for row in rows('fills'):
            source_id=row.pop('id');old_key=row['idempotency_key']
            order=db.execute('SELECT * FROM orders WHERE idempotency_key=?',(old_key,)).fetchone()
            if not order or order['status']!='FILLED':raise ValueError('SOURCE_FILL_WITHOUT_FILLED_ORDER')
            new_key=f'{target}:inherited:{old_key}'
            new_order={**dict(order),'account_id':target,'idempotency_key':new_key}
            insert('orders',new_order)
            row.update(account_id=target,idempotency_key=new_key)
            new_id=insert('fills',row).lastrowid
            db.execute('INSERT INTO paper_inherited_fills VALUES (?,?,?,?)',(target,new_id,source,source_id));inherited+=1
        receipt={'account_id':target,'source_account':source,'branch_date':branch_date,'created_at':datetime.now(UTC).isoformat(),
            'cash':account[0]['cash'],'initial_cash':account[0]['initial_cash'],'inherited_fills':inherited,
            'inherited_history_is_not_new_strategy_execution':True,'real_orders_enabled':False}
        expected_cash=account[0]['initial_cash']
        expected_shares={}
        for fill in db.execute('SELECT * FROM fills WHERE account_id=?',(target,)):
            sign=1 if fill['side']=='BUY' else -1
            expected_cash-=sign*fill['quantity']*fill['price']+fill['fee']
            expected_shares[fill['symbol']]=expected_shares.get(fill['symbol'],0)+sign*fill['quantity']
        for action in db.execute('SELECT * FROM corporate_action_ledger WHERE account_id=?',(target,)):
            if action['action_type']=='CASH_DIVIDEND':expected_cash+=action['cash_amount']
            if action['action_type']=='STOCK_DIVIDEND':expected_shares[action['symbol']]=expected_shares.get(action['symbol'],0)+action['share_quantity']
        actual_shares={r['symbol']:r['quantity'] for r in db.execute('SELECT * FROM positions WHERE account_id=?',(target,)) if r['quantity']}
        if abs(expected_cash-account[0]['cash'])>=.01 or {s:q for s,q in expected_shares.items() if q}!=actual_shares:
            raise ValueError('BRANCH_TRANSACTION_RECONCILIATION_FAILED')
        db.execute('INSERT INTO paper_account_branches VALUES (?,?,?,?,?)',(target,source,branch_date,receipt['created_at'],json.dumps(receipt)))
    if not broker.reconcile(target).matched:raise ValueError('BRANCH_RECONCILIATION_FAILED')
    return receipt
