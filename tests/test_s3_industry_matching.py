import pandas as pd
from moneymore.challenger_attribution import matched_day,linked_attribution


def test_industry_matching_cannot_choose_closer_other_industry_peer():
    names=['HELD','SAME','OTHER'];features=pd.DataFrame({'size':[0.,100.,0.]},index=names)
    args=([{'symbol':'HELD','quantity':100}],1000.,0.,0.,pd.Series(1.,index=names),pd.Series(0.,index=names),features,names)
    old=matched_day(*args,neighbors=1)
    restricted=matched_day(*args,neighbors=1,industries=pd.Series({'HELD':'BANK','SAME':'BANK','OTHER':'CHIP'}))
    assert old['matches'][0]['peers']==['OTHER']
    assert restricted['matches'][0]['peers']==['SAME']


def test_no_same_industry_peer_is_reported_not_silently_substituted():
    names=['HELD','OTHER'];features=pd.DataFrame({'size':[0.,0.]},index=names)
    result=matched_day([{'symbol':'HELD','quantity':100}],1000.,0.,0.,pd.Series(1.,index=names),pd.Series(0.,index=names),features,names,
                       neighbors=1,industries=pd.Series({'HELD':'BANK','OTHER':'CHIP'}))
    assert result['status']=='INSUFFICIENT_STYLE_PEERS'


def test_unknown_industries_are_not_presented_as_real_same_industry_matches():
    names=['HELD','OTHER'];features=pd.DataFrame({'size':[0.,0.]},index=names)
    result=matched_day([{'symbol':'HELD','quantity':100}],1000.,0.,0.,pd.Series(1.,index=names),pd.Series(0.,index=names),features,names,
                       neighbors=1,industries=pd.Series('UNKNOWN',index=names))
    assert result['status']=='UNKNOWN_HELD_INDUSTRY'


def test_unmatched_holding_is_separate_pnl_not_zero_or_cross_industry_match():
    names=['HELD','OTHER'];features=pd.DataFrame({'size':[0.,0.]},index=names)
    result=matched_day([{'symbol':'HELD','quantity':100}],1000.,.02,5.,pd.Series(1.,index=names),pd.Series({'HELD':.1,'OTHER':0.}),features,names,
        neighbors=1,industries=pd.Series({'HELD':'BANK','OTHER':'CHIP'}),allow_partial=True)
    assert result['status']=='PARTIALLY_MATCHED' and result['matches']==[]
    assert abs(result['unmatched_stock']-.01)<1e-12 and result['selection']==0
    assert result['unmatched'][0]['reason']=='INSUFFICIENT_STYLE_PEERS'
    linked=linked_attribution([result],('market','style','selection','unmatched_stock','trading_and_other','costs'))
    assert abs(linked['total_return']-.02)<1e-12 and abs(linked['reconciliation_error'])<1e-12
