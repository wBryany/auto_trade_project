from dataclasses import replace
from unittest.mock import patch
import pytest
from btc_futures_bot.models import Candle,Position,Signal
from btc_futures_bot.strategy import StrategyConfig,_TraditionalSetupState,invalidate_breakout_setup,breakout_failure_exit_reason

def test_persisted_breakout_expires_on_close_but_not_wick():
 s=_TraditionalSetupState(False,False,False,False,True,False,True,False,100,90)
 assert invalidate_breakout_setup(s,[Candle(1,101,103,99,101,1)]).long_ready
 assert not invalidate_breakout_setup(s,[Candle(1,101,103,99,99,1),Candle(2,99,104,98,103,1)]).long_ready
 assert invalidate_breakout_setup(replace(s,golden_cross=True),[Candle(1,101,103,99,99,1)]).golden_cross
 short=replace(s,breakout_long=False,breakout_short=True,breakout_short_raw=True)
 assert not invalidate_breakout_setup(short,[Candle(1,89,92,88,91,1)]).breakout_short

@pytest.mark.parametrize('side',['long','short'])
def test_failure_requires_two_post_entry_closes_momentum_and_current_price(side):
 price=99 if side=='long' else 101
 p=Position(side,1,100,95 if side=='long' else 105,110,1)
 sig=Signal(side,7,0,('5m_breakout' if side=='long' else '5m_breakdown','breakout_level=100'))
 cfg=StrategyConfig(trigger_timeframe='5m',enable_breakout_failure_exit=True)
 bars=[Candle(t,price,price+1,price-1,price,1) for t in [300000,600000]]
 minutes=[Candle(t,price,price+1,price-1,price,1) for t in [780000,840000]]
 data={'5m':bars,'1m':minutes}
 with patch('btc_futures_bot.strategy._one_minute_adverse_confirmation',return_value=True):
  assert breakout_failure_exit_reason(p,sig,data,cfg,price,900000)=='breakout_failure'
  assert not breakout_failure_exit_reason(p,sig,data,cfg,100,900000)
  assert not breakout_failure_exit_reason(p,sig,data,cfg,price,899999)
  assert not breakout_failure_exit_reason(replace(p,opened_at=300001),sig,data,cfg,price,900000)
  assert not breakout_failure_exit_reason(p,replace(sig,reasons=sig.reasons[:1]),data,cfg,price,900000)
  assert not breakout_failure_exit_reason(p,sig,data,cfg,price,1200000)
  assert not breakout_failure_exit_reason(p,sig,{'5m':[replace(bars[0],close=100),bars[1]],'1m':minutes},cfg,price,900000)
 with patch('btc_futures_bot.strategy._one_minute_adverse_confirmation',return_value=False):
  assert not breakout_failure_exit_reason(p,sig,data,cfg,price,900000)
