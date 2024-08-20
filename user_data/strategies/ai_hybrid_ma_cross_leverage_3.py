import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


sys.path.append(str(Path(__file__).parent))

from ai_hybrid_ma_cross_strategy import AIHybridMACrossStrategy  # noqa: E402


class AIHybridMACrossLeverage3Strategy(AIHybridMACrossStrategy):
    # "roi": {
    #   "0": 0.42700000000000005,
    #   "264": 0.158,
    #   "454": 0.055,
    #   "1605": 0
    # },
    # "stoploss": {
    #   "stoploss": -0.055
    minimal_roi = {
        "0": 0.4,
        "300": 0.15,
        "450": 0.05,
        "1500": 0,
        "5000": -1
    }

    # Stoploss:
    stoploss = -0.05

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 3
