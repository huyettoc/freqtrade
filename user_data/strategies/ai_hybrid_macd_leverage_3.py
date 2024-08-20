import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


sys.path.append(str(Path(__file__).parent))

from ai_hybrid_macd_strategy import AIHybridMACDStrategy  # noqa: E402


class AIHybridMACDLeverage3Strategy(AIHybridMACDStrategy):
    # "roi": {
    #   "0": 0.23399999999999999,
    #   "283": 0.079,
    #   "961": 0.053,
    #   "2103": 0
    # },
    # "stoploss": {
    #   "stoploss": -0.324
    # }
    stoploss = -0.32
    minimal_roi = {
        "0": 0.4,
        "333": 0.2,
        "1000": 0.1,
        "2000": 0
    }

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
