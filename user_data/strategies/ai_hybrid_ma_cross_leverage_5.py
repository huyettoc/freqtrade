import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


sys.path.append(str(Path(__file__).parent))

from ai_hybrid_ma_cross_strategy import AIHybridMACrossStrategy  # noqa: E402


class AIHybridMACrossLeverage5Strategy(AIHybridMACrossStrategy):
    # minimal_roi = {
    #     "0": 0.458,
    #     "463": 0.23,
    #     "703": 0.09,
    #     "947": 0
    # }
    #
    # # Stoploss:
    # stoploss = -0.294

    minimal_roi = {
        "0": 0.33,
        "500": 0.15,
        "900": 0.1,
        "1800": 0,
        "5000": -1
    }

    # Stoploss:
    stoploss = -0.04

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
        return 5
