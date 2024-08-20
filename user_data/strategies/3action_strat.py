import logging
from functools import reduce as reducde
from typing import Dict

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class RL3actionsStrategy1h(IStrategy):

    plot_config = {
        "main_plot": {
            "EMA_5": {'color': 'blue'},
            "EMA_10": {'color': 'green'},
            "EMA_20": {'color': 'purple'},
            "EMA_50": {'color': 'red'},
            "SMA_50": {'color': 'yellow'},
        },

        'subplots': {
            # Additional subplot RSI
            "RSI": {
                "RSI": {"color": "green"}
            },
            "CCI": {
                "CCI": {"color": "orange"}
            }
        }
    }

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs):
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame,
                                         metadata: Dict, **kwargs):
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]

        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame,
                                     metadata: Dict, **kwargs):
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame,
                           metadata: Dict, **kwargs):
        dataframe["&-action"] = 0

        return dataframe

    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        dataframe['EMA_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['EMA_10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['EMA_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['EMA_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['SMA_50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['RSI'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['CCI'] = ta.CCI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame,
                             metadata: dict) -> DataFrame:
        """
        if the action is buy, populate buy trend
        """

        buy_conditions = [dataframe["do_predict"] == 1,
                          dataframe["&-action"] == 1,
                          (qtpylib.crossed_above(dataframe['RSI'], 30))
                          ]

        if buy_conditions:
            dataframe.loc[reducde(lambda x, y: x & y, buy_conditions),
                          "buy"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        """
        if the action is sell, populate sell trend
        """

        sell_conditions = [dataframe["do_predict"] == 1,
                           dataframe["&-action"] == 2,
                           (qtpylib.crossed_below(dataframe['RSI'], 70))
                           ]
        if sell_conditions:
            dataframe.loc[reducde(lambda x, y: x & y, sell_conditions),
                          "sell"] = 1

        return dataframe
