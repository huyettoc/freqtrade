from datetime import datetime, timedelta
from typing import Dict, Optional

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy


class OptimizedAIHybridRSIBbandStrategy(IStrategy):
    plot_config = {
        "main_plot": {
            # Configuration for main plot indicators.
            # Specifies `ema10` to be red, and `ema50` to be a shade of gray
            "bb_lowerband": {},
            "bb_middleband": {},
            "bb_upperband": {},
        },
        "subplots": {
            # Additional subplot RSI
            "RSI": {
                "rsi": {"color": "red"},
            },
            "predict": {"&-s_close": {}},
        },
    }

    INTERFACE_VERSION: int = 3
    # ROI table:

    # minimal_roi = {
    #     "0": 0.1,
    #     "5000": -1
    # }
    minimal_roi = {"0": 1, "5000": -1}

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.8

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    timeframe = "15m"

    use_custom_stoploss = True
    optimized_take_profit = DecimalParameter(
        0.05, 0.5, default=0.05, decimals=2, space="sell", optimize=True
    )
    optimized_stoploss = DecimalParameter(
        -0.5, -0.05, default=-0.05, decimals=2, space="sell", optimize=True
    )
    optimized_leverage = CategoricalParameter(
        [2, 3, 5, 7, 10], default=3, space="buy", optimize=True
    )

    thres_enter_long = DecimalParameter(0.0, 0.5, default=0.0, space="buy", optimize=False)
    thres_exit_long = DecimalParameter(-0.5, 0.0, default=-0.0, space="sell", optimize=False)
    rsi_buy = IntParameter(low=5, high=45, default=30, space="buy", optimize=False)
    rsi_sell = IntParameter(low=55, high=95, default=60, space="sell", optimize=False)

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        return self.optimized_stoploss.value

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        return self.optimized_leverage.value

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: Dict, **kwargs
    ) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]

        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = dataframe["close"] / dataframe["bb_lowerband-period"]

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .mean()
            / dataframe["close"]
            - 1
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Add your trend following buy signals here
        dataframe.loc[
            (dataframe["rsi"] < self.rsi_buy.value)
            & (dataframe["close"] < dataframe["bb_lowerband"])
            &
            # (dataframe['obv'] > dataframe['obv'].shift(1)) &
            # (dataframe["do_predict"] == 1) &
            (dataframe["&-s_close"] > self.thres_enter_long.value),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Add your trend following exit signals for long positions here
        dataframe.loc[(dataframe["rsi"] > self.rsi_sell.value), "exit_long"] = 1

        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):

        # dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        #
        # trade_date = timeframe_to_prev_date(
        #     self.timeframe, (trade.open_date_utc - timedelta(minutes=int(self.timeframe[:-1])))
        # )
        # trade_candle = dataframe.loc[(dataframe["date"] == trade_date)]
        # if trade_candle.empty:
        #     return None

        # trade_duration = (current_time - trade.open_date_utc).seconds / 60
        #
        # if trade_duration > 1000:
        #     return "trade expired"

        if current_profit >= self.optimized_take_profit.value:
            return "customized take profit"
