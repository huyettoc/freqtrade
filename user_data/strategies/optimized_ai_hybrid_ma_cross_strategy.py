from datetime import datetime, timedelta
from typing import Dict, Optional

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import CategoricalParameter, DecimalParameter
from freqtrade.strategy.interface import IStrategy


class OptimizedAIHybridMACrossStrategy(IStrategy):
    INTERFACE_VERSION: int = 3
    # ROI table:
    # minimal_roi = {"0": 0.15, "300": 0.1, "600": 0.05, "1200": 0.0}

    plot_config = {
        "main_plot": {
            # Configuration for main plot indicators.
            # Specifies `ema10` to be red, and `ema50` to be a shade of gray
            "ema_5": {},
            "ema_20": {},
            "ema_50": {},
        },
        "subplots": {"predict": {"&-s_close": {}}},
    }
    # minimal_roi = {
    #     "0": 0.05,
    #     "1440": -1
    # }
    minimal_roi = {"0": 1, "1500": -1}
    # Stoploss:
    stoploss = -0.8

    # Trailing stop:
    trailing_stop = False
    # trailing_stop_positive = 0.05
    # trailing_stop_positive_offset = 0.1
    # trailing_only_offset_is_reached = False

    timeframe = "15m"
    can_short = False
    thres_enter_long = DecimalParameter(-0.1, 0.1, default=0.0, space="buy", optimize=False)
    thres_enter_short = DecimalParameter(-0.1, 0.1, default=-0.01, space="buy", optimize=False)
    # long_period = IntParameter(low=20, high=50, space='buy', default=50, optimize=False)
    # short_period = IntParameter(low=5, high=20, space='buy', default=20, optimize=False)
    # adx_buy = IntParameter(low=25, high=50, space='buy', default=25, optimize=True)
    # adx_sell = IntParameter(low=0, high=25, space='sell', default=25, optimize=True)
    # def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
    #                 current_profit: float, **kwargs):
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #
    #     # Sell any positions at a loss if they are held for more than one day.
    #     if current_profit < 0.0 and (current_time - trade.open_date_utc).days >= self.exit_day.value:
    #         return 'unclog'

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

        # # Calculate OBV
        # dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])
        # dataframe['adx'] = ta.ADX(dataframe)
        # Add your trend following indicators here
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_5"] = ta.EMA(dataframe, timeperiod=5)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Add your trend following buy signals here
        dataframe.loc[
            (dataframe["do_predict"] == 1)
            & (dataframe["&-s_close"] > self.thres_enter_long.value)
            & (qtpylib.crossed_above(dataframe["ema_5"], dataframe["ema_20"]))
            & (dataframe["ema_5"] > dataframe["ema_50"])
            & (dataframe["ema_20"] > dataframe["ema_50"])
            & (dataframe["close"] > dataframe["ema_50"]),
            ["enter_long", "enter_tag"],
        ] = (1, "long")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # disable exit trend
        dataframe.loc[(dataframe["ema_20"] < 0), "exit_long"] = 1

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
        # last_candle = dataframe.iloc[-1].squeeze()
        # trade_date = timeframe_to_prev_date(
        #     self.timeframe, (trade.open_date_utc - timedelta(minutes=int(self.timeframe[:-1])))
        # )
        # trade_candle = dataframe.loc[(dataframe["date"] == trade_date)]
        # if trade_candle.empty:
        #     return None
        #
        # trade_duration = (current_time - trade.open_date_utc).seconds / 60
        # if trade_duration > 1000:
        #     return "trade expired"

        if current_profit >= self.optimized_take_profit.value:
            return "customized take profit"
