import logging
import warnings
from datetime import datetime
from typing import List, Optional

# --------------------------------
# Add your lib to import here
# TODO: talib is fast but have not more indicators
import talib.abstract as ta
from pandas import DataFrame, merge, to_timedelta
from technical.util import compute_interval, resample_to_interval

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.optimize.space import Dimension, Integer
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
)


logger = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=FutureWarning)


def _resampled_merge(original: DataFrame, resampled: DataFrame, column: str, fill_na=True):
    """
    Merges a resampled dataset back into the original data set.
    Resampled candle will match OHLC only if full timespan is available in original dataframe.

    :param original: the original non resampled dataset
    :param resampled:  the resampled dataset
    :return: the merged dataset
    """

    original_int = compute_interval(original)
    resampled_int = compute_interval(resampled)

    if original_int < resampled_int:
        # Subtract "small" timeframe so merging is not delayed by 1 small candle.
        # Detailed explanation in https://github.com/freqtrade/freqtrade/issues/4073
        resampled["date"] = (
            resampled["date"] + to_timedelta(resampled_int, "m") - to_timedelta(original_int, "m")
        )
    else:
        raise ValueError(
            "Tried to merge a faster timeframe to a slower timeframe." "Upsampling is not possible."
        )
    resampled = resampled[["date", column]]
    # rename all the columns to the correct interval
    resampled.columns = [f"resample_{resampled_int}_{col}" for col in resampled.columns]

    dataframe = merge(
        original,
        resampled,
        how="left",
        left_on="date",
        right_on=f"resample_{resampled_int}_date",
    )
    # dataframe = dataframe.drop(f"resample_{resampled_int}_date_merge", axis=1)
    dataframe = dataframe.drop(columns=[f"resample_{resampled_int}_date"]).rename(
        columns={f"resample_{resampled_int}_{column}": column}
    )
    if fill_na:
        dataframe[column] = dataframe[column].ffill()
    return dataframe


def gen_indicator(dataframe, indicator, timeframe, period):
    indicator_name = "_".join([indicator, timeframe, str(period)])

    splits = indicator.split("-")
    period = int(period) if period is not None else None

    if len(splits) > 1:
        indicator, col = splits[0], int(splits[1])
        if timeframe != "5m":
            dataframe_sampled = resample_to_interval(dataframe, timeframe)
            dataframe_sampled[indicator_name] = getattr(ta, indicator)(
                dataframe_sampled, period=period
            ).iloc[:, col]
            dataframe = _resampled_merge(dataframe, dataframe_sampled, indicator_name)
            dataframe[indicator_name] = dataframe[indicator_name].ffill()
        else:
            dataframe[indicator_name] = getattr(ta, indicator)(dataframe, period=period).iloc[
                :, col
            ]
    else:
        if timeframe != "5m":
            dataframe_sampled = resample_to_interval(dataframe, timeframe)
            dataframe_sampled[indicator_name] = getattr(ta, indicator)(
                dataframe_sampled, timeperiod=period
            )
            dataframe = _resampled_merge(dataframe, dataframe_sampled, indicator_name)
            dataframe[indicator_name] = dataframe[indicator_name].ffill()
        else:
            dataframe[indicator_name] = getattr(ta, indicator)(dataframe, timeperiod=period)

    return dataframe, indicator_name


def gen_condition(
    dataframe,
    left_indicator,
    left_timeframe,
    left_period,
    _operator,
    right_indicator,
    right_timeframe,
    right_period,
    value,
):
    dataframe, left_indicator_name = gen_indicator(
        dataframe, left_indicator, left_timeframe, left_period
    )
    left_operand = dataframe[left_indicator_name]

    if right_indicator is not None:
        dataframe, right_indicator_name = gen_indicator(
            dataframe, right_indicator, right_timeframe, right_period
        )
        right_operand = dataframe[right_indicator_name]

        # handle in case not cross indicator
        if (
            left_indicator in MT5Generate4CombineStrategy.compare_indicator
            and left_indicator_name != right_indicator_name
        ):
            # normalize left indicator
            if left_indicator == "CCI":
                # min max scaler value CCI in range 0 -> 100 to range -200 -> 200
                value = (value - 50) * 4
            elif left_indicator == "ROC":
                value = (value - 50) * 2
            right_operand = value
    else:
        right_operand = value

    if _operator == ">":
        condition = left_operand > right_operand
    elif _operator == "<":
        condition = left_operand < right_operand
    elif _operator == "cross_above":
        condition = qtpylib.crossed_above(left_operand, right_operand)
    elif _operator == "cross_below":
        condition = qtpylib.crossed_below(left_operand, right_operand)
    elif _operator is None:
        condition = left_operand
    else:
        condition = left_operand > right_operand

    return dataframe, condition


class MT5Generate4CombineStrategy(IStrategy):
    minimal_roi = {"1440": -1}

    stoploss = -0.1

    timeframe = "1h"
    order_types = {
        "entry": "market",
        "exit": "market",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_limit_ratio": 0.99,
    }
    use_exit_signal = True
    startup_candle_count = 1440
    plot_config = {
        "main_plot": {
            # Configuration for main plot indicators.
            # Specifies `ema10` to be red, and `ema50` to be a shade of gray
            "SAR_5m_20": {},
            "EMA_5m_100": {},
            "BBANDS-1_5m_100": {},
            "SMA_4h_5": {},
        }
    }
    cross_indicators = [
        "SMA",
        "EMA",
        "WMA",
        "BBANDS-0",  # Bollinger Bands
        "BBANDS-1",  # Bollinger Bands
        "BBANDS-2",
        "SAR",
        "HT_TRENDLINE",  # Hilbert Transform - Instantaneous Trendline
        "KAMA",  # Kaufman Adaptive Moving Average
    ]
    compare_indicator = [
        "RSI",
        "CCI",
        "MFI",
        "ADX",
        "MINUS_DI",  # Minus Directional Indicator
        "MINUS_DM",
        "ROC",
        "STOCH-0",  # Stochastic
        "STOCH-1",  # Stochastic
        "STOCHF-0",  # Stochastic Fast
        "STOCHF-1",  # Stochastic Fast
        "STOCHRSI-0",  # Stochastic Relative Strength Index
        "STOCHRSI-1",  # Stochastic Relative Strength Index
    ]

    volume_indicator = ["AD", "ADOSC", "OBV"]  # Chaikin A/D Line  # Chaikin A/D Oscillator
    candle_pattern_indicator = [
        "CDLDOJI",  # Doji
        "CDLDOJISTAR",  # Doji Star
        "CDLDRAGONFLYDOJI",  # Dragonfly Doji
        "CDLENGULFING",  # Engulfing Pattern
        "CDLEVENINGDOJISTAR",  # Evening Doji Star
        "CDLEVENINGSTAR",  # Evening Star
        "CDLHAMMER",  # Hammer
        "CDLHANGINGMAN",  # Hanging Man
        "CDLMORNINGDOJISTAR",  # Morning Doji Star
        "CDLMORNINGSTAR",  # Morning Star
    ]

    indicators = []
    indicators += cross_indicators
    indicators += compare_indicator

    self_indicators = []
    # self_indicators += volume_indicator
    self_indicators += candle_pattern_indicator

    time_frames = ["5m", "15m", "1h", "4h"]

    cross_time_periods = [5, 10, 20, 50, 100]

    compare_time_periods = [14, 26, 52]

    time_periods = []
    time_periods += cross_time_periods
    time_periods += compare_time_periods

    operators = [">", "<", "cross_above", "cross_below"]

    self_operators = ["increase", "decrease"]

    cross_left_indicator = CategoricalParameter(
        cross_indicators, default=cross_indicators[0], space="buy", optimize=True
    )
    cross_left_timeframe = CategoricalParameter(
        time_frames, default=time_frames[0], space="buy", optimize=True
    )
    cross_left_period = CategoricalParameter(
        cross_time_periods, default=cross_time_periods[0], space="buy", optimize=True
    )
    cross_operator = CategoricalParameter(operators, default=">", space="buy", optimize=True)
    cross_right_indicator = CategoricalParameter(
        cross_indicators, default=cross_indicators[0], space="buy", optimize=True
    )
    cross_right_timeframe = CategoricalParameter(
        time_frames, default=time_frames[0], space="buy", optimize=True
    )
    cross_right_period = CategoricalParameter(
        cross_time_periods, default=cross_time_periods[0], space="buy", optimize=True
    )

    cross_left_indicator2 = CategoricalParameter(
        cross_indicators, default=cross_indicators[0], space="buy", optimize=True
    )
    cross_left_timeframe2 = CategoricalParameter(
        time_frames, default=time_frames[0], space="buy", optimize=True
    )
    cross_left_period2 = CategoricalParameter(
        cross_time_periods, default=cross_time_periods[0], space="buy", optimize=True
    )
    cross_operator2 = CategoricalParameter(operators, default=">", space="buy", optimize=True)
    cross_right_indicator2 = CategoricalParameter(
        cross_indicators, default=cross_indicators[0], space="buy", optimize=True
    )
    cross_right_timeframe2 = CategoricalParameter(
        time_frames, default=time_frames[0], space="buy", optimize=True
    )
    cross_right_period2 = CategoricalParameter(
        cross_time_periods, default=cross_time_periods[0], space="buy", optimize=True
    )
    cross_combine = CategoricalParameter(
        ["and", "or", "ignore"], default="and", space="buy", optimize=True
    )

    compare_left_indicator = CategoricalParameter(
        compare_indicator, default=compare_indicator[0], space="buy", optimize=True
    )
    compare_left_timeframe = CategoricalParameter(
        time_frames, default=time_frames[0], space="buy", optimize=True
    )
    compare_left_period = CategoricalParameter(
        compare_time_periods, default=compare_time_periods[0], space="buy", optimize=True
    )
    compare_operator = CategoricalParameter(operators, default=">", space="buy", optimize=True)
    compare_value = IntParameter(low=1, high=100, default=50, space="buy", optimize=True)
    compare_combine = CategoricalParameter(
        ["and", "or", "ignore"], default="and", space="buy", optimize=True
    )

    self_left_indicator = CategoricalParameter(
        self_indicators, default=self_indicators[0], space="buy", optimize=True
    )
    self_left_timeframe = CategoricalParameter(
        time_frames, default=time_frames[0], space="buy", optimize=True
    )
    self_combine = CategoricalParameter(
        ["and", "or", "ignore"], default="and", space="buy", optimize=True
    )

    exit_cross_left_indicator = CategoricalParameter(
        cross_indicators, default=cross_indicators[0], space="sell", optimize=True
    )
    exit_cross_left_timeframe = CategoricalParameter(
        time_frames, default=time_frames[0], space="sell", optimize=True
    )
    exit_cross_left_period = CategoricalParameter(
        time_periods, default=cross_time_periods[0], space="sell", optimize=True
    )
    exit_cross_operator = CategoricalParameter(operators, default=">", space="sell", optimize=True)
    exit_cross_right_indicator = CategoricalParameter(
        cross_indicators, default=cross_indicators[0], space="sell", optimize=True
    )
    exit_cross_right_timeframe = CategoricalParameter(
        time_frames, default=time_frames[0], space="sell", optimize=True
    )
    exit_cross_right_period = CategoricalParameter(
        time_periods, default=cross_time_periods[0], space="sell", optimize=True
    )

    take_profit = DecimalParameter(0.05, 0.2, default=0.05, decimals=2, space="sell", optimize=True)

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
        return 100

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            dataframe, condition = gen_condition(
                dataframe=dataframe,
                left_indicator=self.cross_left_indicator.value,
                left_period=self.cross_left_period.value,
                left_timeframe=self.cross_left_timeframe.value,
                _operator=self.cross_operator.value,
                right_indicator=self.cross_right_indicator.value,
                right_period=self.cross_right_period.value,
                right_timeframe=self.cross_right_timeframe.value,
                value=None,
            )
            conditions = condition

            dataframe, cross_condition2 = gen_condition(
                dataframe=dataframe,
                left_indicator=self.cross_left_indicator2.value,
                left_period=self.cross_left_period2.value,
                left_timeframe=self.cross_left_timeframe2.value,
                _operator=self.cross_operator2.value,
                right_indicator=self.cross_right_indicator2.value,
                right_period=self.cross_right_period2.value,
                right_timeframe=self.cross_right_timeframe2.value,
                value=None,
            )
            if self.cross_combine.value == "and":
                conditions &= cross_condition2
            elif self.cross_combine.value == "or":
                conditions |= cross_condition2

            dataframe, compare_condition = gen_condition(
                dataframe=dataframe,
                left_indicator=self.compare_left_indicator.value,
                left_period=self.compare_left_period.value,
                left_timeframe=self.compare_left_timeframe.value,
                _operator=self.compare_operator.value,
                right_indicator=None,
                right_period=None,
                right_timeframe=None,
                value=self.compare_value.value,
            )

            if self.compare_combine.value == "and":
                conditions &= compare_condition
            elif self.compare_combine.value == "or":
                conditions |= compare_condition

            dataframe, self_condition = gen_condition(
                dataframe=dataframe,
                left_indicator=self.self_left_indicator.value,
                left_period=None,
                left_timeframe=self.compare_left_timeframe.value,
                _operator=None,
                right_indicator=None,
                right_period=None,
                right_timeframe=None,
                value=None,
            )
            if self.self_combine.value == "and":
                conditions &= self_condition
            elif self.self_combine.value == "or":
                conditions |= self_condition

            dataframe.loc[conditions, "enter_long"] = 1
        except Exception as e:
            logger.debug(f"populate_entry_trend err: {e}")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            dataframe, condition = gen_condition(
                dataframe=dataframe,
                left_indicator=self.exit_cross_left_indicator.value,
                left_period=self.exit_cross_left_period.value,
                left_timeframe=self.exit_cross_left_timeframe.value,
                _operator=self.exit_cross_operator.value,
                right_indicator=self.exit_cross_right_indicator.value,
                right_period=self.exit_cross_right_period.value,
                right_timeframe=self.exit_cross_right_timeframe.value,
                value=None,
            )
            dataframe.loc[condition, "exit_long"] = 1
        except Exception as e:
            logger.debug(f"populate_exit_trend err: {e}")

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
        if current_profit >= self.take_profit.value:
            return "customized take profit"

    @staticmethod
    def max_open_trades_space() -> List[Dimension]:
        return [
            Integer(-1, 10, name="max_open_trades"),
        ]
