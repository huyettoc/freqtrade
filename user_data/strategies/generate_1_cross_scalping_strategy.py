import logging
import warnings
from datetime import datetime
from typing import Optional

# --------------------------------
# Add your lib to import here
# TODO: talib is fast but have not more indicators
import talib.abstract as ta
from pandas import DataFrame, merge, to_timedelta
from technical.util import compute_interval, resample_to_interval

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, BooleanParameter, IStrategy

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
    period = int(period)
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


class Generate1CrossScalpingStrategy(IStrategy):
    minimal_roi = {"480": -1}

    stoploss = -0.2

    timeframe = "5m"

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
    indicators = [
        "SMA",
        "EMA",
        "WMA",
        "BBANDS-0",  # Bollinger Bands
        "BBANDS-1",  # Bollinger Bands
        "BBANDS-2",
        "SAR",
        "SAREXT",  # Parabolic SAR - Extended
        "HT_TRENDLINE",  # Hilbert Transform - Instantaneous Trendline
        "KAMA",  # Kaufman Adaptive Moving Average
        "AVGPRICE",  # Average Price
        "MEDPRICE",  # Median Price
        "TYPPRICE",  # Typical Price
        "WCLPRICE",  # Weighted Close Price
        "LINEARREG",
    ]

    time_frames = ["5m", "15m", "1h", "4h"]

    time_periods = [5, 10, 20, 50, 100]

    operators = [">", "<", "cross_above", "cross_below"]

    left_indicator = CategoricalParameter(
        indicators, default=indicators[0], space="buy", optimize=True
    )
    left_timeframe = CategoricalParameter(
        time_frames, default=time_frames[0], space="buy", optimize=True
    )
    left_period = CategoricalParameter(
        time_periods, default=time_periods[0], space="buy", optimize=True
    )

    _operator = CategoricalParameter(operators, default=">", space="buy", optimize=True)

    right_indicator = CategoricalParameter(
        indicators, default=indicators[0], space="buy", optimize=True
    )
    right_timeframe = CategoricalParameter(
        time_frames, default=time_frames[0], space="buy", optimize=True
    )
    right_period = CategoricalParameter(
        time_periods, default=time_periods[0], space="buy", optimize=True
    )

    exit_left_indicator = CategoricalParameter(
        indicators, default=indicators[0], space="sell", optimize=True
    )
    exit_left_timeframe = CategoricalParameter(
        time_frames, default=time_frames[0], space="sell", optimize=True
    )
    exit_left_period = CategoricalParameter(
        time_periods, default=time_periods[0], space="sell", optimize=True
    )

    _exit_operator = CategoricalParameter(operators, default=">", space="sell", optimize=True)

    exit_right_indicator = CategoricalParameter(
        indicators, default=indicators[0], space="sell", optimize=True
    )
    exit_right_timeframe = CategoricalParameter(
        time_frames, default=time_frames[0], space="sell", optimize=True
    )
    exit_right_period = CategoricalParameter(
        time_periods, default=time_periods[0], space="sell", optimize=True
    )

    take_profit = DecimalParameter(0.05, 0.2, default=0.05, decimals=2, space="sell", optimize=True)

    leverage_ = IntParameter(1, 10, default=1, space="sell", optimize=False)
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection",
                                     optimize=True)
    stop_duration_stoploss = IntParameter(1, 200, default=5,
                                          space="protection", optimize=True)
    stop_duration_maxdrawdown = IntParameter(1, 200, default=5,
                                             space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection",
                                           optimize=True)
    use_drawdown_protection = BooleanParameter(default=True, space="protection",
                                               optimize=True)

    @property
    def protections(self):

        prot = []
        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value,
        })

        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 480,
                "trade_limit": 4,
                "stop_duration_candles": self.stop_duration_stoploss.value,
                "only_per_pair": False,
            })
        if self.use_drawdown_protection.value:
            prot.append({
                "method": "MaxDrawdown",
                "lookback_period_candles": 480,
                "trade_limit": 4,
                "stop_duration_candles": self.stop_duration_maxdrawdown.value,
                "max_allowed_drawdown": 0.1,
            })

        return prot

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
        return self.leverage_.value
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            dataframe, left_indicator_name = gen_indicator(
                dataframe,
                self.left_indicator.value,
                self.left_timeframe.value,
                self.left_period.value,
            )
            dataframe, right_indicator_name = gen_indicator(
                dataframe,
                self.right_indicator.value,
                self.right_timeframe.value,
                self.right_period.value,
            )
            operator_ = self._operator.value
            if operator_ == ">":
                condition = dataframe[left_indicator_name] > dataframe[right_indicator_name]
            elif operator_ == "<":
                condition = dataframe[left_indicator_name] < dataframe[right_indicator_name]
            elif operator_ == "cross_above":
                condition = qtpylib.crossed_above(
                    dataframe[left_indicator_name], dataframe[right_indicator_name]
                )
            elif operator_ == "cross_below":
                condition = qtpylib.crossed_below(
                    dataframe[left_indicator_name], dataframe[right_indicator_name]
                )
            dataframe.loc[condition, "enter_long"] = 1
        except Exception as e:
            logger.debug(f"populate_entry_trend err: {e}")
            print(self.left_indicator.value, self.left_timeframe.value, self.left_period.value)
            print(self.right_indicator.value, self.right_timeframe.value, self.right_period.value)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            dataframe, left_indicator_name = gen_indicator(
                dataframe,
                self.exit_left_indicator.value,
                self.exit_left_timeframe.value,
                self.exit_left_period.value,
            )
            dataframe, right_indicator_name = gen_indicator(
                dataframe,
                self.exit_right_indicator.value,
                self.exit_right_timeframe.value,
                self.exit_right_period.value,
            )
            operator_ = self._exit_operator.value
            if operator_ == ">":
                condition = dataframe[left_indicator_name] > dataframe[right_indicator_name]
            elif operator_ == "<":
                condition = dataframe[left_indicator_name] < dataframe[right_indicator_name]
            elif operator_ == "cross_above":
                condition = qtpylib.crossed_above(
                    dataframe[left_indicator_name], dataframe[right_indicator_name]
                )
            elif operator_ == "cross_below":
                condition = qtpylib.crossed_below(
                    dataframe[left_indicator_name], dataframe[right_indicator_name]
                )
            dataframe.loc[condition, "exit_long"] = 1
        except Exception as e:
            logger.debug(f"populate_exit_trend err: {e}")
            print(
                self.exit_left_indicator.value,
                self.exit_left_timeframe.value,
                self.exit_left_period.value,
            )
            print(
                self.exit_right_indicator.value,
                self.exit_right_timeframe.value,
                self.exit_right_period.value,
            )

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
