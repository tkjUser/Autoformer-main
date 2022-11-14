from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    返回适用于给定频率（这里为小时h）字符串的时间特征列表。
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {  # 定义一个字典
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],  # 一个类，表示编码为介于[-0.5，0.5]之间的值的月份
        offsets.MonthEnd: [MonthOfYear],  # 一个类，表示编码为介于[-0.5，0.5]之间的值的月份
        offsets.Week: [DayOfMonth, WeekOfYear],  # [该天所在的月，该周所在的年]  编码为[-0.5,0.5]之间的值
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],  # [该天所在的周，该天所在的月，该天所在的年]
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],  # 工作日所在的[周几，一个月的第几天，一年的第几天]
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],  # 小时所在的[一天的第几个小时，一周的星期几，一月的第几天，一年的第几天]
        offsets.Minute: [  # 统计分钟数据
            MinuteOfHour,  # 一个小时的第几分钟
            HourOfDay,  # 一天的第几个小时
            DayOfWeek,  # 一周的第几天（星期几）
            DayOfMonth,  # 一个月的第几天
            DayOfYear,  # 一年的第几天
        ],
        offsets.Second: [  # 统计秒的数据
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)  # 设置采样频率大小，这里是小时：<Hour>

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):  # 这里在Hour时判断成立：<class 'pandas._libs.tslibs.offsets.Hour'>
            # [HourOfDay, DayOfWeek,DayOfMonth, DayOfYear]
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


"""
这个函数是把一个时间戳分解为了四个部分：[HourOfDay, DayOfWeek,DayOfMonth, DayOfYear]
举例，如果是'2016-07-01 02:00:00'会分解为:[2,对应星期几，这个月的第几天，这一年的第几天]，并且把这些数据（整数）缩小到[-0.5,0.5]范围内
dates： datetime的索引：['2016-07-01 02:00:00', '2016-07-01 03:00:00',...]
freq： 数据采样的间隔
"""
def time_features(dates, freq='h'):
    # 把数据在竖直方向堆叠，[4,18412]
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
