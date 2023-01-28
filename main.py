import requests
import pandas as pd
import numpy as np
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import jarque_bera
from numpy import log
from math import sqrt
import matplotlib.pyplot as plt

def send_get_request(url):
    '''
    发送get请求, 获取响应数据
    Parameters
    ----------
    url: 目标url
    Returns
    -------
    content
    '''
    response = requests.get(url)
    return response.json()


def get_return_series(data: np.ndarray) -> pd.core.series.Series:
    """
    计算收益率序列
    Parameters
    ----------
    data : numpy.ndarray
        数据
    Returns
    -------
    pd.core.series.Series
        收益率序列
    """
    data = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])
    data.set_index("date", inplace=True)
    data = data.astype(float)
    return data['close'].pct_change()


def get_return_desc(return_series: pd.core.series.Series) -> pd.core.series.Series:
    """
    计算收益率序列的描述性统计信息
    Parameters
    ----------
    return_series : pd.core.series.Series
        收益率序列
    Returns
    -------
    pd.core.series.Series
        收益率序列的描述性统计信息
    """
    desc = return_series.describe()
    # print(desc.to_string())
    return desc


def jb_test(return_series):
    """
    JB检验函数
    :param return_series: 收益率序列
    :return: jb值，p值
    """
    # 计算jb值和p值
    jb_value, p_value = jarque_bera(return_series)
    # 判断拒绝或接受原假设
    if p_value < 0.05:
        print("拒绝原假设，说明收益率序列不是正态分布的。")
    else:
        print("接受原假设，说明收益率序列是正态分布的。")
    return jb_value, p_value


def calc_hurst_exponent(returns):
    """
    计算Hurst指数
    Args:
        returns: 收益率序列
    Returns:
        Hurst指数
    """
    lags = range(2, 100)
    tau = [np.sqrt(np.std(returns[i:] - returns[i-1])) for i in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]*2

def is_fractal_market(hurst_index):
    """
    判断收益率序列是否符合分形市场假说
    分形市场假说指收益率序列是一个分形时间序列，具有不确定性和自相关性
    :param hurst_index: 收益率序列
    :return: True/False
    """
    if hurst_index < 0.5:
        print("符合分形市场假说")
        return True
    else:
        print("不符合分形市场假说")
        return False

def evaluation(data_RB0, data_I0):
    # 计算RB0的收益率序列
    rb0_returns = get_return_series(data_RB0)

    # 计算I0的收益率序列
    i0_returns = get_return_series(data_I0)

    # 计算RB0的描述性统计信息
    rb0_desc = get_return_desc(rb0_returns)
    print(rb0_desc)

    # 计算I0的描述性统计信息
    i0_desc = get_return_desc(i0_returns)
    print(i0_desc)

    # 计算RB0的JB检验结果
    rb0_jb = jb_test(rb0_returns)
    print(rb0_jb)

    # 计算I0的JB检验结果
    i0_jb = jb_test(i0_returns)
    print(i0_jb)

    # 计算RB0的Hurst指数
    rb0_hurst = calc_hurst_exponent(rb0_returns)
    print(rb0_hurst)

    # 计算I0的Hurst指数
    i0_hurst = calc_hurst_exponent(i0_returns)
    print(i0_hurst)

    # 制定跨品种套利策略并进行回测
    spread_returns = rb0_returns - i0_returns
    spread_desc = get_return_desc(spread_returns)
    print(spread_desc)

    # 对跨品种套利效果进行评估
    # 利用回测结果绘制累计收益曲线
    plt.plot(np.cumsum(spread_returns))
    plt.title('Cumulative Returns of the Spread')
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.show()

    # 寻找改进空间

def main(url):
    data = send_get_request(url)
    # print(data)
    return_series = get_return_series(data)
    # print(return_series)
    get_return_desc(return_series)
    jb_test(return_series)
    hurst_index = calc_hurst_exponent(return_series)
    is_fractal_market(hurst_index)

if __name__ == "__main__":
    url_RB0 = "https://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesDailyKLine?symbol=RB0"
    url_I0 = "https://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesDailyKLine?symbol=I0"
    data_RB0 = send_get_request(url_RB0)
    data_I0 = send_get_request(url_I0)
    evaluation(data_RB0,data_I0)
