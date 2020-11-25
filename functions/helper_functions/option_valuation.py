from math import log, exp, sqrt
from numpy import zeros
from datetime import date
from scipy.stats import norm


def black_scholes_merton(data, column, strike, rf, end_date, start_date=date.today(), call=True):
    returns = log(data.loc[:, column]/data.loc[:, column].shift())
    time_dif = data.index[-1] - data.index[-2]
    quote = data.loc[:, column][-1]
    duration = (start_date - end_date).days
    del data

    if time_dif.days == 1:
        year_periods = 252
    elif time_dif.days < 10:
        year_periods = 52
    elif time_dif.days < 50:
        year_periods = 12
    elif time_dif.days < 120:
        year_periods = 4
    elif time_dif == 0:
        raise ValueError('Must use at least daily returns.')

    returns = returns[int(-(700/time_dif.days)):]
    vol = sqrt(1/(len(returns - 1)) * (((returns - returns.mean()) ** 2).sum()) * sqrt(year_periods))

    d1 = (log(quote / strike) + (rf + (vol ** 2) / 2) * (duration / time_dif / year_periods)) / (vol * sqrt(
        duration / time_dif / year_periods))
    d2 = d1 - vol * sqrt(duration / time_dif / year_periods)

    if call:
        value = quote * norm.cdf(d1) - exp(-rf * (duration / time_dif / year_periods)) * strike * norm.cdf(d2)
    else:
        value = exp(-rf * (duration / time_dif / year_periods)) * strike * norm.cdf(-d2) - quote * norm.cdf(-d1)

    return value


def binomial_pricing_tree(data, column, strike, rf, periods, end_date, start_date=date.today(), tree=False):
    returns = log(data.loc[:, column] / data.loc[:, column].shift())
    time_dif = data.index[-1] - data.index[-2]
    quote = data.loc[:, column][-1]
    duration = (start_date - end_date).days
    del data

    if time_dif.days == 1:
        year_periods = 252
    elif time_dif.days < 10:
        year_periods = 52
    elif time_dif.days < 50:
        year_periods = 12
    elif time_dif.days < 120:
        year_periods = 4
    elif time_dif == 0:
        raise ValueError('Must use at least daily returns.')

    returns = returns[int(-(700 / time_dif.days)):]
    vol = sqrt(1 / (len(returns - 1)) * (((returns - returns.mean()) ** 2).sum()) * sqrt(year_periods))

    u = exp(vol * sqrt(duration/periods))
    d = exp(-vol * sqrt(duration/periods))
    q = (exp(rf * duration / periods) - d) / (u - d)

    stock_tree = zeros((periods+1, periods+1))
    stock_tree[0, 0]
    for ind1 in range(1, periods+1):
        stock_tree[ind1, 0] = stock_tree[ind1-1, 0] * u
        for ind2 in range(1, periods+1):
            stock_tree[ind1, ind2] = stock_tree[ind1-1, ind2-2] * d

    option_tree = zeros((periods+1, periods+1))
    for ind2 in range(periods+1):
        option_tree[periods, ind2] = max(0, strike-stock_tree[periods, ind2])

    for ind1 in range(periods-1, -1, -1):
        for ind2 in range(ind1+1):
            intrinsic = strike - stock_tree[ind1, ind2]
            discounted = exp(-rf * duration / periods) * (q * option_tree[ind1+1, ind2] + (1-q) * option_tree[ind1+1,
                                                                                                              ind2+1])
            option_tree[ind1, ind2] = max(0, intrinsic, discounted)

    if tree:
        return option_tree
    else:
        return option_tree[0, 0]
