from math import exp, sqrt
from numpy import zeros, log
from datetime import date
from scipy.stats import norm


def black_scholes_merton(data, column, strike, rf, end_date, start_date=date.today(), dividend_yield=0, call=True):
    continuous_dividend = log(1+dividend_yield)
    returns = log(data.loc[:, column] / data.loc[:, column].shift())
    time_dif = data.index[-1] - data.index[-2]
    quote = data.loc[:, column][-1]
    duration = (end_date - start_date).days
    del data

    if time_dif.days == 1:
        year_periods = 252
    elif time_dif.days < 10:
        year_periods = 52
    elif time_dif.days < 50:
        year_periods = 12
    elif time_dif.days < 120:
        year_periods = 4
    elif time_dif.days == 0:
        raise ValueError('Must use at least daily returns.')

    modified_quote = quote * exp(-continuous_dividend * duration)
    returns = returns[int(-(700 / time_dif.days)):]
    vol = sqrt(1 / (len(returns - 1)) * (((returns - returns.mean()) ** 2).sum()) * sqrt(year_periods))

    d1 = (log(modified_quote / strike) + (rf + (vol ** 2) / 2) * (duration / time_dif.days / year_periods)) / (vol * sqrt(
             duration / time_dif.days / year_periods))
    d2 = d1 - vol * sqrt(duration / time_dif.days / year_periods)

    if call:
        value = modified_quote * norm.cdf(d1) - exp(-rf * (duration / time_dif.days / year_periods)) * strike * norm.cdf(d2)
    else:
        value = exp(-rf * (duration / time_dif.days / year_periods)) * strike * norm.cdf(-d2) - modified_quote * norm.cdf(-d1)

    return value


def binomial_pricing_tree(data, column, strike, rf, steps, end_date, start_date=date.today(), tree=False):
    returns = log(data.loc[:, column] / data.loc[:, column].shift())
    time_dif = data.index[-1] - data.index[-2]
    quote = data.loc[:, column][-1]
    duration = (end_date - start_date).days
    del data

    if time_dif.days == 1:
        year_periods = 252
    elif time_dif.days < 10:
        year_periods = 52
    elif time_dif.days < 50:
        year_periods = 12
    elif time_dif.days < 120:
        year_periods = 4
    elif time_dif.days == 0:
        raise ValueError('Must use at least daily returns.')

    returns = returns[int(-(700 / time_dif.days)):]
    vol = sqrt(1 / (len(returns) - 1) * (((returns - returns.mean()) ** 2).sum())) * sqrt(year_periods)

    t_years = duration / year_periods

    u = exp(vol * sqrt(t_years / steps))
    d = exp(-vol * sqrt(t_years / steps))
    q = (exp(rf * t_years / steps) - d) / (u - d)

    stock_tree = zeros((steps + 1, steps + 1))
    stock_tree[0, 0] = quote
    for ind1 in range(1, steps + 1):
        for ind2 in range(0, ind1 + 1):
            stock_tree[ind1, ind2] = quote * (u ** (ind1 - ind2)) * (d ** ind2)

    option_tree = zeros((steps + 1, steps + 1))
    for ind2 in range(steps + 1):
        option_tree[steps, ind2] = max(0, strike - stock_tree[steps, ind2])

    for ind1 in range(steps - 1, -1, -1):
        for ind2 in range(ind1 + 1):
            intrinsic = strike - stock_tree[ind1, ind2]
            discounted = exp(-rf * t_years / steps) * (q * option_tree[ind1 + 1, ind2] + (1 - q) *
                                                       option_tree[ind1 + 1, ind2 + 1])
            option_tree[ind1, ind2] = max(0, intrinsic, discounted)

    if tree:
        return stock_tree, option_tree
    else:
        return option_tree[0, 0]
