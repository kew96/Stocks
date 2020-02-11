from alpha_vantage.timeseries import TimeSeries
import errors

alpha_vantage_KEY = '7ZDI2M6PEWCEOSFC'
av_ts = TimeSeries(alpha_vantage_KEY)


def ticker_search(name):
    return av_ts.get_symbol_search(name)


class Stock:

    def __init__(self, ticker=None, name=None):
        self.ticker = ticker
        self.name = name
        self.__set_name()
        # TODO: self.name
        self.data, self.meta = av_ts.get_daily_adjusted(symbol=self.ticker, outputsize='full')
        # av_ts.

    def __set_name(self):
        if self.ticker is None and self.name is None:
            raise errors.NoTickerError()
        elif self.ticker is not None:
            vals = ticker_search(self.ticker)[0]
        else:
            vals = ticker_search(self.name)[0]
        if float(vals[0]['9. matchScore']) > 0.7:
            self.ticker = vals[0]['1. symbol']
            self.name = vals[0]['2. name']
        else:
            choices = []
            for c in vals:
                if float(c['9. matchScore']) > 0.3:
                    choices.append(c)
            string = 'Please select one of the following choices by number:'
            for i, c in enumerate(choices):
                substring = ''
                for key, val in c.items():
                    if key[0] in ['1', '2', '3', '4', '8', '9']:
                        substring = substring + '\t\t' + key[3:] + ': ' + val + ', '
                string = string + '\n' + str(i+1) + '. ' + substring[:-2]
            pick = input(string + '\n' + 'Input: ')
            pick = choices[int(pick)-1]
            self.ticker = pick['1. symbol']
            self.name = pick['2. name']
