class NoTickerError(BaseException):
    def __init__(self):
        print('Please enter a valid ticker or company name.')


class WrongFormatError(BaseException):
    def __init__(self, form):
        print('{} is not a valid format. Please use either "compact" or "full".'.format(form))


class NoDataError(BaseException):
    def __init__(self):
        print('Increase the period to create enough data.')
