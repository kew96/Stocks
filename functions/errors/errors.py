class NoTickerError(BaseException):
    def __init__(self):
        print('Please enter a valid ticker or company name.')

class WrongPeriodError(BaseException):
    def __init__(self, input):
        print('{} is not a vaild period. Please select one of the following:')
