class NoTickerError(BaseException):
    def __init__(self):
        print('Please enter a valid ticker or company name.')
