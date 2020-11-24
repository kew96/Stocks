from django.test import TestCase

from portfolio.models import *
import functions.stock

from datetime import date


# Create your tests here.


class PortfolioTestCase(TestCase):

    def setUp(self):
        test1 = Portfolio.objects.create(name='Test1',
                                         cash=Decimal('2121.21'),
                                         inception=date(2019, 6, 7),
                                         fee_rate=1.2)

        test2 = Portfolio.objects.create(name='Test2',
                                         cash=Decimal('109287.74'),
                                         inception=date(2010, 1, 12),
                                         fee_rate=2.1)

    def test_portfolio_cash(self):
        test1 = Portfolio.objects.get(name='Test1')
        test2 = Portfolio.objects.get(name='Test2')

        self.assertEqual(test1.cash, Decimal('2121.21'))
        self.assertEqual(test2.cash, Decimal('109287.74'))

    def test_portfolio_inception(self):
        test1 = Portfolio.objects.get(name='Test1')
        test2 = Portfolio.objects.get(name='Test2')

        self.assertEqual(test1.inception, date(2019, 6, 7))
        self.assertEqual(test2.inception, date(2010, 1, 12))

    def test_fee_rate(self):
        test1 = Portfolio.objects.get(name='Test1')
        test2 = Portfolio.objects.get(name='Test2')

        self.assertEqual(test1.fee_rate, 1.2)
        self.assertEqual(test2.fee_rate, 2.1)


class StockTest(TestCase):

    def setUp(self):
        aapl = functions.stock.Stock('AAPL')

        Stock.objects.create(
                 ticker=aapl.ticker,
                 name=aapl.name,
                 summary=aapl.summary,
                 sector=aapl.sector,
                 industry=aapl.industry,
                 dividend_rate=aapl.dividend_rate,
                 beta=aapl.beta,
                 trailing_PE=aapl.trailing_PE,
                 market_cap=aapl.market_cap,
                 price_to_sales_12m=aapl.price_to_sales_12m,
                 forward_PE=aapl.forward_PE,
                 tradable=aapl.tradable,
                 dividend_yield=aapl.dividend_yield,
                 forward_EPS=aapl.forward_EPS,
                 profit_margin=aapl.profit_margin,
                 trailing_EPS=aapl.trailing_EPS
                 )

    def test_ticker(self):
        aapl = functions.stock.Stock('AAPL')
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.ticker, aapl.ticker)

    def test_name(self):
        aapl = functions.stock.Stock('AAPL')
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.name, aapl.name)

    def test_summary(self):
        aapl = functions.stock.Stock('AAPL')
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.summary, aapl.summary)

    def test_sector(self):
        aapl = functions.stock.Stock('AAPL')
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.sector, aapl.sector)

    def test_industry(self):
        aapl = functions.stock.Stock('AAPL')
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.industry, aapl.industry)

    def test_dividend_rate(self):
        aapl = functions.stock.Stock('AAPL')
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.dividend_rate, aapl.dividend_rate)

    def test_beta(self):
        aapl = functions.stock.Stock('AAPL')
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.beta, aapl.beta)

    def test_trailing_PE(self):
        aapl = functions.stock.Stock('AAPL')
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.trailing_PE, aapl.trailing_PE)

    def test_market_cap(self):
        aapl = functions.stock.Stock('AAPL')
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.market_cap, aapl.market_cap)

    def test_price_to_sales_12m(self):
        aapl = functions.stock.Stock('AAPL')
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.price_to_sales_12m, aapl.price_to_sales_12m)

    def test_forward_PE(self):
        aapl = functions.stock.Stock('AAPL')
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.forward_PE, aapl.forward_PE)

    def test_tradable(self):
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertTrue(stock1.tradable)

    def test_dividend_yield(self):
        aapl = functions.stock.Stock('AAPL')
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.dividend_yield, aapl.dividend_yield)

    def test_forward_EPS(self):
        aapl = functions.stock.Stock('AAPL')
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.forward_EPS, aapl.forward_EPS)

    def test_profit_margin(self):
        aapl = functions.stock.Stock('AAPL')
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.profit_margin, aapl.profit_margin)

    def test_trailing_EPS(self):
        aapl = functions.stock.Stock('AAPL')
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.trailing_EPS, aapl.trailing_EPS)
