from django.test import TestCase

from portfolio.models import *
import functions.stock

from datetime import date


# Create your tests here.


class PortfolioTestCase(TestCase):

    def setUp(self):
        Portfolio.objects.create(name='Test1',
                                 cash=Decimal('2121.21'),
                                 inception=date(2019, 6, 7),
                                 fee_rate=1.2)

        Portfolio.objects.create(name='Test2',
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
    aapl = functions.stock.Stock(ticker='AAPL', name='Apple Inc.')

    def setUp(self):
        Stock.objects.create(ticker='AAPL')

    def test_ticker(self):
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.ticker, self.aapl.ticker)

    def test_name(self):
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.name, self.aapl.name)

    def test_summary(self):
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.summary, self.aapl.summary)

    def test_sector(self):
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.sector, self.aapl.sector)

    def test_industry(self):
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.industry, self.aapl.industry)

    def test_dividend_rate(self):
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.dividend_rate, Decimal(str(self.aapl.dividend_rate)))

    def test_beta(self):
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.beta, Decimal(str(self.aapl.beta)))

    def test_trailing_PE(self):
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertAlmostEqual(stock1.trailing_PE, Decimal(str(self.aapl.trailing_PE)), 1)

    def test_market_cap(self):
        stock1 = Stock.objects.get(ticker='AAPL', name='Apple Inc.')

        self.assertAlmostEquals(stock1.market_cap, self.aapl.market_cap, places=-12)
        # self.assertEqual(stock1.market_cap, self.aapl.market_cap)

    def test_price_to_sales_12m(self):
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertAlmostEqual(stock1.price_to_sales_12m, Decimal(str(self.aapl.price_to_sales_12m)), 1)

    def test_forward_PE(self):
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertAlmostEqual(stock1.forward_PE, Decimal(str(self.aapl.forward_PE)), 1)

    def test_tradeable(self):
        self.assertTrue(self.aapl.tradable)

    def test_dividend_yield(self):
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertAlmostEqual(stock1.dividend_yield, Decimal(str(self.aapl.dividend_yield)), 6)

    def test_forward_EPS(self):
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertEqual(stock1.forward_EPS, Decimal(str(self.aapl.forward_EPS)))

    def test_profit_margin(self):
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertAlmostEqual(stock1.profit_margin, Decimal(str(self.aapl.profit_margin)), 5)

    def test_trailing_EPS(self):
        stock1 = Stock.objects.get(ticker='AAPL')

        self.assertAlmostEqual(stock1.trailing_EPS, Decimal(str(self.aapl.trailing_EPS)), 2)