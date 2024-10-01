from datetime import datetime, timedelta
from data_manager import DataManager
from portfolio_manager import PortfolioManager
from signal_generator import SignalGenerator
from indicators import Indicators
import pandas as pd
from shared_state import get_shared_state
from backtest_analyzer import BacktestAnalyzer
import matplotlib.pyplot as plt
from strategy_factory import StrategyFactory
from config_loader import CONFIG
from metrics import PerformanceMetrics

class BacktestEngine:
    def __init__(self, initial_cash, min_cash, max_open_positions, start_date, end_date, params, strategy_names):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.params = params
        self.data_manager = DataManager()
        self.portfolio_manager = PortfolioManager(initial_cash, min_cash, max_open_positions, 
                                                  params['transaction_cost'], params['slippage'], 
                                                  self.data_manager, strategy_names)
        self.strategies = {name: StrategyFactory.get_strategy(name, self.data_manager) for name in strategy_names}
        self.signal_generators = {name: SignalGenerator(self.data_manager, strategy) for name, strategy in self.strategies.items()}
        self.PORTFOLIO = []
        self.shared_state = get_shared_state()
        self.BENCHMARK = []
        self.analyzer = BacktestAnalyzer()
        self.performance_metrics = PerformanceMetrics()


    def run_backtest(self):
        current_date = self.start_date
        last_valid_asset_universe = None

        self.analyzer.start_real_time_chart()

        while current_date < self.end_date:
            new_asset_universe = self.data_manager.update_investment_universe(current_date, 
                                                                              self.params['max_MC_rank'], 
                                                                              self.params['min_volume_thres'])
            if new_asset_universe is not None:
                last_valid_asset_universe = new_asset_universe

            ASSET_UNIVERSE = new_asset_universe if new_asset_universe is not None else last_valid_asset_universe

            for strategy_name, signal_generator in self.signal_generators.items():
                prices, eligible_coins = signal_generator.rank_coins(current_date, ASSET_UNIVERSE, self.params['ranking_method'])
                signals = signal_generator.generate_and_prioritize_signals(current_date, eligible_coins, self.portfolio_manager.POSITIONS[strategy_name])
                self.shared_state.update_open_orders({strategy_name: signals})

            self.portfolio_manager.process_orders(current_date)

            if self.params['rebalancing']:
                self.portfolio_manager.process_rebalancing(current_date)
                self.portfolio_manager.check_rebalancing(current_date)

            self.portfolio_manager.update_portfolio(current_date)
            self.PORTFOLIO.append((current_date, self.portfolio_manager.AUM))

            btc_price = self.data_manager.fetch_current_price('BTCUSDT', current_date, 'Close')
            self.BENCHMARK.append((current_date, btc_price))

            self.analyzer.update_data(current_date, self.portfolio_manager.AUM, btc_price)

            current_date += timedelta(days=1)
            plt.pause(0.001)

        self.generate_backtest_report()

        return self.PORTFOLIO

    def generate_backtest_report(self):
        output_dir = 'backtest_results'
        self.analyzer.generate_report(output_dir)
        
        # Convert PORTFOLIO and BENCHMARK to Series
        portfolio_series = pd.Series({date: value for date, value in self.PORTFOLIO}, name='Value')
        portfolio_series.index = pd.to_datetime(portfolio_series.index)
        benchmark_series = pd.Series({date: value for date, value in self.BENCHMARK}, name='Value')
        benchmark_series.index = pd.to_datetime(benchmark_series.index)

        # Get trades from portfolio_manager
        trades = self.portfolio_manager.get_trades()

        try:
            self.performance_metrics.run_full_analysis(
                portfolio_series, 
                benchmark_series, 
                trades, 
                risk_free_rate=0.02, 
                periods_per_year=365
            )
        except Exception as e:
            print(f"An error occurred during the backtest analysis: {str(e)}")
            print("Debug information:")
            print(f"Portfolio Series shape: {portfolio_series.shape}")
            print(f"Benchmark Series shape: {benchmark_series.shape}")
            print(f"Trades keys: {trades.keys()}")
            for strategy, strategy_trades in trades.items():
                print(f"Strategy: {strategy}, Number of trades: {len(strategy_trades)}")

        # Print summary statistics
        print("\nBacktest Summary:")
        print(f"Start Date: {self.start_date.strftime('%Y-%m-%d')}")
        print(f"End Date: {self.end_date.strftime('%Y-%m-%d')}")
        print(f"Initial Portfolio Value: ${portfolio_series.iloc[0]:,.2f}")
        print(f"Final Portfolio Value: ${portfolio_series.iloc[-1]:,.2f}")
        
        total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1) * 100
        print(f"Total Return: {total_return:.2f}%")
        
        # Calculate annualized return
        days = (self.end_date - self.start_date).days
        annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
        print(f"Annualized Return: {annualized_return:.2f}%")
        
        print(f"\nDetailed results saved in {output_dir}")

if __name__ == "__main__":
    
    backtest = BacktestEngine(
        CONFIG['backtest']['initial_cash'],
        CONFIG['backtest']['min_cash'],
        CONFIG['backtest']['max_open_positions'],
        CONFIG['backtest']['start_date'],
        CONFIG['backtest']['end_date'],
        CONFIG['params'],
        CONFIG['strategies']
    )
    
    results = backtest.run_backtest()
