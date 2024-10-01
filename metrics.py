#metrics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

class PerformanceMetrics:
    @staticmethod
    def total_return(series):
        if len(series) < 2:
            return 0
        return (series.iloc[-1] / series.iloc[0]) - 1

    @staticmethod
    def annual_return(returns, periods_per_year=252):
        if len(returns) < 2:
            return 0
        total_return = (1 + returns).prod()
        n_periods = len(returns)
        return total_return ** (periods_per_year / n_periods) - 1

    @staticmethod
    def annual_volatility(returns, periods_per_year=252):
        if len(returns) < 2:
            return 0
        return returns.std() * np.sqrt(periods_per_year)

    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
        if len(returns) < 2:
            return 0
        excess_returns = returns - risk_free_rate / periods_per_year
        return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

    @staticmethod
    def max_drawdown(returns):
        if len(returns) < 2:
            return 0
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns / peak) - 1
        return drawdown.min()

    @staticmethod
    def calmar_ratio(returns, periods_per_year=252):
        if len(returns) < 2:
            return 0
        annual_ret = PerformanceMetrics.annual_return(returns, periods_per_year)
        max_dd = PerformanceMetrics.max_drawdown(returns)
        return -annual_ret / max_dd if max_dd != 0 else np.inf

    @staticmethod
    def sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
        if len(returns) < 2:
            return 0
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(periods_per_year)
        return (np.mean(excess_returns) * periods_per_year) / downside_deviation if downside_deviation != 0 else np.inf

    @staticmethod
    def beta(returns, market_returns):
        if len(returns) < 2 or len(market_returns) < 2:
            return 0
        covariance = np.cov(returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance != 0 else np.nan

    @staticmethod
    def alpha(returns, market_returns, risk_free_rate=0.02, periods_per_year=252):
        if len(returns) < 2 or len(market_returns) < 2:
            return 0
        beta = PerformanceMetrics.beta(returns, market_returns)
        ann_return = PerformanceMetrics.annual_return(returns, periods_per_year)
        ann_market_return = PerformanceMetrics.annual_return(market_returns, periods_per_year)
        return ann_return - (risk_free_rate + beta * (ann_market_return - risk_free_rate))


    @staticmethod
    def calculate_trade_returns(trades_dict):
        all_trade_info = []
        
        for strategy, trades in trades_dict.items():
            if not trades:  # Skip empty trade lists
                continue
            
            trades_df = pd.DataFrame(trades)

            print(f'TRADES_DF: {trades_df}')
            
            # Check if required columns exist
            required_columns = ['date', 'instrument', 'action', 'units', 'price']
            missing_columns = [col for col in required_columns if col not in trades_df.columns]
            if missing_columns:
                print(f"Warning: Missing columns {missing_columns} for strategy {strategy}. Skipping.")
                continue
            
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            trades_df = trades_df.sort_values(['instrument', 'date'])
            
            for instrument, group in trades_df.groupby('instrument'):
                open_trades = []
                
                for _, trade in group.iterrows():
                    if trade['action'] == 'OPEN':
                        open_trades.append(trade)
                    elif trade['action'] == 'CLOSE' and open_trades:
                        open_trade = open_trades.pop(0)
                        
                        entry_price = open_trade['price']
                        exit_price = trade['price']
                        units = open_trade['units']
                        
                        if units > 0:
                            order = 'LONG'
                            trade_return = (exit_price / entry_price - 1) * 100
                        else:
                            order = 'SHORT'
                            trade_return = (entry_price / exit_price - 1) * 100
                        
                        entry_date = open_trade['date']
                        exit_date = trade['date']
                        holding_time = (exit_date - entry_date).days
                        
                        all_trade_info.append({
                            'Strategy': strategy,
                            'Instrument': instrument,
                            'Order': order,
                            'Trade Return (%)': trade_return,
                            'Entry Price': entry_price,
                            'Exit Price': exit_price,
                            'Entry Date': entry_date,
                            'Exit Date': exit_date,
                            'Holding Time (days)': holding_time,
                            'Status': 'Closed'
                        })
                
                for open_trade in open_trades:
                    all_trade_info.append({
                        'Strategy': strategy,
                        'Instrument': instrument,
                        'Order': 'LONG' if open_trade['units'] > 0 else 'SHORT',
                        'Trade Return (%)': np.nan,
                        'Entry Price': open_trade['price'],
                        'Exit Price': np.nan,
                        'Entry Date': open_trade['date'],
                        'Exit Date': np.nan,
                        'Holding Time (days)': np.nan,
                        'Status': 'Open'
                    })
        
        results_df = pd.DataFrame(all_trade_info)
        
        if not results_df.empty and 'Entry Date' in results_df.columns:
            results_df = results_df.sort_values('Entry Date')
        
        return results_df


    @staticmethod
    def analyze_trades(trades_dict):
        if not trades_dict:
            return pd.DataFrame(columns=['Value'])

        trades_df = PerformanceMetrics.calculate_trade_returns(trades_dict)
        print(f'Analyze trades - trades_df: {trades_df}')
        trades_df.to_csv('backtest_results/trades_df.csv')

        closed_trades = trades_df[trades_df['Status'] == 'Closed']
        
        metrics = {
            'Total Trades': len(trades_df),
            'Closed Trades': len(closed_trades),
            'Open Trades': len(trades_df) - len(closed_trades),
        }

        if len(closed_trades) > 0:
            metrics.update({
                'Winning Trades': sum(closed_trades['Trade Return (%)'] > 0),
                'Losing Trades': sum(closed_trades['Trade Return (%)'] <= 0),
                'Win Rate': sum(closed_trades['Trade Return (%)'] > 0) / len(closed_trades),
                'Average Trade Return': closed_trades['Trade Return (%)'].mean(),
                'Average Winning Trade': closed_trades.loc[closed_trades['Trade Return (%)'] > 0, 'Trade Return (%)'].mean(),
                'Average Losing Trade': closed_trades.loc[closed_trades['Trade Return (%)'] <= 0, 'Trade Return (%)'].mean(),
                'Largest Winning Trade': closed_trades['Trade Return (%)'].max(),
                'Largest Losing Trade': closed_trades['Trade Return (%)'].min(),
                'Average Holding Period (days)': closed_trades['Holding Time (days)'].mean(),
            })
        else:
            metrics.update({
                'Winning Trades': np.nan,
                'Losing Trades': np.nan,
                'Win Rate': np.nan,
                'Average Trade Return': np.nan,
                'Average Winning Trade': np.nan,
                'Average Losing Trade': np.nan,
                'Largest Winning Trade': np.nan,
                'Largest Losing Trade': np.nan,
                'Average Holding Period (days)': np.nan,
            })

        metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        print(metrics)
        return metrics

    @staticmethod
    def calculate_metrics(portfolio_values, benchmark_values, trades, risk_free_rate=0.02, periods_per_year=252):
        #print(f"Debug: calculate_metrics - portfolio_values: {portfolio_values}")
        #print(f"Debug: calculate_metrics - benchmark_values: {benchmark_values}")
        #print(f"Debug: calculate_metrics - trades: {trades}")

        # Ensure portfolio_values and benchmark_values have the same index
        common_index = portfolio_values.index.intersection(benchmark_values.index)
        portfolio_values = portfolio_values.loc[common_index]
        benchmark_values = benchmark_values.loc[common_index]

        # Calculate returns
        portfolio_returns = portfolio_values.pct_change().dropna()
        benchmark_returns = benchmark_values.pct_change().dropna()

        # Ensure returns have the same index
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        # Calculate metrics
        overall_metrics = {
            'Total Return': PerformanceMetrics.total_return(portfolio_values),
            'Annual Return': PerformanceMetrics.annual_return(portfolio_returns, periods_per_year),
            'Annual Volatility': PerformanceMetrics.annual_volatility(portfolio_returns, periods_per_year),
            'Sharpe Ratio': PerformanceMetrics.sharpe_ratio(portfolio_returns, risk_free_rate, periods_per_year),
            'Max Drawdown': PerformanceMetrics.max_drawdown(portfolio_returns),
            'Calmar Ratio': PerformanceMetrics.calmar_ratio(portfolio_returns, periods_per_year),
            'Sortino Ratio': PerformanceMetrics.sortino_ratio(portfolio_returns, risk_free_rate, periods_per_year),
            'Beta': PerformanceMetrics.beta(portfolio_returns, benchmark_returns),
            'Alpha': PerformanceMetrics.alpha(portfolio_returns, benchmark_returns, risk_free_rate, periods_per_year)
        }

        #print(f"Debug: Overall metrics: {overall_metrics}")

        trade_metrics = PerformanceMetrics.analyze_trades(trades)
        #print(f"Debug: Trade metrics: {trade_metrics}")

        try:
            trade_frequency = PerformanceMetrics.analyze_trade_frequency(trades)
        except Exception as e:
            print(f"Warning: Error in analyze_trade_frequency: {str(e)}")
            trade_frequency = pd.Series()

        #print(f"Debug: Trade frequency: {trade_frequency}")

        return {
            'Overall Metrics': pd.DataFrame.from_dict(overall_metrics, orient='index', columns=['Value']),
            'Trade Metrics': trade_metrics,
            'Trade Frequency': trade_frequency
        }


    @staticmethod
    def analyze_trade_frequency(trades_dict):
        #print(f"Debug: analyze_trade_frequency - trades_dict: {trades_dict}")
        if not trades_dict:
            print("Warning: Empty trades dictionary")
            return pd.Series()

        all_trades = []
        for strategy, trades in trades_dict.items():
            if not trades:
                print(f"Warning: No trades for strategy {strategy}")
                continue
            
            strategy_trades = pd.DataFrame(trades)
            strategy_trades['strategy'] = strategy
            all_trades.append(strategy_trades)

        if not all_trades:
            print("Warning: No valid trades found")
            return pd.Series()

        trades_df = pd.concat(all_trades, ignore_index=True)
        
        #print(f"Debug: Combined trades_df: {trades_df}")

        if 'instrument' not in trades_df.columns:
            print("Warning: 'instrument' column not found in trades data")
            return pd.Series()

        trade_frequency = trades_df['instrument'].value_counts()
        #print(f"Debug: Trade frequency: {trade_frequency}")

        return trade_frequency


    @staticmethod
    def plot_equity_curve(portfolio_values, benchmark_values):
        #print(f"Debug: plot_equity_curve - portfolio_values: {portfolio_values}")
        #print(f"Debug: plot_equity_curve - benchmark_values: {benchmark_values}")
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values.index, portfolio_values, label='Portfolio')
        plt.plot(benchmark_values.index, benchmark_values, label='Benchmark')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_drawdown(returns):
        #print(f"Debug: plot_drawdown - returns: {returns}")
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns / peak) - 1
        
        plt.figure(figsize=(12, 6))
        plt.plot(drawdown.index, drawdown)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_monthly_returns_heatmap(returns):
        #print(f"Debug: plot_monthly_returns_heatmap - returns: {returns}")
        monthly_returns = returns.resample('M').agg(lambda x: (x + 1).prod() - 1)
        monthly_returns_df = monthly_returns.to_frame()
        monthly_returns_df['Year'] = monthly_returns_df.index.year
        monthly_returns_df['Month'] = monthly_returns_df.index.month
        monthly_returns_pivot = monthly_returns_df.pivot(index='Month', columns='Year', values='Value')
        
        #print(f"Debug: monthly_returns_pivot: {monthly_returns_pivot}")
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_returns_pivot, annot=True, fmt='.2%', cmap='RdYlGn')
        plt.title('Monthly Returns Heatmap')
        plt.show()

    @staticmethod
    def plot_trade_returns_distribution(trades_dict):
        #print(f"Debug: plot_trade_returns_distribution - trades_dict: {trades_dict}")
        trade_returns_df = PerformanceMetrics.calculate_trade_returns(trades_dict)
        
        #print(f"Debug: trade_returns_df: {trade_returns_df}")
        
        plt.figure(figsize=(12, 6))
        
        for strategy in trade_returns_df['Strategy'].unique():
            strategy_data = trade_returns_df[trade_returns_df['Strategy'] == strategy]
            sns.histplot(data=strategy_data, x='Trade Return (%)', kde=True, label=strategy)
        
        plt.axvline(x=0, color='r', linestyle='--', label='Break-even')
        
        mean_return = trade_returns_df['Trade Return (%)'].mean()
        median_return = trade_returns_df['Trade Return (%)'].median()
        
        plt.axvline(x=mean_return, color='g', linestyle='-', label=f'Mean: {mean_return:.2f}%')
        plt.axvline(x=median_return, color='b', linestyle='-', label=f'Median: {median_return:.2f}%')
        
        plt.xlabel('Trade Return (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Trade Returns by Strategy')
        
        plt.legend()
        
        stats = f"Total Trades: {len(trade_returns_df)}\n"
        stats += f"Profitable Trades: {sum(trade_returns_df['Trade Return (%)'] > 0)} ({sum(trade_returns_df['Trade Return (%)'] > 0) / len(trade_returns_df) * 100:.2f}%)\n"
        stats += f"Loss-making Trades: {sum(trade_returns_df['Trade Return (%)'] < 0)} ({sum(trade_returns_df['Trade Return (%)'] < 0) / len(trade_returns_df) * 100:.2f}%)\n"
        stats += f"Max Profit: {trade_returns_df['Trade Return (%)'].max():.2f}%\n"
        stats += f"Max Loss: {trade_returns_df['Trade Return (%)'].min():.2f}%"
        
        plt.text(0.95, 0.95, stats, transform=plt.gca().transAxes, va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_trade_frequency(trade_frequency):
        plt.figure(figsize=(12, 6))
        trade_frequency.plot(kind='bar')
        plt.title('Trade Frequency by Instrument')
        plt.xlabel('Instrument')
        plt.ylabel('Number of Trades')
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def plot_cumulative_returns(portfolio_returns, benchmark_returns):
        cum_portfolio_returns = (1 + portfolio_returns).cumprod()
        cum_benchmark_returns = (1 + benchmark_returns).cumprod()
        
        plt.figure(figsize=(12, 6))
        plt.plot(cum_portfolio_returns.index, cum_portfolio_returns, label='Portfolio')
        plt.plot(cum_benchmark_returns.index, cum_benchmark_returns, label='Benchmark')
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_rolling_sharpe(returns, window=252):
        rolling_sharpe = returns.rolling(window=window).apply(lambda x: PerformanceMetrics.sharpe_ratio(x))
        
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_sharpe.index, rolling_sharpe)
        plt.title(f'Rolling Sharpe Ratio (Window: {window} days)')
        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_exposure_over_time(trades, portfolio_values):
        if not trades:
            print("No trades to plot exposure.")
            return

        trades_df = pd.DataFrame(trades)
        
        if 'entry_date' not in trades_df.columns or 'exposure' not in trades_df.columns:
            print("Required columns 'entry_date' or 'exposure' not found in trades data")
            return

        daily_exposure = trades_df.set_index('entry_date')['exposure'].resample('D').sum()
        daily_exposure = daily_exposure.reindex(portfolio_values.index).fillna(0)
        
        plt.figure(figsize=(12, 6))
        plt.plot(daily_exposure.index, daily_exposure)
        plt.title('Daily Exposure Over Time')
        plt.xlabel('Date')
        plt.ylabel('Exposure')
        plt.grid(True)
        plt.show()

    @staticmethod
    def check_look_ahead_bias(trades, portfolio_values):
        if not trades:
            print("No trades to check for look-ahead bias.")
            return

        trades_df = pd.DataFrame(trades)

        if 'Entry_date' not in trades_df.columns or 'Exit_date' not in trades_df.columns:
            print("Required columns 'Entry_date' or 'Exit_date' not found in trades data")
            return

        for _, trade in trades_df.iterrows():
            entry_date = trade['Entry_date']
            exit_date = trade['Exit_date']
            
            if entry_date not in portfolio_values.index or exit_date not in portfolio_values.index:
                print(f"Warning: Trade dates not in portfolio values index. Entry: {entry_date}, Exit: {exit_date}")
            
            if exit_date <= entry_date:
                print(f"Warning: Exit date {exit_date} is not after entry date {entry_date}")

    @staticmethod
    def check_overfitting(in_sample_returns, out_of_sample_returns):
        in_sample_sharpe = PerformanceMetrics.sharpe_ratio(in_sample_returns)
        out_of_sample_sharpe = PerformanceMetrics.sharpe_ratio(out_of_sample_returns)
        
        print(f"In-sample Sharpe Ratio: {in_sample_sharpe:.2f}")
        print(f"Out-of-sample Sharpe Ratio: {out_of_sample_sharpe:.2f}")
        
        if out_of_sample_sharpe < 0.5 * in_sample_sharpe:
            print("Warning: Potential overfitting detected. Out-of-sample performance is significantly worse.")

    @staticmethod
    def save_results(output_dir, metrics):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save overall metrics
        if metrics['Overall Metrics'] is not None:
            overall_metrics_path = os.path.join(output_dir, 'overall_metrics.json')
            with open(overall_metrics_path, 'w') as f:
                json.dump(metrics['Overall Metrics'].to_dict(), f, indent=4)
        
        # Save trade metrics
        if metrics['Trade Metrics'] is not None:
            trade_metrics_path = os.path.join(output_dir, 'trade_metrics.json')
            with open(trade_metrics_path, 'w') as f:
                json.dump(metrics['Trade Metrics'].to_dict(), f, indent=4)
        
        # Save trade frequency
        if metrics['Trade Frequency'] is not None:
            trade_frequency_path = os.path.join(output_dir, 'trade_frequency.json')
            with open(trade_frequency_path, 'w') as f:
                json.dump(metrics['Trade Frequency'].to_dict(), f, indent=4)
        
        print(f"Results saved in {output_dir}")

    @staticmethod
    def run_full_analysis(portfolio_values, benchmark_values, trades_dict, risk_free_rate=0.0, periods_per_year=365):
        try:
            #print("Debug: Starting run_full_analysis")
            #print(f"Debug: Initial portfolio_values: {portfolio_values}")
            #print(f"Debug: Initial benchmark_values: {benchmark_values}")
            
            # Ensure portfolio_values and benchmark_values are pandas Series with a DatetimeIndex
            if not isinstance(portfolio_values, pd.Series):
                portfolio_values = pd.Series(portfolio_values)
            if not isinstance(benchmark_values, pd.Series):
                benchmark_values = pd.Series(benchmark_values)

            #print(f"Debug: portfolio_values after conversion: {portfolio_values}")
            #print(f"Debug: benchmark_values after conversion: {benchmark_values}")

            if not isinstance(portfolio_values.index, pd.DatetimeIndex):
                portfolio_values.index = pd.to_datetime(portfolio_values.index)
            if not isinstance(benchmark_values.index, pd.DatetimeIndex):
                benchmark_values.index = pd.to_datetime(benchmark_values.index)

            #print(f"Debug: portfolio_values after index conversion: {portfolio_values}")
            #print(f"Debug: benchmark_values after index conversion: {benchmark_values}")

            # Align the data
            common_index = portfolio_values.index.intersection(benchmark_values.index)
            portfolio_values = portfolio_values.loc[common_index]
            benchmark_values = benchmark_values.loc[common_index]

            #print(f"Debug: portfolio_values after alignment: {portfolio_values}")
            #print(f"Debug: benchmark_values after alignment: {benchmark_values}")

            if len(portfolio_values) < 2 or len(benchmark_values) < 2:
                print("Warning: Insufficient data points for meaningful analysis.")
                return

            metrics = PerformanceMetrics.calculate_metrics(portfolio_values, benchmark_values, trades_dict, risk_free_rate, periods_per_year)
            PerformanceMetrics.save_results('backtest_results', metrics)

            
            print("Overall Metrics:")
            print(metrics['Overall Metrics'])
            print("\nTrade Metrics:")
            print(metrics['Trade Metrics'])
            
            portfolio_returns = portfolio_values.pct_change().dropna()
            benchmark_returns = benchmark_values.pct_change().dropna()
            
            #print(f"Debug: portfolio_returns: {portfolio_returns}")
            #print(f"Debug: benchmark_returns: {benchmark_returns}")

            if len(portfolio_returns) >= 2 and len(benchmark_returns) >= 2:
                #print("Debug: Attempting to plot equity curve")
                PerformanceMetrics.plot_equity_curve(portfolio_values, benchmark_values)
                #print("Debug: Attempting to plot drawdown")
                PerformanceMetrics.plot_drawdown(portfolio_returns)
                #print("Debug: Attempting to plot monthly returns heatmap")
                PerformanceMetrics.plot_monthly_returns_heatmap(portfolio_returns)
                #print("Debug: Attempting to plot trade returns distribution")
                PerformanceMetrics.plot_trade_returns_distribution(trades_dict)
            else:
                print("Warning: Insufficient data points for plotting.")

        except Exception as e:
            print(f"An error occurred during the analysis: {str(e)}")
            print("Debug information:")
            print(f"Portfolio values shape: {portfolio_values.shape}")
            print(f"Benchmark values shape: {benchmark_values.shape}")
            print(f"Trades dict keys: {trades_dict.keys()}")
            for strategy, trades in trades_dict.items():
                print(f"Strategy: {strategy}, Number of trades: {len(trades)}")
            
            # Print stack trace for more detailed error information
            import traceback
            traceback.print_exc()