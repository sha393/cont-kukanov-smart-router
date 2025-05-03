#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from itertools import product
from typing import List, Dict, Tuple, Any
import random

BUY_ORDER_SIZE = 5000
TWAP_BUCKET_SIZE = 60
DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def allocate(order_size: int, venues: List[Dict], lambda_over: float, lambda_under: float, 
             theta_queue: float) -> Tuple[List[int], float]:
    if not venues:
        return [], float('inf')
    
    step = min(100, max(1, order_size // 50))
    debug_print(f"Using step size: {step}")
    
    splits = [[]]
    
    for v in range(len(venues)):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            if used >= order_size:
                new_splits.append(alloc)
                continue
            
            max_v = min(order_size - used, venues[v]['ask_size'])
            for q in range(0, max_v + 1, step):
                if used + q <= order_size:
                    new_splits.append(alloc + [q])
            if max_v > 0 and (len(alloc) == 0 or alloc[-1] != max_v):
                new_splits.append(alloc + [max_v])
        splits = new_splits
    
    best_cost = float('inf')
    best_split = []
    
    for alloc in splits:
        if sum(alloc) > 0:
            cost = compute_cost(alloc, venues, order_size, lambda_over, lambda_under, theta_queue)
            if cost < best_cost:
                best_cost = cost
                best_split = alloc
    
    debug_print(f"Best allocation: {best_split} with cost: {best_cost}")
    return best_split, best_cost

def compute_cost(split: List[int], venues: List[Dict], order_size: int, 
                 lambda_over: float, lambda_under: float, theta: float) -> float:
    executed = 0
    cash_spent = 0
    
    for i in range(len(venues)):
        exe = min(split[i], venues[i]['ask_size'])
        executed += exe
        
        price_with_fee = venues[i]['ask'] + venues[i]['fee']
        cost_component = exe * price_with_fee
        cash_spent += cost_component
        
        if split[i] > 0:
            rebate_amount = min(split[i], venues[i]['ask_size']) * venues[i]['rebate']
            cash_spent -= rebate_amount
    
    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    
    risk_pen = theta * (underfill + overfill)
    cost_pen = lambda_under * underfill + lambda_over * overfill
    
    total_cost = cash_spent + risk_pen + cost_pen
    
    debug_print(f"Total cost: {total_cost}")
    return total_cost

def single_venue_optimize(order_size: int, venue: Dict, lambda_over: float, lambda_under: float, 
                          theta_queue: float, queue_distrib_func=None) -> Tuple[int, int]:
    """
    Single venue optimization based on Proposition 3.
    Returns: (market_order_size, limit_order_size)
    """
    # For single venue, M + L = S (no oversizing)
    Q = venue.get('queue_position', 0)
    h = 0.005  # half-spread (simplified)
    f = venue['fee']
    r = venue['rebate']
    
    # Simplified probability model for queue outflow
    # We'll assume a simple model where fill probability depends on queue position
    if Q == 0:
        # No queue - limit orders fill with high probability
        fill_prob_limit = 0.9
    else:
        # Fill probability decreases with queue size
        fill_prob_limit = max(0.1, 1.0 - Q / (2 * venue['ask_size']))
    
    # Cost comparison
    # Market order cost
    market_cost = h + f
    
    # Expected limit order cost (accounting for non-fill risk)
    expected_limit_cost = -(h + r) * fill_prob_limit + lambda_under * (1 - fill_prob_limit)
    
    # Add impact costs
    total_market_cost = market_cost + theta_queue
    total_limit_cost = expected_limit_cost + theta_queue
    
    # Decision logic
    if total_limit_cost <= total_market_cost:
        # Limit orders are cheaper
        return 0, order_size
    else:
        # Market orders are cheaper
        return order_size, 0
    
def process_level1_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values(['ts_event', 'publisher_id']).drop_duplicates(subset=['ts_event', 'publisher_id'])
    
    snapshot_data = df[['ts_event', 'publisher_id', 'ask_px_00', 'ask_sz_00']].copy()
    snapshot_data = snapshot_data.rename(columns={
        'ask_px_00': 'ask',
        'ask_sz_00': 'ask_size'
    })
    
    snapshot_data = snapshot_data.sort_values('ts_event')
    snapshot_data['queue_position'] = 0  # Set all queue positions to 0
    #snapshot_data['queue_position'] = snapshot_data['ask_size'].apply(lambda x: random.randint(0, max(1, int(x * 0.1))))
    
    return snapshot_data

def run_backtest(snapshots: pd.DataFrame, lambda_over: float, lambda_under: float, 
                 theta_queue: float, fees: Dict[int, float], rebates: Dict[int, float]) -> Dict:
    remaining_size = BUY_ORDER_SIZE
    total_cash_spent = 0
    executed_shares = 0
    fill_history = []
    
    grouped = snapshots.groupby('ts_event')
    
    cumulative_fills = []
    cumulative_costs = []
    timestamps = []
    
    for ts, group in grouped:
        if remaining_size <= 0:
            break
            
        venues = []
        for _, row in group.iterrows():
            venue_id = row['publisher_id']
            venues.append({
                'ask': row['ask'],
                'ask_size': row['ask_size'],
                'fee': fees.get(venue_id, 0.0),
                'rebate': rebates.get(venue_id, 0.0),
                'queue_position': row.get('queue_position', 0)
            })
        
        if not venues:
            continue
            
        allocation, _ = allocate(
            remaining_size, 
            venues, 
            lambda_over, 
            lambda_under, 
            theta_queue
        )
        
        for i, alloc in enumerate(allocation):
            effective_ask_size = max(0, venues[i]['ask_size'] - venues[i]['queue_position'])
            fill_size = min(alloc, effective_ask_size)
            
            if fill_size > 0:
                price = venues[i]['ask']
                fee = venues[i]['fee']
                rebate = venues[i]['rebate']
                
                cost = fill_size * (price + fee) - (fill_size * rebate)
                
                total_cash_spent += cost
                executed_shares += fill_size
                
                fill_info = {
                    'timestamp': ts,
                    'venue': venues[i].get('id', i),
                    'size': fill_size,
                    'price': price,
                    'fee': fee,
                    'rebate': rebate,
                    'cost': cost
                }
                fill_history.append(fill_info)
        
        remaining_size -= sum(min(alloc, max(0, venues[i]['ask_size'] - venues[i]['queue_position'])) 
                           for i, alloc in enumerate(allocation))
        
        cumulative_fills.append(BUY_ORDER_SIZE - remaining_size)
        if executed_shares > 0:
            cumulative_costs.append(total_cash_spent / executed_shares)
        else:
            cumulative_costs.append(0)
        timestamps.append(ts)
    
    average_price = total_cash_spent / executed_shares if executed_shares > 0 else 0
    
    return {
        'executed_shares': executed_shares,
        'remaining_size': remaining_size,
        'total_cash_spent': total_cash_spent,
        'average_price': average_price,
        'fill_history': fill_history,
        'cumulative_fills': cumulative_fills,
        'cumulative_costs': cumulative_costs,
        'timestamps': timestamps
    }

def run_backtest(snapshots: pd.DataFrame, lambda_over: float, lambda_under: float, 
                 theta_queue: float, fees: Dict[int, float], rebates: Dict[int, float]) -> Dict:
    # Initialize variables
    remaining_size = BUY_ORDER_SIZE
    total_cash_spent = 0
    executed_shares = 0
    fill_history = []
    
    grouped = snapshots.groupby('ts_event')
    
    cumulative_fills = []
    cumulative_costs = []
    timestamps = []
    
    for ts, group in grouped:
        if remaining_size <= 0:
            break
            
        venues = []
        for _, row in group.iterrows():
            venue_id = row['publisher_id']
            venues.append({
                'ask': row['ask'],
                'ask_size': row['ask_size'],
                'fee': fees.get(venue_id, 0.0),
                'rebate': rebates.get(venue_id, 0.0),
                'queue_position': row.get('queue_position', 0)
            })
        
        if not venues:
            continue
            
        allocation, _ = allocate(
            remaining_size, 
            venues, 
            lambda_over, 
            lambda_under, 
            theta_queue
        )
        
        for i, alloc in enumerate(allocation):
            effective_ask_size = max(0, venues[i]['ask_size'] - venues[i]['queue_position'])
            fill_size = min(alloc, effective_ask_size)
            
            if fill_size > 0:
                price = venues[i]['ask']
                fee = venues[i]['fee']
                rebate = venues[i]['rebate']
                
                cost = fill_size * (price + fee) - (fill_size * rebate)
                
                total_cash_spent += cost
                executed_shares += fill_size
                
                fill_info = {
                    'timestamp': ts,
                    'venue': venues[i].get('id', i),
                    'size': fill_size,
                    'price': price,
                    'fee': fee,
                    'rebate': rebate,
                    'cost': cost
                }
                fill_history.append(fill_info)
        
        remaining_size -= sum(min(alloc, max(0, venues[i]['ask_size'] - venues[i]['queue_position'])) 
                           for i, alloc in enumerate(allocation))
        
        cumulative_fills.append(BUY_ORDER_SIZE - remaining_size)
        if executed_shares > 0:
            cumulative_costs.append(total_cash_spent / executed_shares)
        else:
            cumulative_costs.append(0)
        timestamps.append(ts)
    
    average_price = total_cash_spent / executed_shares if executed_shares > 0 else 0
    
    return {
        'executed_shares': executed_shares,
        'remaining_size': remaining_size,
        'total_cash_spent': total_cash_spent,
        'average_price': average_price,
        'fill_history': fill_history,
        'cumulative_fills': cumulative_fills,
        'cumulative_costs': cumulative_costs,
        'timestamps': timestamps
    }

# ADD THIS NEW FUNCTION HERE:
def run_optimized_single_venue(snapshots: pd.DataFrame, lambda_over: float, lambda_under: float, 
                               theta_queue: float, fees: Dict[int, float], rebates: Dict[int, float]) -> Dict:
    remaining_size = BUY_ORDER_SIZE
    total_cash_spent = 0
    executed_shares = 0
    fill_history = []
    
    grouped = snapshots.groupby('ts_event')
    
    cumulative_fills = []
    cumulative_costs = []
    timestamps = []
    
    for ts, group in grouped:
        if remaining_size <= 0:
            break
            
        # Single venue case
        if len(group) == 1:
            row = group.iloc[0]
            venue_id = row['publisher_id']
            venue = {
                'ask': row['ask'],
                'ask_size': row['ask_size'],
                'fee': fees.get(venue_id, 0.0),
                'rebate': rebates.get(venue_id, 0.0),
                'queue_position': row.get('queue_position', 0)
            }
            
            # Use single-venue optimization
            M_opt, L_opt = single_venue_optimize(
                remaining_size, 
                venue, 
                lambda_over, 
                lambda_under, 
                theta_queue
            )
            
            # Execute market orders first
            market_fill = M_opt
            market_cost = market_fill * (venue['ask'] + venue['fee'])
            total_cash_spent += market_cost
            executed_shares += market_fill
            
            # Then try limit orders (with queue position)
            effective_ask_size = max(0, venue['ask_size'] - venue['queue_position'])
            limit_fill = min(L_opt, effective_ask_size)
            limit_cost = limit_fill * (venue['ask'] + venue['fee'] - venue['rebate'])
            total_cash_spent += limit_cost
            executed_shares += limit_fill
            
            remaining_size -= (market_fill + limit_fill)
            
            fill_history.append({
                'timestamp': ts,
                'venue': venue_id,
                'market_size': market_fill,
                'limit_size': limit_fill,
                'total_cost': market_cost + limit_cost
            })
        
        cumulative_fills.append(BUY_ORDER_SIZE - remaining_size)
        if executed_shares > 0:
            cumulative_costs.append(total_cash_spent / executed_shares)
        else:
            cumulative_costs.append(0)
        timestamps.append(ts)
    
    average_price = total_cash_spent / executed_shares if executed_shares > 0 else 0
    
    return {
        'executed_shares': executed_shares,
        'remaining_size': remaining_size,
        'total_cash_spent': total_cash_spent,
        'average_price': average_price,
        'fill_history': fill_history,
        'cumulative_fills': cumulative_fills,
        'cumulative_costs': cumulative_costs,
        'timestamps': timestamps
    }


def run_best_ask_baseline(snapshots: pd.DataFrame, fees: Dict[int, float]) -> Dict:
    remaining_size = BUY_ORDER_SIZE
    total_cash_spent = 0
    executed_shares = 0
    fill_history = []
    
    grouped = snapshots.groupby('ts_event')
    
    cumulative_fills = []
    cumulative_costs = []
    timestamps = []
    
    for ts, group in grouped:
        if remaining_size <= 0:
            break
            
        best_ask_row = group.loc[group['ask'].idxmin()]
        best_ask = best_ask_row['ask']
        venue_id = best_ask_row['publisher_id']
        available_size = best_ask_row['ask_size']
        
        queue_position = best_ask_row.get('queue_position', 0)
        effective_size = max(0, available_size - queue_position)
        
        fill_size = min(remaining_size, effective_size)
        if fill_size > 0:
            fee = fees.get(venue_id, 0.0)
            price = best_ask
            cost = fill_size * (price + fee)
            
            total_cash_spent += cost
            executed_shares += fill_size
            remaining_size -= fill_size
            
            fill_history.append({
                'timestamp': ts,
                'venue': venue_id,
                'size': fill_size,
                'price': price,
                'fee': fee,
                'cost': cost
            })
        
        cumulative_fills.append(BUY_ORDER_SIZE - remaining_size)
        if executed_shares > 0:
            cumulative_costs.append(total_cash_spent / executed_shares)
        else:
            cumulative_costs.append(0)
        timestamps.append(ts)
    
    average_price = total_cash_spent / executed_shares if executed_shares > 0 else 0
    
    return {
        'executed_shares': executed_shares,
        'remaining_size': remaining_size,
        'total_cash_spent': total_cash_spent,
        'average_price': average_price,
        'fill_history': fill_history,
        'cumulative_fills': cumulative_fills,
        'cumulative_costs': cumulative_costs,
        'timestamps': timestamps
    }

def run_twap_baseline(snapshots: pd.DataFrame, fees: Dict[int, float]) -> Dict:
    snapshots = snapshots.copy()
    start_time = snapshots['ts_event'].min()
    snapshots['bucket'] = ((snapshots['ts_event'] - start_time).dt.total_seconds() / TWAP_BUCKET_SIZE).astype(int)
    
    total_buckets = snapshots['bucket'].max() + 1
    shares_per_bucket = BUY_ORDER_SIZE / total_buckets
    
    remaining_size = BUY_ORDER_SIZE
    total_cash_spent = 0
    executed_shares = 0
    fill_history = []
    
    cumulative_fills = []
    cumulative_costs = []
    timestamps = []
    
    for bucket in range(total_buckets):
        bucket_snapshots = snapshots[snapshots['bucket'] == bucket]
        
        if remaining_size <= 0 or bucket_snapshots.empty:
            continue
        
        bucket_shares = min(remaining_size, round(shares_per_bucket))
        
        first_snapshot = bucket_snapshots.iloc[0]
        ts = first_snapshot['ts_event']
        
        group = bucket_snapshots[bucket_snapshots['ts_event'] == ts]
        
        best_ask_row = group.loc[group['ask'].idxmin()]
        best_ask = best_ask_row['ask']
        venue_id = best_ask_row['publisher_id']
        available_size = best_ask_row['ask_size']
        
        queue_position = best_ask_row.get('queue_position', 0)
        effective_size = max(0, available_size - queue_position)
        # Add before fill_size calculation
        print(f"TWAP: bucket_shares={bucket_shares}, effective_size={effective_size}")
        
        fill_size = min(bucket_shares, effective_size)
        if fill_size > 0:
            fee = fees.get(venue_id, 0.0)
            price = best_ask
            cost = fill_size * (price + fee)
            
            total_cash_spent += cost
            executed_shares += fill_size
            remaining_size -= fill_size
            
            fill_history.append({
                'timestamp': ts,
                'venue': venue_id,
                'size': fill_size,
                'price': price,
                'fee': fee,
                'cost': cost
            })
        
        cumulative_fills.append(BUY_ORDER_SIZE - remaining_size)
        if executed_shares > 0:
            cumulative_costs.append(total_cash_spent / executed_shares)
        else:
            cumulative_costs.append(0)
        timestamps.append(ts)
    
    average_price = total_cash_spent / executed_shares if executed_shares > 0 else 0
    
    return {
        'executed_shares': executed_shares,
        'remaining_size': remaining_size,
        'total_cash_spent': total_cash_spent,
        'average_price': average_price,
        'fill_history': fill_history,
        'cumulative_fills': cumulative_fills,
        'cumulative_costs': cumulative_costs,
        'timestamps': timestamps
    }

def run_vwap_baseline(snapshots: pd.DataFrame, fees: Dict[int, float]) -> Dict:
    remaining_size = BUY_ORDER_SIZE
    total_cash_spent = 0
    executed_shares = 0
    fill_history = []
    
    grouped = snapshots.groupby('ts_event')
    
    cumulative_fills = []
    cumulative_costs = []
    timestamps = []
    
    total_volume = snapshots['ask_size'].sum()
    
    for ts, group in grouped:
        if remaining_size <= 0:
            break
            
        timestamp_volume = group['ask_size'].sum()
        
        proportion = timestamp_volume / total_volume
        shares_to_execute = min(remaining_size, round(BUY_ORDER_SIZE * proportion))
        
        if shares_to_execute <= 0:
            continue
        
        for _, row in group.iterrows():
            venue_id = row['publisher_id']
            venue_proportion = row['ask_size'] / timestamp_volume
            venue_shares = round(shares_to_execute * venue_proportion)
            
            if venue_shares <= 0:
                continue
                
            queue_position = row.get('queue_position', 0)
            effective_size = max(0, row['ask_size'] - queue_position)
            
            fill_size = min(venue_shares, effective_size)
            if fill_size > 0:
                fee = fees.get(venue_id, 0.0)
                price = row['ask']
                cost = fill_size * (price + fee)
                
                total_cash_spent += cost
                executed_shares += fill_size
                remaining_size -= fill_size
                
                fill_history.append({
                    'timestamp': ts,
                    'venue': venue_id,
                    'size': fill_size,
                    'price': price,
                    'fee': fee,
                    'cost': cost
                })
        
        cumulative_fills.append(BUY_ORDER_SIZE - remaining_size)
        if executed_shares > 0:
            cumulative_costs.append(total_cash_spent / executed_shares)
        else:
            cumulative_costs.append(0)
        timestamps.append(ts)
    
    average_price = total_cash_spent / executed_shares if executed_shares > 0 else 0
    
    return {
        'executed_shares': executed_shares,
        'remaining_size': remaining_size,
        'total_cash_spent': total_cash_spent,
        'average_price': average_price,
        'fill_history': fill_history,
        'cumulative_fills': cumulative_fills,
        'cumulative_costs': cumulative_costs,
        'timestamps': timestamps
    }

def hybrid_search(snapshots: pd.DataFrame, fees: Dict[int, float], rebates: Dict[int, float]) -> Dict:
    print("Phase 1: Random search to identify promising parameters...")
    
    lambda_over_range = (0.01, 1.0)
    lambda_under_range = (0.01, 1.0)
    theta_queue_range = (0.001, 0.1)
    num_samples = 15
    
    best_avg_price = float('inf')
    best_params = {}
    best_result = {}
    
    print(f"Starting random search with {num_samples} parameter combinations...")
    
    # Check venue count for appropriate function selection
    unique_venues = snapshots['publisher_id'].unique()
    is_single_venue = len(unique_venues) == 1
    
    for i in range(num_samples):
        lambda_over = np.random.uniform(*lambda_over_range)
        lambda_under = np.random.uniform(*lambda_under_range)
        theta_queue = np.random.uniform(*theta_queue_range)
        
        print(f"Testing combination {i+1}/{num_samples}: λ_over={lambda_over:.3f}, λ_under={lambda_under:.3f}, θ_queue={theta_queue:.3f}")
        
        # Use appropriate function based on venue count
        if is_single_venue:
            result = run_optimized_single_venue(
                snapshots, 
                lambda_over, 
                lambda_under, 
                theta_queue, 
                fees, 
                rebates
            )
        else:
            result = run_backtest(
                snapshots, 
                lambda_over, 
                lambda_under, 
                theta_queue, 
                fees, 
                rebates
            )
        
        avg_price = result['average_price']
        executed_shares = result['executed_shares']
        
        print(f"  Result: {executed_shares} shares executed at avg price {avg_price:.4f}")
        
        if executed_shares > 0 and avg_price < best_avg_price:
            best_avg_price = avg_price
            best_params = {
                'lambda_over': lambda_over,
                'lambda_under': lambda_under,
                'theta_queue': theta_queue
            }
            best_result = result
            print(f"  New best result found!")
    
    print(f"Random search complete. Best average price: {best_avg_price}")
    
    if not best_params:
        print("Random search failed to find valid parameters.")
        return {
            'params': {},
            'result': {}
        }
    
    print(f"Phase 2: Focused grid search around best random parameters: {best_params}")
    
    lambda_over = best_params['lambda_over']
    lambda_under = best_params['lambda_under']
    theta_queue = best_params['theta_queue']
    
    lambda_over_min = max(0.01, lambda_over * 0.5)
    lambda_over_max = min(1.0, lambda_over * 1.5)
    
    lambda_under_min = max(0.01, lambda_under * 0.5)
    lambda_under_max = min(1.0, lambda_under * 1.5)
    
    theta_queue_min = max(0.001, theta_queue * 0.5)
    theta_queue_max = min(0.1, theta_queue * 1.5)
    
    lambda_over_values = np.linspace(lambda_over_min, lambda_over_max, 5)
    lambda_under_values = np.linspace(lambda_under_min, lambda_under_max, 5)
    theta_queue_values = np.linspace(theta_queue_min, theta_queue_max, 3)
    
    total_combinations = len(lambda_over_values) * len(lambda_under_values) * len(theta_queue_values)
    print(f"Running focused grid search with {total_combinations} parameter combinations...")
    
    completed = 0
    
    for lo in lambda_over_values:
        for lu in lambda_under_values:
            for tq in theta_queue_values:
                completed += 1
                print(f"Testing combination {completed}/{total_combinations}: λ_over={lo:.3f}, λ_under={lu:.3f}, θ_queue={tq:.3f}")
                
                # Use appropriate function based on venue count
                if is_single_venue:
                    result = run_optimized_single_venue(
                        snapshots, 
                        lo, 
                        lu, 
                        tq, 
                        fees, 
                        rebates
                    )
                else:
                    result = run_backtest(
                        snapshots, 
                        lo, 
                        lu, 
                        tq, 
                        fees, 
                        rebates
                    )
                
                avg_price = result['average_price']
                executed_shares = result['executed_shares']
                
                print(f"  Result: {executed_shares} shares executed at avg price {avg_price:.4f}")
                
                if executed_shares > 0 and avg_price < best_avg_price:
                    best_avg_price = avg_price
                    best_params = {
                        'lambda_over': lo,
                        'lambda_under': lu,
                        'theta_queue': tq
                    }
                    best_result = result
                    print(f"  New best result found!")
    
    print(f"Hybrid search complete. Best average price: {best_avg_price}")
    return {
        'params': best_params,
        'result': best_result
    }

def calculate_savings_bps(optimized: float, baseline: float) -> float:
    if baseline == 0 or optimized == 0:
        return 0
    return 10000 * (baseline - optimized) / baseline

def plot_cumulative_costs(
    opt_results: Dict, 
    best_ask_results: Dict, 
    twap_results: Dict, 
    vwap_results: Dict,
    params: Dict
) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    start_time = min(opt_results['timestamps'][0],
                    best_ask_results['timestamps'][0],
                    twap_results['timestamps'][0],
                    vwap_results['timestamps'][0])
    
    opt_times = [(t - start_time).total_seconds() for t in opt_results['timestamps']]
    best_ask_times = [(t - start_time).total_seconds() for t in best_ask_results['timestamps']]
    twap_times = [(t - start_time).total_seconds() for t in twap_results['timestamps']]
    vwap_times = [(t - start_time).total_seconds() for t in vwap_results['timestamps']]
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Average Price')
    
    ax1.plot(opt_times, opt_results['cumulative_costs'], label='Optimized', linewidth=2)
    ax1.plot(best_ask_times, best_ask_results['cumulative_costs'], label='Best Ask', linestyle='--')
    ax1.plot(twap_times, twap_results['cumulative_costs'], label='TWAP', linestyle=':')
    ax1.plot(vwap_times, vwap_results['cumulative_costs'], label='VWAP', linestyle='-.')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Shares Filled')
    
    ax2.plot(opt_times, opt_results['cumulative_fills'], color='gray', alpha=0.5, label='Shares Filled')
    
    param_str = f"λ_over={params['lambda_over']:.3f}, λ_under={params['lambda_under']:.3f}, θ_queue={params['theta_queue']:.3f}"
    plt.title(f"Cumulative Cost Comparison\n{param_str}")
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('results.png', dpi=150)

def main():
    global DEBUG
    DEBUG = False
    
    print("Processing L1 data...")
    snapshots = process_level1_data('c:/Users/moham/OneDrive/Desktop/l1_day.csv')
    
    print(f"\nData info: {len(snapshots)} total records")
    first_ts = snapshots['ts_event'].min()
    first_snap = snapshots[snapshots['ts_event'] == first_ts]
    print(f"First timestamp: {first_ts}")
    print(f"Venues at first timestamp: {len(first_snap)}")
    print(f"Total liquidity at first timestamp: {first_snap['ask_size'].sum()}")
    
    unique_venues = snapshots['publisher_id'].unique()
    fees = {}
    rebates = {}
    
    for i, venue_id in enumerate(unique_venues):
        fees[venue_id] = 0.00005 + (i % 3) * 0.00005
        rebates[venue_id] = 0.00005 + (i % 2) * 0.000025
    
    print(f"\nVenue fee structure:")
    for venue_id in unique_venues[:5]:
        print(f"  Venue {venue_id}: Fee={fees[venue_id]*10000:.1f}bps, Rebate={rebates[venue_id]*10000:.1f}bps")
    
    print("\nSearching for optimal parameters...")
    search_result = hybrid_search(snapshots, fees, rebates)
    
    if not search_result['params']:
        print("Hybrid search failed to find valid parameters. Please check your data or adjust parameter ranges.")
        return
    
    best_params = search_result['params']
    
    # Check if we have single venue data
    if len(unique_venues) == 1:
        print("\nDetected single venue data - using single-venue optimization")
        
        # Run optimized strategy with single-venue logic
        opt_results = run_optimized_single_venue(
            snapshots, 
            best_params['lambda_over'],
            best_params['lambda_under'],
            best_params['theta_queue'],
            fees,
            rebates
        )
    else:
        print(f"\nMulti-venue data detected ({len(unique_venues)} venues) - using routing optimization")
        
        # Use regular backtest for multi-venue data
        opt_results = run_backtest(
            snapshots, 
            best_params['lambda_over'],
            best_params['lambda_under'],
            best_params['theta_queue'],
            fees,
            rebates
        )
    
    print("\nRunning baseline strategies...")
    best_ask_results = run_best_ask_baseline(snapshots, fees)
    twap_results = run_twap_baseline(snapshots, fees)
    vwap_results = run_vwap_baseline(snapshots, fees)
    
    best_ask_savings = calculate_savings_bps(opt_results['average_price'], best_ask_results['average_price'])
    twap_savings = calculate_savings_bps(opt_results['average_price'], twap_results['average_price'])
    vwap_savings = calculate_savings_bps(opt_results['average_price'], vwap_results['average_price'])
    
    output = {
        "best_parameters": {
            "lambda_over": best_params.get('lambda_over', 0),
            "lambda_under": best_params.get('lambda_under', 0),
            "theta_queue": best_params.get('theta_queue', 0)
        },
        "optimized_strategy": {
            "total_cash_spent": opt_results['total_cash_spent'],
            "average_price": opt_results['average_price'],
            "is_single_venue": len(unique_venues) == 1
        },
        "best_ask_baseline": {
            "total_cash_spent": best_ask_results['total_cash_spent'],
            "average_price": best_ask_results['average_price']
        },
        "twap_baseline": {
            "total_cash_spent": twap_results['total_cash_spent'],
            "average_price": twap_results['average_price']
        },
        "vwap_baseline": {
            "total_cash_spent": vwap_results['total_cash_spent'],
            "average_price": vwap_results['average_price']
        },
        "savings_bps": {
            "vs_best_ask": best_ask_savings,
            "vs_twap": twap_savings,
            "vs_vwap": vwap_savings
        }
    }
    
    print(json.dumps(output, indent=2))
    
    plot_cumulative_costs(opt_results, best_ask_results, twap_results, vwap_results, best_params)
    print("Plot saved as results.png")

if __name__ == "__main__":
    main()