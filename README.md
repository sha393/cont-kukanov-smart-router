# Cont-Kukanov-Smart-Router
# Smart Order Router with Cont-Kukanov Allocation

This repository contains a back-testing implementation of a smart order router based on the static cost model by Cont & Kukanov ("Optimal Order Placement in Limit Order Markets"). The system optimizes order execution across multiple venues by balancing execution risk, market impact, and liquidity costs.

## Overview

The system implements the Cont-Kukanov allocator to split a 5,000-share buy order across venues, optimizing for:
- Execution certainty vs. cost trade-off
- Risk penalties for over/underfilling targets
- Queue position and market impact minimization
- Venue-specific fee structures

### Key Features

- **Dynamic Allocation**: Uses the Cont-Kukanov model to dynamically allocate orders based on venue conditions
- **Multi-Strategy Comparison**: Benchmarks against best-ask, TWAP, and VWAP baseline strategies
- **Single/Multi-Venue Support**: Automatically adapts to single or multiple venue scenarios
- **Parameter Optimization**: Hybrid search combining random and grid search for optimal parameters

## Technical Implementation

### Core Components

1. **Allocator Engine**: Implements the exact pseudocode from Cont-Kukanov paper
   - Iteratively builds splits across venues
   - Computes expected costs considering fees, rebates, and queue position
   - Minimizes total cost function including risk penalties

2. **Optimization Module**: Finds optimal risk parameters
   - Random search over parameter space (λ_over, λ_under, θ_queue)
   - Focused grid search around promising regions
   - Single-venue optimization using Proposition 3 when applicable

3. **Baseline Strategies**:
   - **Best Ask**: Takes liquidity at the best available price
   - **TWAP**: Splits order equally across time buckets
   - **VWAP**: Allocates proportionally to displayed size

### Parameter Space

- **λ_over**: Cost penalty for exceeding target (0.01 - 1.0)
- **λ_under**: Cost penalty for underfilling (0.01 - 1.0)  
- **θ_queue**: Market impact coefficient (0.001 - 0.1)

## Architecture

### Data Processing Pipeline

1. **Level-1 Data Ingestion**:
   - Reads timestamped MBP (Market By Price) data
   - Extracts ask prices and sizes per venue
   - Maintains order book state snapshots

2. **Execution Simulation**:
   - Models queue position and fill dynamics
   - Accounts for venue-specific fees and rebates
   - Tracks cumulative costs and fills

3. **Performance Analysis**:
   - Calculates average execution price
   - Measures savings in basis points vs baselines
   - Generates cumulative cost visualizations

### Core Algorithms

```python
# Allocator (as per Cont-Kukanov pseudocode)
function allocate(order_size, venues, λ_over, λ_under, θ_queue):
    splits = [[]]  # Start with empty allocation
    
    # Iteratively build splits for each venue
    for v in venues:
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size - used, venue[v].ask_size)
            
            # Step through possible allocations
            for q in 0 to max_v step 100:
                new_splits.append(alloc + [q])
        
        splits = new_splits
    
    # Find minimum cost allocation
    best_cost = infinity
    best_split = []
    
    for alloc in splits:
        if sum(alloc) == order_size:
            cost = compute_cost(alloc, venues, ...)
            if cost < best_cost:
                best_cost = cost
                best_split = alloc
    
    return best_split, best_cost
```

## Results Analysis

Based on the test run provided:

### Performance Metrics

| Strategy     | Average Price | Savings vs Optimized (bps) |
|--------------|---------------|---------------------------|
| Optimized    | 222.8205      | -                         |
| Best Ask     | 222.8205      | 0.002                     |
| TWAP         | 223.0904      | 12.101                    |
| VWAP         | 222.8592      | 1.740                     |

### Key Observations

1. **Single Venue Detection**: The system correctly identified single-venue data and adapted its strategy accordingly

2. **Parameter Optimization**: The hybrid search found optimal parameters:
   - λ_over: 0.622
   - λ_under: 0.045
   - θ_queue: 0.035

3. **Execution Results**: 
   - Successfully executed all 5,000 shares
   - Achieved significant savings vs TWAP (12.1 bps)
   - Competitive with best-ask strategy

4. **Performance Characteristics**:
   - Optimized strategy shows immediate execution in the cumulative cost plot
   - TWAP shows gradual accumulation over 9-minute window
   - VWAP demonstrates volume-proportional execution pattern

## Implementation Notes

### Fee Structure Modeling

The system models venue economics with:
- Trading fees (0.5-1.5 basis points)
- Rebates for providing liquidity (0.5-1.0 basis points)
- Queue position effects on fill probability

### Queue Position Effects

In single-venue scenarios, the optimizer considers:
- Queue position at bid/ask levels
- Fill probability based on order flow
- Trade-off between immediate execution and better price

### Risk Model

The cost function incorporates:
- Cash cost: execution price × size + fees - rebates
- Risk penalty: θ × (underfill + overfill)
- Execution shortfall: λ_under × underfill + λ_over × overfill

## Usage

```python
# Basic usage
snapshots = process_level1_data('l1_day.csv')
search_result = hybrid_search(snapshots, fees, rebates)
best_params = search_result['params']

# Run optimized strategy
opt_results = run_backtest(
    snapshots, 
    best_params['lambda_over'],
    best_params['lambda_under'],
    best_params['theta_queue'],
    fees,
    rebates
)
```

## Suggested Improvements

1. **Queue Simulation**: Implement stochastic order flow model for more realistic queue dynamics
2. **Latency Modeling**: Add network latency effects between venues
3. **Adverse Selection**: Model price impact based on fill timing
4. **Market State Detection**: Adapt parameters based on market regime (volatile vs. stable)
5. **Cancel/Replace Logic**: Add order cancellation and replacement strategies

## Performance Optimization

The implementation is designed for:
- Sub-second execution time for parameter search
- Minimal memory footprint for order book state
- Scalable to multiple venues with efficient allocation algorithms

## Dependencies

- pandas: Data processing and manipulation
- numpy: Numerical computations
- matplotlib: Visualization
- Standard library: itertools, typing, random

## License

This implementation is for educational and research purposes. The Cont-Kukanov model and paper are subject to their respective licenses.
