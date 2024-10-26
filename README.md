# Dynamic Security Hash

A dynamic hashing implementation featuring multiple security levels, caching, and parallel processing capabilities.

## Features

1. Multiple Security Levels: From fastest to ultra_secure.
2. Memory-Hard Hashing Functions: Enhances resistance against hardware attacks.
3. Parallel Batch Processing: Efficiently handles multiple hashing tasks simultaneously.
4. Automatic Performance Tuning: Optimizes performance based on the system's capabilities.
5. Built-in Caching System: Improves speed by storing previously computed hashes.
6. Comprehensive Benchmarking: Allows for performance evaluation under different scenarios.

## Installation

You can install the package via pip:

```bash
pip install git+https://github.com/vincentiusyoshua/dynamic_security_hash.git
```

Alternatively, install from a requirements.txt file:

```bash
pip install -r requirements.txt
```

## Quick Start

Here’s a quick example to get you started:

```python
from dynamic_security_hash import UltraOptimizedHash

# Initialize the hasher
hasher = UltraOptimizedHash(
    security_level='balanced',
    cache_size=50000,
    optimize_memory=True,
    auto_tune=True
)

# Single hash
result = hasher.hash("Hello, World!")
print(result)

# Batch processing
data_list = ["Data 1", "Data 2", "Data 3"]
results = hasher.parallel_batch_hash(data_list)
print(results)
```

## Security Levels

The package offers five security levels:

1. fastest: Minimal security, maximum speed.
2. fast: Good balance for non-critical applications.
3. balanced: Default level, offers a good security/performance trade-off.
4. secure: High security level for sensitive data.
5. ultra_secure: Maximum security settings, suitable for the most critical applications.


# Benchmarking

To run comprehensive benchmarks:

```python
# Run comprehensive benchmarks
bench_results = hasher.benchmark("test_data")
print(bench_results)
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
