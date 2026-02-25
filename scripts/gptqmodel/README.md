# GPTQModel Benchmark Results

## GSM8K Performance

| Bits | Exact Match | Throughput (tok/s) | Status |
|------|-------------|-------------------|--------|
| 2bit | 0.0000 | TBU | BROKEN |
| 3bit | TBU | TBU | - |
| 4bit | 0.6562 | 11940.46 |  |

## Configuration Summary

| Bits | MSE | Group Size | SmoothMSE | EoRA Rank | Samples |
|------|-----|------------|-----------|-----------|---------|
| 2bit | 1.5 | 32 | steps=96, maxshrink=0.65 | 128 | 1024 |
| 3bit | 2.0 | 128 | steps=64, maxshrink=0.70 | 96 | 512 |
| 4bit | 2.4 | 128 | steps=64, maxshrink=0.75 | 64 | 512 |
