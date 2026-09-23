## ADDED Requirements

### Requirement: Empirical Benchmark Artifact Alignment
The public reproduction tables in `README.md` and `site/index.html` SHALL reflect exact numbers from `examples/harness_benchmark_results.json`, including P50 and mean latency, in-domain accuracy ratios (30/30 vs 29/30), and empirical ECE values.

#### Scenario: Documentation matches committed raw artifact
- **WHEN** a reader reviews the benchmark tables in `README.md` and `site/index.html`
- **THEN** reported latencies and accuracies match `examples/harness_benchmark_results.json` without rounding inversion or transposition errors

### Requirement: Open Source Attribution
The project documentation SHALL explicitly cite and link upstream and third-party benchmark references, specifically identifying `ekzhang/openjev-sglang` as the author of the JevBench v1 replication.

#### Scenario: Reader views JevBench v1 replication credit
- **WHEN** reading the empirical replication section in `README.md` or `site/index.html`
- **THEN** explicit credit and link to `https://github.com/ekzhang/openjev-sglang` are displayed
