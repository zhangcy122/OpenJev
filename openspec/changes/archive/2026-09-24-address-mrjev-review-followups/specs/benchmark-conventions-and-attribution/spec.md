## ADDED Requirements

### Requirement: P95 Latency Formula Specification
The documentation and benchmark reports SHALL explicitly define the $P95$ percentile formula as $P95 = \text{sorted}(L)[\lfloor 0.95 \cdot N \rfloor]$ (index 34 for $N=36$) and document nearest-rank values for cross-tool comparability.

#### Scenario: Reader compares published P95 against alternative percentile methods
- **WHEN** an external reviewer or engineer compares published P95 numbers against nearest-rank or linear interpolation
- **THEN** the README benchmark table footnote explains that rank index 34 is used, and provides nearest-rank reference figures (OpenJevPro 1655.9ms, Direct LLM 1120.3ms, TypeSafe Jev 805.5ms)

### Requirement: External Baseline Attribution
The comparison table and overview SHALL explicitly attribute `openjev-sglang` as an external, independent reference architecture developed by Eric Zhang (`@ekzhang`).

#### Scenario: Reader inspects the first comparison table in README
- **WHEN** the reader views the zero-shot baseline table
- **THEN** the entry is titled "Independent Reference: openjev-sglang (@ekzhang)" and clearly distinguished from this repository's codebase
