## Context

OpenJevPro began as an open-source research and engineering effort to reproduce and extend TypeSafe Jev functionality using open LLMs, ModernBERT 322M (Laya), temperature scaling, and guard harnesses. Over rapid iterations, it introduced:
1. Decode-time grammar-masked likelihood scoring (sub-35ms).
2. Mathematical order invariance (`order_invariant=True`).
3. Dual-cutoff guardrails for 100% selective accuracy.
4. Two-tier hybrid routing & fallback circuit breaking.
5. The Deliberative Decision Flywheel ("Explore First, Crystallize Later").

However, presenting all these developments in parallel created a "tri-head" narrative on the landing page and README: visitors are hit with Mode A (local open alternative), Mode B (Jev enterprise harness), and Mode C (cognitive flywheel), accompanied by heavy mathematical and infrastructural jargon (ModernBERT 322M, vLLM, SGLang, ECE calibration, RLCD, Banking77, SLA states).

The result is cognitive overload: visitors cannot decipher within 10 seconds whether OpenJev is a library, a proxy service, a fine-tuning dataset, or an academic paper.

## Goals / Non-Goals

**Goals:**
- **Single Core Identity**: Position OpenJevPro as "The High-Throughput Deterministic Decision Engine for AI Agents".
- **Time-to-Aha $\le 8$ seconds**: Visitors understand the exact problem solved and see a clean 5-line Python snippet on the first screen.
- **Hierarchical Reframe**: Reframe Mode A and Mode B into concrete *Deployment Strategies* (Pure Open-Source vs. Enterprise Cloud Gateway), while elevating Order Invariance and the Deliberative Decision Flywheel as the twin killer capabilities.
- **Maintain Full Technical Depth**: Retain all empirical benchmarks, API references, mathematical proofs, and architectural details in organized lower sections.
- **Synchronized Bilateral Copy**: Align `README.md`, `README_zh.md`, and `site/index.html` (both `zh` and `en` i18n dictionaries).

**Non-Goals:**
- Refactoring internal algorithm implementations or breaking public Python APIs.
- Removing or deprecating the Banking77 benchmark data or commercial licensing options.

## Decisions

### Decision 1: "Primitives + Twin Flagships + Deployment Strategies" Taxonomy
- **Rationale**: Rather than treating Mode A/B/C as three competing product identities, we unify them under the Agent Decision Engine umbrella:
  - **Primitives**: `Choice<T>`, `Noul`, `Score` (<35ms, typed, non-autoregressive).
  - **Twin Flagships**:
    1. *Mathematical Permutation Invariance*: 100% order-invariant scoring via isolated forward passes.
    2. *Deliberative Decision Flywheel*: System 2 (Thinking LLMs) offline distillation into System 1 rules.
  - **Deployment Strategies**:
    - Strategy 1: Pure Local / Open-Source (vLLM / Laya 322M).
    - Strategy 2: Enterprise Gateway / Cost Arbitrage (60%+ savings & Circuit Breaker SLA).

### Decision 2: Elevating Interactive REPL Playground on the Landing Page
- **Rationale**: An interactive widget showing probability distributions and toggleable option ordering proves the core value in seconds, far more persuasively than paragraphs of marketing text. Moving `#playground` right after the Hero section establishes instant credibility.

### Decision 3: 5-Line Quickstart Code Card on Hero
- **Rationale**: Developers buy code, not slides. Embedding a syntax-highlighted code block immediately below the Hero CTA demonstrates that OpenJev is a drop-in Python library with zero boilerplate.

## Risks / Trade-offs

- **[Risk] Existing users looking specifically for "Jev Harness" or "Laya 322M" might think those features were removed.**
  → **Mitigation**: Dedicated badges and clear subsections in the Deployment section explicitly highlighting Laya ModernBERT-large 322M and TypeSafe Jev API compatibility.
- **[Risk] Squeezing too much content into the Hero code card could cause responsive wrapping issues on mobile.**
  → **Mitigation**: Keep the snippet concise (6 lines max), with mobile-optimized horizontal overflow (`overflow-x-auto text-[11px] sm:text-xs`).

## Migration Plan

1. Create formal capabilities specifications in OpenSpec (`specs/`).
2. Update `site/index.html`:
   - Hero headline, subtitle, and instant code snippet.
   - Reorder `#playground` to directly follow `#hero`.
   - Update `window.I18N` dictionaries for Chinese and English.
3. Update `README.md` and `README_zh.md` with unified structure.
4. Run regression suite (`pytest`), test responsive visual rendering via Playwright CDP, and verify zero regressions.
