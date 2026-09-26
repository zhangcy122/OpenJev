## Why

OpenJevPro currently suffers from severe positioning diffusion and cognitive friction on both its documentation (`README.md`, `README_zh.md`) and official site (`site/index.html`). By simultaneously presenting Mode A (open-source local alternative), Mode B (commercial Jev enterprise harness), and Mode C (self-evolving cognitive flywheel) as equal peers alongside heavy technical jargon (ModernBERT 322M, vLLM grammar masking, RLCD calibration, ECE, Banking77), visitors experience cognitive overload within the critical first 10 seconds, unsure if OpenJev is a replacement, a proxy gateway, or an academic research project. 

This change establishes a single unified identity: **"The High-Throughput Deterministic Decision Engine for AI Agents"**, organized around a hierarchical "Core Primitives + Twin Differentiators (100% Invariance & Deliberative Flywheel) + Deployment Spectrum" architecture with an immediate 5-line code quickstart above the fold.

## What Changes

- **Unified Product Positioning**: Retire the fragmented "Tri-Head Mode A/B/C" hero framing in favor of a single clear proposition: OpenJev turns slow, expensive, prompt-biased LLM calls into sub-35ms, type-safe, 100% order-invariant, self-crystallizing decision primitives for AI agents.
- **Hero & Above-The-Fold Restructure in `site/index.html`**:
  - Replace the 15-word jargon title with a crisp 5-second hook.
  - Insert an immediate, copy-pastable 5-line interactive code block right below/beside the hero CTA.
  - Elevate the Interactive REPL Playground directly following the hero section so visitors experience order invariance and sub-35ms speed before reading deep architecture text.
- **Hierarchy Refactor for Delivery Modes**:
  - Reframe Mode A and Mode B from competing product modes into two pragmatic deployment strategies: **Pure Open-Source / Local** (vLLM / Laya 322M) vs. **Enterprise Gateway / Cost Arbitrage** (saving 60%+ commercial API costs).
  - Center Mode C (Deliberative Decision Flywheel) as the flagship cognitive evolution capability ("Explore First, Crystallize Later").
- **Documentation Restructure (`README.md` & `README_zh.md`)**:
  - Re-sequence sections into: Problem Hook $\to$ 60-Second Quickstart $\to$ Core Primitives $\to$ Twin Flagship Capabilities (Order Invariance & Flywheel) $\to$ Deployment Strategies $\to$ Empirical Proof.

## Capabilities

### New Capabilities
- `decision-engine-positioning-narrative`: Defines the core single-anchor narrative, terminology taxonomy, and value proposition contracts for OpenJevPro across all developer touchpoints.
- `site-information-architecture-restructure`: Defines the visual information hierarchy, hero code preview card, and section flow on `site/index.html` to achieve $<8$s Time-to-Aha.

### Modified Capabilities
<!-- None: Requirement contracts of existing runtime algorithms and harnesses remain intact. -->

## Impact

- `site/index.html`: Hero copy, subheaders, code preview widget, section reordering, and i18n dictionary updates (`zh` and `en`).
- `README.md` & `README_zh.md`: Top-level positioning statement, architecture diagram, quickstart section, and table of contents.
- `pyproject.toml` / package description: Synchronized tagline alignment.
- Zero breaking changes to public APIs (`openjevpro.client`, `openjevpro.flywheel`, `openjevpro.gateway`, `openjevpro.guard`).
