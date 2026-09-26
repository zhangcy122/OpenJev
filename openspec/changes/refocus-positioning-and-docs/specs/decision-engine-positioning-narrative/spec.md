## ADDED Requirements

### Requirement: Unified Agent Decision Engine Positioning
All user-facing documentation and site headers SHALL describe OpenJevPro as a high-throughput, deterministic, calibrated decision engine designed specifically for AI Agents, converting slow generative LLM calls into sub-35ms typed primitives.

#### Scenario: Visitor reads the primary tagline
- **WHEN** a visitor reads the hero title or README header
- **THEN** the text clearly identifies OpenJevPro as the deterministic decision engine for AI agents rather than a collection of disparate modes

### Requirement: Reframe Modes into Unified Hierarchy
The documentation and landing page SHALL present OpenJev's capabilities through a structured hierarchy consisting of Core Primitives, Twin Flagship Capabilities (100% Order Invariance & Deliberative Flywheel), and Two Flexible Deployment Strategies (Pure Open-Source vs. Enterprise Cloud Gateway).

#### Scenario: Developer reviews architectural deployment options
- **WHEN** a developer inspects deployment modes
- **THEN** Mode A and Mode B are presented as complementary deployment strategies (standalone local vs. gateway co-existence) rather than mutually exclusive products

### Requirement: 60-Second Quickstart Prominence
The documentation SHALL provide a clean, copy-pastable 5-line Python snippet illustrating `Choice<T>` routing with `order_invariant=True` before introducing complex architectural or mathematical concepts.

#### Scenario: First-time user opens README
- **WHEN** a user visits `README.md` or `README_zh.md`
- **THEN** they can read and execute the 5-line quickstart code block within 60 seconds
