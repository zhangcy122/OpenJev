## 1. Documentation & Positioning Narrative Alignment

- [ ] 1.1 Refactor `README.md` to lead with the single-anchor hook: "The High-Throughput Deterministic Decision Engine for AI Agents"
- [ ] 1.2 Add the 60-Second Quickstart with a 5-line `Choice<T>` code block immediately above the fold in `README.md`
- [ ] 1.3 Restructure `README.md` architecture section into Primitives, Twin Flagships (Order Invariance & Flywheel), and Two Deployment Strategies
- [ ] 1.4 Synchronize `README_zh.md` with the identical Chinese narrative, headings, and code snippets
- [ ] 1.5 Update `pyproject.toml` package description to reflect the unified deterministic decision engine identity

## 2. Website Information Architecture & Hero Restructure

- [ ] 2.1 Refactor Hero copy in `site/index.html` with the focused 5-second hook and sharp value proposition
- [ ] 2.2 Embed a syntax-highlighted 5-line quickstart code preview widget directly inside the Hero container
- [ ] 2.3 Reorder sections in `site/index.html` to place `#playground` directly after Hero for instant tactile verification
- [ ] 2.4 Update the Bento Grid section in `site/index.html` to clearly frame Mode A and Mode B as deployment strategies and highlight the Deliberative Flywheel as the cognitive evolutionary tier
- [ ] 2.5 Update bilingual i18n dictionaries (`zh` and `en`) in `site/index.html` for all revised titles, subtitles, and badges

## 3. Responsive Verification & Visual Polish

- [ ] 3.1 Verify desktop, tablet, and mobile responsiveness via Playwright CDP headless screenshots to ensure zero overflow and clean typography
- [ ] 3.2 Verify language switching (`ZH` ↔ `EN`) maintains perfect layout integrity across all viewports
- [ ] 3.3 Run pytest regression suite (`pytest`) to ensure all existing test suites pass cleanly with zero failures
