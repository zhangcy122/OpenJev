## ADDED Requirements

### Requirement: Hero Quickstart Code Preview Card
The landing page hero section SHALL display an interactive, styled Python snippet demonstrating `from openjevpro import Choice` directly adjacent to or beneath the primary CTA buttons.

#### Scenario: Visitor lands on openjev.pro
- **WHEN** the homepage loads on any desktop or mobile device
- **THEN** the visitor sees an authentic syntax-highlighted 5-line code snippet showing immediate usage without scrolling past the fold

### Requirement: REPL Playground Elevation
The interactive REPL decision simulator section (`#playground`) SHALL be positioned immediately following the Hero section to provide hands-on tactile verification before detailed text descriptions.

#### Scenario: User navigates down the page
- **WHEN** user scrolls past the Hero section
- **THEN** the next major section is the interactive decision simulator allowing live testing of candidate permutation invariance

### Requirement: Synchronized Bilingual i18n Dictionary
All restructured hero headlines, subheadings, code labels, and narrative descriptions SHALL be completely translated in both `zh` and `en` dictionaries in `site/index.html`.

#### Scenario: User toggles language switcher
- **WHEN** user clicks the `ZH` or `EN` button in the navbar
- **THEN** all restructured positioning text updates seamlessly without untranslated keys or layout breakage
