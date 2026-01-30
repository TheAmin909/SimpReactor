SCRIBE'S JOURNAL - ARCHIVE:
Before starting, read .jules/scribe.md (create if missing).
⚠️ ONLY add journal entries when you discover:
- A frequently asked question by the team
- A part of the setup process that developers struggle with
- Discrepancies between docs and code
Format: `## YYYY-MM-DD - [Topic]
**Confusion:** [What was unclear]
**Clarification:** [How it is now documented]`

## 2026-01-30 - Reactor Stoichiometry
**Confusion:** The `build_cstr_rhs` function hardcodes reaction stoichiometry (index 0 is reactant, index 1 is product) but doesn't document this constraint, potentially leading to incorrect results for other reaction types.
**Clarification:** Documented the assumption in `build_cstr_rhs` docstring and clarified that it supports A -> B type reactions.
