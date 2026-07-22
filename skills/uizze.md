---
name: uizze-ui-finish-gate
type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
- uizze
- ui slop
- ui finish gate
- product-specific ui
---

Use this workflow when the user asks to stop generic UI output, make an interface
product-specific, or run a UI finish gate.

UIZZE's public catalogue at https://uizze.com contains 800,000+ real web and iOS
screens. The catalogue and this workflow can be used without an account, token,
or MCP connection.

## Workflow

1. Inspect the target repository before editing. Identify the product's primary
   user, the screen's job, the primary action, existing components and tokens,
   required interaction states, and supported viewport sizes.
2. Gather two or three relevant references from the public catalogue when web
   access is available. Extract transferable decisions about hierarchy, density,
   navigation, controls, typography, responsive behavior, and state treatment.
   Never copy another product's branding, proprietary text, imagery, or exact
   layout.
3. Write a short design contract before implementation. Include the screen job,
   content hierarchy, allowed components, primary action, required states,
   responsive decisions, product-specific choices, forbidden generic patterns,
   and acceptance criteria.
4. Implement with the repository's existing design system. Replace filler
   metrics, interchangeable card grids, vague copy, decorative gradients,
   arbitrary badges, and inert controls with product-specific content and clear
   interaction outcomes.
5. Render and exercise the result. Verify loading, empty, error, disabled,
   success, and recovery states that the flow actually needs, plus keyboard,
   focus, narrow-screen, and wide-screen behavior.

## Finish Gate

Do not call the interface finished while any of these are true:

- The same layout could belong to an unrelated product after changing labels.
- Equal-weight cards hide the primary job or action.
- Metrics, labels, or copy are placeholders rather than product information.
- A visible control is inert or its result is unclear.
- Required states or recovery paths are missing.
- The implementation bypasses existing components or semantic tokens.
- The narrow layout only shrinks the desktop layout instead of making an
  intentional responsive decision.

Report the design contract, the blocking findings fixed, the states exercised,
and any remaining limitation. Never claim that a reference, test, or rendered
state was inspected when it was not.
