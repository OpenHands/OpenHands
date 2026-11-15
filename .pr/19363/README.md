# Mobile automation draft preservation

Baseline: `57cd74aaafc5254fa7760db4b444bdd9468f5b7f`.
Product fix: `cbbfefbb78d7c37de1a9634cadee17559bd9aa31`.

The real production Canvas build ran with Agent Server 1.59.0 and automation 1.26.1 in isolated Ubuntu WSL stacks. Chrome ran on Windows at 390 × 844. Both builds used Node 24.21.0 and npm 11.12.1. Browser requests and server responses were not intercepted.

Create a setup conversation through Automate > New Automation > Event > Ask Agent, then return to Form. Enter a name, prompt and event key. Switch to Conversation and back. Repeat with different edits. `observations.json` contains the actual browser-read values for both attempts:

- Main resets the name to New Automation, clears the prompt, selects Schedule and removes the event-key field.
- The fix retains all four values on both round trips.
- Leaving the conversation and reopening it still discards the unsaved form.

`base-fields-proof.webm` and `fixed-fields-proof.webm` record these transitions. The PNGs show the actual UI, including the discarded and retained forms. A validation-only LLM profile pointed at unreachable loopback port 9 to permit conversation creation. Model completion is unnecessary for this form-lifetime check; no paid inference was used.

The component regression fails before the fix. The existing conversation-layout and real setup-panel suites pass 94 tests after it. Application and library builds pass. Full Windows testing reported 10,020 passed and 61 failed: 60 failing names reproduce in the current-main control; the remaining library-import 30-second timeout also reproduces in a separate current-main control. Full lint fails on 14 formatting errors in four unchanged main files. The upstream main CI has the same formatting failures. No unrelated test, timeout or verification skill was changed.
