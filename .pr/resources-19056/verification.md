# Conversation resources verification

Real app: local Agent Canvas production build from this checkout, Agent Server 1.59.0, automation 1.26.1, Chrome, macOS. The isolated run was launched with the repository's `control-openhands launch --new` workflow. Its API arranged a saved LLM profile (`openai/gpt-4o-mini`, dummy key), then created two conversations through the real Agent Server. No model call was made; this feature reads configuration and does not depend on model execution.

Browser actions:
1. Dismiss onboarding using `control-openhands onboard --skip`.
2. Open a conversation, click **Show panel**, select **Resources**, and expand the profile, model and condenser rows.
3. Observe the launch revision and the reported model, then the stored condenser's enabled state and settings. `desktop.png` captures a conversation whose profile revision matches.
4. The first conversation was created before the normal first-visit profile migration updated the default profile. `changed-profile.png` captures its launch revision 0 and the changed-profile warning; profile-derived values are unavailable rather than attributed from revision 1.
5. Switch to the phone viewport and navigate to `/conversations/<id>/panel`. `mobile.png` captures the same Resources panel and navigation in the mobile route.
6. Reload: Resources remains selected; the desktop drawer follows the existing session-only closed-on-load behavior.
7. After the stack restart and reload, `browser errors --app-only` reported 0 page errors and 0 app errors.

Cloud and ACP gating is covered by component tests with real consumer hooks and mocked underlying services. No live Cloud or ACP account was used.

The component renders explicit scalar fields rather than serializing profile objects. Tests include a secret-bearing condenser object and verify its key is never rendered, and verify that launch-scoped names do not trigger secret-value fetches.
