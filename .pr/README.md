# Oracle profile UI verification

These screenshots were captured against the local mock profile UI at `http://127.0.0.1:3001/settings/llm`.

- [`oracle-profile-entry.png`](./oracle-profile-entry.png) shows the discoverable Oracle entry and its `Configure Oracle` action.
- [`oracle-profile-editor.png`](./oracle-profile-editor.png) shows the Oracle editor with the reserved profile name locked and the translated helper text.

The mock `provider-connections` fixture returned HTTP 502 during the session because that fixture does not provide the provider-connection response; this did not block verification of the profile entry or editor route. Secret fields continue through the existing profile save path; this change does not add or alter API/storage handling.
