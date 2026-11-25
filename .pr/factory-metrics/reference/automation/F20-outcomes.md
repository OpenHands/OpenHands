# Automation outcome values

Outcome facts belong to a factory and persist across automation replacements.
The factory registers the type of each accepted key. User credentials and run
tokens submit flat payloads or explicit typed facts; the credential determines
whether the source is a user or an agent. Analytics reads the factory's history.

Source: `openhands/automation/outcome_router.py`, `openhands/automation/outcome_schemas.py`, `openhands/automation/outcome_values.py`, `openhands/automation/router.py`, `openhands/automation/schemas.py`

Routes: `POST /api/automation/v1/outcomes`,
`GET /api/automation/v1/factories/{factory_id}/outcomes/values`,
`GET /api/automation/v1/factories/{factory_id}/outcomes/aggregate`

## Sub-features

- `F20.submit-flat`: strict flat scalar payloads become typed facts; nested, null, non-finite, unregistered or oversized values answer 422.
- `F20.submit-typed`: `values` carries `{key, type, value}` facts, including datetime and JSON, without requiring parsers.
- `F20.run-scope`: run tokens derive `source=agent` and the run claim; user credentials require `automation_id` and record `source=user`. An automation without a factory answers 409.
- `F20.list-values`: factory-scoped newest-first rows, filtered by key, automation, run, source or time. Other organizations cannot read them.
- `F20.aggregate-values`: per-factory count, numeric and boolean aggregates, optionally grouped by day, week or month; the same filters apply.

## How to get to it (agent POV)

- Create a factory with `outcome_keys` and attach an automation before submitting.
- POST a flat `{"payload":{"score":1}}` or typed `{"values":[{"key":"score","type":"number","value":1}]}`. Use a run token or normal user auth with `automation_id`.
- Read the authenticated factory routes to fetch raw or aggregate facts across automations. The stored factory id is a snapshot, even if the automation later moves.

## Driving it with control-automation

Preconditions:

- A live run (`launch --new`) and a clean `doctor`.
- Use the baseline local session key for user-authenticated examples. Token-auth coverage is handled by service tests because the token is minted per dispatched run.

- **Submit flat and typed (`F20.submit-flat`, `F20.submit-typed`, `F20.run-scope`).** Register keys before creating the automation.
  ```sh
  FID=$(control-automation api POST /v1/factories --json '{"name":"QA Outcomes","outcome_keys":{"score":{"type":"number"},"passed":{"type":"boolean"},"metadata":{"type":"json"}}}' --expect 201 --field id)
  OID=$(control-automation api POST /v1 --json "{\"name\":\"qa-outcomes\",\"factory_id\":\"$FID\",\"trigger\":{\"type\":\"cron\",\"schedule\":\"0 0 1 1 *\"},\"tarball_path\":\"https://qa.invalid/x.tgz\",\"entrypoint\":\"bash run.sh\"}" --expect 201 --field id)
  control-automation api POST "/v1/outcomes?automation_id=$OID" --json '{"payload":{"score":2,"passed":true}}' --expect 201 --check stored eq 2 --save F20.submit-flat/submit
  control-automation api POST "/v1/outcomes?automation_id=$OID" --json '{"values":[{"key":"metadata","type":"json","value":{"review":"complete"}}]}' --expect 201 --check stored eq 1 --save F20.submit-typed/submit
  control-automation api POST "/v1/outcomes?automation_id=$OID" --json '{"payload":{"score":{"nested":2}}}' --expect 422
  control-automation api POST "/v1/outcomes?automation_id=$OID" --json '{"payload":{"unknown":1}}' --expect 422
  NID=$(control-automation fixture automation --kind echo --field id)
  control-automation api POST "/v1/outcomes?automation_id=$NID" --json '{"payload":{"score":1}}' --expect 409 --save F20.run-scope/no-factory
  ```
- **List and aggregate (`F20.list-values`, `F20.aggregate-values`).** Scope reads by factory, not automation.
  Requires: `F20.submit-flat`
  ```sh
  control-automation api GET "/v1/factories/$FID/outcomes/values" --query key=score --query source=user --expect 200 --check total eq 1 --check values.0.key eq score --save F20.list-values/list
  control-automation api GET "/v1/factories/$FID/outcomes/aggregate" --query key=score --query operation=sum --expect 200 --check value eq 2 --save F20.aggregate-values/sum
  control-automation api GET "/v1/factories/$FID/outcomes/aggregate" --query key=passed --query operation=count_true --expect 200 --check value eq 1
  ```

## Gotchas

- Outcome tokens are intentionally the same scoped token family as KV tokens, so a run gets one secret with an automation and run claim rather than a second auth system.
- Typed items use `type`, not the storage column `value_type`. Source is derived from authentication, never the request body. Use typed values for datetime or JSON; parser-based nested payloads are a separate layer.
