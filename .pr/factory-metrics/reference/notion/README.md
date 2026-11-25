# Notion pages (internal)

The design drew on four internal Notion pages. They are drafts, marked "for discussion", and are not copied here.
Ask the owner for access if you cannot open them.

| Page | Link |
| --- | --- |
| Software Factory Metrics (hub) | https://www.notion.so/41a7be798a17802ba0fbcc14f9c7503a |
| Software Factory metrics: examples | https://www.notion.so/41a7be798a1780beb271f1d9e9866335 |
| Metrics API (proposal) | https://www.notion.so/41a7be798a1780799277e4db3ca722e3 |
| Metrics Learnings from other Products | https://www.notion.so/41a7be798a17803f85f0f8a4def63c20 |

## Digest (what the sheet takes from them)

- **Consumers.** Leaders want value, cost and effectiveness. Operators want throughput, errors, LLM cost and load, and
  A/B tests of automation versions. Other roles need workflow views, not metrics.
- **Three kinds of data, kept apart.** OTEL events (granular, ClickHouse and Laminar), worklog events (narrative), and
  metrics (numeric KPIs and cards).
- **Model.** A *metric* is a definition. A *point* is one timestamped value, append-only. A *card* is a saved query over a
  metric plus display settings. Platform-derived metrics use the reserved `platform.` key prefix.
- **Card fields.** `title metric window compare display group_by filter link order`. Display types in the proposal:
  `number number_trend sparkline percent progress`. The sheet adds a breakdown, a variant comparison, a duration split and a
  series chart.
- **First increment.** A count card with an optional link out, with points stored append-only from the start.
- **First release.** About 8 leader metrics and 10 operator metrics. Sheet 10 maps each to a card type.
- **Decisions the pages leave open.** Factory identifier in API paths (slug or id). Which metrics are derived and which
  are pushed. What a Factory's KPI is (goal text, names, or a metric reference with target). How variants are identified
  and assigned for A/B tests.
- **Dependency.** Several leader metrics need PR state and timestamps. The Issues and PRs milestone stores only id, title
  and URL today.
