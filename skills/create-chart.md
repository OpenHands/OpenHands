---
name: create-chart
description: Create a B1 chart (bar, line, pie, doughnut, radar, polarArea, scatter) connected to an existing data source (DSO). Uses CLOB-stored Chart.js config for appearance and JMESPath dataConversionExpression to transform DSO rows into Chart.js format.
---

# Create Chart with DSO Data

Guides creation of `b1_chart` objects connected to an existing data source (DSO). A CLOB-stored Chart.js config controls appearance (`extendedConfig`), and a JMESPath `dataConversionExpression` transforms DSO rows into Chart.js format.

**Reference implementation:** `InvoiceDueDateStatusChartScreen` in the Samples module.

---

## Trigger Phrases

Use this skill when the user says any of the following:
- "create a chart"
- "add a chart"
- "create a bar chart" / "create a pie chart" / "create a line chart" (any chart type)
- "add a chart to a screen"
- "create chart visualization"

---

## Phase 1: Gather Requirements

Collect the following from the user before creating any objects.

| Item | Example | Default |
|------|---------|---------|
| Chart type | `bar`, `line`, `pie`, `doughnut`, `radar`, `polarArea`, `scatter` | `bar` |
| Chart object name | `InvoiceSalesChart` | — (required) |
| Config CLOB name | `InvoiceSalesChartConfig` | `{chartName}Config` |
| Module | `Invoicing` | — (required) |
| Data source name | `invoiceDSO` | — (required) |
| Data conversion expression | JMESPath expression | — (required) |
| DSO filter expression | Inline JS for `eventBeforeFetch` | `""` (no filter) |
| Chart title | `"Monthly Sales"` | `""` (no title) |
| Show legend | `true` / `false` | `true` |
| Legend position | `top`, `bottom`, `left`, `right` | `top` (bar/line/radar/scatter), `bottom` (pie/doughnut/polarArea) |
| Create sample screen | `yes` / `no` | `no` |
| Screen name (if yes) | `InvoiceSalesChartScreen` | `{chartName}Screen` |

---

## Data Conversion Expression

The DSO returns raw tabular rows. A JMESPath `dataConversionExpression` on the chart object transforms these into Chart.js format `{ labels, datasets }`.

A custom `group_by(array, 'fieldName')` function is available that groups array elements by a field value and returns an array of arrays.

**Common patterns:**

Group by a field and count:
```
{labels: group_by(@, 'status')[*][0].status, datasets: [{label: 'Count', data: group_by(@, 'status')[*].length(@)}]}
```

Group by a field and sum a value:
```
{labels: group_by(@, 'category')[*][0].category, datasets: [{label: 'Total', data: group_by(@, 'category')[*][*].amount | [*].sum(@)}]}
```

Simple field mapping (no aggregation):
```
{labels: [*].month, datasets: [{label: 'Revenue', data: [*].revenue}]}
```

### DSO filtering and row limits

Filter rows via inline `eventBeforeFetch` on the DSO instance (plain JavaScript, no `#.` prefix):
```
eventSource.addFilter('status', 'ne', 'draft')
```

By default a DSO fetches limited rows (default 50). To fetch all records for a chart, disable the limit:
```
eventSource.setFilters({ top: -1 })
```

Combined example:
```
eventSource.addFilter('status', 'ne', 'draft'); eventSource.setFilters({ top: -1 })
```

---

## Phase 2: Create Config CLOB

Create a `b1_clob` object containing the Chart.js extendedConfig JSON.

1. Select the default config template for the chosen chart type (see [Default Config Templates](#default-config-templates))
2. Apply user preferences:
   - Set `plugins.title.display` to `true` and `plugins.title.text` to the user's title (if provided)
   - Set `plugins.legend.display` and `plugins.legend.position` per user input
3. Call `mcp__B1_Blueprint__create_blueprint`:

```
objectTypeName: "b1_clob"
objectName: {configClobName}
moduleName: {module}
objectDescription: "{chartName} extended config"
attributes:
  content: {stringified config JSON}
  contentType: "json"
```

---

## Phase 3: Create Chart Object

Create the `b1_chart` object with `extendedConfig` pointing to the config CLOB and `dataConversionExpression` for data transformation.

Call `mcp__B1_Blueprint__create_blueprint`:

```
objectTypeName: "b1_chart"
objectName: {chartName}
moduleName: {module}
objectDescription: {user description or chartName}
attributes:
  extendedConfig: "/service/app/data/clob/{configClobName}"
  dataConversionExpression: {JMESPath expression}
```

**Important:** The `extendedConfig` value must use the full CLOB endpoint path with the `/service/app/` prefix.

---

## Phase 4: Create Sample Screen (optional)

**Skip unless** the user explicitly requested a sample screen.

**To get the DSO's objectMasterGuid**, call `mcp__B1_Blueprint__get_object` with `detail: "full"` on the data source name.

Create the screen with `mcp__B1_Blueprint__create_blueprint`:

```
b1_screen "{screenName}"
  attributes:
    title: "{chart title or chartName}"
  instances:
    1. b1_data_source "{chartName}DSO"
       objectMasterGuid: {existing DSO objectMasterGuid}
       attributes:
         eventBeforeFetch: {inline filter JS or empty}
    2. b1_panel "a"
       layoutPosition: "a"
       objectMasterGuid: "31572117-2fe3-4262-ba03-20a74c6b59ca"
       instances:
         3. b1_chart "{chartName}"
            objectMasterGuid: {chart objectMasterGuid from Phase 3}
            attributes:
              extendedConfig: "/service/app/data/clob/{configClobName}"
              dataConversionExpression: {JMESPath expression}
  links:
    - DATA link from DSO instance to chart instance
      linkTypeGuid: "d37727fb-24ff-7ea3-0e14-a5d9d8526f38"
      linkName: "Data"
```

---

## Phase 5: Preview & Verify

If a screen was created:
- Call `mcp__B1_Blueprint__preview_screen` with the screen name

If no screen was created, explain how to embed the chart into an existing screen:
1. Add a `b1_data_source` instance (the existing DSO, with optional `eventBeforeFetch`)
2. Add a `b1_panel` instance at a layout position
3. Add the chart as an instance inside the panel
4. Create a DATA link from the data source to the chart

---

## Default Config Templates

### Bar

```json
{
  "type": "bar",
  "options": {
    "responsive": true,
    "maintainAspectRatio": false,
    "aspectRatio": 0.6,
    "plugins": {
      "legend": { "display": true, "position": "top", "labels": { "usePointStyle": true } },
      "tooltip": { "enabled": true },
      "title": { "display": false, "text": "" }
    },
    "scales": {
      "x": { "ticks": { "font": { "weight": 500 } }, "grid": { "display": false, "drawBorder": false } },
      "y": { "ticks": {}, "grid": { "drawBorder": false } }
    }
  }
}
```

### Line

```json
{
  "type": "line",
  "options": {
    "responsive": true,
    "maintainAspectRatio": false,
    "aspectRatio": 0.6,
    "plugins": {
      "legend": { "display": true, "position": "top", "labels": { "usePointStyle": true } },
      "tooltip": { "enabled": true },
      "title": { "display": false, "text": "" }
    },
    "scales": {
      "x": { "ticks": { "font": { "weight": 500 } }, "grid": { "display": false, "drawBorder": false } },
      "y": { "ticks": {}, "grid": { "drawBorder": false } }
    }
  }
}
```

> **Tip:** For smooth curves, add `"tension": 0.4` to each dataset.

### Pie

```json
{
  "type": "pie",
  "options": {
    "responsive": true,
    "maintainAspectRatio": false,
    "plugins": {
      "legend": { "display": true, "position": "bottom", "labels": { "usePointStyle": true, "padding": 20 } },
      "tooltip": { "enabled": true },
      "title": { "display": false, "text": "" }
    }
  }
}
```

### Doughnut

```json
{
  "type": "doughnut",
  "options": {
    "responsive": true,
    "maintainAspectRatio": false,
    "plugins": {
      "legend": { "display": true, "position": "bottom", "labels": { "usePointStyle": true, "padding": 20 } },
      "tooltip": { "enabled": true },
      "title": { "display": false, "text": "" }
    }
  }
}
```

### Radar

```json
{
  "type": "radar",
  "options": {
    "responsive": true,
    "maintainAspectRatio": false,
    "plugins": {
      "legend": { "display": true, "position": "top", "labels": { "usePointStyle": true, "padding": 20 } },
      "tooltip": { "enabled": true },
      "title": { "display": false, "text": "" }
    },
    "scales": {
      "r": { "beginAtZero": true, "ticks": {}, "grid": {}, "pointLabels": {} }
    }
  }
}
```

### Polar Area

```json
{
  "type": "polarArea",
  "options": {
    "responsive": true,
    "maintainAspectRatio": false,
    "plugins": {
      "legend": { "display": true, "position": "bottom", "labels": { "usePointStyle": true, "padding": 20 } },
      "tooltip": { "enabled": true },
      "title": { "display": false, "text": "" }
    },
    "scales": {
      "r": { "ticks": {}, "grid": {} }
    }
  }
}
```

### Scatter

```json
{
  "type": "scatter",
  "options": {
    "responsive": true,
    "maintainAspectRatio": false,
    "aspectRatio": 0.6,
    "plugins": {
      "legend": { "display": true, "position": "top", "labels": { "usePointStyle": true } },
      "tooltip": { "enabled": true },
      "title": { "display": false, "text": "" }
    },
    "scales": {
      "x": { "ticks": { "font": { "weight": 500 } }, "grid": { "display": false, "drawBorder": false } },
      "y": { "ticks": {}, "grid": { "drawBorder": false } }
    }
  }
}
```

> **Tip:** Scatter datasets use `{ x, y }` point objects instead of flat number arrays.

---

## Important Rules

1. **Always use CLOB for extendedConfig** — Store chart configuration in a `b1_clob` and set `extendedConfig` to the CLOB endpoint path. Do not inline large JSON directly.
2. **Resource path format** — `extendedConfig` must use the full proxy path: `/service/app/data/clob/{clobName}`.
3. **DATA link is required** — A chart must have a DATA link from a data source to receive data. Without it, the chart renders empty.
4. **No scales for pie/doughnut** — Pie and doughnut charts do NOT use `scales` in their config.
5. **Always use B1 MCP** — Use `mcp__B1_Blueprint__create_blueprint` and `mcp__B1_Blueprint__patch_blueprint` for all blueprint operations. Never edit JSON files directly.
6. **Naming conventions** — Config CLOB: `{ChartName}Config`, DSO instance: `{ChartName}DSO`, Screen: `{ChartName}Screen`.
7. **group_by syntax** — The custom `group_by` function takes a string field name: `group_by(@, 'fieldName')`. Do NOT use expression references like `&fieldName`.

---

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| Chart renders empty / no data | Missing DATA link between DSO and chart | Add a DATA link (linkTypeGuid `d37727fb-24ff-7ea3-0e14-a5d9d8526f38`) from the DSO instance to the chart instance |
| Chart shows wrong type | Incorrect `type` in config CLOB | Update the `content` attribute of the config CLOB with the correct `type` value |
| Config not applied | Wrong `extendedConfig` path | Verify path starts with `/service/app/data/clob/` (not `/resources/chart/`) |
| CLOB returns 404 | Object name mismatch | Verify the `b1_clob` object exists with `mcp__B1_Blueprint__get_object` |
| dataConversionExpression not working | Expression syntax error | Check browser console for `[jmespath] Transformation failed` warnings. Verify `group_by` uses string field name syntax |
| Chart empty with existing DSO | DSO returns no records | Check that `eventBeforeFetch` filter isn't too restrictive. Consider adding `eventSource.setFilters({ top: -1 })` |

---

## Hints and External Resources

- **Chart.js documentation** — Full configuration reference for chart types, scales, plugins, animations, and dataset options: https://www.chartjs.org/docs/
- **JMESPath documentation** — Expression syntax tutorial for writing `dataConversionExpression`: https://jmespath.org/tutorial.html
- **Custom group_by** — `group_by(@, 'fieldName')` is a Build.One extension, not part of standard JMESPath. See `src/web-core/src/utils/jmespath.ts`.
- **eventDataPointClick** — Chart objects support a click event handler for data point interactions. See `b1_chart` object type docs.
- **Static templates** — Base configs exist at `src/web-framework/layer/public/resources/chart/*.json` but prefer CLOB configs for customization.

---

## Reference

| Purpose | Resource |
|---------|----------|
| DSO chart reference screen | `InvoiceDueDateStatusChartScreen` (Samples module) |
| Chart with dataConversionExpression | `InvoiceDueDateStatusChart` (Samples module) |
| Chart object type docs | `src/web-docs/docs/object-types/b1_chart.md` |
| CLOB object type docs | `src/web-docs/docs/object-types/b1_clob.md` |
| JMESPath utility (group_by) | `src/web-core/src/utils/jmespath.ts` |
