{{/*
Expand the name of the chart.
*/}}
{{- define "agent-canvas.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a fully qualified app name.
*/}}
{{- define "agent-canvas.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "agent-canvas.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "agent-canvas.labels" -}}
helm.sh/chart: {{ include "agent-canvas.chart" . }}
{{ include "agent-canvas.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "agent-canvas.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agent-canvas.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Immutable labels — the subset of common labels that are safe to stamp
into resource fields that Kubernetes rejects modifications to after
creation. Excludes `helm.sh/chart` and `app.kubernetes.io/version`,
both of which change whenever the chart version or appVersion is
bumped.

Use this (not `agent-canvas.labels`) inside anything under
`StatefulSet.spec.volumeClaimTemplates[].metadata` — that subtree is
immutable, and any diff there causes `helm upgrade` to fail with:

    StatefulSet.apps ... is invalid: spec: Forbidden: updates to
    statefulset spec for fields other than 'replicas', 'ordinals',
    'template', ... are forbidden

Object-level `metadata.labels` (on the STS itself, Services, etc.)
are mutable, so those keep the full `agent-canvas.labels` set.
*/}}
{{- define "agent-canvas.immutableLabels" -}}
{{ include "agent-canvas.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
ServiceAccount name to use.
*/}}
{{- define "agent-canvas.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "agent-canvas.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Effective image reference.
*/}}
{{- define "agent-canvas.image" -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end -}}

{{/*
PVC name — either the user-provided existing claim or the templated one
that StatefulSet.volumeClaimTemplates generates as "<claimName>-<pod>".
When existingClaim is set we mount it as a plain volume (not via
volumeClaimTemplates), which is why the StatefulSet template branches.
*/}}
{{- define "agent-canvas.pvcClaimName" -}}
{{- if .Values.persistence.existingClaim -}}
{{- .Values.persistence.existingClaim -}}
{{- else -}}
data
{{- end -}}
{{- end -}}
