{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "CurriculumFL.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "CurriculumFL.labels" -}}
helm.sh/chart: {{ include "CurriculumFL.chart" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: "{{ .Chart.AppVersion }}"
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}
    