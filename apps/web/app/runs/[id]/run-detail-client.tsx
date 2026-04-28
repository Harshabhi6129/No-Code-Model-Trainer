"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { formatDistanceToNow } from "date-fns"
import {
  ArrowLeft, CheckCircle2, XCircle, Activity, Clock,
  Database, BrainCircuit, Cpu, BarChart3, Rocket,
  AlertTriangle, RefreshCw, FileText,
} from "lucide-react"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartsTooltip, ResponsiveContainer,
} from "recharts"
import { createClient } from "@/lib/supabase/client"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { Run, RunEvent } from "@/lib/supabase/types"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AgentStep {
  name: string
  icon: React.ElementType
  fields: { label: string; value: string }[]
  message: string
}

interface MetricPoint {
  name: string
  value: number
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const STATUS_CONFIG: Record<
  string,
  {
    label: string
    color: string
    bg: string
    icon: React.ElementType
    badge: "default" | "secondary" | "destructive" | "outline"
  }
> = {
  completed: {
    label: "Completed",
    color: "text-emerald-400",
    bg: "bg-emerald-500/10 border-emerald-500/20",
    icon: CheckCircle2,
    badge: "default",
  },
  running: {
    label: "Running",
    color: "text-blue-400",
    bg: "bg-blue-500/10 border-blue-500/20",
    icon: Activity,
    badge: "secondary",
  },
  pending: {
    label: "Pending",
    color: "text-yellow-400",
    bg: "bg-yellow-500/10 border-yellow-500/20",
    icon: Clock,
    badge: "outline",
  },
  failed: {
    label: "Failed",
    color: "text-destructive",
    bg: "bg-destructive/10 border-destructive/20",
    icon: XCircle,
    badge: "destructive",
  },
  cancelled: {
    label: "Cancelled",
    color: "text-muted-foreground",
    bg: "bg-secondary border-border",
    icon: XCircle,
    badge: "outline",
  },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function asRecord(v: unknown): Record<string, unknown> {
  return v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {}
}

function buildAgentSteps(run: Run): AgentStep[] {
  const intent = asRecord(run.intent_spec)
  const recipe = asRecord(run.model_recipe)
  const steps: AgentStep[] = []

  if (Object.keys(intent).length > 0) {
    const labelVal = Array.isArray(intent.label_names)
      ? (intent.label_names as string[]).join(", ")
      : typeof intent.num_labels === "number"
      ? String(intent.num_labels)
      : "—"

    steps.push({
      name: "Intent",
      icon: BrainCircuit,
      fields: [
        { label: "Task type", value: String(intent.task_type ?? "—").replace(/_/g, " ") },
        { label: "Input column", value: String(intent.input_column ?? "—") },
        { label: "Label column", value: String(intent.label_column ?? "—") },
        { label: "Labels", value: labelVal },
        {
          label: "Confidence",
          value:
            typeof intent.confidence === "number"
              ? `${(intent.confidence * 100).toFixed(0)}%`
              : "—",
        },
      ].filter((f) => f.value !== "—"),
      message: `Detected a ${String(intent.task_type ?? "").replace(/_/g, " ")} task. Base model hint: ${String(intent.base_model_hint ?? "auto-select")}.`,
    })
  }

  if (run.dataset_filename) {
    steps.push({
      name: "Data",
      icon: Database,
      fields: [
        { label: "File", value: run.dataset_filename },
        { label: "Rows", value: run.dataset_rows?.toLocaleString() ?? "—" },
      ].filter((f) => f.value !== "—"),
      message: `Loaded ${run.dataset_rows?.toLocaleString() ?? "?"} rows from ${run.dataset_filename}. Dataset profiled and validated.`,
    })
  }

  if (Object.keys(recipe).length > 0) {
    steps.push({
      name: "Model",
      icon: Cpu,
      fields: [
        { label: "Base model", value: String(recipe.base_model ?? "—") },
        {
          label: "Approach",
          value: String(recipe.training_approach ?? "—")
            .replace(/_/g, " ")
            .toUpperCase(),
        },
        { label: "Epochs", value: String(recipe.num_epochs ?? "—") },
        { label: "Learning rate", value: String(recipe.learning_rate ?? "—") },
        { label: "Batch size", value: String(recipe.batch_size ?? "—") },
        {
          label: "LoRA rank",
          value: recipe.lora_r != null ? String(recipe.lora_r) : "—",
        },
      ].filter((f) => f.value !== "—"),
      message: String(recipe.reasoning ?? ""),
    })
  }

  return steps
}

function buildMetrics(run: Run): MetricPoint[] {
  const m = asRecord(run.metrics)
  const candidates: [string, string][] = [
    ["accuracy", "Accuracy"],
    ["f1", "F1"],
    ["precision", "Precision"],
    ["recall", "Recall"],
  ]
  return candidates
    .filter(([k]) => typeof m[k] === "number")
    .map(([k, label]) => ({
      name: label,
      value: +((m[k] as number) * 100).toFixed(1),
    }))
}

function elapsedLabel(run: Run): string {
  if (!run.created_at) return "—"
  const start = new Date(run.created_at).getTime()
  const end = run.completed_at
    ? new Date(run.completed_at).getTime()
    : Date.now()
  const secs = Math.round((end - start) / 1000)
  if (secs < 60) return `${secs}s`
  const m = Math.floor(secs / 60)
  const s = secs % 60
  return `${m}m ${s}s`
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatCard({
  label,
  value,
  icon: Icon,
  mono = false,
}: {
  label: string
  value: string
  icon: React.ElementType
  mono?: boolean
}) {
  return (
    <Card>
      <CardContent className="pt-5 pb-4">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className="text-xs text-muted-foreground mb-1">{label}</p>
            <p
              className={`text-sm font-semibold truncate ${
                mono ? "font-mono text-[11px]" : ""
              }`}
            >
              {value}
            </p>
          </div>
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-primary/10">
            <Icon className="h-3.5 w-3.5 text-primary" />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function SpecCard({
  title,
  icon: Icon,
  fields,
}: {
  title: string
  icon: React.ElementType
  fields: { label: string; value: string }[]
}) {
  if (fields.length === 0) return null
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <Icon className="h-4 w-4 text-primary" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2.5">
        {fields.map(({ label, value }) => (
          <div
            key={label}
            className="flex items-start justify-between gap-4 text-sm"
          >
            <span className="text-muted-foreground shrink-0 text-xs">{label}</span>
            <span className="font-mono text-xs text-right break-all">{value}</span>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function RunDetailClient({
  run: initialRun,
}: {
  run: Run
  events: RunEvent[]
}) {
  const [run, setRun] = useState<Run>(initialRun)
  const supabase = createClient()

  useEffect(() => {
    if (run.status !== "running") return
    const iv = setInterval(async () => {
      const { data } = await supabase
        .from("runs")
        .select("*")
        .eq("id", run.id)
        .single()
      if (data) setRun(data as Run)
    }, 3000)
    return () => clearInterval(iv)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run.id, run.status])

  const cfg = STATUS_CONFIG[run.status] ?? STATUS_CONFIG.pending
  const StatusIcon = cfg.icon
  const agentSteps = buildAgentSteps(run)
  const metricData = buildMetrics(run)
  const hasMetrics = metricData.length > 0

  const intentStep = agentSteps.find((s) => s.name === "Intent")
  const modelStep = agentSteps.find((s) => s.name === "Model")

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-6">
      {/* Back link */}
      <Link
        href="/runs"
        className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        All runs
      </Link>

      {/* Page header */}
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2.5 flex-wrap">
            <h1 className="text-2xl font-bold capitalize">
              {run.task_type?.replace(/_/g, " ") ?? "Training run"}
            </h1>
            <Badge
              variant={cfg.badge}
              className="text-xs h-5 flex items-center gap-1 px-2"
            >
              <StatusIcon className="h-3 w-3" />
              {cfg.label}
            </Badge>
          </div>
          {run.model_id && (
            <p className="text-sm text-muted-foreground font-mono">{run.model_id}</p>
          )}
          <p className="text-xs text-muted-foreground/60 font-mono">{run.id}</p>
        </div>

        {run.status === "completed" && run.hf_model_url && (
          <Button asChild size="sm">
            <a href={run.hf_model_url} target="_blank" rel="noreferrer">
              <Rocket className="h-4 w-4 mr-2" />
              View on HF Hub
            </a>
          </Button>
        )}
      </div>

      {/* Live indicator */}
      {run.status === "running" && (
        <div className="flex items-center gap-2 text-sm text-blue-400 animate-pulse">
          <RefreshCw className="h-3.5 w-3.5" />
          Live — updating every 3 s
        </div>
      )}

      {/* Error banner */}
      {run.status === "failed" && run.error_message && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-destructive/10 border border-destructive/30">
          <AlertTriangle className="h-4 w-4 text-destructive mt-0.5 shrink-0" />
          <div className="text-sm">
            <p className="font-medium text-destructive mb-0.5">Run failed</p>
            <p className="text-muted-foreground whitespace-pre-wrap">
              {run.error_message}
            </p>
          </div>
        </div>
      )}

      {/* Stat overview */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Dataset"
          value={run.dataset_filename ?? "No file"}
          icon={FileText}
          mono
        />
        <StatCard
          label="Rows"
          value={run.dataset_rows?.toLocaleString() ?? "—"}
          icon={Database}
        />
        <StatCard
          label="Started"
          value={formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
          icon={Clock}
        />
        <StatCard
          label={run.status === "running" ? "Elapsed" : "Duration"}
          value={elapsedLabel(run)}
          icon={Activity}
        />
      </div>

      {/* Timeline + Spec panels */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Agent timeline */}
        <div className="lg:col-span-2 space-y-3">
          <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Agent timeline
          </h2>

          {agentSteps.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-14 gap-2 rounded-xl border border-dashed text-center">
              <BrainCircuit className="h-6 w-6 text-muted-foreground/40" />
              <p className="text-sm text-muted-foreground">No agent data yet</p>
            </div>
          ) : (
            <div className="relative space-y-2">
              {/* Vertical connector line */}
              <div className="absolute left-[15px] top-8 bottom-8 w-px bg-border z-0" />

              {agentSteps.map((step) => {
                const Icon = step.icon
                return (
                  <div
                    key={step.name}
                    className="relative z-10 rounded-xl border border-border bg-card p-4 space-y-2"
                  >
                    <div className="flex items-center gap-2">
                      <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/10 shrink-0">
                        <Icon className="h-3.5 w-3.5 text-primary" />
                      </div>
                      <span className="text-sm font-medium">{step.name} Agent</span>
                      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 ml-auto shrink-0" />
                    </div>
                    {step.message && (
                      <p className="text-xs text-muted-foreground leading-relaxed pl-9">
                        {step.message}
                      </p>
                    )}
                  </div>
                )
              })}

              {run.status === "running" && (
                <div className="relative z-10 flex items-center gap-2 p-3 rounded-xl border border-dashed border-primary/40 text-xs text-primary animate-pulse">
                  <RefreshCw className="h-3.5 w-3.5" />
                  Training in progress…
                </div>
              )}
            </div>
          )}
        </div>

        {/* Spec cards */}
        <div className="lg:col-span-3 space-y-4">
          <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Specs
          </h2>
          {intentStep && (
            <SpecCard
              title="Intent Spec"
              icon={BrainCircuit}
              fields={intentStep.fields}
            />
          )}
          {modelStep && (
            <SpecCard
              title="Model Recipe"
              icon={Cpu}
              fields={modelStep.fields}
            />
          )}
          {!intentStep && !modelStep && (
            <div className="flex flex-col items-center justify-center py-16 gap-2 rounded-xl border border-dashed text-center">
              <p className="text-sm text-muted-foreground">
                Spec data will appear after the pipeline runs.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Metrics */}
      {hasMetrics && (
        <div className="space-y-3">
          <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Evaluation metrics
          </h2>
          <Card>
            <CardContent className="pt-6">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart
                  data={metricData}
                  margin={{ top: 0, right: 8, bottom: 0, left: 0 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(224 18% 14%)"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 12, fill: "hsl(215 16% 55%)" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fontSize: 11, fill: "hsl(215 16% 55%)" }}
                    axisLine={false}
                    tickLine={false}
                    tickFormatter={(v) => `${v}%`}
                    width={36}
                  />
                  <RechartsTooltip
                    formatter={(v) => [typeof v === "number" ? `${v}%` : "—", ""]}
                    contentStyle={{
                      background: "hsl(224 20% 9%)",
                      border: "1px solid hsl(224 18% 14%)",
                      borderRadius: "8px",
                      fontSize: "12px",
                      color: "hsl(213 31% 91%)",
                    }}
                    cursor={{ fill: "hsl(224 18% 14%)" }}
                  />
                  <Bar
                    dataKey="value"
                    radius={[4, 4, 0, 0]}
                    fill="hsl(245 58% 63%)"
                    maxBarSize={64}
                  />
                </BarChart>
              </ResponsiveContainer>

              {/* Raw numbers below chart */}
              <div className="flex gap-6 mt-4 pt-4 border-t border-border flex-wrap">
                {metricData.map(({ name, value }) => (
                  <div key={name} className="text-center">
                    <p className="text-lg font-bold">{value}%</p>
                    <p className="text-xs text-muted-foreground">{name}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* HF artifact link */}
      {run.artifact_path && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-secondary/50 border border-border text-sm">
          <BarChart3 className="h-4 w-4 text-primary shrink-0" />
          <span className="text-muted-foreground">Artifact:</span>
          <span className="font-mono text-xs break-all">{run.artifact_path}</span>
        </div>
      )}
    </div>
  )
}
