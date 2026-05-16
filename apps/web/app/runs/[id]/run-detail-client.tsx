"use client"

import { useState, useEffect, useRef } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { formatDistanceToNow } from "date-fns"
import {
  ArrowLeft, CheckCircle2, XCircle, Activity, Clock,
  Database, BrainCircuit, Cpu, BarChart3, Rocket,
  AlertTriangle, RefreshCw, FileText, Copy, Check,
  ChevronDown, ChevronUp, Zap, ExternalLink, Info,
  Download,
} from "lucide-react"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartsTooltip, ResponsiveContainer,
} from "recharts"
import { createClient } from "@/lib/supabase/client"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import type { Run, RunEvent } from "@/lib/supabase/types"

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AgentStep {
  name: string
  icon: React.ElementType
  fields: { label: string; value: string }[]
  message: string
}

interface MetricPoint { name: string; value: number }

interface InferScore { label: string; score: number; pct: number }
interface InferResult { predicted_label: string; confidence: number; all_scores: InferScore[] }

// ---------------------------------------------------------------------------
// Status config
// ---------------------------------------------------------------------------

const STATUS_CONFIG: Record<string, {
  label: string; color: string; bg: string
  icon: React.ElementType
  badge: "default" | "secondary" | "destructive" | "outline"
}> = {
  completed: { label: "Completed", color: "text-emerald-400", bg: "bg-emerald-500/10 border-emerald-500/20", icon: CheckCircle2, badge: "default" },
  running:   { label: "Running",   color: "text-blue-400",    bg: "bg-blue-500/10 border-blue-500/20",    icon: Activity,     badge: "secondary" },
  pending:   { label: "Pending",   color: "text-yellow-400",  bg: "bg-yellow-500/10 border-yellow-500/20", icon: Clock,       badge: "outline" },
  failed:    { label: "Failed",    color: "text-destructive", bg: "bg-destructive/10 border-destructive/20", icon: XCircle,   badge: "destructive" },
  cancelled: { label: "Cancelled", color: "text-muted-foreground", bg: "bg-secondary border-border", icon: XCircle,         badge: "outline" },
}

const GRADE_CONFIG: Record<string, { ring: string; text: string; bg: string }> = {
  A: { ring: "border-emerald-500", text: "text-emerald-400", bg: "bg-emerald-500/10" },
  B: { ring: "border-blue-500",    text: "text-blue-400",    bg: "bg-blue-500/10"    },
  C: { ring: "border-yellow-500",  text: "text-yellow-400",  bg: "bg-yellow-500/10"  },
  D: { ring: "border-orange-500",  text: "text-orange-400",  bg: "bg-orange-500/10"  },
  F: { ring: "border-destructive", text: "text-destructive", bg: "bg-destructive/10" },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function asRecord(v: unknown): Record<string, unknown> {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {}
}
function asStrArray(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []
}

function buildAgentSteps(run: Run): AgentStep[] {
  const intent = asRecord(run.intent_spec)
  const recipe = asRecord(run.model_recipe)
  const steps: AgentStep[] = []

  if (Object.keys(intent).length > 0) {
    const labelVal = Array.isArray(intent.label_names)
      ? (intent.label_names as string[]).join(", ")
      : typeof intent.num_labels === "number" ? String(intent.num_labels) : "—"
    steps.push({
      name: "Intent", icon: BrainCircuit,
      fields: [
        { label: "Task type",    value: String(intent.task_type ?? "—").replace(/_/g, " ") },
        { label: "Input column", value: String(intent.input_column ?? "—") },
        { label: "Label column", value: String(intent.label_column ?? "—") },
        { label: "Labels",       value: labelVal },
        { label: "Confidence",   value: typeof intent.confidence === "number" ? `${(intent.confidence * 100).toFixed(0)}%` : "—" },
      ].filter(f => f.value !== "—"),
      message: `Detected a ${String(intent.task_type ?? "").replace(/_/g, " ")} task. Base model hint: ${String(intent.base_model_hint ?? "auto-select")}.`,
    })
  }
  if (run.dataset_filename) {
    steps.push({
      name: "Data", icon: Database,
      fields: [
        { label: "File", value: run.dataset_filename },
        { label: "Rows", value: run.dataset_rows?.toLocaleString() ?? "—" },
      ].filter(f => f.value !== "—"),
      message: `Loaded ${run.dataset_rows?.toLocaleString() ?? "?"} rows from ${run.dataset_filename}.`,
    })
  }
  if (Object.keys(recipe).length > 0) {
    steps.push({
      name: "Model", icon: Cpu,
      fields: [
        { label: "Base model",   value: String(recipe.base_model ?? "—") },
        { label: "Approach",     value: String(recipe.training_approach ?? "—").replace(/_/g, " ").toUpperCase() },
        { label: "Epochs",       value: String(recipe.num_epochs ?? "—") },
        { label: "Learning rate",value: String(recipe.learning_rate ?? "—") },
        { label: "Batch size",   value: String(recipe.batch_size ?? "—") },
        { label: "LoRA rank",    value: recipe.lora_r != null ? String(recipe.lora_r) : "—" },
      ].filter(f => f.value !== "—"),
      message: String(recipe.reasoning ?? ""),
    })
  }
  return steps
}

function buildMetrics(run: Run): MetricPoint[] {
  const m = asRecord(run.metrics)
  return (["accuracy", "f1", "precision", "recall"] as const)
    .filter(k => typeof m[k] === "number")
    .map(k => ({ name: k.charAt(0).toUpperCase() + k.slice(1), value: +((m[k] as number) * 100).toFixed(1) }))
}

function elapsedLabel(run: Run): string {
  if (!run.created_at) return "—"
  const end = run.completed_at ? new Date(run.completed_at).getTime() : Date.now()
  const secs = Math.round((end - new Date(run.created_at).getTime()) / 1000)
  const m = Math.floor(secs / 60); const s = secs % 60
  return m ? `${m}m ${s}s` : `${secs}s`
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatCard({ label, value, icon: Icon, mono = false }: { label: string; value: string; icon: React.ElementType; mono?: boolean }) {
  return (
    <Card><CardContent className="pt-5 pb-4">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-xs text-muted-foreground mb-1">{label}</p>
          <p className={`text-sm font-semibold truncate ${mono ? "font-mono text-[11px]" : ""}`}>{value}</p>
        </div>
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-primary/10">
          <Icon className="h-3.5 w-3.5 text-primary" />
        </div>
      </div>
    </CardContent></Card>
  )
}

function SpecCard({ title, icon: Icon, fields }: { title: string; icon: React.ElementType; fields: { label: string; value: string }[] }) {
  if (!fields.length) return null
  return (
    <Card>
      <CardHeader className="pb-3"><CardTitle className="text-sm flex items-center gap-2"><Icon className="h-4 w-4 text-primary" />{title}</CardTitle></CardHeader>
      <CardContent className="space-y-2.5">
        {fields.map(({ label, value }) => (
          <div key={label} className="flex items-start justify-between gap-4 text-sm">
            <span className="text-muted-foreground shrink-0 text-xs">{label}</span>
            <span className="font-mono text-xs text-right break-all">{value}</span>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <button onClick={copy} className="p-1.5 rounded-md hover:bg-secondary transition-colors" title="Copy">
      {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5 text-muted-foreground" />}
    </button>
  )
}

function CodeBlock({ code, lang = "python" }: { code: string; lang?: string }) {
  return (
    <div className="relative rounded-lg bg-[hsl(224_25%_4%)] border border-border">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border">
        <span className="text-[10px] text-muted-foreground font-mono uppercase tracking-wider">{lang}</span>
        <CopyButton text={code} />
      </div>
      <pre className="p-4 text-xs font-mono text-foreground overflow-x-auto leading-relaxed whitespace-pre">{code}</pre>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Retrain Button
// ---------------------------------------------------------------------------

interface RetrainPrefill {
  intent: string
  selectedModelId: string
  hyperParams: {
    task_type: string; num_epochs: number; learning_rate: number
    batch_size: number; max_length: number; training_approach: string
    lora_r: number; weight_decay: number; warmup_ratio: number
  }
  datasetFilename: string | null
  datasetRows: number | null
  textColumns: string[]; labelColumns: string[]; uniqueLabels: string[]
  label: string
  sourceRunId: string
}

function RetrainButton({ run }: { run: Run }) {
  const router = useRouter()
  const recipe = asRecord(run.model_recipe)
  const intent = asRecord(run.intent_spec)

  if (!run.status || !["completed", "failed"].includes(run.status)) return null

  function handleRetrain() {
    const prefill: RetrainPrefill = {
      intent: typeof intent.user_intent === "string"
        ? intent.user_intent
        : `Train a ${run.task_type?.replace(/_/g, " ") ?? "classification"} model on this dataset`,
      selectedModelId:   String(recipe.base_model ?? "roberta-base"),
      hyperParams: {
        task_type:         String(intent.task_type ?? run.task_type ?? "text_classification"),
        num_epochs:        Number(recipe.num_epochs   ?? 3),
        learning_rate:     Number(recipe.learning_rate ?? 2e-5),
        batch_size:        Number(recipe.batch_size   ?? 16),
        max_length:        Number(recipe.max_length   ?? 128),
        training_approach: String(recipe.training_approach ?? "full_finetune"),
        lora_r:            Number(recipe.lora_r        ?? 8),
        weight_decay:      Number(recipe.weight_decay  ?? 0.01),
        warmup_ratio:      Number(recipe.warmup_ratio  ?? 0.1),
      },
      datasetFilename: run.dataset_filename,
      datasetRows:     run.dataset_rows,
      textColumns:     typeof intent.input_column === "string" ? [intent.input_column] : [],
      labelColumns:    typeof intent.label_column === "string" ? [intent.label_column] : [],
      uniqueLabels:    [],
      label:           `Retrain: ${(run.task_type ?? "run").replace(/_/g, " ")}`.slice(0, 30),
      sourceRunId:     run.id,
    }
    localStorage.setItem("modelforge_retrain_prefill", JSON.stringify(prefill))
    router.push("/train")
  }

  return (
    <Button variant="outline" size="sm" onClick={handleRetrain} className="gap-2">
      <RefreshCw className="h-3.5 w-3.5" /> Retrain with tweaks
    </Button>
  )
}

// ---------------------------------------------------------------------------
// Export Section
// ---------------------------------------------------------------------------

function ExportSection({ run }: { run: Run }) {
  const [format, setFormat]   = useState<"onnx" | "torchscript">("onnx")
  const [exporting, setExporting] = useState(false)
  const [error, setError]     = useState<string | null>(null)

  if (run.status !== "completed" || !run.artifact_path) return null

  async function handleExport() {
    setExporting(true)
    setError(null)
    try {
      const res = await fetch(`${API_URL}/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: run.id, artifact_path: run.artifact_path, format }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: "Export failed" }))
        throw new Error(body.detail ?? "Export failed")
      }
      const blob = await res.blob()
      const url  = URL.createObjectURL(blob)
      const a    = document.createElement("a")
      a.href     = url
      a.download = `model_${run.id.slice(0, 8)}.${format === "onnx" ? "onnx" : "pt"}`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="space-y-3">
      <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Export Model</h2>
      <Card>
        <CardContent className="pt-5 space-y-4">
          <div className="flex gap-2">
            {(["onnx", "torchscript"] as const).map(fmt => (
              <button
                key={fmt}
                onClick={() => setFormat(fmt)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
                  format === fmt
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-border text-muted-foreground hover:text-foreground hover:bg-secondary"
                }`}
              >
                {fmt === "onnx" ? "ONNX" : "TorchScript"}
              </button>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            {format === "onnx"
              ? "Universal format — runs on ONNX Runtime, OpenVINO, TensorRT, and Core ML."
              : "PyTorch-optimized — ideal for production PyTorch servers and mobile deployment."}
          </p>
          {error && (
            <div className="flex items-start gap-2.5 p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-xs text-destructive">
              <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
              {error}
            </div>
          )}
          <Button onClick={handleExport} disabled={exporting} size="sm" className="gap-2">
            {exporting
              ? <><RefreshCw className="h-3.5 w-3.5 animate-spin" />Converting…</>
              : <><Download className="h-3.5 w-3.5" />Export {format.toUpperCase()}</>
            }
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Eval Report Card
// ---------------------------------------------------------------------------

function EvalReportCard({ metrics }: { metrics: Record<string, unknown> }) {
  const grade = typeof metrics.evaluation_grade === "string" ? metrics.evaluation_grade : null
  const summary = typeof metrics.summary === "string" ? metrics.summary : null
  const strengths = asStrArray(metrics.strengths)
  const concerns  = asStrArray(metrics.concerns)
  const nextSteps = asStrArray(metrics.next_steps)

  if (!grade && !summary) return null

  const gc = GRADE_CONFIG[grade ?? ""] ?? GRADE_CONFIG["C"]

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-primary" /> Evaluation Report
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Grade + summary */}
        <div className="flex items-start gap-4">
          {grade && (
            <div className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-xl border-2 font-bold text-2xl ${gc.ring} ${gc.text} ${gc.bg}`}>
              {grade}
            </div>
          )}
          {summary && <p className="text-sm text-muted-foreground leading-relaxed">{summary}</p>}
        </div>

        {/* Strengths + Concerns */}
        {(strengths.length > 0 || concerns.length > 0) && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {strengths.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">Strengths</p>
                {strengths.map((s, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 mt-0.5 shrink-0" />
                    <span>{s}</span>
                  </div>
                ))}
              </div>
            )}
            {concerns.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-xs font-semibold text-yellow-400 uppercase tracking-wider">Concerns</p>
                {concerns.map((c, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                    <AlertTriangle className="h-3.5 w-3.5 text-yellow-400 mt-0.5 shrink-0" />
                    <span>{c}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Next Steps */}
        {nextSteps.length > 0 && (
          <div className="space-y-1.5">
            <p className="text-xs font-semibold text-primary uppercase tracking-wider">Next Steps</p>
            {nextSteps.map((step, i) => (
              <div key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary text-[9px] font-bold mt-0.5">{i + 1}</span>
                <span>{step}</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Deploy Section
// ---------------------------------------------------------------------------

function DeploySection({ run }: { run: Run }) {
  const [cardOpen, setCardOpen] = useState(false)
  const [activeSnippet, setActiveSnippet] = useState<"pipeline" | "python">("pipeline")

  const metrics = asRecord(run.metrics)
  const trainSkipped = metrics.status === "skipped"
  if (!run.artifact_path && !run.hf_model_url && !run.deploy_status) return null
  if (trainSkipped) return null

  const status = run.deploy_status ?? "not_deployed"
  const snippets = run.model_card ? extractSnippets(run.model_card) : null

  return (
    <div className="space-y-3">
      <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Deploy</h2>
      <Card>
        <CardContent className="pt-5 space-y-4">
          {/* Status row */}
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-2.5">
              <div className={`flex h-8 w-8 items-center justify-center rounded-lg ${
                status === "deployed" ? "bg-emerald-500/10" :
                status === "failed"   ? "bg-destructive/10" :
                "bg-secondary"
              }`}>
                <Rocket className={`h-4 w-4 ${
                  status === "deployed" ? "text-emerald-400" :
                  status === "failed"   ? "text-destructive" :
                  "text-muted-foreground"
                }`} />
              </div>
              <div>
                <p className="text-sm font-medium capitalize">{
                  status === "deployed"     ? "Deployed to HuggingFace Hub" :
                  status === "skipped"      ? "Deploy skipped" :
                  status === "failed"       ? "Deploy failed" :
                  status === "deploying"    ? "Deploying…" :
                  "Not deployed"
                }</p>
                {run.hf_repo_id && <p className="text-xs text-muted-foreground font-mono">{run.hf_repo_id}</p>}
              </div>
            </div>
            {run.hf_model_url && (
              <Button asChild size="sm" variant="outline">
                <a href={run.hf_model_url} target="_blank" rel="noreferrer" className="flex items-center gap-1.5">
                  <ExternalLink className="h-3.5 w-3.5" /> View on HF Hub
                </a>
              </Button>
            )}
          </div>

          {/* Usage snippets */}
          {snippets && (
            <div className="space-y-3">
              <Separator />
              <div className="flex gap-1">
                {(["pipeline", "python"] as const).map(tab => (
                  <button key={tab} onClick={() => setActiveSnippet(tab)}
                    className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                      activeSnippet === tab ? "bg-primary/10 text-primary" : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                    }`}>
                    {tab === "pipeline" ? "Pipeline API" : "Manual inference"}
                  </button>
                ))}
              </div>
              <CodeBlock code={activeSnippet === "pipeline" ? (snippets.pipeline ?? "") : (snippets.python ?? "")} />
            </div>
          )}

          {/* Model card toggle */}
          {run.model_card && (
            <div className="space-y-2">
              <Separator />
              <button onClick={() => setCardOpen(o => !o)}
                className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors w-full text-left">
                {cardOpen ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                {cardOpen ? "Hide" : "Show"} model card (README.md)
                <CopyButton text={run.model_card} />
              </button>
              {cardOpen && (
                <div className="rounded-lg border border-border bg-[hsl(224_25%_4%)] p-4 text-xs font-mono leading-relaxed whitespace-pre-wrap text-muted-foreground max-h-96 overflow-y-auto">
                  {run.model_card}
                </div>
              )}
            </div>
          )}

          {/* No token / skipped instructions */}
          {status === "skipped" && !run.hf_model_url && run.artifact_path && (
            <div className="flex items-start gap-2.5 p-3 rounded-lg bg-secondary/50 border border-border text-xs text-muted-foreground">
              <Info className="h-3.5 w-3.5 mt-0.5 shrink-0 text-primary" />
              <div>
                <p className="font-medium text-foreground mb-1">Deploy manually</p>
                <p>Set <code className="text-primary">HF_TOKEN</code> in your <code>.env</code> and re-run the pipeline, or push directly:</p>
                <code className="block mt-2 text-[10px] bg-background/50 rounded p-2 font-mono">
                  {`huggingface-cli login\nhuggingface-cli upload ${run.artifact_path} your-username/your-model-name`}
                </code>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function extractSnippets(modelCard: string): { pipeline?: string; python?: string } | null {
  const blocks: string[] = []
  const re = /```python\n([\s\S]*?)```/g
  let m: RegExpExecArray | null
  while ((m = re.exec(modelCard)) !== null) blocks.push(m[1].trim())
  if (!blocks.length) return null
  return {
    pipeline: blocks.find(b => b.includes("pipeline(")) ?? blocks[0],
    python:   blocks.find(b => b.includes("AutoModel"))  ?? blocks[blocks.length - 1],
  }
}

// ---------------------------------------------------------------------------
// Inference Playground
// ---------------------------------------------------------------------------

function InferencePlayground({ run }: { run: Run }) {
  const [text, setText] = useState("")
  const [result, setResult] = useState<InferResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const metrics = asRecord(run.metrics)
  const labelNames = asStrArray(metrics.label_names)

  const canInfer = run.status === "completed" && !!run.artifact_path
  if (!canInfer) return null

  async function handleInfer() {
    if (!text.trim()) return
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await fetch(`${API_URL}/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: run.id,
          text: text.trim(),
          artifact_path: run.artifact_path,
          label_names: labelNames,
        }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: "Inference failed" }))
        throw new Error(body.detail ?? "Inference failed")
      }
      setResult(await res.json())
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err))
    } finally {
      setLoading(false)
    }
  }

  const maxChars = 2000
  const charsLeft = maxChars - text.length
  const overLimit = charsLeft < 0

  return (
    <div className="space-y-3">
      <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
        <Zap className="h-3.5 w-3.5 text-primary" /> Inference Playground
      </h2>
      <Card>
        <CardContent className="pt-5 space-y-4">
          {/* Input */}
          <div className="space-y-2">
            <div className="relative">
              <textarea
                ref={textareaRef}
                value={text}
                onChange={e => setText(e.target.value)}
                placeholder="Type or paste text to classify…"
                rows={4}
                disabled={loading}
                className="w-full resize-none rounded-lg border border-input bg-card px-4 py-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary transition-colors disabled:opacity-50"
              />
            </div>
            <div className="flex items-center justify-between">
              <p className={`text-xs ${overLimit ? "text-destructive" : "text-muted-foreground"}`}>
                {Math.abs(charsLeft)} {overLimit ? "chars over limit" : "chars remaining"}
              </p>
              <Button onClick={handleInfer} disabled={loading || !text.trim() || overLimit} size="sm" className="gap-2">
                {loading ? <><RefreshCw className="h-3.5 w-3.5 animate-spin" />Running…</> : <><Zap className="h-3.5 w-3.5" />Run Inference</>}
              </Button>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-start gap-2.5 p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-xs text-destructive">
              <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
              {error}
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="space-y-3 pt-1">
              <Separator />
              <div className="flex items-center gap-3">
                <p className="text-xs text-muted-foreground shrink-0">Predicted:</p>
                <Badge className="text-xs capitalize bg-primary/10 text-primary border-primary/20 border">
                  {result.predicted_label}
                </Badge>
                <span className="text-xs text-muted-foreground ml-auto">
                  {(result.confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
              {/* Confidence bars */}
              <div className="space-y-2">
                {result.all_scores.map(({ label, pct }) => (
                  <div key={label} className="space-y-1">
                    <div className="flex justify-between text-[11px]">
                      <span className={`capitalize font-medium ${label === result.predicted_label ? "text-primary" : "text-muted-foreground"}`}>{label}</span>
                      <span className="text-muted-foreground font-mono">{pct.toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${label === result.predicted_label ? "bg-primary" : "bg-muted-foreground/40"}`}
                        style={{ width: `${Math.max(pct, 0.5)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function RunDetailClient({ run: initialRun }: { run: Run; events: RunEvent[] }) {
  const [run, setRun] = useState<Run>(initialRun)
  const supabase = createClient()

  useEffect(() => {
    if (run.status !== "running") return
    const iv = setInterval(async () => {
      const { data } = await supabase.from("runs").select("*").eq("id", run.id).single()
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
  const metricsObj = asRecord(run.metrics)

  const intentStep = agentSteps.find(s => s.name === "Intent")
  const modelStep  = agentSteps.find(s => s.name === "Model")

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-6">
      {/* Back */}
      <Link href="/runs" className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors">
        <ArrowLeft className="h-3.5 w-3.5" /> All runs
      </Link>

      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2.5 flex-wrap">
            <h1 className="text-2xl font-bold capitalize">{run.task_type?.replace(/_/g, " ") ?? "Training run"}</h1>
            <Badge variant={cfg.badge} className="text-xs h-5 flex items-center gap-1 px-2">
              <StatusIcon className="h-3 w-3" />{cfg.label}
            </Badge>
          </div>
          {run.model_id && <p className="text-sm text-muted-foreground font-mono">{run.model_id}</p>}
          <p className="text-xs text-muted-foreground/60 font-mono">{run.id}</p>
        </div>
        <div className="flex gap-2 flex-wrap">
          <RetrainButton run={run} />
          {run.status === "completed" && run.hf_model_url && (
            <Button asChild size="sm">
              <a href={run.hf_model_url} target="_blank" rel="noreferrer">
                <Rocket className="h-4 w-4 mr-2" /> View on HF Hub
              </a>
            </Button>
          )}
        </div>
      </div>

      {run.status === "running" && (
        <div className="flex items-center gap-2 text-sm text-blue-400 animate-pulse">
          <RefreshCw className="h-3.5 w-3.5" /> Live — updating every 3s
        </div>
      )}

      {run.status === "failed" && run.error_message && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-destructive/10 border border-destructive/30">
          <AlertTriangle className="h-4 w-4 text-destructive mt-0.5 shrink-0" />
          <div className="text-sm">
            <p className="font-medium text-destructive mb-0.5">Run failed</p>
            <p className="text-muted-foreground whitespace-pre-wrap">{run.error_message}</p>
          </div>
        </div>
      )}

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Dataset" value={run.dataset_filename ?? "No file"} icon={FileText} mono />
        <StatCard label="Rows"    value={run.dataset_rows?.toLocaleString() ?? "—"}       icon={Database} />
        <StatCard label="Started" value={formatDistanceToNow(new Date(run.created_at), { addSuffix: true })} icon={Clock} />
        <StatCard label={run.status === "running" ? "Elapsed" : "Duration"} value={elapsedLabel(run)} icon={Activity} />
      </div>

      {/* Timeline + Specs */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-2 space-y-3">
          <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Agent timeline</h2>
          {agentSteps.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-14 gap-2 rounded-xl border border-dashed text-center">
              <BrainCircuit className="h-6 w-6 text-muted-foreground/40" />
              <p className="text-sm text-muted-foreground">No agent data yet</p>
            </div>
          ) : (
            <div className="relative space-y-2">
              <div className="absolute left-[15px] top-8 bottom-8 w-px bg-border z-0" />
              {agentSteps.map(step => {
                const Icon = step.icon
                return (
                  <div key={step.name} className="relative z-10 rounded-xl border border-border bg-card p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/10 shrink-0">
                        <Icon className="h-3.5 w-3.5 text-primary" />
                      </div>
                      <span className="text-sm font-medium">{step.name} Agent</span>
                      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 ml-auto shrink-0" />
                    </div>
                    {step.message && <p className="text-xs text-muted-foreground leading-relaxed pl-9">{step.message}</p>}
                  </div>
                )
              })}
              {run.status === "running" && (
                <div className="relative z-10 flex items-center gap-2 p-3 rounded-xl border border-dashed border-primary/40 text-xs text-primary animate-pulse">
                  <RefreshCw className="h-3.5 w-3.5" /> Training in progress…
                </div>
              )}
            </div>
          )}
        </div>

        <div className="lg:col-span-3 space-y-4">
          <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Specs</h2>
          {intentStep && <SpecCard title="Intent Spec" icon={BrainCircuit} fields={intentStep.fields} />}
          {modelStep  && <SpecCard title="Model Recipe" icon={Cpu}         fields={modelStep.fields}  />}
          {!intentStep && !modelStep && (
            <div className="flex flex-col items-center justify-center py-16 gap-2 rounded-xl border border-dashed text-center">
              <p className="text-sm text-muted-foreground">Spec data will appear after the pipeline runs.</p>
            </div>
          )}
        </div>
      </div>

      {/* Eval Report Card */}
      <EvalReportCard metrics={metricsObj} />

      {/* Metrics chart */}
      {hasMetrics && (
        <div className="space-y-3">
          <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Evaluation metrics</h2>
          <Card>
            <CardContent className="pt-6">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={metricData} margin={{ top: 0, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(224 18% 14%)" vertical={false} />
                  <XAxis dataKey="name" tick={{ fontSize: 12, fill: "hsl(215 16% 55%)" }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: "hsl(215 16% 55%)" }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} width={36} />
                  <RechartsTooltip
                    formatter={(v) => [typeof v === "number" ? `${v}%` : "—", ""]}
                    contentStyle={{ background: "hsl(224 20% 9%)", border: "1px solid hsl(224 18% 14%)", borderRadius: "8px", fontSize: "12px", color: "hsl(213 31% 91%)" }}
                    cursor={{ fill: "hsl(224 18% 14%)" }}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]} fill="hsl(245 58% 63%)" maxBarSize={64} />
                </BarChart>
              </ResponsiveContainer>
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

      {/* Deploy Section */}
      <DeploySection run={run} />

      {/* Export Section */}
      <ExportSection run={run} />

      {/* Inference Playground */}
      <InferencePlayground run={run} />

      {/* Artifact path */}
      {run.artifact_path && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-secondary/50 border border-border text-sm">
          <FileText className="h-4 w-4 text-primary shrink-0" />
          <span className="text-muted-foreground text-xs">Saved model:</span>
          <span className="font-mono text-xs break-all">{run.artifact_path}</span>
        </div>
      )}
    </div>
  )
}
