"use client"

import { useState, useRef, useCallback } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import {
  CloudUpload, Loader2, CheckCircle2, XCircle, AlertTriangle,
  Database, BrainCircuit, Cpu, BarChart3, Rocket, FileText,
  Settings2, Play, ChevronRight, ArrowRight,
} from "lucide-react"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartsTooltip, ResponsiveContainer, Cell,
} from "recharts"
import { toast } from "sonner"
import { createClient } from "@/lib/supabase/client"

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TextLengthStats {
  min: number; max: number; mean: number; p50: number; p90: number; p99: number
}

interface UploadResult {
  file_id: string
  filename: string
  rows: number
  columns: string[]
  text_columns: string[]
  label_columns: string[]
  unique_labels: string[]
  class_distribution: Record<string, number>
  text_length_stats: TextLengthStats | Record<string, never>
  text_length_histogram: { bin_start: number; bin_end: number; count: number }[]
  data_warnings: string[]
  duplicate_count: number
  null_count: number
  sample_rows: Record<string, unknown>[]
}

interface AgentMessage {
  agent: string
  success: boolean
  message: string
  output: Record<string, unknown>
}

interface ModelEntry {
  model_id: string
  display_name: string
  param_count: string
  description: string
  quality_tier: "excellent" | "good" | "fast"
  inference_speed: "fast" | "medium" | "slow"
  lora_compatible: boolean
  best_for: string
}

interface HyperParams {
  task_type: string
  num_epochs: number
  learning_rate: number
  batch_size: number
  max_length: number
  training_approach: string
  lora_r: number
  weight_decay: number
  warmup_ratio: number
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WIZARD_STEPS = [
  { key: "upload",    label: "Upload",    icon: CloudUpload },
  { key: "analyze",   label: "Analyze",   icon: BarChart3 },
  { key: "configure", label: "Configure", icon: Settings2 },
  { key: "train",     label: "Train",     icon: Cpu },
  { key: "results",   label: "Results",   icon: CheckCircle2 },
]

const MODELS: ModelEntry[] = [
  {
    model_id: "distilbert-base-uncased",
    display_name: "DistilBERT (66M)",
    param_count: "66M",
    description: "Lightweight BERT distillation. Fast training and inference, great for small datasets.",
    quality_tier: "good",
    inference_speed: "fast",
    lora_compatible: true,
    best_for: "Small datasets, fast prototyping",
  },
  {
    model_id: "bert-base-uncased",
    display_name: "BERT Base (110M)",
    param_count: "110M",
    description: "Google's original BERT. Solid baseline across most classification tasks.",
    quality_tier: "good",
    inference_speed: "medium",
    lora_compatible: true,
    best_for: "General text classification",
  },
  {
    model_id: "roberta-base",
    display_name: "RoBERTa Base (125M)",
    param_count: "125M",
    description: "Robustly optimized BERT. Consistently outperforms vanilla BERT on benchmarks.",
    quality_tier: "excellent",
    inference_speed: "medium",
    lora_compatible: true,
    best_for: "Accuracy-critical tasks",
  },
  {
    model_id: "distilroberta-base",
    display_name: "DistilRoBERTa (82M)",
    param_count: "82M",
    description: "Distilled RoBERTa. Best speed/accuracy tradeoff for production deployments.",
    quality_tier: "good",
    inference_speed: "fast",
    lora_compatible: true,
    best_for: "Speed-sensitive applications",
  },
  {
    model_id: "microsoft/deberta-v3-small",
    display_name: "DeBERTa v3 Small (142M)",
    param_count: "142M",
    description: "Microsoft's top small model. Leads classification benchmarks despite its size.",
    quality_tier: "excellent",
    inference_speed: "medium",
    lora_compatible: true,
    best_for: "Maximum accuracy",
  },
  {
    model_id: "cardiffnlp/twitter-roberta-base-sentiment-latest",
    display_name: "Twitter RoBERTa Sentiment",
    param_count: "125M",
    description: "Pre-fine-tuned on 124M tweets. Ideal jump-start for sentiment tasks.",
    quality_tier: "excellent",
    inference_speed: "medium",
    lora_compatible: true,
    best_for: "Social media & sentiment",
  },
]

const TASK_TYPES = [
  { value: "text_classification", label: "Text Classification" },
  { value: "sentiment_analysis",  label: "Sentiment Analysis" },
  { value: "ner",                 label: "Named Entity Recognition" },
  { value: "question_answering",  label: "Question Answering" },
]

const LR_OPTIONS = [
  { value: 1e-5,  label: "1e-5  (conservative)" },
  { value: 2e-5,  label: "2e-5  (recommended)" },
  { value: 3e-5,  label: "3e-5  (aggressive)" },
  { value: 5e-5,  label: "5e-5  (very aggressive)" },
]

const APPROACH_OPTIONS = [
  { value: "full_finetune", label: "Full Fine-tune", desc: "All weights updated — best quality" },
  { value: "lora",          label: "LoRA",           desc: "Parameter-efficient, faster" },
  { value: "qlora",         label: "QLoRA",          desc: "Quantized LoRA — memory-efficient" },
]

const BAR_COLORS = ["#7c6fcd","#5ea5f8","#4ade80","#f97316","#e879f9","#facc15","#38bdf8","#fb7185"]

const AGENT_ICONS: Record<string, React.ElementType> = {
  Intent: BrainCircuit, Data: Database, Clean: Settings2, Model: Cpu,
  Train: Cpu, Eval: BarChart3, Deploy: Rocket, System: XCircle,
}

const DEFAULT_HYPERPARAMS: HyperParams = {
  task_type: "text_classification",
  num_epochs: 3,
  learning_rate: 2e-5,
  batch_size: 16,
  max_length: 128,
  training_approach: "full_finetune",
  lora_r: 8,
  weight_decay: 0.01,
  warmup_ratio: 0.1,
}

// ---------------------------------------------------------------------------
// Step 2 — Dataset Analysis
// ---------------------------------------------------------------------------

function DataAnalysisPanel({ upload, onNext }: { upload: UploadResult; onNext: () => void }) {
  const classData = Object.entries(upload.class_distribution)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12)
    .map(([name, value]) => ({ name: name.length > 14 ? name.slice(0, 12) + "…" : name, value }))

  const histData = upload.text_length_histogram.map(b => ({
    name: String(b.bin_start),
    count: b.count,
    range: `${b.bin_start}–${b.bin_end}`,
  }))

  const stats = "mean" in upload.text_length_stats ? (upload.text_length_stats as TextLengthStats) : null

  const chartHeight = Math.max(classData.length * 28, 160)

  return (
    <div className="space-y-6">
      {/* Summary stat row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {([
          { label: "Total rows",    value: upload.rows.toLocaleString(),            icon: Database },
          { label: "Columns",       value: String(upload.columns.length),           icon: FileText },
          { label: "Unique labels", value: String(upload.unique_labels.length),     icon: BarChart3 },
          { label: "Duplicates",    value: String(upload.duplicate_count),          icon: AlertTriangle },
        ] as const).map(({ label, value, icon: Icon }) => (
          <Card key={label}>
            <CardContent className="pt-4 pb-3">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-xs text-muted-foreground">{label}</p>
                  <p className="text-xl font-bold mt-0.5">{value}</p>
                </div>
                <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/10">
                  <Icon className="h-3.5 w-3.5 text-primary" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Warnings */}
      {upload.data_warnings.length > 0 && (
        <div className="space-y-2">
          {upload.data_warnings.map((w, i) => (
            <div key={i} className="flex items-start gap-2.5 p-3 rounded-lg bg-yellow-500/5 border border-yellow-500/20 text-xs">
              <AlertTriangle className="h-3.5 w-3.5 text-yellow-400 mt-0.5 shrink-0" />
              <span className="text-muted-foreground">{w}</span>
            </div>
          ))}
        </div>
      )}

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {classData.length > 0 && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-primary" /> Class Distribution
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={chartHeight}>
                <BarChart data={classData} layout="vertical" margin={{ top: 0, right: 24, bottom: 0, left: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(224 18% 14%)" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 11, fill: "hsl(215 16% 55%)" }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: "hsl(215 16% 55%)" }} axisLine={false} tickLine={false} width={80} />
                  <RechartsTooltip
                    formatter={(v) => [v, "Samples"]}
                    contentStyle={{ background: "hsl(224 20% 9%)", border: "1px solid hsl(224 18% 14%)", borderRadius: "8px", fontSize: "12px" }}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={20}>
                    {classData.map((_, i) => <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {histData.length > 0 && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-primary" /> Text Length Distribution
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={chartHeight}>
                <BarChart data={histData} margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(224 18% 14%)" vertical={false} />
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: "hsl(215 16% 55%)" }} axisLine={false} tickLine={false}
                    label={{ value: "characters", position: "insideBottom", offset: -12, fontSize: 10, fill: "hsl(215 16% 55%)" }} />
                  <YAxis tick={{ fontSize: 11, fill: "hsl(215 16% 55%)" }} axisLine={false} tickLine={false} width={32} />
                  <RechartsTooltip
                    formatter={(v, _, p) => [v, `${(p.payload as { range: string }).range} chars`]}
                    contentStyle={{ background: "hsl(224 20% 9%)", border: "1px solid hsl(224 18% 14%)", borderRadius: "8px", fontSize: "12px" }}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]} fill="hsl(245 58% 63%)" maxBarSize={32} />
                </BarChart>
              </ResponsiveContainer>
              {stats && (
                <div className="flex gap-5 mt-3 pt-3 border-t border-border flex-wrap text-xs">
                  {([
                    { label: "Min", value: stats.min },
                    { label: "Avg", value: stats.mean.toFixed(0) },
                    { label: "p50", value: stats.p50.toFixed(0) },
                    { label: "p90", value: stats.p90.toFixed(0) },
                    { label: "Max", value: stats.max },
                  ] as const).map(({ label, value }) => (
                    <div key={label} className="text-center">
                      <p className="font-mono font-semibold">{value}</p>
                      <p className="text-muted-foreground">{label}</p>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>

      {/* Column tags */}
      <Card>
        <CardHeader className="pb-3"><CardTitle className="text-sm">Detected Columns</CardTitle></CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {upload.columns.map(col => {
              const isText  = upload.text_columns.includes(col)
              const isLabel = upload.label_columns.includes(col)
              return (
                <Badge key={col} variant="secondary"
                  className={`text-xs gap-1 ${isText ? "bg-primary/15 text-primary border-primary/25" : isLabel ? "border-emerald-500/30 text-emerald-400 bg-emerald-500/10" : ""}`}>
                  {col}
                  {isText  && <span className="opacity-60 text-[9px]">TEXT</span>}
                  {isLabel && <span className="opacity-60 text-[9px]">LABEL</span>}
                </Badge>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Sample rows */}
      {upload.sample_rows.length > 0 && (
        <Card>
          <CardHeader className="pb-3"><CardTitle className="text-sm">Sample Rows</CardTitle></CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border">
                    {upload.columns.slice(0, 5).map(col => (
                      <th key={col} className="text-left py-2 px-3 font-medium text-muted-foreground">{col}</th>
                    ))}
                    {upload.columns.length > 5 && (
                      <th className="text-left py-2 px-3 font-medium text-muted-foreground">+{upload.columns.length - 5}</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {upload.sample_rows.map((row, i) => (
                    <tr key={i} className="border-b border-border/50 hover:bg-secondary/30">
                      {upload.columns.slice(0, 5).map(col => (
                        <td key={col} className="py-2 px-3 max-w-[200px] truncate text-muted-foreground font-mono">
                          {String(row[col] ?? "")}
                        </td>
                      ))}
                      {upload.columns.length > 5 && <td className="py-2 px-3 text-muted-foreground/40">…</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="flex justify-end">
        <Button onClick={onNext} className="gap-2">
          Configure Training <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 3 — Configure
// ---------------------------------------------------------------------------

function ConfigurePanel({
  upload, hyperparams, selectedModel, intent,
  onHyperparams, onModel, onIntent, onStart, starting,
}: {
  upload: UploadResult
  hyperparams: HyperParams
  selectedModel: string
  intent: string
  onHyperparams: (h: HyperParams) => void
  onModel: (m: string) => void
  onIntent: (s: string) => void
  onStart: () => void
  starting: boolean
}) {
  const hp = hyperparams
  const setHp = (patch: Partial<HyperParams>) => onHyperparams({ ...hp, ...patch })
  const isLoRA = hp.training_approach === "lora" || hp.training_approach === "qlora"
  const suggestedId = upload.rows < 500 ? "distilbert-base-uncased" : "roberta-base"

  return (
    <div className="space-y-6">
      {/* Task description */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <BrainCircuit className="h-4 w-4 text-primary" /> Describe Your Task
            <span className="text-muted-foreground font-normal text-xs ml-1">(optional — AI will infer from your data)</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <textarea
            value={intent}
            onChange={e => onIntent(e.target.value)}
            placeholder={`e.g. "Classify customer support tickets by urgency: low, medium, high"`}
            rows={3}
            className="w-full resize-none rounded-lg border border-input bg-card px-4 py-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary transition-colors"
          />
          <div className="flex items-center gap-3">
            <span className="text-xs text-muted-foreground shrink-0">Task type:</span>
            <select
              value={hp.task_type}
              onChange={e => setHp({ task_type: e.target.value })}
              className="text-xs h-7 rounded-md border border-input bg-card px-2 text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            >
              {TASK_TYPES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
            </select>
          </div>
        </CardContent>
      </Card>

      {/* Model selection */}
      <div className="space-y-3">
        <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Choose a Base Model</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {MODELS.map(model => {
            const isSelected  = selectedModel === model.model_id
            const isSuggested = model.model_id === suggestedId
            return (
              <button key={model.model_id} onClick={() => onModel(model.model_id)}
                className={`text-left p-4 rounded-xl border transition-all ${
                  isSelected ? "border-primary bg-primary/5 ring-1 ring-primary"
                              : "border-border hover:border-primary/40 hover:bg-secondary/30"
                }`}>
                <div className="flex items-start justify-between gap-2 mb-1.5">
                  <p className="text-sm font-semibold">{model.display_name}</p>
                  <div className="flex gap-1 shrink-0">
                    {isSuggested && (
                      <Badge className="text-[10px] h-4 px-1.5 bg-primary/15 text-primary border-primary/25">
                        Recommended
                      </Badge>
                    )}
                    <Badge variant="secondary" className={`text-[10px] h-4 px-1.5 ${model.quality_tier === "excellent" ? "text-emerald-400" : ""}`}>
                      {model.quality_tier}
                    </Badge>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed mb-2">{model.description}</p>
                <div className="flex items-center gap-2 text-[10px] text-muted-foreground/70">
                  <span className="font-mono">{model.param_count} params</span>
                  <span>·</span>
                  <span>{model.inference_speed} inference</span>
                  {model.lora_compatible && <><span>·</span><span className="text-primary">LoRA ✓</span></>}
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {/* Hyperparameter playground */}
      <div className="space-y-3">
        <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Training Configuration</h2>
        <Card>
          <CardContent className="pt-5 space-y-5">
            {/* Training approach */}
            <div className="space-y-2">
              <label className="text-xs font-medium">Training Approach</label>
              <div className="grid grid-cols-3 gap-2">
                {APPROACH_OPTIONS.map(opt => (
                  <button key={opt.value} onClick={() => setHp({ training_approach: opt.value })}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      hp.training_approach === opt.value ? "border-primary bg-primary/5" : "border-border hover:border-primary/40"
                    }`}>
                    <p className="text-xs font-medium">{opt.label}</p>
                    <p className="text-[10px] text-muted-foreground mt-0.5 leading-tight">{opt.desc}</p>
                  </button>
                ))}
              </div>
            </div>

            <Separator />

            <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
              {/* Epochs */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-xs font-medium">Epochs</label>
                  <span className="text-xs font-mono font-semibold text-primary">{hp.num_epochs}</span>
                </div>
                <input type="range" min={1} max={20} step={1} value={hp.num_epochs}
                  onChange={e => setHp({ num_epochs: Number(e.target.value) })}
                  className="w-full h-1.5 cursor-pointer accent-violet-500" />
                <div className="flex justify-between text-[10px] text-muted-foreground/60">
                  <span>1</span><span>20</span>
                </div>
              </div>

              {/* Batch size */}
              <div className="space-y-2">
                <label className="text-xs font-medium">Batch Size</label>
                <select value={hp.batch_size} onChange={e => setHp({ batch_size: Number(e.target.value) })}
                  className="w-full h-8 rounded-md border border-input bg-card px-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary">
                  {[4, 8, 16, 32, 64].map(v => <option key={v} value={v}>{v}</option>)}
                </select>
              </div>

              {/* Max length */}
              <div className="space-y-2">
                <label className="text-xs font-medium">Max Length</label>
                <select value={hp.max_length} onChange={e => setHp({ max_length: Number(e.target.value) })}
                  className="w-full h-8 rounded-md border border-input bg-card px-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary">
                  {[64, 128, 256, 512].map(v => <option key={v} value={v}>{v} tokens</option>)}
                </select>
              </div>

              {/* Learning rate */}
              <div className="space-y-2">
                <label className="text-xs font-medium">Learning Rate</label>
                <select value={hp.learning_rate} onChange={e => setHp({ learning_rate: Number(e.target.value) })}
                  className="w-full h-8 rounded-md border border-input bg-card px-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary">
                  {LR_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
            </div>

            {/* LoRA rank */}
            {isLoRA && (
              <>
                <Separator />
                <div className="space-y-2 max-w-xs">
                  <div className="flex justify-between">
                    <label className="text-xs font-medium">LoRA Rank (r)</label>
                    <span className="text-xs font-mono font-semibold text-primary">{hp.lora_r}</span>
                  </div>
                  <input type="range" min={4} max={64} step={4} value={hp.lora_r}
                    onChange={e => setHp({ lora_r: Number(e.target.value) })}
                    className="w-full h-1.5 cursor-pointer accent-violet-500" />
                  <p className="text-[10px] text-muted-foreground">Higher rank = more parameters adapted = higher quality but slower</p>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Launch row */}
      <div className="flex items-center justify-between gap-4 pt-1">
        <div className="text-xs text-muted-foreground space-y-0.5">
          <p>Model: <span className="font-mono text-foreground">{selectedModel}</span></p>
          <p>
            {hp.training_approach} · {hp.num_epochs} epochs · lr {hp.learning_rate} · batch {hp.batch_size}
          </p>
        </div>
        <Button onClick={onStart} disabled={starting} size="lg" className="gap-2 shrink-0">
          {starting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          Start Training
        </Button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 4 — Training stream
// ---------------------------------------------------------------------------

function TrainingPanel({ messages, streaming }: { messages: AgentMessage[]; streaming: boolean }) {
  const TOTAL = 7
  const completed = messages.filter(m => m.output.final !== false && m.success).map(m => m.agent)
  const progress = (completed.length / TOTAL) * 100
  const activeAgent = streaming ? messages[messages.length - 1]?.agent : null

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="pt-5 pb-4 space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">{streaming ? "Pipeline running…" : "Pipeline complete"}</span>
            <span className="text-muted-foreground text-xs">{completed.length} / {TOTAL} agents</span>
          </div>
          <Progress value={progress} className="h-2" />
          <div className="flex gap-2 flex-wrap">
            {["Intent","Data","Clean","Model","Train","Eval","Deploy"].map(name => {
              const done   = completed.includes(name)
              const active = activeAgent === name
              return (
                <div key={name} className={`flex items-center gap-1.5 text-xs px-2 py-1 rounded-md ${
                  done   ? "bg-emerald-500/10 text-emerald-400" :
                  active ? "bg-primary/10 text-primary animate-pulse" :
                           "bg-secondary/50 text-muted-foreground"
                }`}>
                  {done   ? <CheckCircle2 className="h-3 w-3" /> :
                   active ? <Loader2 className="h-3 w-3 animate-spin" /> :
                            <div className="h-3 w-3 rounded-full border border-current opacity-30" />}
                  {name}
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>

      <div className="space-y-3">
        {messages.map((msg, i) => {
          const AgentIcon = AGENT_ICONS[msg.agent] ?? BrainCircuit
          const isKeepalive = msg.output.final === false
          return (
            <div key={i} className={`rounded-xl p-4 border transition-opacity ${
              msg.success ? "bg-secondary/50 border-border" : "bg-destructive/5 border-destructive/30"
            } ${isKeepalive ? "opacity-50" : ""}`}>
              <div className="flex items-center gap-2 mb-2">
                <div className={`flex h-6 w-6 items-center justify-center rounded-md ${msg.success ? "bg-primary/10" : "bg-destructive/10"}`}>
                  <AgentIcon className={`h-3.5 w-3.5 ${msg.success ? "text-primary" : "text-destructive"}`} />
                </div>
                <Badge variant={msg.success ? "outline" : "destructive"} className="text-[10px] h-4 px-1.5">
                  {msg.agent} Agent
                </Badge>
                <span className="ml-auto">
                  {msg.success
                    ? <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                    : <XCircle className="h-3.5 w-3.5 text-destructive" />
                  }
                </span>
              </div>
              <p className="text-sm leading-relaxed whitespace-pre-wrap pl-8">{msg.message}</p>
            </div>
          )
        })}
        {streaming && (
          <div className="flex items-center gap-3 text-sm text-muted-foreground animate-pulse">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            Processing…
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 5 — Results
// ---------------------------------------------------------------------------

function ResultsPanel({ messages, runId }: { messages: AgentMessage[]; runId: string | null }) {
  const router = useRouter()
  const success = messages.length > 0 && messages.every(m => m.success)
  const evalMsg  = messages.find(m => m.agent === "Eval")
  const trainMsg = messages.find(m => m.agent === "Train" && m.output.final !== false)
  const metrics  = (evalMsg?.output ?? trainMsg?.output ?? {}) as Record<string, unknown>

  const accuracy = typeof metrics.accuracy === "number" ? (metrics.accuracy * 100).toFixed(1) : null
  const f1       = typeof metrics.f1       === "number" ? metrics.f1.toFixed(3)               : null
  const grade    = typeof metrics.evaluation_grade === "string" ? metrics.evaluation_grade    : null

  const GRADE_COLORS: Record<string, string> = {
    A: "text-emerald-400 border-emerald-500 bg-emerald-500/10",
    B: "text-blue-400 border-blue-500 bg-blue-500/10",
    C: "text-yellow-400 border-yellow-500 bg-yellow-500/10",
    D: "text-orange-400 border-orange-500 bg-orange-500/10",
    F: "text-destructive border-destructive bg-destructive/10",
  }
  const gradeStyle = GRADE_COLORS[grade ?? ""] ?? "text-muted-foreground border-border bg-secondary"

  return (
    <div className="flex flex-col items-center gap-6 py-8 text-center">
      {success ? (
        <>
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500/10 border-2 border-emerald-500">
            <CheckCircle2 className="h-8 w-8 text-emerald-400" />
          </div>
          <div className="space-y-1">
            <h2 className="text-2xl font-bold">Training Complete</h2>
            <p className="text-sm text-muted-foreground">Your model is trained and ready.</p>
          </div>

          {grade && (
            <div className={`flex h-14 w-14 items-center justify-center rounded-xl border-2 font-bold text-2xl ${gradeStyle}`}>
              {grade}
            </div>
          )}

          {(accuracy || f1) && (
            <div className="flex gap-10 justify-center">
              {accuracy && (
                <div>
                  <p className="text-4xl font-bold text-primary">{accuracy}%</p>
                  <p className="text-xs text-muted-foreground mt-1">Accuracy</p>
                </div>
              )}
              {f1 && (
                <div>
                  <p className="text-4xl font-bold">{f1}</p>
                  <p className="text-xs text-muted-foreground mt-1">F1 Score</p>
                </div>
              )}
            </div>
          )}

          {runId && (
            <Button onClick={() => router.push(`/runs/${runId}`)} className="gap-2 mt-2">
              View Full Report & Inference Playground <ArrowRight className="h-4 w-4" />
            </Button>
          )}
        </>
      ) : (
        <>
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-destructive/10 border-2 border-destructive">
            <XCircle className="h-8 w-8 text-destructive" />
          </div>
          <div className="space-y-1">
            <h2 className="text-2xl font-bold">Pipeline Failed</h2>
            <p className="text-sm text-muted-foreground max-w-md">
              {messages.find(m => !m.success)?.message ?? "An error occurred. See the training step for details."}
            </p>
          </div>
          <Button variant="outline" onClick={() => window.location.reload()} className="gap-2 mt-2">
            Start Over
          </Button>
        </>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main TrainClient
// ---------------------------------------------------------------------------

export function TrainClient() {
  const supabase = createClient()
  const fileRef  = useRef<HTMLInputElement>(null)

  const [step,          setStep]          = useState(0)
  const [upload,        setUpload]        = useState<UploadResult | null>(null)
  const [uploading,     setUploading]     = useState(false)
  const [dragOver,      setDragOver]      = useState(false)
  const [intent,        setIntent]        = useState("")
  const [selectedModel, setSelectedModel] = useState("roberta-base")
  const [hyperparams,   setHyperparams]   = useState<HyperParams>(DEFAULT_HYPERPARAMS)
  const [messages,      setMessages]      = useState<AgentMessage[]>([])
  const [streaming,     setStreaming]     = useState(false)
  const [runId,         setRunId]         = useState<string | null>(null)

  const handleFile = useCallback(async (file: File) => {
    setUploading(true)
    const form = new FormData()
    form.append("file", file)
    try {
      const res = await fetch(`${API_URL}/upload`, { method: "POST", body: form })
      if (!res.ok) throw new Error(await res.text())
      const data: UploadResult = await res.json()
      setUpload(data)
      setStep(1)
      toast.success(`Loaded ${data.rows.toLocaleString()} rows from ${data.filename}`)
    } catch (err) {
      toast.error(`Upload failed: ${err}`)
    } finally {
      setUploading(false)
    }
  }, [])

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  async function startTraining() {
    if (!upload) return
    setStep(3)
    setStreaming(true)
    setMessages([])

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { data: { user } } = await (supabase as any).auth.getUser()
    let newRunId: string | null = null
    if (user) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: run } = await (supabase as any).from("runs").insert({
        user_id:          user.id,
        status:           "running",
        dataset_filename: upload.filename,
        dataset_rows:     upload.rows,
      }).select().single()
      newRunId = run?.id ?? null
      setRunId(newRunId)
    }

    const userIntent = intent.trim() ||
      `Train a ${hyperparams.task_type.replace(/_/g, " ")} model on this dataset`

    const overrides = {
      model_id:          selectedModel,
      num_epochs:        hyperparams.num_epochs,
      learning_rate:     hyperparams.learning_rate,
      batch_size:        hyperparams.batch_size,
      max_length:        hyperparams.max_length,
      training_approach: hyperparams.training_approach,
      lora_r:            hyperparams.lora_r,
      weight_decay:      hyperparams.weight_decay,
      warmup_ratio:      hyperparams.warmup_ratio,
    }

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          message:                  userIntent,
          file_id:                  upload.file_id,
          run_id:                   newRunId,
          hyperparameter_overrides: overrides,
        }),
      })
      if (!res.ok) throw new Error(await res.text())

      const reader  = res.body!.getReader()
      const decoder = new TextDecoder()
      let buf = ""
      const allMessages: AgentMessage[] = []

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split("\n")
        buf = lines.pop() ?? ""
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          const payload = line.slice(6).trim()
          if (payload === "[DONE]") break
          try {
            const msg: AgentMessage = JSON.parse(payload)
            const lastIdx = allMessages.findLastIndex(m => m.agent === msg.agent)
            const lastWasKeepalive = lastIdx >= 0 && allMessages[lastIdx].output.final === false
            if (lastWasKeepalive) {
              allMessages[lastIdx] = msg
            } else {
              allMessages.push(msg)
            }
            setMessages([...allMessages])
          } catch { /* ignore parse errors */ }
        }
      }

      // Persist to Supabase
      const pipelineSuccess = allMessages.every(m => m.success)
      if (newRunId) {
        const intentOut  = (allMessages.find(m => m.agent === "Intent")?.output ?? {}) as Record<string, unknown>
        const modelOut   = (allMessages.find(m => m.agent === "Model")?.output  ?? {}) as Record<string, unknown>
        const trainOut   = (allMessages.find(m => m.agent === "Train" && m.output.final !== false)?.output ?? {}) as Record<string, unknown>
        const evalOut    = (allMessages.find(m => m.agent === "Eval")?.output    ?? {}) as Record<string, unknown>
        const deployOut  = (allMessages.find(m => m.agent === "Deploy")?.output  ?? {}) as Record<string, unknown>
        const mSrc       = Object.keys(evalOut).length > 0 ? evalOut : trainOut
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (supabase as any).from("runs").update({
          status:        pipelineSuccess ? "completed" : "failed",
          task_type:     intentOut.task_type   as string ?? null,
          model_id:      (modelOut.base_model  ?? intentOut.base_model_hint) as string ?? null,
          intent_spec:   intentOut,
          model_recipe:  modelOut,
          metrics: {
            accuracy:         mSrc.accuracy         ?? null,
            f1:               mSrc.f1               ?? null,
            precision:        mSrc.precision        ?? null,
            recall:           mSrc.recall           ?? null,
            per_class_f1:     mSrc.per_class_f1     ?? {},
            evaluation_grade: evalOut.evaluation_grade ?? null,
            summary:          evalOut.summary          ?? null,
            strengths:        evalOut.strengths        ?? [],
            concerns:         evalOut.concerns         ?? [],
            next_steps:       evalOut.next_steps       ?? [],
          },
          artifact_path: trainOut.model_path   as string ?? null,
          hf_model_url:  deployOut.hf_url      as string ?? null,
          hf_repo_id:    deployOut.hf_repo_id  as string ?? null,
          model_card:    deployOut.model_card  as string ?? null,
          deploy_status: deployOut.status      as string ?? "not_deployed",
          completed_at:  new Date().toISOString(),
          error_message: pipelineSuccess ? null : (allMessages.find(m => !m.success)?.message ?? null),
        }).eq("id", newRunId)
      }

      setStep(4)
      if (!pipelineSuccess) toast.error("Pipeline encountered an error — see details below")
    } catch (err) {
      toast.error(`Pipeline error: ${err}`)
      setMessages(prev => [...prev, { agent: "System", success: false, message: String(err), output: {} }])
      if (newRunId) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (supabase as any).from("runs").update({ status: "failed", error_message: String(err) }).eq("id", newRunId)
      }
      setStep(4)
    } finally {
      setStreaming(false)
    }
  }

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold mb-1">New Training Run</h1>
        <p className="text-sm text-muted-foreground">Upload a dataset, analyze it, configure training, and watch your model train.</p>
      </div>

      {/* Wizard indicator */}
      <div className="flex items-center">
        {WIZARD_STEPS.map((s, i) => (
          <div key={s.key} className="flex items-center flex-1 min-w-0">
            <div className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors shrink-0 ${
              i === step ? "bg-primary/10 text-primary" :
              i < step   ? "text-emerald-400" :
                           "text-muted-foreground"
            }`}>
              {i < step
                ? <CheckCircle2 className="h-4 w-4 shrink-0" />
                : <s.icon className="h-4 w-4 shrink-0" />
              }
              <span className="text-xs font-medium hidden sm:block">{s.label}</span>
            </div>
            {i < WIZARD_STEPS.length - 1 && (
              <div className={`flex-1 h-px mx-1 transition-colors min-w-[8px] ${i < step ? "bg-emerald-500/40" : "bg-border"}`} />
            )}
          </div>
        ))}
      </div>

      {/* Step 0: Upload */}
      {step === 0 && (
        <div className="space-y-6">
          <input ref={fileRef} type="file" accept=".csv,.json,.jsonl" className="hidden"
            onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
          <div
            onDrop={onDrop}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onClick={() => !uploading && fileRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-16 flex flex-col items-center gap-4 cursor-pointer transition-all select-none ${
              dragOver  ? "border-primary bg-primary/5 scale-[1.01]" :
              uploading ? "border-primary/40 cursor-default" :
                          "border-border hover:border-primary/50 hover:bg-secondary/20"
            }`}
          >
            {uploading
              ? <Loader2 className="h-10 w-10 text-primary animate-spin" />
              : <CloudUpload className="h-10 w-10 text-muted-foreground" />
            }
            <div className="text-center">
              <p className="text-base font-semibold">{uploading ? "Analyzing your dataset…" : "Drop your dataset here"}</p>
              <p className="text-sm text-muted-foreground mt-1">CSV, JSON, or JSONL · up to 50 MB</p>
            </div>
            {!uploading && (
              <Button variant="outline" size="sm" onClick={e => { e.stopPropagation(); fileRef.current?.click() }}>
                Browse files
              </Button>
            )}
          </div>

          <Card className="bg-secondary/20 border-dashed">
            <CardContent className="pt-5">
              <div className="grid grid-cols-3 gap-6 text-xs text-muted-foreground">
                <div><p className="font-medium text-foreground mb-1">CSV</p><p>Comma-separated with a header row. Best supported.</p></div>
                <div><p className="font-medium text-foreground mb-1">JSON</p><p>Array of objects or newline-delimited records.</p></div>
                <div><p className="font-medium text-foreground mb-1">JSONL</p><p>One JSON object per line — common ML export format.</p></div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Step 1: Analyze */}
      {step === 1 && upload && (
        <DataAnalysisPanel upload={upload} onNext={() => setStep(2)} />
      )}

      {/* Step 2: Configure */}
      {step === 2 && upload && (
        <ConfigurePanel
          upload={upload}
          hyperparams={hyperparams}
          selectedModel={selectedModel}
          intent={intent}
          onHyperparams={setHyperparams}
          onModel={setSelectedModel}
          onIntent={setIntent}
          onStart={startTraining}
          starting={streaming}
        />
      )}

      {/* Step 3: Training */}
      {step === 3 && (
        <TrainingPanel messages={messages} streaming={streaming} />
      )}

      {/* Step 4: Results */}
      {step === 4 && (
        <ResultsPanel messages={messages} runId={runId} />
      )}
    </div>
  )
}
