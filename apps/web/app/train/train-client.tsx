"use client"

import {
  useState, useReducer, useRef, useCallback, useEffect,
} from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import {
  CloudUpload, Loader2, CheckCircle2, XCircle, AlertTriangle,
  Database, BrainCircuit, Cpu, BarChart3, Rocket, FileText,
  Settings2, Play, ChevronRight, Plus, Clock, Zap,
  TrendingUp, Download, ArrowRight, Pencil, Check,
} from "lucide-react"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartsTooltip, ResponsiveContainer, Cell,
} from "recharts"
import { toast } from "sonner"
import { createClient } from "@/lib/supabase/client"

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
const LS_KEY = "modelforge_sessions_v2"

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

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

type SetupSubstep = "upload" | "analyzing" | "configure"
type SessionStatus = "setup" | "training" | "completed" | "failed"

interface TrainingSession {
  id: string
  label: string
  status: SessionStatus
  setupSubstep: SetupSubstep
  createdAt: number
  uploadResult: UploadResult | null
  intent: string
  selectedModelId: string
  hyperParams: HyperParams
  runId: string | null
  grade: string | null
  accuracy: number | null
  f1: number | null
  artifactPath: string | null
  errorMessage: string | null
}

type Action =
  | { type: "HYDRATE"; sessions: TrainingSession[]; activeId: string | null }
  | { type: "ADD"; session: TrainingSession }
  | { type: "SELECT"; id: string }
  | { type: "UPDATE"; id: string; patch: Partial<TrainingSession> }

interface WsState {
  sessions: TrainingSession[]
  activeId: string | null
}

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

const DEFAULT_PARAMS: HyperParams = {
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

const APPROACH_OPTIONS = [
  { value: "full_finetune", label: "Full Fine-tune", desc: "All weights updated — best quality" },
  { value: "lora",          label: "LoRA",           desc: "Parameter-efficient, faster" },
  { value: "qlora",         label: "QLoRA",          desc: "4-bit quantized — memory-efficient" },
]

const LR_OPTIONS = [
  { value: 1e-5, label: "1e-5  (conservative)" },
  { value: 2e-5, label: "2e-5  (recommended)" },
  { value: 3e-5, label: "3e-5  (aggressive)" },
  { value: 5e-5, label: "5e-5  (very aggressive)" },
]

const TASK_TYPES = [
  { value: "text_classification", label: "Text Classification" },
  { value: "sentiment_analysis",  label: "Sentiment Analysis" },
  { value: "ner",                 label: "Named Entity Recognition" },
  { value: "question_answering",  label: "Question Answering" },
]

const BAR_COLORS = ["#7c6fcd","#5ea5f8","#4ade80","#f97316","#e879f9","#facc15","#38bdf8","#fb7185"]

const AGENT_ICONS: Record<string, React.ElementType> = {
  Intent: BrainCircuit, Data: Database, Clean: Settings2, Model: Cpu,
  Train: Cpu, Eval: BarChart3, Deploy: Rocket, System: XCircle,
}

const AGENT_ORDER = ["Intent","Data","Clean","Model","Train","Eval","Deploy"]

const STATUS_BADGE: Record<SessionStatus, { label: string; cls: string }> = {
  setup:     { label: "Setup",     cls: "bg-secondary text-muted-foreground" },
  training:  { label: "Training",  cls: "bg-primary/15 text-primary animate-pulse" },
  completed: { label: "Done",      cls: "bg-emerald-500/15 text-emerald-500" },
  failed:    { label: "Failed",    cls: "bg-destructive/15 text-destructive" },
}

const GRADE_STYLES: Record<string, string> = {
  A: "text-emerald-400 border-emerald-500 bg-emerald-500/10",
  B: "text-blue-400 border-blue-500 bg-blue-500/10",
  C: "text-yellow-400 border-yellow-500 bg-yellow-500/10",
  D: "text-orange-400 border-orange-500 bg-orange-500/10",
  F: "text-destructive border-destructive bg-destructive/10",
}

// ──────────────────────────────────────────────────────────────────────────────
// State helpers
// ──────────────────────────────────────────────────────────────────────────────

function makeSession(overrides: Partial<TrainingSession> = {}): TrainingSession {
  return {
    id: crypto.randomUUID(),
    label: "New Session",
    status: "setup",
    setupSubstep: "upload",
    createdAt: Date.now(),
    uploadResult: null,
    intent: "",
    selectedModelId: "roberta-base",
    hyperParams: { ...DEFAULT_PARAMS },
    runId: null,
    grade: null,
    accuracy: null,
    f1: null,
    artifactPath: null,
    errorMessage: null,
    ...overrides,
  }
}

function reducer(state: WsState, action: Action): WsState {
  switch (action.type) {
    case "HYDRATE":
      return { sessions: action.sessions, activeId: action.activeId }
    case "ADD":
      return { ...state, sessions: [...state.sessions, action.session] }
    case "SELECT":
      return { ...state, activeId: action.id }
    case "UPDATE":
      return {
        ...state,
        sessions: state.sessions.map(s =>
          s.id === action.id ? { ...s, ...action.patch } : s
        ),
      }
    default:
      return state
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Session Sidebar
// ──────────────────────────────────────────────────────────────────────────────

function SessionSidebar({
  sessions, activeId, onSelect, onAdd,
}: {
  sessions: TrainingSession[]
  activeId: string | null
  onSelect: (id: string) => void
  onAdd: () => void
}) {
  return (
    <div className="w-52 shrink-0 flex flex-col border-r border-border bg-card/50 h-full">
      <div className="flex items-center justify-between px-3 py-3 border-b border-border">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Sessions</span>
        <button
          onClick={onAdd}
          className="h-6 w-6 flex items-center justify-center rounded-md hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
          title="New session"
        >
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {sessions.length === 0 && (
          <p className="text-xs text-muted-foreground px-2 py-3 text-center">
            No sessions yet.<br />Click + to start.
          </p>
        )}
        {sessions.map(s => {
          const badge = STATUS_BADGE[s.status]
          const isActive = s.id === activeId
          return (
            <button
              key={s.id}
              onClick={() => onSelect(s.id)}
              className={`w-full text-left rounded-lg px-2.5 py-2 transition-colors ${
                isActive ? "bg-primary/10 text-primary" : "hover:bg-secondary text-foreground"
              }`}
            >
              <div className="flex items-center justify-between gap-1 mb-0.5">
                <span className="text-xs font-medium truncate">{s.label}</span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium shrink-0 ${badge.cls}`}>
                  {badge.label}
                </span>
              </div>
              {s.uploadResult && (
                <p className="text-[10px] text-muted-foreground truncate">{s.uploadResult.filename}</p>
              )}
              {!s.uploadResult && (
                <p className="text-[10px] text-muted-foreground">No dataset</p>
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Data Analysis Panel (shown in Setup column after upload)
// ──────────────────────────────────────────────────────────────────────────────

function DataAnalysisPanel({ upload, onNext }: { upload: UploadResult; onNext: () => void }) {
  const classData = Object.entries(upload.class_distribution)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([name, value]) => ({ name: name.length > 14 ? name.slice(0, 12) + "…" : name, value }))

  const histData = upload.text_length_histogram.map(b => ({
    name: String(b.bin_start),
    count: b.count,
    range: `${b.bin_start}–${b.bin_end}`,
  }))

  const stats = "mean" in upload.text_length_stats ? (upload.text_length_stats as TextLengthStats) : null
  const chartH = Math.max(classData.length * 26, 140)

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        {([
          { label: "Rows",    value: upload.rows.toLocaleString(), icon: Database },
          { label: "Labels",  value: String(upload.unique_labels.length), icon: BarChart3 },
          { label: "Columns", value: String(upload.columns.length), icon: FileText },
          { label: "Dupes",   value: String(upload.duplicate_count), icon: AlertTriangle },
        ] as const).map(({ label, value, icon: Icon }) => (
          <Card key={label}>
            <CardContent className="p-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[10px] text-muted-foreground">{label}</p>
                  <p className="text-lg font-bold mt-0.5">{value}</p>
                </div>
                <div className="h-7 w-7 flex items-center justify-center rounded-md bg-primary/10">
                  <Icon className="h-3.5 w-3.5 text-primary" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {upload.data_warnings.length > 0 && (
        <div className="space-y-1.5">
          {upload.data_warnings.map((w, i) => (
            <div key={i} className="flex gap-2 p-2.5 rounded-lg bg-yellow-500/5 border border-yellow-500/20 text-xs">
              <AlertTriangle className="h-3.5 w-3.5 text-yellow-400 mt-0.5 shrink-0" />
              <span className="text-muted-foreground">{w}</span>
            </div>
          ))}
        </div>
      )}

      {classData.length > 0 && (
        <Card>
          <CardHeader className="pb-2 pt-3 px-4">
            <CardTitle className="text-xs flex items-center gap-1.5">
              <BarChart3 className="h-3.5 w-3.5 text-primary" /> Class Distribution
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-4">
            <ResponsiveContainer width="100%" height={chartH}>
              <BarChart data={classData} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(224 18% 14%)" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 10, fill: "hsl(215 16% 55%)" }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: "hsl(215 16% 55%)" }} axisLine={false} tickLine={false} width={70} />
                <RechartsTooltip
                  formatter={(v) => [v, "Samples"]}
                  contentStyle={{ background: "hsl(224 20% 9%)", border: "1px solid hsl(224 18% 14%)", borderRadius: "8px", fontSize: "11px" }}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={18}>
                  {classData.map((_, i) => <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {histData.length > 0 && (
        <Card>
          <CardHeader className="pb-2 pt-3 px-4">
            <CardTitle className="text-xs flex items-center gap-1.5">
              <BarChart3 className="h-3.5 w-3.5 text-primary" /> Text Length Distribution
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-4">
            <ResponsiveContainer width="100%" height={140}>
              <BarChart data={histData} margin={{ top: 4, right: 8, bottom: 18, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(224 18% 14%)" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 9, fill: "hsl(215 16% 55%)" }} axisLine={false} tickLine={false}
                  label={{ value: "chars", position: "insideBottom", offset: -12, fontSize: 9, fill: "hsl(215 16% 55%)" }} />
                <YAxis tick={{ fontSize: 10, fill: "hsl(215 16% 55%)" }} axisLine={false} tickLine={false} width={28} />
                <RechartsTooltip
                  formatter={(v, _, p) => [v, `${(p.payload as { range: string }).range} chars`]}
                  contentStyle={{ background: "hsl(224 20% 9%)", border: "1px solid hsl(224 18% 14%)", borderRadius: "8px", fontSize: "11px" }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]} fill="hsl(245 58% 63%)" maxBarSize={28} />
              </BarChart>
            </ResponsiveContainer>
            {stats && (
              <div className="flex gap-4 mt-2 pt-2 border-t border-border flex-wrap text-[10px]">
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

      {/* Column tags */}
      <Card>
        <CardContent className="p-3">
          <p className="text-[10px] font-medium text-muted-foreground mb-2">Detected Columns</p>
          <div className="flex flex-wrap gap-1.5">
            {upload.columns.map(col => {
              const isText  = upload.text_columns.includes(col)
              const isLabel = upload.label_columns.includes(col)
              return (
                <span key={col} className={`text-[10px] px-2 py-0.5 rounded border font-medium ${
                  isText  ? "bg-primary/10 text-primary border-primary/25" :
                  isLabel ? "bg-emerald-500/10 text-emerald-500 border-emerald-500/25" :
                            "bg-secondary text-muted-foreground border-border"
                }`}>
                  {col}
                  {isText  && <span className="opacity-60 ml-1">TEXT</span>}
                  {isLabel && <span className="opacity-60 ml-1">LABEL</span>}
                </span>
              )
            })}
          </div>
        </CardContent>
      </Card>

      <Button onClick={onNext} size="sm" className="w-full gap-2">
        Configure Training <ChevronRight className="h-3.5 w-3.5" />
      </Button>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Configure Panel (model picker + hyperparams in Setup column)
// ──────────────────────────────────────────────────────────────────────────────

const QUICK_MODELS = [
  { model_id: "distilbert-base-uncased", display_name: "DistilBERT (67M)", param_count: "67M", quality_tier: "good",      inference_speed: "fast",   lora_compatible: true,  desc: "Lightweight & fast, great for small datasets." },
  { model_id: "roberta-base",            display_name: "RoBERTa Base (125M)", param_count: "125M", quality_tier: "excellent", inference_speed: "medium", lora_compatible: true,  desc: "Robustly optimized BERT. Best all-round choice." },
  { model_id: "bert-base-uncased",       display_name: "BERT Base (110M)", param_count: "110M", quality_tier: "good",      inference_speed: "medium", lora_compatible: true,  desc: "Google's original BERT. Solid baseline." },
  { model_id: "microsoft/deberta-v3-small", display_name: "DeBERTa v3 Small (142M)", param_count: "142M", quality_tier: "excellent", inference_speed: "medium", lora_compatible: true,  desc: "Microsoft's top small model. Leads benchmarks." },
  { model_id: "albert-base-v2",          display_name: "ALBERT Base (11M)", param_count: "11M", quality_tier: "fast",      inference_speed: "fast",   lora_compatible: true,  desc: "Ultra-lightweight. For edge / quick experiments." },
  { model_id: "distilroberta-base",      display_name: "DistilRoBERTa (82M)", param_count: "82M", quality_tier: "good",      inference_speed: "fast",   lora_compatible: true,  desc: "Best speed/accuracy tradeoff." },
]

function ConfigurePanel({
  session, onUpdate, onStart, starting,
}: {
  session: TrainingSession
  onUpdate: (patch: Partial<TrainingSession>) => void
  onStart: () => void
  starting: boolean
}) {
  const hp = session.hyperParams
  const setHp = (p: Partial<HyperParams>) => onUpdate({ hyperParams: { ...hp, ...p } })
  const isLoRA = hp.training_approach === "lora" || hp.training_approach === "qlora"
  const suggestedId = (session.uploadResult?.rows ?? 0) < 500 ? "distilbert-base-uncased" : "roberta-base"

  return (
    <div className="space-y-5">
      {/* Task description */}
      <div className="space-y-2">
        <label className="text-xs font-medium text-muted-foreground">Describe your task (optional)</label>
        <textarea
          value={session.intent}
          onChange={e => onUpdate({ intent: e.target.value })}
          placeholder={`e.g. "Classify support tickets by urgency: low, medium, high"`}
          rows={2}
          className="w-full resize-none rounded-lg border border-input bg-card px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary transition-colors"
        />
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Task:</span>
          <select
            value={hp.task_type}
            onChange={e => setHp({ task_type: e.target.value })}
            className="text-xs h-7 rounded-md border border-input bg-card px-2 text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
          >
            {TASK_TYPES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
          </select>
        </div>
      </div>

      <Separator />

      {/* Model picker */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Base Model</label>
          <a href="/models" className="text-[10px] text-primary hover:underline">Browse full catalog →</a>
        </div>
        <div className="space-y-2">
          {QUICK_MODELS.map(m => {
            const isSelected  = session.selectedModelId === m.model_id
            const isSuggested = m.model_id === suggestedId
            return (
              <button
                key={m.model_id}
                onClick={() => onUpdate({ selectedModelId: m.model_id })}
                className={`w-full text-left p-3 rounded-xl border transition-all ${
                  isSelected ? "border-primary bg-primary/5 ring-1 ring-primary"
                             : "border-border hover:border-primary/40 hover:bg-secondary/30"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <p className="text-xs font-semibold">{m.display_name}</p>
                  <div className="flex gap-1 shrink-0">
                    {isSuggested && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-primary/15 text-primary border border-primary/25 font-medium">
                        Recommended
                      </span>
                    )}
                    <span className={`text-[9px] px-1.5 py-0.5 rounded border font-medium ${
                      m.quality_tier === "excellent" ? "bg-violet-500/10 text-violet-500 border-violet-500/25" : "bg-secondary text-muted-foreground"
                    }`}>
                      {m.quality_tier}
                    </span>
                  </div>
                </div>
                <p className="text-[10px] text-muted-foreground leading-relaxed">{m.desc}</p>
                <div className="flex gap-2 mt-1 text-[9px] text-muted-foreground/60">
                  <span className="font-mono">{m.param_count}</span>
                  <span>·</span>
                  <span>{m.inference_speed} inference</span>
                  {m.lora_compatible && <><span>·</span><span className="text-primary">LoRA ✓</span></>}
                </div>
              </button>
            )
          })}
        </div>
      </div>

      <Separator />

      {/* Hyperparams */}
      <div className="space-y-4">
        <label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Training Config</label>

        {/* Approach */}
        <div className="space-y-1.5">
          <p className="text-xs font-medium">Approach</p>
          <div className="grid grid-cols-3 gap-2">
            {APPROACH_OPTIONS.map(opt => (
              <button
                key={opt.value}
                onClick={() => setHp({ training_approach: opt.value })}
                className={`p-2.5 rounded-lg border text-left transition-all ${
                  hp.training_approach === opt.value ? "border-primary bg-primary/5" : "border-border hover:border-primary/40"
                }`}
              >
                <p className="text-[10px] font-semibold">{opt.label}</p>
                <p className="text-[9px] text-muted-foreground mt-0.5 leading-tight">{opt.desc}</p>
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          {/* Epochs */}
          <div className="space-y-1.5">
            <div className="flex justify-between">
              <label className="text-xs font-medium">Epochs</label>
              <span className="text-xs font-mono font-semibold text-primary">{hp.num_epochs}</span>
            </div>
            <input type="range" min={1} max={20} step={1} value={hp.num_epochs}
              onChange={e => setHp({ num_epochs: Number(e.target.value) })}
              className="w-full h-1.5 cursor-pointer accent-violet-500" />
          </div>

          {/* Batch size */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium">Batch Size</label>
            <select value={hp.batch_size} onChange={e => setHp({ batch_size: Number(e.target.value) })}
              className="w-full h-8 rounded-md border border-input bg-card px-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary">
              {[4, 8, 16, 32, 64].map(v => <option key={v}>{v}</option>)}
            </select>
          </div>

          {/* Max length */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium">Max Length</label>
            <select value={hp.max_length} onChange={e => setHp({ max_length: Number(e.target.value) })}
              className="w-full h-8 rounded-md border border-input bg-card px-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary">
              {[64, 128, 256, 512].map(v => <option key={v} value={v}>{v} tokens</option>)}
            </select>
          </div>

          {/* Learning rate */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium">Learning Rate</label>
            <select value={hp.learning_rate} onChange={e => setHp({ learning_rate: Number(e.target.value) })}
              className="w-full h-8 rounded-md border border-input bg-card px-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary">
              {LR_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </div>
        </div>

        {/* LoRA rank */}
        {isLoRA && (
          <div className="space-y-1.5">
            <div className="flex justify-between">
              <label className="text-xs font-medium">LoRA Rank (r)</label>
              <span className="text-xs font-mono font-semibold text-primary">{hp.lora_r}</span>
            </div>
            <input type="range" min={4} max={64} step={4} value={hp.lora_r}
              onChange={e => setHp({ lora_r: Number(e.target.value) })}
              className="w-full h-1.5 cursor-pointer accent-violet-500" />
            <p className="text-[10px] text-muted-foreground">Higher rank = more adapted params = slower but higher quality</p>
          </div>
        )}
      </div>

      {/* Config summary + launch */}
      <div className="rounded-lg bg-secondary/50 px-3 py-2.5 text-[10px] text-muted-foreground space-y-0.5">
        <p>Model: <span className="font-mono text-foreground">{session.selectedModelId}</span></p>
        <p>{hp.training_approach} · {hp.num_epochs} epochs · lr {hp.learning_rate} · batch {hp.batch_size}</p>
      </div>

      <Button onClick={onStart} disabled={starting} className="w-full gap-2">
        {starting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
        Start Training
      </Button>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Setup Column
// ──────────────────────────────────────────────────────────────────────────────

function SetupColumn({
  session, onUpdate, onStartTraining, streaming,
}: {
  session: TrainingSession
  onUpdate: (patch: Partial<TrainingSession>) => void
  onStartTraining: () => void
  streaming: boolean
}) {
  const fileRef  = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver]   = useState(false)

  const handleFile = useCallback(async (file: File) => {
    setUploading(true)
    const form = new FormData()
    form.append("file", file)
    try {
      const res = await fetch(`${API_URL}/upload`, { method: "POST", body: form })
      if (!res.ok) throw new Error(await res.text())
      const data: UploadResult = await res.json()
      const label = file.name.replace(/\.(csv|json|jsonl)$/i, "").replace(/[_-]/g, " ").slice(0, 28)
      onUpdate({ uploadResult: data, setupSubstep: "analyzing", label })
      toast.success(`Loaded ${data.rows.toLocaleString()} rows from ${data.filename}`)
    } catch (err) {
      toast.error(`Upload failed: ${err}`)
    } finally {
      setUploading(false)
    }
  }, [onUpdate])

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const isLocked = session.status === "training" || session.status === "completed"

  const subStep = session.setupSubstep

  return (
    <div className="flex flex-col h-full min-w-[280px]">
      {/* Column header */}
      <div className="px-4 py-3 border-b border-border shrink-0">
        <div className="flex items-center gap-2">
          <Settings2 className="h-4 w-4 text-primary" />
          <span className="text-sm font-semibold">Setup</span>
        </div>
        {!isLocked && (
          <div className="flex items-center gap-1.5 mt-2">
            {(["upload","analyzing","configure"] as SetupSubstep[]).map((s, i) => {
              const done   = i < ["upload","analyzing","configure"].indexOf(subStep)
              const active = s === subStep
              return (
                <div key={s} className="flex items-center gap-1">
                  <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                    active ? "bg-primary/15 text-primary" :
                    done   ? "bg-emerald-500/15 text-emerald-500" :
                             "bg-secondary text-muted-foreground"
                  }`}>
                    {s === "upload" ? "Upload" : s === "analyzing" ? "Analyze" : "Configure"}
                  </span>
                  {i < 2 && <ChevronRight className="h-2.5 w-2.5 text-muted-foreground/40" />}
                </div>
              )
            })}
          </div>
        )}
        {isLocked && session.uploadResult && (
          <p className="text-[10px] text-muted-foreground mt-1 truncate">
            {session.uploadResult.filename} · {session.uploadResult.rows.toLocaleString()} rows
          </p>
        )}
      </div>

      {/* Column content */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLocked ? (
          <SetupSummary session={session} />
        ) : subStep === "upload" ? (
          <div className="space-y-4">
            <input ref={fileRef} type="file" accept=".csv,.json,.jsonl" className="hidden"
              onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
            <div
              onDrop={onDrop}
              onDragOver={e => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onClick={() => !uploading && fileRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-10 flex flex-col items-center gap-3 cursor-pointer transition-all select-none ${
                dragOver  ? "border-primary bg-primary/5 scale-[1.01]" :
                uploading ? "border-primary/40 cursor-default" :
                            "border-border hover:border-primary/50 hover:bg-secondary/20"
              }`}
            >
              {uploading
                ? <Loader2 className="h-8 w-8 text-primary animate-spin" />
                : <CloudUpload className="h-8 w-8 text-muted-foreground" />
              }
              <div className="text-center">
                <p className="text-sm font-semibold">
                  {uploading ? "Analyzing dataset…" : "Drop dataset here"}
                </p>
                <p className="text-xs text-muted-foreground mt-1">CSV, JSON, or JSONL</p>
              </div>
              {!uploading && (
                <Button variant="outline" size="sm" onClick={e => { e.stopPropagation(); fileRef.current?.click() }}>
                  Browse files
                </Button>
              )}
            </div>
            <div className="grid grid-cols-3 gap-3 text-[10px] text-muted-foreground">
              <div><p className="font-medium text-foreground mb-0.5">CSV</p><p>Header row, comma-separated</p></div>
              <div><p className="font-medium text-foreground mb-0.5">JSON</p><p>Array of objects</p></div>
              <div><p className="font-medium text-foreground mb-0.5">JSONL</p><p>One record per line</p></div>
            </div>
          </div>
        ) : subStep === "analyzing" ? (
          <DataAnalysisPanel
            upload={session.uploadResult!}
            onNext={() => onUpdate({ setupSubstep: "configure" })}
          />
        ) : (
          <ConfigurePanel
            session={session}
            onUpdate={onUpdate}
            onStart={onStartTraining}
            starting={streaming}
          />
        )}
      </div>
    </div>
  )
}

function SetupSummary({ session }: { session: TrainingSession }) {
  const hp = session.hyperParams
  return (
    <div className="space-y-3 text-xs">
      {session.uploadResult && (
        <div className="p-3 rounded-lg bg-secondary/50 space-y-1">
          <p className="text-muted-foreground font-medium">Dataset</p>
          <p className="font-mono">{session.uploadResult.filename}</p>
          <p className="text-muted-foreground">{session.uploadResult.rows.toLocaleString()} rows · {session.uploadResult.unique_labels.length} labels</p>
        </div>
      )}
      <div className="p-3 rounded-lg bg-secondary/50 space-y-1">
        <p className="text-muted-foreground font-medium">Model</p>
        <p className="font-mono">{session.selectedModelId}</p>
      </div>
      <div className="p-3 rounded-lg bg-secondary/50 space-y-1">
        <p className="text-muted-foreground font-medium">Config</p>
        <p>{hp.training_approach} · {hp.num_epochs} epochs</p>
        <p>lr {hp.learning_rate} · batch {hp.batch_size} · max {hp.max_length} tokens</p>
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Training Column
// ──────────────────────────────────────────────────────────────────────────────

function TrainingColumn({
  session, messages, streaming,
}: {
  session: TrainingSession
  messages: AgentMessage[]
  streaming: boolean
}) {
  const completed = messages.filter(m => m.output.final !== false && m.success).map(m => m.agent)
  const progress  = (completed.length / AGENT_ORDER.length) * 100
  const activeAgent = streaming ? messages.at(-1)?.agent : null

  const isIdle = session.status === "setup"

  return (
    <div className="flex flex-col h-full min-w-[280px]">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border shrink-0">
        <div className="flex items-center gap-2">
          <Cpu className="h-4 w-4 text-primary" />
          <span className="text-sm font-semibold">Training</span>
          {streaming && (
            <span className="ml-auto text-[10px] text-primary animate-pulse flex items-center gap-1">
              <div className="h-1.5 w-1.5 rounded-full bg-primary animate-ping" />
              Live
            </span>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {isIdle ? (
          <div className="flex flex-col items-center justify-center h-40 gap-3 text-center">
            <div className="h-10 w-10 rounded-full border-2 border-dashed border-border flex items-center justify-center">
              <Clock className="h-5 w-5 text-muted-foreground/40" />
            </div>
            <p className="text-xs text-muted-foreground">Configure setup and click<br />Start Training to begin.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Progress bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span>{streaming ? "Pipeline running…" : session.status === "completed" ? "Completed" : "Failed"}</span>
                <span className="text-muted-foreground">{completed.length} / {AGENT_ORDER.length}</span>
              </div>
              <Progress value={progress} className="h-2" />
              <div className="flex gap-1.5 flex-wrap">
                {AGENT_ORDER.map(name => {
                  const done   = completed.includes(name)
                  const active = activeAgent === name
                  return (
                    <div key={name} className={`flex items-center gap-1 text-[10px] px-2 py-1 rounded-md ${
                      done   ? "bg-emerald-500/10 text-emerald-400" :
                      active ? "bg-primary/10 text-primary animate-pulse" :
                               "bg-secondary/50 text-muted-foreground"
                    }`}>
                      {done   ? <CheckCircle2 className="h-2.5 w-2.5" /> :
                       active ? <Loader2 className="h-2.5 w-2.5 animate-spin" /> :
                                <div className="h-2.5 w-2.5 rounded-full border border-current opacity-30" />}
                      {name}
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Agent messages */}
            <div className="space-y-2">
              {messages.map((msg, i) => {
                const AgentIcon = AGENT_ICONS[msg.agent] ?? BrainCircuit
                const isKeepalive = msg.output.final === false
                return (
                  <div key={i} className={`rounded-xl p-3 border text-xs ${
                    msg.success ? "bg-secondary/50 border-border" : "bg-destructive/5 border-destructive/30"
                  } ${isKeepalive ? "opacity-50" : ""}`}>
                    <div className="flex items-center gap-2 mb-1.5">
                      <div className={`h-5 w-5 flex items-center justify-center rounded-md ${msg.success ? "bg-primary/10" : "bg-destructive/10"}`}>
                        <AgentIcon className={`h-3 w-3 ${msg.success ? "text-primary" : "text-destructive"}`} />
                      </div>
                      <span className="font-medium">{msg.agent}</span>
                      <span className="ml-auto">
                        {msg.success
                          ? <CheckCircle2 className="h-3 w-3 text-emerald-400" />
                          : <XCircle className="h-3 w-3 text-destructive" />
                        }
                      </span>
                    </div>
                    <p className="leading-relaxed text-muted-foreground pl-7 whitespace-pre-wrap">{msg.message}</p>
                  </div>
                )
              })}
              {streaming && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground animate-pulse">
                  <Loader2 className="h-3 w-3 animate-spin text-primary" />
                  Processing…
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Results Column
// ──────────────────────────────────────────────────────────────────────────────

function ResultsColumn({ session }: { session: TrainingSession }) {
  const router = useRouter()
  const isEmpty = session.status !== "completed" && session.status !== "failed"

  const gradeStyle = GRADE_STYLES[session.grade ?? ""] ?? "text-muted-foreground border-border bg-secondary"

  return (
    <div className="flex flex-col h-full min-w-[280px]">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border shrink-0">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-primary" />
          <span className="text-sm font-semibold">Results</span>
          {session.grade && (
            <span className={`ml-auto text-sm font-bold px-2 py-0.5 rounded-lg border ${gradeStyle}`}>
              {session.grade}
            </span>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {isEmpty ? (
          <div className="flex flex-col items-center justify-center h-40 gap-3 text-center">
            <div className="h-10 w-10 rounded-full border-2 border-dashed border-border flex items-center justify-center">
              <BarChart3 className="h-5 w-5 text-muted-foreground/40" />
            </div>
            <p className="text-xs text-muted-foreground">Results will appear here<br />after training completes.</p>
          </div>
        ) : session.status === "failed" ? (
          <div className="space-y-4">
            <div className="flex flex-col items-center gap-3 py-6 text-center">
              <div className="h-12 w-12 flex items-center justify-center rounded-full bg-destructive/10 border-2 border-destructive">
                <XCircle className="h-6 w-6 text-destructive" />
              </div>
              <div>
                <p className="font-semibold">Pipeline Failed</p>
                <p className="text-xs text-muted-foreground mt-1 max-w-[200px]">
                  {session.errorMessage ?? "An error occurred during training."}
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Grade + metrics */}
            <div className="flex flex-col items-center gap-3 py-4 text-center">
              <div className="h-14 w-14 flex items-center justify-center rounded-full bg-emerald-500/10 border-2 border-emerald-500">
                <CheckCircle2 className="h-7 w-7 text-emerald-400" />
              </div>
              <div>
                <p className="font-bold text-lg">Training Complete</p>
                <p className="text-xs text-muted-foreground">Your model is ready.</p>
              </div>

              {session.grade && (
                <div className={`flex h-12 w-12 items-center justify-center rounded-xl border-2 font-bold text-xl ${gradeStyle}`}>
                  {session.grade}
                </div>
              )}
            </div>

            {(session.accuracy !== null || session.f1 !== null) && (
              <div className="flex gap-6 justify-center">
                {session.accuracy !== null && (
                  <div className="text-center">
                    <p className="text-3xl font-bold text-primary">{(session.accuracy * 100).toFixed(1)}%</p>
                    <p className="text-[10px] text-muted-foreground mt-0.5">Accuracy</p>
                  </div>
                )}
                {session.f1 !== null && (
                  <div className="text-center">
                    <p className="text-3xl font-bold">{session.f1.toFixed(3)}</p>
                    <p className="text-[10px] text-muted-foreground mt-0.5">F1 Score</p>
                  </div>
                )}
              </div>
            )}

            {session.runId && (
              <Button
                onClick={() => router.push(`/runs/${session.runId}`)}
                className="w-full gap-2"
                size="sm"
              >
                <ArrowRight className="h-3.5 w-3.5" />
                View Full Report
              </Button>
            )}

            {session.runId && (
              <Button
                variant="outline"
                onClick={() => router.push(`/runs`)}
                className="w-full gap-2"
                size="sm"
              >
                <Download className="h-3.5 w-3.5" />
                All Runs
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Session label editor (inline)
// ──────────────────────────────────────────────────────────────────────────────

function SessionLabelEditor({
  label, onSave,
}: {
  label: string
  onSave: (v: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [val, setVal]         = useState(label)

  function commit() {
    const trimmed = val.trim()
    if (trimmed) onSave(trimmed)
    setEditing(false)
  }

  if (editing) {
    return (
      <div className="flex items-center gap-1.5">
        <input
          autoFocus
          value={val}
          onChange={e => setVal(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter") commit(); if (e.key === "Escape") setEditing(false) }}
          onBlur={commit}
          className="text-base font-bold bg-transparent border-b border-primary focus:outline-none"
        />
        <button onClick={commit}><Check className="h-4 w-4 text-primary" /></button>
      </div>
    )
  }

  return (
    <button
      onClick={() => { setVal(label); setEditing(true) }}
      className="flex items-center gap-1.5 group"
    >
      <span className="text-base font-bold">{label}</span>
      <Pencil className="h-3.5 w-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
    </button>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Session Workspace (3-column view for a session)
// ──────────────────────────────────────────────────────────────────────────────

function SessionWorkspace({
  session, messages, streaming, onUpdate, onStartTraining,
}: {
  session: TrainingSession
  messages: AgentMessage[]
  streaming: boolean
  onUpdate: (patch: Partial<TrainingSession>) => void
  onStartTraining: () => void
}) {
  return (
    <div className="flex flex-col h-full">
      {/* Workspace header */}
      <div className="px-5 py-3 border-b border-border shrink-0 flex items-center gap-3">
        <SessionLabelEditor label={session.label} onSave={label => onUpdate({ label })} />
        <div className={`ml-2 text-[10px] px-2 py-0.5 rounded-full font-medium ${STATUS_BADGE[session.status].cls}`}>
          {STATUS_BADGE[session.status].label}
        </div>
        {session.uploadResult && (
          <span className="text-xs text-muted-foreground ml-auto">
            {session.uploadResult.filename} · {session.uploadResult.rows.toLocaleString()} rows
          </span>
        )}
      </div>

      {/* 3-column grid */}
      <div className="flex flex-1 overflow-hidden divide-x divide-border">
        <div className="flex-1 overflow-y-auto min-w-[260px]">
          <SetupColumn
            session={session}
            onUpdate={onUpdate}
            onStartTraining={onStartTraining}
            streaming={streaming}
          />
        </div>
        <div className="flex-1 overflow-y-auto min-w-[260px]">
          <TrainingColumn session={session} messages={messages} streaming={streaming} />
        </div>
        <div className="flex-1 overflow-y-auto min-w-[260px]">
          <ResultsColumn session={session} />
        </div>
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Empty workspace state
// ──────────────────────────────────────────────────────────────────────────────

function EmptyWorkspace({ onAdd }: { onAdd: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-5 text-center px-8">
      <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center">
        <Zap className="h-8 w-8 text-primary" />
      </div>
      <div className="space-y-1">
        <h2 className="text-xl font-bold">No training sessions</h2>
        <p className="text-sm text-muted-foreground max-w-xs">
          Create a session to upload a dataset, configure your model, and start training.
        </p>
      </div>
      <Button onClick={onAdd} size="lg" className="gap-2">
        <Plus className="h-4 w-4" />
        New Training Session
      </Button>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Main export
// ──────────────────────────────────────────────────────────────────────────────

export function TrainClient() {
  const supabase = createClient()
  const [state, dispatch] = useReducer(reducer, { sessions: [], activeId: null })
  // Messages stored outside reducer: not serialized to localStorage
  const [liveMessages, setLiveMessages] = useState<Record<string, AgentMessage[]>>({})
  const [streamingId,  setStreamingId]  = useState<string | null>(null)
  const hydratedRef = useRef(false)

  // Hydrate from localStorage once on mount
  useEffect(() => {
    if (hydratedRef.current) return
    hydratedRef.current = true

    try {
      const raw = localStorage.getItem(LS_KEY)
      if (raw) {
        const parsed: TrainingSession[] = JSON.parse(raw).map((s: TrainingSession) => ({
          ...s,
          // Recover sessions that were mid-stream when page was closed
          status: s.status === "training" ? "failed" : s.status,
          errorMessage: s.status === "training"
            ? (s.errorMessage ?? "Session interrupted. Try starting training again.")
            : s.errorMessage,
        }))
        const firstId = parsed[0]?.id ?? null
        dispatch({ type: "HYDRATE", sessions: parsed, activeId: firstId })
      }
    } catch { /* corrupt localStorage — ignore */ }

    // Check if user arrived from Model Catalog with a pre-selected model
    try {
      const raw = localStorage.getItem("modelforge_preselect_model")
      if (raw) {
        const model = JSON.parse(raw) as { model_id: string; display_name: string }
        localStorage.removeItem("modelforge_preselect_model")
        const session = makeSession({
          selectedModelId: model.model_id,
          label: `Train ${model.display_name}`.slice(0, 30),
        })
        dispatch({ type: "ADD", session })
        dispatch({ type: "SELECT", id: session.id })
      }
    } catch { /* ignore */ }

    // Check if user arrived from Dataset Library with a pre-selected dataset
    try {
      const raw = localStorage.getItem("modelforge_preselect_dataset")
      if (raw) {
        const uploadResult = JSON.parse(raw) as UploadResult
        localStorage.removeItem("modelforge_preselect_dataset")
        const label = uploadResult.filename
          .replace(/\.(csv|json|jsonl)$/i, "")
          .replace(/[_-]/g, " ")
          .slice(0, 28)
        const session = makeSession({ uploadResult, setupSubstep: "analyzing", label })
        dispatch({ type: "ADD", session })
        dispatch({ type: "SELECT", id: session.id })
      }
    } catch { /* ignore */ }
  }, [])

  // Persist sessions to localStorage on every change
  useEffect(() => {
    if (!hydratedRef.current) return
    if (state.sessions.length === 0) return
    localStorage.setItem(LS_KEY, JSON.stringify(state.sessions))
  }, [state.sessions])

  function addSession() {
    const session = makeSession()
    dispatch({ type: "ADD", session })
    dispatch({ type: "SELECT", id: session.id })
  }

  function updateSession(id: string, patch: Partial<TrainingSession>) {
    dispatch({ type: "UPDATE", id, patch })
  }

  async function startTraining(sessionId: string) {
    const session = state.sessions.find(s => s.id === sessionId)
    if (!session?.uploadResult) return

    if (streamingId) {
      toast.error("Another session is already training. Wait for it to finish.")
      return
    }

    setStreamingId(sessionId)
    setLiveMessages(prev => ({ ...prev, [sessionId]: [] }))
    updateSession(sessionId, { status: "training", errorMessage: null })

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { data: { user } } = await (supabase as any).auth.getUser()
    let newRunId: string | null = null
    if (user) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: run } = await (supabase as any)
        .from("runs")
        .insert({
          user_id:          user.id,
          status:           "running",
          dataset_filename: session.uploadResult.filename,
          dataset_rows:     session.uploadResult.rows,
        })
        .select()
        .single()
      newRunId = run?.id ?? null
      updateSession(sessionId, { runId: newRunId })
    }

    const userIntent = session.intent.trim() ||
      `Train a ${session.hyperParams.task_type.replace(/_/g, " ")} model on this dataset`

    const overrides = {
      model_id:          session.selectedModelId,
      num_epochs:        session.hyperParams.num_epochs,
      learning_rate:     session.hyperParams.learning_rate,
      batch_size:        session.hyperParams.batch_size,
      max_length:        session.hyperParams.max_length,
      training_approach: session.hyperParams.training_approach,
      lora_r:            session.hyperParams.lora_r,
      weight_decay:      session.hyperParams.weight_decay,
      warmup_ratio:      session.hyperParams.warmup_ratio,
    }

    const hfToken = localStorage.getItem("modelforge_hf_token") || null

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          message:                  userIntent,
          file_id:                  session.uploadResult.file_id,
          run_id:                   newRunId,
          hyperparameter_overrides: overrides,
          hf_token:                 hfToken,
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
            setLiveMessages(prev => ({ ...prev, [sessionId]: [...allMessages] }))
          } catch { /* ignore parse errors */ }
        }
      }

      // Persist to Supabase
      const pipelineSuccess = allMessages.every(m => m.success)
      const intentOut = (allMessages.find(m => m.agent === "Intent")?.output ?? {}) as Record<string, unknown>
      const modelOut  = (allMessages.find(m => m.agent === "Model")?.output  ?? {}) as Record<string, unknown>
      const trainOut  = (allMessages.find(m => m.agent === "Train" && m.output.final !== false)?.output ?? {}) as Record<string, unknown>
      const evalOut   = (allMessages.find(m => m.agent === "Eval")?.output   ?? {}) as Record<string, unknown>
      const deployOut = (allMessages.find(m => m.agent === "Deploy")?.output ?? {}) as Record<string, unknown>
      const mSrc      = Object.keys(evalOut).length > 0 ? evalOut : trainOut

      if (newRunId) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (supabase as any).from("runs").update({
          status:        pipelineSuccess ? "completed" : "failed",
          task_type:     intentOut.task_type  as string ?? null,
          model_id:      (modelOut.base_model ?? intentOut.base_model_hint) as string ?? null,
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
          artifact_path: trainOut.model_path  as string ?? null,
          hf_model_url:  deployOut.hf_url     as string ?? null,
          hf_repo_id:    deployOut.hf_repo_id as string ?? null,
          model_card:    deployOut.model_card as string ?? null,
          deploy_status: deployOut.status     as string ?? "not_deployed",
          completed_at:  new Date().toISOString(),
          error_message: pipelineSuccess ? null : (allMessages.find(m => !m.success)?.message ?? null),
        }).eq("id", newRunId)
      }

      updateSession(sessionId, {
        status:      pipelineSuccess ? "completed" : "failed",
        grade:       typeof evalOut.evaluation_grade === "string" ? evalOut.evaluation_grade : null,
        accuracy:    typeof mSrc.accuracy === "number" ? mSrc.accuracy : null,
        f1:          typeof mSrc.f1       === "number" ? mSrc.f1       : null,
        artifactPath: trainOut.model_path as string ?? null,
        errorMessage: pipelineSuccess ? null : (allMessages.find(m => !m.success)?.message ?? null),
      })

      if (!pipelineSuccess) toast.error("Pipeline encountered an error — see training column")
      else toast.success("Training complete!")
    } catch (err) {
      const msg = String(err)
      updateSession(sessionId, { status: "failed", errorMessage: msg })
      setLiveMessages(prev => ({
        ...prev,
        [sessionId]: [
          ...(prev[sessionId] ?? []),
          { agent: "System", success: false, message: msg, output: {} },
        ],
      }))
      if (newRunId) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await (supabase as any).from("runs").update({ status: "failed", error_message: msg }).eq("id", newRunId)
      }
      toast.error(`Pipeline error: ${err}`)
    } finally {
      setStreamingId(null)
    }
  }

  const activeSession = state.sessions.find(s => s.id === state.activeId) ?? null
  const activeMsgs    = activeSession ? (liveMessages[activeSession.id] ?? []) : []

  return (
    <div className="flex h-screen overflow-hidden">
      <SessionSidebar
        sessions={state.sessions}
        activeId={state.activeId}
        onSelect={id => dispatch({ type: "SELECT", id })}
        onAdd={addSession}
      />
      <div className="flex-1 overflow-hidden">
        {activeSession ? (
          <SessionWorkspace
            session={activeSession}
            messages={activeMsgs}
            streaming={streamingId === activeSession.id}
            onUpdate={patch => updateSession(activeSession.id, patch)}
            onStartTraining={() => startTraining(activeSession.id)}
          />
        ) : (
          <EmptyWorkspace onAdd={addSession} />
        )}
      </div>
    </div>
  )
}
