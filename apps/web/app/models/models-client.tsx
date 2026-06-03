"use client"

import { useState, useEffect, useMemo } from "react"
import { useRouter } from "next/navigation"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import {
  Search, Zap, BookOpen, BrainCircuit, Cpu, Globe2, Eye,
  Cloud, ArrowRight, Lock, Loader2, XCircle, Filter,
  Trophy, Users, Clock, BarChart3,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { formatDistanceToNow } from "date-fns"

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

type View       = "catalog" | "leaderboard"
type Category   = "all" | "encoder" | "seq2seq" | "decoder" | "embedding" | "vision" | "api"
type Provider   = "all" | "huggingface" | "openai" | "cohere" | "google"
type QualityTier = "excellent" | "good" | "balanced" | "fast"

export interface CatalogModel {
  model_id: string
  display_name: string
  category: string
  provider: string
  param_count: string
  param_count_m: number
  task_types: string[]
  quality_tier: QualityTier
  inference_speed: string
  lora_compatible: boolean
  qlora_compatible: boolean
  languages: string[]
  description: string
  best_for: string
  tags: string[]
  requires_token: boolean
  license: string
}

interface LeaderboardEntry {
  rank: number
  model_id: string
  display_name: string
  category: string
  provider: string
  param_count: string
  quality_tier: string
  lora_compatible: boolean
  run_count: number
  best_f1: number | null
  avg_f1: number | null
  avg_accuracy: number | null
  task_types: string[]
  last_run_at: string | null
}

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

const CATEGORY_TABS: { key: Category; label: string; icon: React.ElementType }[] = [
  { key: "all",       label: "All Models",  icon: BookOpen },
  { key: "encoder",   label: "Encoder",     icon: BrainCircuit },
  { key: "seq2seq",   label: "Seq2Seq",     icon: Filter },
  { key: "decoder",   label: "Decoder/LLM", icon: Cpu },
  { key: "embedding", label: "Embedding",   icon: Zap },
  { key: "vision",    label: "Vision",      icon: Eye },
  { key: "api",       label: "Cloud API",   icon: Cloud },
]

const QUALITY_STYLES: Record<string, string> = {
  excellent: "bg-violet-500/10 text-violet-500 border-violet-500/25",
  good:      "bg-blue-500/10 text-blue-500 border-blue-500/25",
  balanced:  "bg-teal-500/10 text-teal-500 border-teal-500/25",
  fast:      "bg-amber-500/10 text-amber-500 border-amber-500/25",
}

const PROVIDER_STYLES: Record<string, string> = {
  huggingface: "bg-orange-500/10 text-orange-500 border-orange-500/25",
  openai:      "bg-green-500/10 text-green-500 border-green-500/25",
  cohere:      "bg-purple-500/10 text-purple-500 border-purple-500/25",
  google:      "bg-blue-500/10 text-blue-500 border-blue-500/25",
}

const PROVIDER_LABELS: Record<string, string> = {
  huggingface: "HuggingFace",
  openai:      "OpenAI",
  cohere:      "Cohere",
  google:      "Google",
}

// ──────────────────────────────────────────────────────────────────────────────
// Main component
// ──────────────────────────────────────────────────────────────────────────────

export function ModelsClient() {
  const router = useRouter()

  // View toggle
  const [view, setView] = useState<View>("catalog")

  // Catalog state
  const [catalog, setCatalog] = useState<CatalogModel[]>([])
  const [loading, setLoading]  = useState(true)
  const [error, setError]      = useState<string | null>(null)

  // Catalog filters
  const [q, setQ]               = useState("")
  const [category, setCategory] = useState<Category>("all")
  const [provider, setProvider] = useState<Provider>("all")
  const [loraOnly, setLoraOnly] = useState(false)

  // Leaderboard state
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([])
  const [lbLoading, setLbLoading]     = useState(false)
  const [lbError, setLbError]         = useState<string | null>(null)

  // Fetch catalog once on mount
  useEffect(() => {
    fetch(`${API_URL}/models`)
      .then(r => {
        if (!r.ok) throw new Error(`API returned ${r.status}`)
        return r.json() as Promise<CatalogModel[]>
      })
      .then(data => { setCatalog(data); setLoading(false) })
      .catch(e => { setError(String(e)); setLoading(false) })
  }, [])

  // Fetch leaderboard when tab is active; poll every 30s
  useEffect(() => {
    if (view !== "leaderboard") return
    let alive = true

    const load = async () => {
      if (!alive) return
      setLbLoading(prev => leaderboard.length === 0 ? true : prev)
      try {
        const r = await fetch(`${API_URL}/leaderboard`)
        if (!r.ok) throw new Error(`API returned ${r.status}`)
        const data = await r.json() as LeaderboardEntry[]
        if (alive) { setLeaderboard(data); setLbError(null) }
      } catch (e) {
        if (alive) setLbError(String(e))
      } finally {
        if (alive) setLbLoading(false)
      }
    }

    load()
    const iv = setInterval(load, 30_000)
    return () => { alive = false; clearInterval(iv) }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view])

  const filtered = useMemo(() => {
    let results = catalog
    if (category !== "all") results = results.filter(m => m.category === category)
    if (provider !== "all") results = results.filter(m => m.provider === provider)
    if (loraOnly)           results = results.filter(m => m.lora_compatible)
    if (q.trim()) {
      const ql = q.toLowerCase()
      results = results.filter(m =>
        m.display_name.toLowerCase().includes(ql) ||
        m.description.toLowerCase().includes(ql) ||
        m.best_for.toLowerCase().includes(ql) ||
        m.model_id.toLowerCase().includes(ql) ||
        m.tags.some(t => t.includes(ql))
      )
    }
    return results
  }, [catalog, category, provider, loraOnly, q])

  // Build a CatalogModel from leaderboard entry + catalog lookup for trainWithModel
  function trainWithLeaderboardEntry(entry: LeaderboardEntry) {
    const catalogMatch = catalog.find(m => m.model_id === entry.model_id)
    const synthetic: CatalogModel = catalogMatch ?? {
      model_id:       entry.model_id,
      display_name:   entry.display_name,
      category:       entry.category,
      provider:       entry.provider,
      param_count:    entry.param_count,
      param_count_m:  0,
      task_types:     entry.task_types,
      quality_tier:   (entry.quality_tier as QualityTier) || "balanced",
      inference_speed: "medium",
      lora_compatible: entry.lora_compatible,
      qlora_compatible: false,
      languages:      ["en"],
      description:    "",
      best_for:       "",
      tags:           [],
      requires_token: false,
      license:        "",
    }
    localStorage.setItem("modelforge_preselect_model", JSON.stringify(synthetic))
    router.push("/train")
  }

  function trainWithModel(model: CatalogModel) {
    localStorage.setItem("modelforge_preselect_model", JSON.stringify(model))
    router.push("/train")
  }

  // ── Global catalog loading/error states ────────────────────────────────────
  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-7 w-7 animate-spin text-primary" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-3 text-center px-4">
        <XCircle className="h-8 w-8 text-destructive" />
        <p className="text-sm text-muted-foreground max-w-xs">
          Could not load catalog. Make sure the backend is running.<br />
          <span className="text-xs font-mono opacity-60">{error}</span>
        </p>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-5">
      {/* Header + view toggle */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            {view === "catalog" ? "Model Catalog" : "Community Leaderboard"}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            {view === "catalog"
              ? `${catalog.length} models — encoders, LLMs, embedding, vision, and cloud APIs.`
              : "Base models ranked by best F1 across all platform training runs · refreshes every 30s"}
          </p>
        </div>

        {/* View toggle pills */}
        <div className="flex rounded-lg border border-border overflow-hidden shrink-0">
          <button
            onClick={() => setView("catalog")}
            className={cn(
              "flex items-center gap-1.5 px-4 py-2 text-sm font-medium transition-colors",
              view === "catalog"
                ? "bg-primary text-primary-foreground"
                : "bg-secondary text-muted-foreground hover:text-foreground hover:bg-secondary/70"
            )}
          >
            <BookOpen className="h-3.5 w-3.5" />
            Model Catalog
          </button>
          <button
            onClick={() => setView("leaderboard")}
            className={cn(
              "flex items-center gap-1.5 px-4 py-2 text-sm font-medium transition-colors border-l border-border",
              view === "leaderboard"
                ? "bg-primary text-primary-foreground"
                : "bg-secondary text-muted-foreground hover:text-foreground hover:bg-secondary/70"
            )}
          >
            <Trophy className="h-3.5 w-3.5" />
            Community Leaderboard
          </button>
        </div>
      </div>

      {/* ── CATALOG VIEW ─────────────────────────────────────────────────────── */}
      {view === "catalog" && (
        <>
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
            <Input
              placeholder="Search by name, task, or keyword…"
              value={q}
              onChange={e => setQ(e.target.value)}
              className="pl-9"
            />
          </div>

          {/* Category tabs */}
          <div className="flex gap-2 flex-wrap">
            {CATEGORY_TABS.map(tab => {
              const Icon   = tab.icon
              const active = category === tab.key
              return (
                <button
                  key={tab.key}
                  onClick={() => setCategory(tab.key)}
                  className={cn(
                    "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors",
                    active
                      ? "bg-primary text-primary-foreground"
                      : "bg-secondary text-muted-foreground hover:text-foreground hover:bg-secondary/70"
                  )}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {tab.label}
                </button>
              )
            })}
          </div>

          {/* Sub-filters */}
          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={() => setLoraOnly(v => !v)}
              className={cn(
                "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium border transition-colors",
                loraOnly
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border text-muted-foreground hover:text-foreground hover:bg-secondary/50"
              )}
            >
              <Zap className="h-3.5 w-3.5" />
              LoRA compatible
            </button>

            <Separator orientation="vertical" className="h-5" />

            {(["all", "huggingface", "openai", "cohere", "google"] as Provider[]).map(p => (
              <button
                key={p}
                onClick={() => setProvider(p)}
                className={cn(
                  "px-3 py-1.5 rounded-lg text-sm font-medium border transition-colors",
                  provider === p
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-border text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                )}
              >
                {p === "all" ? "All Providers" : PROVIDER_LABELS[p] ?? p}
              </button>
            ))}

            <span className="ml-auto text-xs text-muted-foreground">
              {filtered.length} model{filtered.length !== 1 ? "s" : ""}
            </span>
          </div>

          {/* Model grid */}
          {filtered.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 gap-3 text-center">
              <BookOpen className="h-10 w-10 text-muted-foreground/40" />
              <p className="text-sm text-muted-foreground">No models match your filters.</p>
              <Button variant="outline" size="sm" onClick={() => { setQ(""); setCategory("all"); setProvider("all"); setLoraOnly(false) }}>
                Clear filters
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {filtered.map(model => (
                <ModelCard key={model.model_id} model={model} onTrain={trainWithModel} />
              ))}
            </div>
          )}
        </>
      )}

      {/* ── LEADERBOARD VIEW ─────────────────────────────────────────────────── */}
      {view === "leaderboard" && (
        <LeaderboardView
          entries={leaderboard}
          loading={lbLoading}
          error={lbError}
          onTrain={trainWithLeaderboardEntry}
        />
      )}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Leaderboard view
// ──────────────────────────────────────────────────────────────────────────────

function LeaderboardView({
  entries,
  loading,
  error,
  onTrain,
}: {
  entries: LeaderboardEntry[]
  loading: boolean
  error: string | null
  onTrain: (e: LeaderboardEntry) => void
}) {
  if (loading && entries.length === 0) {
    return (
      <div className="space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-24 rounded-xl bg-secondary/40 animate-pulse" />
        ))}
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-3 text-center">
        <XCircle className="h-8 w-8 text-destructive" />
        <p className="text-sm text-muted-foreground max-w-xs">
          Could not load leaderboard. Make sure the backend is running.<br />
          <span className="text-xs font-mono opacity-60">{error}</span>
        </p>
      </div>
    )
  }

  if (entries.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-4 text-center">
        <Trophy className="h-12 w-12 text-muted-foreground/30" />
        <div>
          <p className="font-semibold text-foreground/80">No completed runs yet</p>
          <p className="text-sm text-muted-foreground mt-1">
            Train your first model to claim the #1 spot.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => window.location.href = "/train"}>
          Start training
          <ArrowRight className="h-3.5 w-3.5 ml-1.5" />
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* Column header */}
      <div className="grid grid-cols-[3rem_1fr_auto] md:grid-cols-[3rem_1fr_14rem_auto] items-center gap-4 px-4 text-[11px] font-medium text-muted-foreground uppercase tracking-wide">
        <span className="text-center">Rank</span>
        <span>Model</span>
        <span className="hidden md:block text-right">F1 / Accuracy</span>
        <span />
      </div>

      {entries.map(entry => (
        <LeaderboardRow key={entry.model_id} entry={entry} onTrain={onTrain} />
      ))}

      {/* Staleness hint */}
      <p className="text-center text-[11px] text-muted-foreground/50 pt-2">
        <Clock className="inline h-3 w-3 mr-1 opacity-60" />
        Auto-refreshes every 30 seconds
      </p>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Leaderboard row
// ──────────────────────────────────────────────────────────────────────────────

const RANK_STYLES: Record<number, { bg: string; text: string; label: string }> = {
  1: { bg: "bg-amber-500/15 border-amber-500/40",   text: "text-amber-400 font-bold", label: "#1" },
  2: { bg: "bg-slate-400/10  border-slate-400/30",  text: "text-slate-300 font-bold", label: "#2" },
  3: { bg: "bg-orange-700/15 border-orange-700/40", text: "text-orange-500 font-bold", label: "#3" },
}

function fmtF1(v: number | null): string {
  if (v === null) return "—"
  return (v * 100).toFixed(1) + "%"
}

function LeaderboardRow({
  entry,
  onTrain,
}: {
  entry: LeaderboardEntry
  onTrain: (e: LeaderboardEntry) => void
}) {
  const rank        = RANK_STYLES[entry.rank]
  const qualStyle   = QUALITY_STYLES[entry.quality_tier] ?? "bg-secondary text-muted-foreground border-border"
  const provStyle   = PROVIDER_STYLES[entry.provider]    ?? "bg-secondary text-muted-foreground border-border"

  return (
    <Card className={cn(
      "hover:border-primary/40 transition-colors",
      entry.rank <= 3 ? "border-primary/20" : ""
    )}>
      <CardContent className="p-4">
        <div className="grid grid-cols-[3rem_1fr_auto] md:grid-cols-[3rem_1fr_14rem_auto] items-center gap-4">
          {/* Rank badge */}
          <div className="flex justify-center">
            <span className={cn(
              "inline-flex items-center justify-center w-9 h-9 rounded-lg border text-sm",
              rank
                ? `${rank.bg} ${rank.text}`
                : "bg-secondary/60 border-border text-muted-foreground font-medium"
            )}>
              {rank ? rank.label : `#${entry.rank}`}
            </span>
          </div>

          {/* Model info */}
          <div className="min-w-0 space-y-2">
            <div>
              <p className="font-semibold text-sm leading-snug">{entry.display_name}</p>
              <p className="text-[11px] text-muted-foreground font-mono mt-0.5 truncate">{entry.model_id}</p>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {entry.quality_tier && (
                <span className={cn("text-[11px] px-2 py-0.5 rounded border font-medium capitalize", qualStyle)}>
                  {entry.quality_tier}
                </span>
              )}
              {entry.provider && (
                <span className={cn("text-[11px] px-2 py-0.5 rounded border font-medium", provStyle)}>
                  {PROVIDER_LABELS[entry.provider] ?? entry.provider}
                </span>
              )}
              {entry.param_count && (
                <span className="text-[11px] px-2 py-0.5 rounded border bg-secondary/80 text-muted-foreground font-mono">
                  {entry.param_count}
                </span>
              )}
              {entry.lora_compatible && (
                <span className="text-[11px] px-2 py-0.5 rounded border bg-emerald-500/10 text-emerald-600 border-emerald-500/25 flex items-center gap-1">
                  <Zap className="h-2.5 w-2.5" /> LoRA
                </span>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {/* Task type chips */}
              <div className="flex flex-wrap gap-1">
                {entry.task_types.slice(0, 3).map(t => (
                  <span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-secondary text-muted-foreground">
                    {t.replace(/_/g, " ")}
                  </span>
                ))}
                {entry.task_types.length > 3 && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-secondary text-muted-foreground">
                    +{entry.task_types.length - 3}
                  </span>
                )}
              </div>
              {/* Run count */}
              <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
                <Users className="h-3 w-3" />
                {entry.run_count} run{entry.run_count !== 1 ? "s" : ""}
              </span>
              {/* Last trained */}
              {entry.last_run_at && (
                <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  {formatDistanceToNow(new Date(entry.last_run_at), { addSuffix: true })}
                </span>
              )}
            </div>
          </div>

          {/* Metrics column (md+) */}
          <div className="hidden md:flex flex-col items-end gap-1">
            <div className="flex items-baseline gap-1.5">
              <BarChart3 className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
              <span className="text-xl font-bold text-emerald-400 tabular-nums">
                {fmtF1(entry.best_f1)}
              </span>
            </div>
            <div className="text-[11px] text-muted-foreground space-x-2 text-right">
              <span>avg F1 {fmtF1(entry.avg_f1)}</span>
              <span>·</span>
              <span>acc {entry.avg_accuracy !== null ? (entry.avg_accuracy * 100).toFixed(1) + "%" : "—"}</span>
            </div>
          </div>

          {/* CTA */}
          <Button size="sm" className="gap-1.5 shrink-0" onClick={() => onTrain(entry)}>
            <span className="hidden sm:inline">Train</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </Button>
        </div>

        {/* Metrics row on mobile */}
        <div className="md:hidden mt-3 pt-3 border-t border-border flex items-center gap-4">
          <div className="flex items-baseline gap-1">
            <span className="text-xs text-muted-foreground">Best F1</span>
            <span className="text-base font-bold text-emerald-400 tabular-nums ml-1.5">
              {fmtF1(entry.best_f1)}
            </span>
          </div>
          <div className="text-[11px] text-muted-foreground">
            avg {fmtF1(entry.avg_f1)} · acc {entry.avg_accuracy !== null ? (entry.avg_accuracy * 100).toFixed(1) + "%" : "—"}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Model card (catalog view)
// ──────────────────────────────────────────────────────────────────────────────

function ModelCard({ model, onTrain }: { model: CatalogModel; onTrain: (m: CatalogModel) => void }) {
  const qualityStyle  = QUALITY_STYLES[model.quality_tier] ?? "bg-secondary text-muted-foreground border-border"
  const providerStyle = PROVIDER_STYLES[model.provider]    ?? "bg-secondary text-muted-foreground border-border"

  return (
    <Card className="group flex flex-col hover:border-primary/40 transition-colors">
      <CardContent className="flex flex-col gap-3 p-4 flex-1">
        {/* Name + provider */}
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className="font-semibold text-sm leading-snug truncate">{model.display_name}</p>
            <p className="text-[11px] text-muted-foreground font-mono mt-0.5 truncate">{model.model_id}</p>
          </div>
          <span className={cn("shrink-0 text-[11px] px-2 py-0.5 rounded border font-medium", providerStyle)}>
            {PROVIDER_LABELS[model.provider] ?? model.provider}
          </span>
        </div>

        {/* Badge row */}
        <div className="flex flex-wrap gap-1.5">
          <span className={cn("text-[11px] px-2 py-0.5 rounded border font-medium capitalize", qualityStyle)}>
            {model.quality_tier}
          </span>
          <span className="text-[11px] px-2 py-0.5 rounded border bg-secondary/80 text-muted-foreground font-mono">
            {model.param_count}
          </span>
          <span className="text-[11px] px-2 py-0.5 rounded border bg-secondary/80 text-muted-foreground capitalize">
            {model.category}
          </span>
          {model.lora_compatible && (
            <span className="text-[11px] px-2 py-0.5 rounded border bg-emerald-500/10 text-emerald-600 border-emerald-500/25 flex items-center gap-1">
              <Zap className="h-2.5 w-2.5" /> LoRA
            </span>
          )}
          {model.requires_token && (
            <span className="text-[11px] px-2 py-0.5 rounded border bg-amber-500/10 text-amber-600 border-amber-500/25 flex items-center gap-1">
              <Lock className="h-2.5 w-2.5" /> Gated
            </span>
          )}
          {model.languages.includes("multilingual") && (
            <span className="text-[11px] px-2 py-0.5 rounded border bg-blue-500/10 text-blue-600 border-blue-500/25 flex items-center gap-1">
              <Globe2 className="h-2.5 w-2.5" /> Multilingual
            </span>
          )}
        </div>

        {/* Description */}
        <p className="text-xs text-muted-foreground leading-relaxed line-clamp-2">{model.description}</p>

        {/* Best for */}
        <div className="rounded-md bg-secondary/50 px-2.5 py-2 text-xs">
          <span className="font-medium text-muted-foreground">Best for: </span>
          <span className="text-foreground/80">{model.best_for}</span>
        </div>

        {/* Task types */}
        <div className="flex flex-wrap gap-1">
          {model.task_types.slice(0, 3).map(t => (
            <span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-secondary text-muted-foreground">
              {t.replace(/_/g, " ")}
            </span>
          ))}
          {model.task_types.length > 3 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-secondary text-muted-foreground">
              +{model.task_types.length - 3}
            </span>
          )}
        </div>

        {/* CTA */}
        <Button size="sm" className="mt-auto w-full gap-1.5" onClick={() => onTrain(model)}>
          Train with this model
          <ArrowRight className="h-3.5 w-3.5" />
        </Button>
      </CardContent>
    </Card>
  )
}
