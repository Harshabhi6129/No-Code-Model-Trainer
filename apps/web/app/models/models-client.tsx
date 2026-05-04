"use client"

import { useState, useEffect, useMemo } from "react"
import { useRouter } from "next/navigation"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import {
  Search, Zap, BookOpen, BrainCircuit, Cpu, Globe2, Eye,
  Cloud, ArrowRight, Lock, Loader2, XCircle, Filter,
} from "lucide-react"
import { cn } from "@/lib/utils"

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

type Category = "all" | "encoder" | "seq2seq" | "decoder" | "embedding" | "vision" | "api"
type Provider = "all" | "huggingface" | "openai" | "cohere" | "google"
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

const QUALITY_STYLES: Record<QualityTier, string> = {
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
  const [catalog, setCatalog] = useState<CatalogModel[]>([])
  const [loading, setLoading]  = useState(true)
  const [error, setError]      = useState<string | null>(null)

  // Filters
  const [q, setQ]               = useState("")
  const [category, setCategory] = useState<Category>("all")
  const [provider, setProvider] = useState<Provider>("all")
  const [loraOnly, setLoraOnly] = useState(false)

  useEffect(() => {
    fetch(`${API_URL}/models`)
      .then(r => {
        if (!r.ok) throw new Error(`API returned ${r.status}`)
        return r.json() as Promise<CatalogModel[]>
      })
      .then(data => { setCatalog(data); setLoading(false) })
      .catch(e => { setError(String(e)); setLoading(false) })
  }, [])

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

  function trainWithModel(model: CatalogModel) {
    localStorage.setItem("modelforge_preselect_model", JSON.stringify(model))
    router.push("/train")
  }

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
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Model Catalog</h1>
        <p className="text-sm text-muted-foreground mt-1">
          {catalog.length} models — encoders, LLMs, embedding, vision, and cloud APIs.
        </p>
      </div>

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
          const Icon = tab.icon
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
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Model card
// ──────────────────────────────────────────────────────────────────────────────

function ModelCard({ model, onTrain }: { model: CatalogModel; onTrain: (m: CatalogModel) => void }) {
  const qualityStyle  = QUALITY_STYLES[model.quality_tier] ?? "bg-secondary text-muted-foreground"
  const providerStyle = PROVIDER_STYLES[model.provider]    ?? "bg-secondary text-muted-foreground"

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
