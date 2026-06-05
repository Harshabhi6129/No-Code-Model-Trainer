"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import Link from "next/link"
import { formatDistanceToNow, format } from "date-fns"
import {
  CheckCircle2, Activity, Clock, XCircle,
  ArrowRight, Search, SlidersHorizontal, Plus,
} from "lucide-react"
import type { Run } from "@/lib/supabase/types"

/* ── Status config ──────────────────────────────────────────────── */
const statusCfg: Record<string, {
  icon: React.ElementType
  textClass: string
  badge: React.ComponentProps<typeof Badge>["variant"]
  label: string
}> = {
  completed: { icon: CheckCircle2, textClass: "text-emerald-400", badge: "success",     label: "Completed" },
  running:   { icon: Activity,     textClass: "text-cyan-400",    badge: "running",     label: "Running"   },
  pending:   { icon: Clock,        textClass: "text-amber-400",   badge: "warning",     label: "Pending"   },
  failed:    { icon: XCircle,      textClass: "text-rose-400",    badge: "destructive", label: "Failed"    },
  cancelled: { icon: XCircle,      textClass: "text-muted-foreground", badge: "secondary", label: "Cancelled" },
}

const STATUS_FILTERS = ["all", "completed", "running", "failed", "pending"] as const
type StatusFilter = typeof STATUS_FILTERS[number]

const SORT_OPTIONS = [
  { value: "newest",   label: "Newest" },
  { value: "best_f1",  label: "Best F1" },
  { value: "best_acc", label: "Best Accuracy" },
] as const
type SortOption = typeof SORT_OPTIONS[number]["value"]

const PAGE_SIZE = 20

function RunRow({ run }: { run: Run }) {
  const cfg     = statusCfg[run.status] ?? statusCfg.pending
  const Icon    = cfg.icon
  const metrics = run.metrics as Record<string, unknown>
  const accuracy = typeof metrics?.accuracy === "number"
    ? `${(metrics.accuracy * 100).toFixed(1)}%` : null
  const f1Score  = typeof metrics?.f1 === "number" ? metrics.f1.toFixed(3) : null
  const loss     = typeof metrics?.final_loss === "number" ? metrics.final_loss.toFixed(4) : null

  return (
    <Link
      href={`/runs/${run.id}`}
      className="group relative grid grid-cols-[auto_1fr_auto_auto_auto_auto] items-center gap-4 px-6 py-4 border-b border-border/40 last:border-0 transition-all duration-200 hover:bg-surface-elevated/50"
    >
      <div
        className="absolute left-0 top-4 bottom-4 w-0.5 rounded-r opacity-0 group-hover:opacity-100 transition-opacity duration-200"
        style={{ background: "linear-gradient(180deg, var(--primary), var(--accent))" }}
      />
      <div className={`relative shrink-0 ${cfg.textClass}`}>
        <Icon className="h-4 w-4" />
        {run.status === "running" && (
          <span className="ping-dot absolute inset-[-3px] rounded-full" style={{ color: "var(--cyan)" }} />
        )}
      </div>
      <div className="min-w-0 space-y-1">
        <p className="text-sm font-medium capitalize text-foreground">
          {run.task_type?.replace(/_/g, " ") ?? "Training run"}
        </p>
        <p className="text-xs text-muted-foreground font-mono truncate">
          <span className="text-foreground/55">{run.model_id ?? "—"}</span>
          <span className="mx-1.5 opacity-30">·</span>
          <span>{run.dataset_filename ?? "no dataset"}</span>
          {run.dataset_rows && (
            <><span className="mx-1.5 opacity-30">·</span><span>{run.dataset_rows.toLocaleString()} rows</span></>
          )}
        </p>
      </div>
      <div className="hidden lg:flex flex-col items-end gap-0.5 shrink-0 min-w-[80px]">
        {accuracy ? (
          <>
            <span className="text-sm font-bold font-mono text-foreground">{accuracy}</span>
            <div className="flex items-center gap-2">
              {f1Score && <span className="text-[10px] text-muted-foreground font-mono">F1 {f1Score}</span>}
              {loss    && <span className="text-[10px] text-muted-foreground font-mono">loss {loss}</span>}
            </div>
          </>
        ) : (
          <span className="text-sm text-muted-foreground/30 font-mono">—</span>
        )}
      </div>
      <Badge variant={cfg.badge} dot pulse={run.status === "running"} className="shrink-0">{cfg.label}</Badge>
      <div className="hidden md:flex flex-col items-end gap-0.5 shrink-0">
        <span className="text-[11px] text-muted-foreground font-mono">
          {formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
        </span>
        <span className="text-[10px] text-muted-foreground/40 font-mono">
          {format(new Date(run.created_at), "MMM d, HH:mm")}
        </span>
      </div>
      <ArrowRight className="h-3.5 w-3.5 text-muted-foreground/25 group-hover:text-muted-foreground/55 group-hover:translate-x-0.5 transition-all duration-200 shrink-0" />
    </Link>
  )
}

export function RunsList({ runs }: { runs: Run[] }) {
  const [search,      setSearch]      = useState("")
  const [statusFilter,setStatusFilter]= useState<StatusFilter>("all")
  const [sortBy,      setSortBy]      = useState<SortOption>("newest")
  const [limit,       setLimit]       = useState(PAGE_SIZE)

  const filtered = runs
    .filter(r => {
      if (statusFilter !== "all" && r.status !== statusFilter) return false
      if (!search.trim()) return true
      const q = search.toLowerCase()
      return (
        r.task_type?.toLowerCase().includes(q) ||
        r.model_id?.toLowerCase().includes(q) ||
        r.dataset_filename?.toLowerCase().includes(q)
      )
    })
    .sort((a, b) => {
      if (sortBy === "best_f1") {
        const fa = (a.metrics as Record<string, unknown>)?.f1 as number ?? -1
        const fb = (b.metrics as Record<string, unknown>)?.f1 as number ?? -1
        return fb - fa
      }
      if (sortBy === "best_acc") {
        const aa = (a.metrics as Record<string, unknown>)?.accuracy as number ?? -1
        const ab = (b.metrics as Record<string, unknown>)?.accuracy as number ?? -1
        return ab - aa
      }
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    })

  const visible = filtered.slice(0, limit)
  const hasMore = filtered.length > limit

  if (runs.length === 0) {
    return (
      <div
        className="rounded-xl border border-border overflow-hidden"
        style={{ background: "color-mix(in srgb, var(--surface) 60%, transparent)" }}
      >
        <div className="flex flex-col items-center justify-center py-24 gap-5 text-center px-8">
          <div
            className="flex h-16 w-16 items-center justify-center rounded-2xl border border-border"
            style={{ background: "linear-gradient(135deg, var(--primary-10), color-mix(in srgb, var(--accent) 6%, transparent))" }}
          >
            <Search className="h-7 w-7 text-indigo-400" />
          </div>
          <div className="space-y-1.5">
            <p className="font-semibold text-foreground">No training runs yet</p>
            <p className="text-sm text-muted-foreground max-w-xs">Start a training run and it will appear here with real-time metrics.</p>
          </div>
          <Button asChild>
            <Link href="/train"><Plus className="h-4 w-4" />Start first run</Link>
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Filter bar */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
          <Input
            value={search}
            onChange={e => { setSearch(e.target.value); setLimit(PAGE_SIZE) }}
            placeholder="Search runs by task, model, or dataset…"
            className="pl-9 h-9 text-sm bg-surface/50 border-border/60"
          />
        </div>

        {/* Status filter chips */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {STATUS_FILTERS.map(s => (
            <button
              key={s}
              onClick={() => { setStatusFilter(s); setLimit(PAGE_SIZE) }}
              className={`px-2.5 py-1 rounded-full text-[11px] font-medium capitalize border transition-all ${
                statusFilter === s
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border/50 text-muted-foreground hover:border-primary/40 hover:text-foreground"
              }`}
            >
              {s}
            </button>
          ))}
        </div>

        {/* Sort dropdown */}
        <div className="flex items-center gap-1.5 ml-auto">
          <SlidersHorizontal className="h-3.5 w-3.5 text-muted-foreground" />
          <select
            value={sortBy}
            onChange={e => setSortBy(e.target.value as SortOption)}
            className="h-8 rounded-lg border border-border/60 bg-surface/50 px-2 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
          >
            {SORT_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </div>
      </div>

      {/* Count */}
      <p className="text-[11px] text-muted-foreground font-mono">
        {filtered.length === runs.length
          ? `${runs.length} run${runs.length !== 1 ? "s" : ""}`
          : `${filtered.length} of ${runs.length} runs`}
      </p>

      {/* List */}
      <div
        className="rounded-xl border border-border overflow-hidden"
        style={{ background: "color-mix(in srgb, var(--surface) 60%, transparent)" }}
      >
        {filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
            <Search className="h-8 w-8 text-muted-foreground/30" />
            <p className="text-sm text-muted-foreground">No runs match your filters.</p>
            <button
              onClick={() => { setSearch(""); setStatusFilter("all"); setLimit(PAGE_SIZE) }}
              className="text-xs text-primary hover:underline"
            >
              Clear filters
            </button>
          </div>
        ) : (
          <>
            {/* Column header */}
            <div className="grid grid-cols-[auto_1fr_auto_auto_auto_auto] items-center gap-4 px-6 py-2.5 border-b border-border/60 bg-surface-elevated/25">
              <span className="w-4" />
              <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground/45 font-mono">Task · Model · Dataset</span>
              <span className="hidden lg:block text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground/45 font-mono text-right min-w-[80px]">Metrics</span>
              <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground/45 font-mono">Status</span>
              <span className="hidden md:block text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground/45 font-mono text-right">Started</span>
              <span className="w-3.5" />
            </div>
            {visible.map(run => <RunRow key={run.id} run={run} />)}
          </>
        )}
      </div>

      {/* Load more */}
      {hasMore && (
        <div className="flex justify-center">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setLimit(l => l + PAGE_SIZE)}
            className="gap-2"
          >
            Load {Math.min(PAGE_SIZE, filtered.length - limit)} more
          </Button>
        </div>
      )}
    </div>
  )
}
