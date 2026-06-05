"use client"

import { useState, useMemo } from "react"
import Link from "next/link"
import { formatDistanceToNow, format } from "date-fns"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Plus, Search, X, Activity, CheckCircle2, Clock, XCircle, ArrowRight, ListOrdered } from "lucide-react"
import type { Run } from "@/lib/supabase/types"
import { SweepGroup } from "./sweep-group"

const statusCfg: Record<string, {
  icon: React.ElementType; textClass: string
  badge: React.ComponentProps<typeof Badge>["variant"]; label: string
}> = {
  completed: { icon: CheckCircle2, textClass: "text-emerald-400",      badge: "success",     label: "Completed" },
  running:   { icon: Activity,     textClass: "text-cyan-400",         badge: "running",     label: "Running"   },
  pending:   { icon: Clock,        textClass: "text-amber-400",        badge: "warning",     label: "Pending"   },
  failed:    { icon: XCircle,      textClass: "text-rose-400",         badge: "destructive", label: "Failed"    },
  cancelled: { icon: XCircle,      textClass: "text-muted-foreground", badge: "secondary",   label: "Cancelled" },
}

function RunRow({ run }: { run: Run }) {
  const cfg     = statusCfg[run.status] ?? statusCfg.pending
  const Icon    = cfg.icon
  const metrics = run.metrics as Record<string, unknown>
  const accuracy = typeof metrics?.accuracy === "number" ? `${(metrics.accuracy * 100).toFixed(1)}%` : null
  const f1Score  = typeof metrics?.f1       === "number" ? metrics.f1.toFixed(3)                     : null
  const loss     = typeof metrics?.final_loss === "number" ? metrics.final_loss.toFixed(4)            : null

  return (
    <Link href={`/runs/${run.id}`}
      className="group relative grid grid-cols-[auto_1fr_auto_auto_auto_auto] items-center gap-4 px-6 py-4 border-b border-border/40 last:border-0 transition-all duration-200 hover:bg-surface-elevated/50">
      <div className="absolute left-0 top-4 bottom-4 w-0.5 rounded-r opacity-0 group-hover:opacity-100 transition-opacity duration-200"
        style={{ background: "linear-gradient(180deg, var(--primary), var(--accent))" }} />

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
          <span className="text-foreground/60">{run.model_id ?? "—"}</span>
          <span className="mx-1.5 opacity-30">·</span>
          {run.dataset_filename ?? "no dataset"}
          {run.dataset_rows && (
            <><span className="mx-1.5 opacity-30">·</span>{run.dataset_rows.toLocaleString()} rows</>
          )}
        </p>
      </div>

      <div className="hidden lg:flex flex-col items-end gap-0.5 shrink-0 min-w-[56px]">
        {accuracy ? (
          <>
            <span className="text-sm font-bold font-mono text-foreground">{accuracy}</span>
            {f1Score && <span className="text-[10px] text-muted-foreground font-mono">F1 {f1Score}</span>}
            {!f1Score && loss && <span className="text-[10px] text-muted-foreground font-mono">loss {loss}</span>}
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
        <span className="text-[10px] text-muted-foreground/40 font-mono hidden lg:block">
          {format(new Date(run.created_at), "MMM d, yyyy")}
        </span>
      </div>

      <ArrowRight className="h-3.5 w-3.5 text-muted-foreground/30 group-hover:text-muted-foreground/60 group-hover:translate-x-0.5 transition-all duration-200 shrink-0" />
    </Link>
  )
}

const STATUS_OPTIONS = [
  { value: "all",       label: "All"       },
  { value: "running",   label: "Running"   },
  { value: "completed", label: "Completed" },
  { value: "failed",    label: "Failed"    },
  { value: "pending",   label: "Pending"   },
]

const SORT_OPTIONS = [
  { value: "newest",   label: "Newest first"   },
  { value: "oldest",   label: "Oldest first"   },
  { value: "best_f1",  label: "Best F1"        },
  { value: "best_acc", label: "Best accuracy"  },
]

interface Props {
  allRuns: Run[]
  sweepGroups: Run[][]
}

const PAGE_SIZE = 30

export function RunsClient({ allRuns, sweepGroups }: Props) {
  const [query,      setQuery]      = useState("")
  const [statusFilter, setStatus]   = useState("all")
  const [sort,       setSort]       = useState("newest")
  const [page,       setPage]       = useState(1)

  const standaloneRuns = useMemo(() => allRuns.filter(r => !r.sweep_id), [allRuns])

  const filtered = useMemo(() => {
    let runs = standaloneRuns

    if (statusFilter !== "all") {
      runs = runs.filter(r => r.status === statusFilter)
    }

    if (query.trim()) {
      const q = query.toLowerCase()
      runs = runs.filter(r =>
        (r.task_type ?? "").toLowerCase().includes(q) ||
        (r.model_id  ?? "").toLowerCase().includes(q) ||
        (r.dataset_filename ?? "").toLowerCase().includes(q)
      )
    }

    return [...runs].sort((a, b) => {
      const mA = a.metrics as Record<string, unknown>
      const mB = b.metrics as Record<string, unknown>
      if (sort === "oldest") return new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
      if (sort === "best_f1") return (typeof mB?.f1 === "number" ? mB.f1 : -1) - (typeof mA?.f1 === "number" ? mA.f1 : -1)
      if (sort === "best_acc") return (typeof mB?.accuracy === "number" ? mB.accuracy : -1) - (typeof mA?.accuracy === "number" ? mA.accuracy : -1)
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    })
  }, [standaloneRuns, query, statusFilter, sort])

  const paginated = filtered.slice(0, page * PAGE_SIZE)
  const hasMore   = paginated.length < filtered.length
  const isFiltered = query.trim() || statusFilter !== "all"

  function clearFilters() { setQuery(""); setStatus("all"); setSort("newest"); setPage(1) }

  return (
    <div className="px-8 py-7 max-w-5xl mx-auto space-y-6 animate-fade-in">

      {/* ── Filter bar ──────────────────────────────────────── */}
      {allRuns.length > 0 && (
        <div className="flex flex-wrap items-center gap-3">
          {/* Search */}
          <div className="relative flex-1 min-w-48">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
            <input
              value={query}
              onChange={e => { setQuery(e.target.value); setPage(1) }}
              placeholder="Search runs…"
              className="w-full h-9 pl-9 pr-3 rounded-xl text-sm outline-none transition-all"
              style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", color: "var(--body)" }}
              onFocus={e  => { e.currentTarget.style.borderColor = "rgba(99,102,241,0.4)"; e.currentTarget.style.background = "rgba(99,102,241,0.05)" }}
              onBlur={e   => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)"; e.currentTarget.style.background = "rgba(255,255,255,0.04)" }}
            />
            {query && (
              <button onClick={() => setQuery("")} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground">
                <X className="h-3 w-3" />
              </button>
            )}
          </div>

          {/* Status pills */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {STATUS_OPTIONS.map(opt => (
              <button key={opt.value}
                onClick={() => { setStatus(opt.value); setPage(1) }}
                className="text-xs px-2.5 py-1.5 rounded-lg font-medium transition-all"
                style={statusFilter === opt.value
                  ? { background: "rgba(99,102,241,0.2)", border: "1px solid rgba(99,102,241,0.4)", color: "#A5B4FC" }
                  : { background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", color: "rgba(203,213,225,0.6)" }
                }>
                {opt.label}
              </button>
            ))}
          </div>

          {/* Sort */}
          <select value={sort} onChange={e => { setSort(e.target.value); setPage(1) }}
            className="h-9 rounded-xl px-3 text-sm outline-none"
            style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", color: "rgba(203,213,225,0.7)" }}>
            {SORT_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>

          {/* Count + clear */}
          <span className="text-xs text-muted-foreground font-mono ml-auto">
            {isFiltered ? `${filtered.length} of ${standaloneRuns.length}` : `${standaloneRuns.length}`} run{standaloneRuns.length !== 1 ? "s" : ""}
          </span>
          {isFiltered && (
            <button onClick={clearFilters} className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors">
              Clear filters
            </button>
          )}
        </div>
      )}

      {/* ── Sweep groups ────────────────────────────────────── */}
      {sweepGroups.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Sweeps</span>
            <span className="font-mono text-[10px] text-muted-foreground/50">{sweepGroups.length}</span>
          </div>
          <div className="rounded-xl border border-border overflow-hidden"
            style={{ background: "color-mix(in srgb, var(--surface) 60%, transparent)" }}>
            {sweepGroups.map(group => <SweepGroup key={group[0]?.sweep_id} runs={group} />)}
          </div>
        </div>
      )}

      {/* ── Standalone runs ─────────────────────────────────── */}
      <div className="rounded-xl border border-border overflow-hidden"
        style={{ background: "color-mix(in srgb, var(--surface) 60%, transparent)" }}>

        {paginated.length > 0 && (
          <div className="grid grid-cols-[auto_1fr_auto_auto_auto_auto] items-center gap-4 px-6 py-2.5 border-b border-border/60 bg-surface-elevated/25">
            <span className="w-4" />
            <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground/45 font-mono">Task · Model · Dataset</span>
            <span className="hidden lg:block text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground/45 font-mono text-right min-w-[80px]">Metrics</span>
            <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground/45 font-mono">Status</span>
            <span className="hidden md:block text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground/45 font-mono text-right">Started</span>
            <span className="w-3.5" />
          </div>
        )}

        {allRuns.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-24 gap-5 text-center px-8">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-border"
              style={{ background: "linear-gradient(135deg, var(--primary-10), color-mix(in srgb, var(--accent) 6%, transparent))" }}>
              <ListOrdered className="h-7 w-7 text-indigo-400" />
            </div>
            <div className="space-y-1.5">
              <p className="font-semibold text-foreground">No training runs yet</p>
              <p className="text-sm text-muted-foreground max-w-xs">Start a training run and it will appear here with real-time metrics.</p>
            </div>
            <Button asChild><Link href="/train"><Plus className="h-4 w-4" />Start first run</Link></Button>
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 gap-4 text-center px-8">
            <Search className="h-8 w-8 text-muted-foreground/30" />
            <p className="text-sm font-medium text-foreground">No runs match your filters</p>
            <button onClick={clearFilters} className="text-sm text-indigo-400 hover:text-indigo-300 transition-colors">Clear filters</button>
          </div>
        ) : standaloneRuns.length === 0 && sweepGroups.length > 0 ? (
          <div className="flex items-center justify-center py-10 text-sm text-muted-foreground">
            All runs are part of sweeps above.
          </div>
        ) : (
          paginated.map(run => <RunRow key={run.id} run={run} />)
        )}
      </div>

      {/* ── Load more ───────────────────────────────────────── */}
      {hasMore && (
        <div className="flex justify-center">
          <Button variant="outline" size="sm" onClick={() => setPage(p => p + 1)}>
            Load {Math.min(PAGE_SIZE, filtered.length - paginated.length)} more
          </Button>
        </div>
      )}
    </div>
  )
}
