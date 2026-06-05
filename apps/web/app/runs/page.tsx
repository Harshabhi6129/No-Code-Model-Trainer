import { AppShell } from "@/components/layout/app-shell"
import { createClient } from "@/lib/supabase/server"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { PageHeader } from "@/components/ui/page-header"
import Link from "next/link"
import { formatDistanceToNow, format } from "date-fns"
import {
  Plus, Activity, CheckCircle2, Clock, XCircle,
  ArrowRight, GitCompare, ListOrdered, Filter,
} from "lucide-react"
import type { Run } from "@/lib/supabase/types"
import { SweepGroup } from "./sweep-group"

/* ── Status config ──────────────────────────────────────────────── */
const statusCfg: Record<string, {
  icon: React.ElementType
  textClass: string
  badge: React.ComponentProps<typeof Badge>["variant"]
  label: string
}> = {
  completed: { icon: CheckCircle2, textClass: "text-emerald-400",      badge: "success",     label: "Completed" },
  running:   { icon: Activity,     textClass: "text-cyan-400",         badge: "running",     label: "Running"   },
  pending:   { icon: Clock,        textClass: "text-amber-400",        badge: "warning",     label: "Pending"   },
  failed:    { icon: XCircle,      textClass: "text-rose-400",         badge: "destructive", label: "Failed"    },
  cancelled: { icon: XCircle,      textClass: "text-muted-foreground", badge: "secondary",   label: "Cancelled" },
}

/* ── Run row ────────────────────────────────────────────────────── */
function RunRow({ run, index }: { run: Run; index: number }) {
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
      {/* Left accent bar */}
      <div
        className="absolute left-0 top-4 bottom-4 w-0.5 rounded-r opacity-0 group-hover:opacity-100 transition-opacity duration-200"
        style={{ background: "linear-gradient(180deg, var(--primary), var(--accent))" }}
      />

      {/* Status icon */}
      <div className={`relative shrink-0 ${cfg.textClass}`}>
        <Icon className="h-4 w-4" />
        {run.status === "running" && (
          <span
            className="ping-dot absolute inset-[-3px] rounded-full"
            style={{ color: "var(--cyan)" }}
          />
        )}
      </div>

      {/* Task + model info */}
      <div className="min-w-0 space-y-1">
        <p className="text-sm font-medium capitalize text-foreground">
          {run.task_type?.replace(/_/g, " ") ?? "Training run"}
        </p>
        <p className="text-xs text-muted-foreground font-mono truncate">
          <span className="text-foreground/55">{run.model_id ?? "—"}</span>
          <span className="mx-1.5 opacity-30">·</span>
          <span>{run.dataset_filename ?? "no dataset"}</span>
          {run.dataset_rows && (
            <>
              <span className="mx-1.5 opacity-30">·</span>
              <span>{run.dataset_rows.toLocaleString()} rows</span>
            </>
          )}
        </p>
      </div>

      {/* Metrics */}
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

      {/* Status badge */}
      <Badge variant={cfg.badge} dot pulse={run.status === "running"} className="shrink-0">
        {cfg.label}
      </Badge>

      {/* Date */}
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

/* ── Status summary pill ────────────────────────────────────────── */
function StatusPill({
  count,
  label,
  variant,
}: {
  count: number
  label: string
  variant: React.ComponentProps<typeof Badge>["variant"]
}) {
  if (count === 0) return null
  return (
    <Badge variant={variant} className="gap-1.5 px-3 py-1">
      <span className="font-bold font-mono">{count}</span>
      <span className="font-normal opacity-80">{label}</span>
    </Badge>
  )
}

/* ── Page ───────────────────────────────────────────────────────── */
export default async function RunsPage() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  const userId = user?.id ?? ""

  const { data: runs } = userId
    ? await supabase.from("runs").select("*").eq("user_id", userId).order("created_at", { ascending: false })
    : { data: [] }

  const allRuns   = (runs ?? []) as Run[]

  // Separate sweep child runs from standalone runs
  const sweepMap = new Map<string, Run[]>()
  const standaloneRuns: Run[] = []
  for (const run of allRuns) {
    if (run.sweep_id) {
      const group = sweepMap.get(run.sweep_id) ?? []
      group.push(run)
      sweepMap.set(run.sweep_id, group)
    } else {
      standaloneRuns.push(run)
    }
  }
  const sweepGroups = Array.from(sweepMap.values())

  const completed = standaloneRuns.filter((r) => r.status === "completed").length
  const running   = standaloneRuns.filter((r) => r.status === "running").length
  const pending   = standaloneRuns.filter((r) => r.status === "pending").length
  const failed    = standaloneRuns.filter((r) => r.status === "failed").length

  return (
    <AppShell>
      <PageHeader
        icon={ListOrdered}
        iconColor="text-amber-400"
        title="All Runs"
        description={`${allRuns.length} run${allRuns.length !== 1 ? "s" : ""} total${sweepGroups.length > 0 ? ` · ${sweepGroups.length} sweep${sweepGroups.length !== 1 ? "s" : ""}` : ""}`}
        actions={
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" asChild>
              <Link href="/runs/compare">
                <GitCompare className="h-4 w-4" />
                Compare
              </Link>
            </Button>
            <Button size="sm" asChild>
              <Link href="/train">
                <Plus className="h-4 w-4" />
                New Run
              </Link>
            </Button>
          </div>
        }
      />

      <div className="px-8 py-7 max-w-5xl mx-auto space-y-6 animate-fade-in">

        {/* Status summary pills */}
        {allRuns.length > 0 && (
          <div className="flex items-center gap-2 flex-wrap">
            <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground/50 font-mono mr-1">
              <Filter className="h-3 w-3" />
              Summary
            </span>
            <StatusPill count={running}   label="running"   variant="running"     />
            <StatusPill count={pending}   label="pending"   variant="warning"     />
            <StatusPill count={completed} label="completed" variant="success"     />
            <StatusPill count={failed}    label="failed"    variant="destructive" />
          </div>
        )}

        {/* Sweep groups (collapsed by default, shown before standalone runs) */}
        {sweepGroups.length > 0 && (
          <div className="space-y-0">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Sweeps</span>
              <span className="font-mono text-[10px] text-muted-foreground/50">{sweepGroups.length}</span>
            </div>
            <div
              className="rounded-xl border border-border overflow-hidden"
              style={{ background: "color-mix(in srgb, var(--surface) 60%, transparent)" }}
            >
              {sweepGroups.map((group) => (
                <SweepGroup key={group[0]?.sweep_id} runs={group} />
              ))}
            </div>
          </div>
        )}

        {/* Standalone runs list */}
        <div
          className="rounded-xl border border-border overflow-hidden"
          style={{ background: "color-mix(in srgb, var(--surface) 60%, transparent)" }}
        >
          {/* Column header row */}
          {standaloneRuns.length > 0 && (
            <div className="grid grid-cols-[auto_1fr_auto_auto_auto_auto] items-center gap-4 px-6 py-2.5 border-b border-border/60 bg-surface-elevated/25">
              <span className="w-4" />
              <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground/45 font-mono">
                Task · Model · Dataset
              </span>
              <span className="hidden lg:block text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground/45 font-mono text-right min-w-[80px]">
                Metrics
              </span>
              <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground/45 font-mono">
                Status
              </span>
              <span className="hidden md:block text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground/45 font-mono text-right">
                Started
              </span>
              <span className="w-3.5" />
            </div>
          )}

          {allRuns.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-24 gap-5 text-center px-8">
              <div
                className="flex h-16 w-16 items-center justify-center rounded-2xl border border-border"
                style={{
                  background: "linear-gradient(135deg, var(--primary-10), color-mix(in srgb, var(--accent) 6%, transparent))",
                }}
              >
                <ListOrdered className="h-7 w-7 text-indigo-400" />
              </div>
              <div className="space-y-1.5">
                <p className="font-semibold text-foreground">No training runs yet</p>
                <p className="text-sm text-muted-foreground max-w-xs">
                  Start a training run and it will appear here with real-time metrics.
                </p>
              </div>
              <Button asChild>
                <Link href="/train">
                  <Plus className="h-4 w-4" />
                  Start first run
                </Link>
              </Button>
            </div>
          ) : standaloneRuns.length === 0 && sweepGroups.length > 0 ? (
            <div className="flex items-center justify-center py-10 text-sm text-muted-foreground">
              All runs are part of sweeps above.
            </div>
          ) : (
            standaloneRuns.map((run, i) => <RunRow key={run.id} run={run} index={i} />)
          )}
        </div>
      </div>
    </AppShell>
  )
}
