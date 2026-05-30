"use client"

import { useState } from "react"
import Link from "next/link"
import {
  ChevronDown, ChevronRight, CheckCircle2, XCircle, Activity,
  Clock, Zap, Star, ArrowRight,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import type { Run } from "@/lib/supabase/types"

// ── helpers ─────────────────────────────────────────────────────────────────

function getF1(run: Run): number | null {
  const m = run.metrics as Record<string, unknown>
  return typeof m?.f1 === "number" ? m.f1 : null
}

function getAccuracy(run: Run): number | null {
  const m = run.metrics as Record<string, unknown>
  return typeof m?.accuracy === "number" ? m.accuracy : null
}

function getGrade(run: Run): string | null {
  const m = run.metrics as Record<string, unknown>
  return typeof m?.evaluation_grade === "string" ? m.evaluation_grade : null
}

function formatCombo(cfg: Record<string, unknown>): string {
  const parts: string[] = []
  if (cfg.learning_rate != null) parts.push(`lr=${Number(cfg.learning_rate).toExponential(0)}`)
  if (cfg.batch_size    != null) parts.push(`bs=${cfg.batch_size}`)
  if (cfg.num_epochs    != null) parts.push(`ep=${cfg.num_epochs}`)
  if (cfg.lora_r        != null) parts.push(`r=${cfg.lora_r}`)
  return parts.join(" · ") || "—"
}

const STATUS_CFG: Record<string, { icon: React.ElementType; cls: string; label: string }> = {
  completed: { icon: CheckCircle2, cls: "text-emerald-400", label: "Done"    },
  running:   { icon: Activity,     cls: "text-cyan-400",    label: "Running" },
  pending:   { icon: Clock,        cls: "text-amber-400",   label: "Pending" },
  failed:    { icon: XCircle,      cls: "text-rose-400",    label: "Failed"  },
  cancelled: { icon: XCircle,      cls: "text-muted-foreground", label: "Cancelled" },
}

const GRADE_CLS: Record<string, string> = {
  A: "text-emerald-400", B: "text-blue-400",
  C: "text-yellow-400",  D: "text-orange-400", F: "text-destructive",
}

// ── SweepGroup ────────────────────────────────────────────────────────────────

export function SweepGroup({ runs }: { runs: Run[] }) {
  const [open, setOpen] = useState(false)

  // Sort children: best F1 first (desc), then by created_at
  const sorted = [...runs].sort((a, b) => {
    const fa = getF1(a) ?? -1
    const fb = getF1(b) ?? -1
    if (fb !== fa) return fb - fa
    return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  })

  const bestRun    = sorted[0]
  const bestF1     = bestRun ? getF1(bestRun) : null
  const bestAcc    = bestRun ? getAccuracy(bestRun) : null

  const doneCount    = runs.filter(r => r.status === "completed").length
  const runningCount = runs.filter(r => r.status === "running" || r.status === "pending").length
  const failedCount  = runs.filter(r => r.status === "failed").length

  const overallStatus =
    runningCount > 0 ? "running" :
    failedCount === runs.length ? "failed" :
    doneCount > 0 ? "completed" : "pending"

  const osCfg  = STATUS_CFG[overallStatus] ?? STATUS_CFG.pending
  const OsIcon = osCfg.icon

  const taskLabel = (runs[0]?.task_type ?? "training run").replace(/_/g, " ")

  return (
    <div className="border-b border-border/40 last:border-0">
      {/* ── Collapsed header ── */}
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full group flex items-center gap-3 px-6 py-3.5 hover:bg-surface-elevated/40 transition-colors text-left"
      >
        {/* Expand icon */}
        <span className="shrink-0 text-muted-foreground/50 group-hover:text-muted-foreground transition-colors">
          {open
            ? <ChevronDown className="h-3.5 w-3.5" />
            : <ChevronRight className="h-3.5 w-3.5" />
          }
        </span>

        {/* Sweep icon */}
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-violet-500/15">
          <Zap className="h-3 w-3 text-violet-400" />
        </span>

        {/* Label */}
        <div className="flex-1 min-w-0">
          <span className="text-sm font-medium capitalize text-foreground">{taskLabel}</span>
          <span className="ml-2 font-mono text-[11px] text-muted-foreground">
            Sweep · {runs.length} run{runs.length !== 1 ? "s" : ""}
          </span>
        </div>

        {/* Best result */}
        {bestF1 !== null && (
          <div className="hidden lg:flex flex-col items-end gap-0.5 shrink-0 min-w-[80px]">
            <span className="text-sm font-bold font-mono text-emerald-400">
              {bestAcc !== null ? `${(bestAcc * 100).toFixed(1)}%` : "—"}
            </span>
            <span className="text-[10px] text-muted-foreground font-mono">
              best F1 {bestF1.toFixed(3)}
            </span>
          </div>
        )}

        {/* Progress pills */}
        <div className="hidden md:flex items-center gap-1.5 shrink-0">
          {doneCount > 0 && (
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              {doneCount} done
            </span>
          )}
          {runningCount > 0 && (
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 animate-pulse">
              {runningCount} running
            </span>
          )}
          {failedCount > 0 && (
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20">
              {failedCount} failed
            </span>
          )}
        </div>

        {/* Overall status */}
        <OsIcon className={`h-3.5 w-3.5 shrink-0 ${osCfg.cls}`} />
      </button>

      {/* ── Expanded child runs table ── */}
      {open && (
        <div className="border-t border-border/30 bg-surface-elevated/20">
          {/* Table header */}
          <div className="grid grid-cols-[1.6fr_0.6fr_0.7fr_0.7fr_0.6fr_auto] gap-3 px-8 py-2 border-b border-border/30">
            {["Parameters", "Grade", "Accuracy", "F1", "Status", ""].map(h => (
              <span key={h} className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground/50 font-mono">
                {h}
              </span>
            ))}
          </div>

          {sorted.map((run, i) => {
            const isBest  = i === 0 && getF1(run) !== null
            const cfg     = STATUS_CFG[run.status] ?? STATUS_CFG.pending
            const RunIcon = cfg.icon
            const combo   = run.sweep_config as Record<string, unknown> | null
            const grade   = getGrade(run)
            const acc     = getAccuracy(run)
            const f1      = getF1(run)

            return (
              <Link
                key={run.id}
                href={`/runs/${run.id}`}
                className="group grid grid-cols-[1.6fr_0.6fr_0.7fr_0.7fr_0.6fr_auto] gap-3 items-center px-8 py-3 border-b border-border/20 last:border-0 hover:bg-surface-elevated/50 transition-colors"
              >
                {/* Params */}
                <div className="flex items-center gap-2 min-w-0">
                  {isBest && (
                    <Star className="h-3 w-3 text-amber-400 fill-amber-400 shrink-0" />
                  )}
                  <span className={`font-mono text-[11px] truncate ${isBest ? "text-foreground font-semibold" : "text-muted-foreground"}`}>
                    {combo ? formatCombo(combo) : "—"}
                  </span>
                </div>

                {/* Grade */}
                <span className={`font-mono text-sm font-bold ${grade ? (GRADE_CLS[grade] ?? "text-muted-foreground") : "text-muted-foreground/30"}`}>
                  {grade ?? "—"}
                </span>

                {/* Accuracy */}
                <span className={`font-mono text-sm ${acc !== null ? "text-foreground" : "text-muted-foreground/30"}`}>
                  {acc !== null ? `${(acc * 100).toFixed(1)}%` : "—"}
                </span>

                {/* F1 */}
                <span className={`font-mono text-sm ${f1 !== null ? "text-foreground" : "text-muted-foreground/30"}`}>
                  {f1 !== null ? f1.toFixed(3) : "—"}
                </span>

                {/* Status */}
                <div className={`flex items-center gap-1.5 text-xs ${cfg.cls}`}>
                  <RunIcon className="h-3 w-3 shrink-0" />
                  <span className="hidden md:block">{cfg.label}</span>
                </div>

                {/* Arrow */}
                <ArrowRight className="h-3.5 w-3.5 text-muted-foreground/25 group-hover:text-muted-foreground/55 group-hover:translate-x-0.5 transition-all duration-200 shrink-0" />
              </Link>
            )
          })}
        </div>
      )}
    </div>
  )
}
