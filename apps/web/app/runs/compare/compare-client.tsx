"use client"

import { useState, useEffect, useMemo } from "react"
import Link from "next/link"
import { formatDistanceToNow } from "date-fns"
import { createClient } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartsTooltip, ResponsiveContainer,
  Legend, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts"
import {
  CheckCircle2, XCircle, Activity, Clock, ArrowLeft,
  Loader2, BarChart3, BrainCircuit, GitCompare,
} from "lucide-react"
import type { Run } from "@/lib/supabase/types"
import { cn } from "@/lib/utils"

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

interface RunMetrics {
  accuracy: number | null
  f1: number | null
  precision: number | null
  recall: number | null
  evaluation_grade: string | null
  summary: string | null
}

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

const STATUS_CFG: Record<string, { icon: React.ElementType; color: string }> = {
  completed: { icon: CheckCircle2, color: "text-emerald-400" },
  running:   { icon: Activity,     color: "text-blue-400" },
  pending:   { icon: Clock,        color: "text-yellow-400" },
  failed:    { icon: XCircle,      color: "text-destructive" },
}

const GRADE_STYLES: Record<string, string> = {
  A: "text-emerald-400 border-emerald-500 bg-emerald-500/10",
  B: "text-blue-400 border-blue-500 bg-blue-500/10",
  C: "text-yellow-400 border-yellow-500 bg-yellow-500/10",
  D: "text-orange-400 border-orange-500 bg-orange-500/10",
  F: "text-destructive border-destructive bg-destructive/10",
}

// Palette for up to 6 compared runs
const PALETTE = ["#7c6fcd","#5ea5f8","#f97316","#4ade80","#e879f9","#facc15"]

function pct(v: number | null): string {
  return v !== null ? `${(v * 100).toFixed(1)}%` : "—"
}
function fmt(v: number | null, decimals = 3): string {
  return v !== null ? v.toFixed(decimals) : "—"
}

function parseMetrics(run: Run): RunMetrics {
  const m = (run.metrics ?? {}) as Record<string, unknown>
  return {
    accuracy:         typeof m.accuracy         === "number" ? m.accuracy         : null,
    f1:               typeof m.f1               === "number" ? m.f1               : null,
    precision:        typeof m.precision        === "number" ? m.precision        : null,
    recall:           typeof m.recall           === "number" ? m.recall           : null,
    evaluation_grade: typeof m.evaluation_grade === "string" ? m.evaluation_grade : null,
    summary:          typeof m.summary          === "string" ? m.summary          : null,
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Main component
// ──────────────────────────────────────────────────────────────────────────────

export function CompareClient() {
  const supabase = createClient()
  const [runs, setRuns]         = useState<Run[]>([])
  const [loading, setLoading]   = useState(true)
  const [selected, setSelected] = useState<Set<string>>(new Set())

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  useEffect(() => {
    async function fetchRuns() {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data: { user } } = await (supabase as any).auth.getUser()
      if (!user) { setLoading(false); return }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { data } = await (supabase as any)
        .from("runs")
        .select("*")
        .eq("user_id", user.id)
        .order("created_at", { ascending: false })
      setRuns(data ?? [])
      setLoading(false)
    }
    fetchRuns()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function toggleRun(id: string) {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else if (next.size < 6) {
        next.add(id)
      } else {
        // replace oldest selection — just show a visual hint instead
      }
      return next
    })
  }

  const selectedRuns = useMemo(
    () => runs.filter(r => selected.has(r.id)),
    [runs, selected]
  )

  // Chart data
  const barData = useMemo(() => {
    if (selectedRuns.length === 0) return []
    return ["Accuracy", "F1", "Precision", "Recall"].map(metric => {
      const key = metric.toLowerCase() as "accuracy" | "f1" | "precision" | "recall"
      const entry: Record<string, string | number> = { metric }
      selectedRuns.forEach((r, i) => {
        const m = parseMetrics(r)
        entry[`run${i}`] = m[key] !== null ? +(m[key]! * 100).toFixed(2) : 0
      })
      return entry
    })
  }, [selectedRuns])

  const radarData = useMemo(() => {
    if (selectedRuns.length < 2) return []
    return ["Accuracy","F1","Precision","Recall"].map(metric => {
      const key = metric.toLowerCase() as "accuracy" | "f1" | "precision" | "recall"
      const entry: Record<string, string | number> = { metric }
      selectedRuns.forEach((r, i) => {
        const m = parseMetrics(r)
        entry[`run${i}`] = m[key] !== null ? +(m[key]! * 100).toFixed(1) : 0
      })
      return entry
    })
  }, [selectedRuns])

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Button variant="outline" size="sm" asChild>
          <Link href="/runs"><ArrowLeft className="h-3.5 w-3.5 mr-1" /> Back</Link>
        </Button>
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Compare Runs</h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            Select up to 6 runs to compare metrics side-by-side.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left: run selector */}
        <div className="xl:col-span-1 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">All Runs</h2>
            {selected.size > 0 && (
              <button
                onClick={() => setSelected(new Set())}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                Clear ({selected.size})
              </button>
            )}
          </div>

          {loading ? (
            <div className="flex items-center gap-2 py-8 justify-center text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-sm">Loading runs…</span>
            </div>
          ) : runs.length === 0 ? (
            <div className="text-center py-10 text-sm text-muted-foreground">
              No completed runs yet.
            </div>
          ) : (
            <div className="space-y-2 max-h-[70vh] overflow-y-auto pr-1">
              {runs.map((run, idx) => {
                const isSelected = selected.has(run.id)
                const cfg = STATUS_CFG[run.status] ?? STATUS_CFG.pending
                const StatusIcon = cfg.icon
                const m = parseMetrics(run)
                const colorIdx = [...selected].indexOf(run.id)

                return (
                  <button
                    key={run.id}
                    onClick={() => toggleRun(run.id)}
                    className={cn(
                      "w-full text-left rounded-xl border p-3 transition-all",
                      isSelected
                        ? "border-primary bg-primary/5 ring-1 ring-primary"
                        : "border-border hover:border-primary/40 hover:bg-secondary/30"
                    )}
                  >
                    <div className="flex items-center gap-2">
                      <div
                        className="h-3 w-3 rounded-full shrink-0"
                        style={{ background: isSelected ? PALETTE[colorIdx] : "transparent", border: `2px solid ${isSelected ? PALETTE[colorIdx] : "hsl(var(--border))"}` }}
                      />
                      <StatusIcon className={cn("h-3.5 w-3.5 shrink-0", cfg.color)} />
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-medium truncate capitalize">
                          {run.task_type?.replace(/_/g, " ") ?? "Training run"}
                        </p>
                        <p className="text-[10px] text-muted-foreground font-mono truncate">
                          {run.model_id ?? "—"} · {run.dataset_rows?.toLocaleString() ?? "?"} rows
                        </p>
                      </div>
                      {m.evaluation_grade && (
                        <span className={cn("text-xs font-bold px-1.5 py-0.5 rounded border shrink-0", GRADE_STYLES[m.evaluation_grade] ?? "")}>
                          {m.evaluation_grade}
                        </span>
                      )}
                    </div>
                    {m.accuracy !== null && (
                      <p className="text-[10px] text-muted-foreground mt-1.5 pl-5">
                        Acc {pct(m.accuracy)} · F1 {fmt(m.f1)}
                      </p>
                    )}
                    <p className="text-[10px] text-muted-foreground/60 mt-0.5 pl-5">
                      {formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
                    </p>
                  </button>
                )
              })}
            </div>
          )}
        </div>

        {/* Right: comparison */}
        <div className="xl:col-span-2 space-y-5">
          {selectedRuns.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 gap-4 text-center rounded-xl border-2 border-dashed border-border">
              <GitCompare className="h-8 w-8 text-muted-foreground/40" />
              <div className="space-y-1">
                <p className="font-medium">Select runs to compare</p>
                <p className="text-sm text-muted-foreground">Choose 2 or more runs from the list.</p>
              </div>
            </div>
          ) : (
            <>
              {/* Metric table */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-primary" /> Metrics Comparison
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border">
                          <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground w-32">Metric</th>
                          {selectedRuns.map((run, i) => (
                            <th key={run.id} className="text-right py-2 px-3 text-xs font-medium">
                              <div className="flex items-center justify-end gap-1.5">
                                <div className="h-2.5 w-2.5 rounded-full" style={{ background: PALETTE[i] }} />
                                <span className="truncate max-w-[120px]">
                                  {run.task_type?.replace(/_/g, " ") ?? run.model_id ?? run.id.slice(0, 8)}
                                </span>
                              </div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {/* Grade */}
                        <tr className="border-b border-border/50">
                          <td className="py-2.5 px-3 text-xs text-muted-foreground">Grade</td>
                          {selectedRuns.map((run, i) => {
                            const m = parseMetrics(run)
                            return (
                              <td key={run.id} className="py-2.5 px-3 text-right">
                                {m.evaluation_grade ? (
                                  <span className={cn("text-sm font-bold px-2 py-0.5 rounded border", GRADE_STYLES[m.evaluation_grade] ?? "")}>
                                    {m.evaluation_grade}
                                  </span>
                                ) : "—"}
                              </td>
                            )
                          })}
                        </tr>
                        {/* Accuracy */}
                        <tr className="border-b border-border/50">
                          <td className="py-2.5 px-3 text-xs text-muted-foreground">Accuracy</td>
                          {selectedRuns.map((run, i) => {
                            const m = parseMetrics(run)
                            const best = Math.max(...selectedRuns.map(r => parseMetrics(r).accuracy ?? -1))
                            return (
                              <td key={run.id} className={cn("py-2.5 px-3 text-right font-mono text-sm", m.accuracy === best && m.accuracy !== null ? "text-emerald-400 font-bold" : "")}>
                                {pct(m.accuracy)}
                              </td>
                            )
                          })}
                        </tr>
                        {/* F1 */}
                        <tr className="border-b border-border/50">
                          <td className="py-2.5 px-3 text-xs text-muted-foreground">F1 Score</td>
                          {selectedRuns.map((run, i) => {
                            const m = parseMetrics(run)
                            const best = Math.max(...selectedRuns.map(r => parseMetrics(r).f1 ?? -1))
                            return (
                              <td key={run.id} className={cn("py-2.5 px-3 text-right font-mono text-sm", m.f1 === best && m.f1 !== null ? "text-emerald-400 font-bold" : "")}>
                                {fmt(m.f1)}
                              </td>
                            )
                          })}
                        </tr>
                        {/* Precision */}
                        <tr className="border-b border-border/50">
                          <td className="py-2.5 px-3 text-xs text-muted-foreground">Precision</td>
                          {selectedRuns.map((run, i) => {
                            const m = parseMetrics(run)
                            return (
                              <td key={run.id} className="py-2.5 px-3 text-right font-mono text-sm text-muted-foreground">
                                {pct(m.precision)}
                              </td>
                            )
                          })}
                        </tr>
                        {/* Recall */}
                        <tr className="border-b border-border/50">
                          <td className="py-2.5 px-3 text-xs text-muted-foreground">Recall</td>
                          {selectedRuns.map((run, i) => {
                            const m = parseMetrics(run)
                            return (
                              <td key={run.id} className="py-2.5 px-3 text-right font-mono text-sm text-muted-foreground">
                                {pct(m.recall)}
                              </td>
                            )
                          })}
                        </tr>
                        {/* Model */}
                        <tr className="border-b border-border/50">
                          <td className="py-2.5 px-3 text-xs text-muted-foreground">Model</td>
                          {selectedRuns.map(run => (
                            <td key={run.id} className="py-2.5 px-3 text-right font-mono text-xs text-muted-foreground max-w-[140px] truncate">
                              {run.model_id ?? "—"}
                            </td>
                          ))}
                        </tr>
                        {/* Dataset */}
                        <tr className="border-b border-border/50">
                          <td className="py-2.5 px-3 text-xs text-muted-foreground">Dataset</td>
                          {selectedRuns.map(run => (
                            <td key={run.id} className="py-2.5 px-3 text-right text-xs text-muted-foreground">
                              {run.dataset_rows?.toLocaleString() ?? "—"} rows
                            </td>
                          ))}
                        </tr>
                        {/* Link */}
                        <tr>
                          <td className="py-2.5 px-3 text-xs text-muted-foreground">Report</td>
                          {selectedRuns.map(run => (
                            <td key={run.id} className="py-2.5 px-3 text-right">
                              <Link href={`/runs/${run.id}`} className="text-xs text-primary hover:underline">
                                View →
                              </Link>
                            </td>
                          ))}
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>

              {/* Bar chart */}
              {selectedRuns.length >= 2 && barData.length > 0 && (
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-primary" /> Metric Breakdown (%)
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={barData} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(224 18% 14%)" vertical={false} />
                        <XAxis dataKey="metric" tick={{ fontSize: 11, fill: "hsl(215 16% 55%)" }} axisLine={false} tickLine={false} />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "hsl(215 16% 55%)" }} axisLine={false} tickLine={false} width={28} unit="%" />
                        <RechartsTooltip
                          formatter={(v, name) => [`${v}%`, `Run ${String(name).replace("run", "")}`]}
                          contentStyle={{ background: "hsl(224 20% 9%)", border: "1px solid hsl(224 18% 14%)", borderRadius: "8px", fontSize: "12px" }}
                        />
                        <Legend formatter={(v) => `Run ${v.replace("run", "")}`} iconType="circle" wrapperStyle={{ fontSize: "11px" }} />
                        {selectedRuns.map((_, i) => (
                          <Bar key={i} dataKey={`run${i}`} fill={PALETTE[i]} radius={[4, 4, 0, 0]} maxBarSize={32} />
                        ))}
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}

              {/* Radar chart (3+ runs) */}
              {selectedRuns.length >= 2 && radarData.length > 0 && (
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <BrainCircuit className="h-4 w-4 text-primary" /> Performance Radar
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="flex justify-center">
                    <RadarChart outerRadius={90} width={380} height={260} data={radarData}>
                      <PolarGrid stroke="hsl(224 18% 18%)" />
                      <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: "hsl(215 16% 55%)" }} />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9, fill: "hsl(215 16% 45%)" }} />
                      {selectedRuns.map((_, i) => (
                        <Radar
                          key={i}
                          name={`Run ${i + 1}`}
                          dataKey={`run${i}`}
                          stroke={PALETTE[i]}
                          fill={PALETTE[i]}
                          fillOpacity={0.15}
                          strokeWidth={2}
                        />
                      ))}
                      <Legend iconType="circle" wrapperStyle={{ fontSize: "11px" }} />
                    </RadarChart>
                  </CardContent>
                </Card>
              )}

              {/* Eval summaries */}
              {selectedRuns.some(r => parseMetrics(r).summary) && (
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">AI Evaluation Summaries</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {selectedRuns.map((run, i) => {
                      const m = parseMetrics(run)
                      if (!m.summary) return null
                      return (
                        <div key={run.id} className="rounded-lg bg-secondary/50 p-3">
                          <div className="flex items-center gap-2 mb-1.5">
                            <div className="h-2.5 w-2.5 rounded-full" style={{ background: PALETTE[i] }} />
                            <p className="text-xs font-medium capitalize">
                              {run.task_type?.replace(/_/g, " ") ?? run.model_id ?? `Run ${i + 1}`}
                            </p>
                            {m.evaluation_grade && (
                              <span className={cn("text-[10px] font-bold px-1.5 py-0.5 rounded border ml-auto", GRADE_STYLES[m.evaluation_grade] ?? "")}>
                                {m.evaluation_grade}
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground leading-relaxed">{m.summary}</p>
                        </div>
                      )
                    })}
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
