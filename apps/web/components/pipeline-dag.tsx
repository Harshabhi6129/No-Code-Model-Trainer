"use client"

/**
 * PipelineDAG — visual 7-node agent pipeline for the training page.
 *
 * Shows: Intent → Data → Clean → Model → Train → Eval → Deploy
 * Each node displays: status icon, elapsed time, token cost, data-flow label on arrow.
 *
 * Replaces the text-only agent progress list in train-client.tsx.
 */

import { useMemo } from "react"
import {
  BrainCircuit, Database, Settings2, Cpu, BarChart3,
  Rocket, CheckCircle2, XCircle, Clock, Loader2, SkipForward,
} from "lucide-react"
import { cn } from "@/lib/utils"

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

type NodeStatus = "pending" | "running" | "completed" | "failed" | "skipped"

interface AgentMessage {
  agent: string
  success: boolean
  message: string
  output: Record<string, unknown>
  metadata?: Record<string, unknown>
}

interface PipelineDAGProps {
  /** All SSE messages received so far */
  messages: AgentMessage[]
  /** True while the SSE stream is still open */
  streaming: boolean
}

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

const AGENTS: { name: string; icon: React.ElementType; dataLabel: string }[] = [
  { name: "Intent", icon: BrainCircuit, dataLabel: "TaskSpec" },
  { name: "Data",   icon: Database,     dataLabel: "DataProfile" },
  { name: "Clean",  icon: Settings2,    dataLabel: "CleanResult" },
  { name: "Model",  icon: Cpu,          dataLabel: "ModelRecipe" },
  { name: "Train",  icon: Cpu,          dataLabel: "TrainedModel" },
  { name: "Eval",   icon: BarChart3,    dataLabel: "EvalReport" },
  { name: "Deploy", icon: Rocket,       dataLabel: "— done —" },
]

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

function fmt_cost(usd: number): string {
  if (usd < 0.0001) return "<$0.0001"
  if (usd < 0.01)   return `$${usd.toFixed(4)}`
  return `$${usd.toFixed(3)}`
}

function fmt_ms(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

// Derive per-agent status from the message list
function deriveStatuses(
  messages: AgentMessage[],
  streaming: boolean,
): Record<string, NodeStatus> {
  const result: Record<string, NodeStatus> = {}

  // Map each agent name to its FINAL message (last one that isn't a keepalive)
  const finalMsg: Record<string, AgentMessage> = {}
  for (const m of messages) {
    if (m.agent === "Pipeline") continue  // summary event — skip
    if (m.output?.final === false) {
      // keepalive / progress event — agent is running
      if (!finalMsg[m.agent]) finalMsg[m.agent] = m
    } else {
      finalMsg[m.agent] = m
    }
  }

  let failedSeen = false
  for (const { name } of AGENTS) {
    const m = finalMsg[name]
    if (!m) {
      result[name] = failedSeen ? "skipped" : "pending"
      continue
    }
    if (!m.success) {
      result[name] = "failed"
      failedSeen = true
      continue
    }
    // Check if it's still running (keepalive present but final not yet received)
    const isFinalMsg = m.output?.final !== false
    if (!isFinalMsg && streaming) {
      result[name] = "running"
    } else if (m.agent === "Train" && streaming && m.output?.status === "training") {
      result[name] = "running"
    } else if (m.agent === "Train" && streaming && !isFinalMsg) {
      result[name] = "running"
    } else {
      result[name] = "completed"
    }
  }

  // The currently-streaming agent is "running" if not yet final
  if (streaming && !failedSeen) {
    for (const { name } of AGENTS) {
      if (result[name] === "pending") {
        // The first pending agent after all completed ones is the running one
        const prevAllDone = AGENTS
          .slice(0, AGENTS.findIndex(a => a.name === name))
          .every(a => result[a.name] === "completed" || result[a.name] === "skipped")
        if (prevAllDone && finalMsg[name]?.output?.final === false) {
          result[name] = "running"
        }
        break
      }
    }
  }

  return result
}

// ──────────────────────────────────────────────────────────────────────────────
// Sub-components
// ──────────────────────────────────────────────────────────────────────────────

function StatusIcon({ status }: { status: NodeStatus }) {
  switch (status) {
    case "completed": return <CheckCircle2 className="h-4 w-4 text-emerald-400" />
    case "failed":    return <XCircle      className="h-4 w-4 text-destructive" />
    case "skipped":   return <SkipForward  className="h-4 w-4 text-muted-foreground" />
    case "running":   return <Loader2      className="h-4 w-4 text-primary animate-spin" />
    default:          return <Clock        className="h-4 w-4 text-muted-foreground/40" />
  }
}

function NodeBadge({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-1 bg-secondary/60 border border-border/50 rounded px-1.5 py-0.5">
      <span className="text-[9px] text-muted-foreground">{label}</span>
      <span className="text-[10px] font-mono text-foreground/80">{value}</span>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Main component
// ──────────────────────────────────────────────────────────────────────────────

export function PipelineDAG({ messages, streaming }: PipelineDAGProps) {
  const statuses = useMemo(() => deriveStatuses(messages, streaming), [messages, streaming])

  // Pipeline summary (last message from "Pipeline" agent)
  const summaryMsg = messages.findLast(m => m.agent === "Pipeline")
  const perStage: Record<string, { cost: number; latency: number }> = useMemo(() => {
    const out: Record<string, { cost: number; latency: number }> = {}
    const stageList = (summaryMsg?.metadata?.stage_metrics ?? []) as {
      agent: string; estimated_cost_usd: number; latency_ms: number
    }[]
    for (const s of stageList) {
      out[s.agent] = { cost: s.estimated_cost_usd, latency: s.latency_ms }
    }
    return out
  }, [summaryMsg])

  return (
    <div className="w-full overflow-x-auto">
      <div className="flex items-start justify-start gap-0 min-w-[700px] pb-2">
        {AGENTS.map(({ name, icon: Icon, dataLabel }, idx) => {
          const status  = statuses[name] ?? "pending"
          const metrics = perStage[name]
          const isLast  = idx === AGENTS.length - 1

          return (
            <div key={name} className="flex items-start">
              {/* Node */}
              <div className="flex flex-col items-center gap-1.5">
                <div
                  className={cn(
                    "relative flex flex-col items-center gap-1 px-3 py-2.5 rounded-xl border transition-all duration-300",
                    "bg-card min-w-[88px]",
                    status === "completed" && "border-emerald-500/40 bg-emerald-500/5",
                    status === "running"   && "border-primary/60 bg-primary/5 shadow-[0_0_12px_rgba(124,111,205,0.25)] animate-pulse",
                    status === "failed"    && "border-destructive/40 bg-destructive/5",
                    status === "skipped"   && "border-border/30 opacity-40",
                    status === "pending"   && "border-border/40 opacity-60",
                  )}
                >
                  {/* Icon + status */}
                  <div className="flex items-center gap-1.5">
                    <Icon className={cn(
                      "h-3.5 w-3.5",
                      status === "completed" && "text-emerald-400",
                      status === "running"   && "text-primary",
                      status === "failed"    && "text-destructive",
                      (status === "pending" || status === "skipped") && "text-muted-foreground/50",
                    )} />
                    <StatusIcon status={status} />
                  </div>

                  {/* Agent name */}
                  <span className={cn(
                    "text-[11px] font-semibold tracking-wide",
                    status === "completed" && "text-emerald-400",
                    status === "running"   && "text-primary",
                    status === "failed"    && "text-destructive",
                    (status === "pending" || status === "skipped") && "text-muted-foreground/50",
                  )}>
                    {name}
                  </span>

                  {/* Metrics badges (only if available) */}
                  {metrics && (
                    <div className="flex flex-col items-center gap-0.5 mt-0.5">
                      {metrics.latency > 0 && (
                        <NodeBadge label="⏱" value={fmt_ms(metrics.latency)} />
                      )}
                      {metrics.cost > 0 && (
                        <NodeBadge label="$" value={fmt_cost(metrics.cost)} />
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Arrow + data label */}
              {!isLast && (
                <div className="flex flex-col items-center justify-center pt-4 mx-0.5">
                  <span className="text-[9px] text-muted-foreground/50 mb-0.5 whitespace-nowrap">
                    {dataLabel}
                  </span>
                  <div className="flex items-center gap-0">
                    <div className="h-px w-6 bg-border/40" />
                    <div className="border-l border-t border-border/40 h-2 w-2 rotate-45 translate-x-[-3px]" />
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Pipeline summary row (total cost + cache hit) */}
      {summaryMsg && (
        <div className="flex items-center gap-3 mt-2 text-[10px] text-muted-foreground/60">
          <span>Pipeline total:</span>
          {summaryMsg.output.total_cost_usd !== undefined && (
            <span className="font-mono">{fmt_cost(summaryMsg.output.total_cost_usd as number)}</span>
          )}
          {summaryMsg.output.overall_cache_hit_ratio !== undefined && (
            <span>
              Cache hit: {((summaryMsg.output.overall_cache_hit_ratio as number) * 100).toFixed(0)}%
            </span>
          )}
          {summaryMsg.output.llm_stages_called !== undefined && (
            <span>{summaryMsg.output.llm_stages_called as number} LLM stages</span>
          )}
        </div>
      )}
    </div>
  )
}
