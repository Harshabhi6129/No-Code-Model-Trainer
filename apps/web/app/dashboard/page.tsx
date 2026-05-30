import { AppShell } from "@/components/layout/app-shell"
import { createClient } from "@/lib/supabase/server"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { StatRing } from "@/components/ui/stat-ring"
import { PageHeader } from "@/components/ui/page-header"
import Link from "next/link"
import { formatDistanceToNow } from "date-fns"
import {
  Plus, Activity, CheckCircle2, Clock, XCircle, Zap, ArrowRight,
  BarChart3, Cpu, Sparkles, FlaskConical, TrendingUp, Database,
  LayoutDashboard, DollarSign,
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

/* ── Run row ────────────────────────────────────────────────────── */
function RunRow({ run, index }: { run: Run; index: number }) {
  const cfg     = statusCfg[run.status] ?? statusCfg.pending
  const Icon    = cfg.icon
  const metrics = run.metrics as Record<string, unknown>
  const accuracy = typeof metrics?.accuracy === "number"
    ? `${(metrics.accuracy * 100).toFixed(1)}%` : null
  const f1Score  = typeof metrics?.f1 === "number" ? metrics.f1.toFixed(3) : null

  return (
    <Link
      href={`/runs/${run.id}`}
      className="group relative flex items-center gap-4 px-6 py-4 border-b border-border/40 last:border-0 transition-all duration-200 hover:bg-surface-elevated/50"
    >
      {/* Left accent bar on hover */}
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

      {/* Task info */}
      <div className="flex-1 min-w-0 space-y-1">
        <p className="text-sm font-medium capitalize text-foreground">
          {run.task_type?.replace(/_/g, " ") ?? "Training run"}
        </p>
        <p className="text-xs text-muted-foreground font-mono truncate">
          <span className="text-foreground/60">{run.model_id ?? "—"}</span>
          <span className="mx-1.5 opacity-30">·</span>
          {run.dataset_filename ?? "no dataset"}
          {run.dataset_rows && (
            <>
              <span className="mx-1.5 opacity-30">·</span>
              {run.dataset_rows.toLocaleString()} rows
            </>
          )}
        </p>
      </div>

      {/* Metrics */}
      <div className="hidden md:flex flex-col items-end gap-0.5 shrink-0 min-w-[56px]">
        {accuracy ? (
          <>
            <span className="text-sm font-bold font-mono text-foreground">{accuracy}</span>
            {f1Score && (
              <span className="text-[10px] text-muted-foreground font-mono">F1 {f1Score}</span>
            )}
          </>
        ) : (
          <span className="text-sm text-muted-foreground/30 font-mono">—</span>
        )}
      </div>

      {/* Status badge */}
      <Badge variant={cfg.badge} dot pulse={run.status === "running"} className="shrink-0">
        {cfg.label}
      </Badge>

      {/* Time */}
      <span className="text-[11px] text-muted-foreground hidden lg:block shrink-0 w-24 text-right font-mono">
        {formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
      </span>

      <ArrowRight className="h-3.5 w-3.5 text-muted-foreground/30 group-hover:text-muted-foreground/60 group-hover:translate-x-0.5 transition-all duration-200 shrink-0" />
    </Link>
  )
}

/* ── Quick-start cards ──────────────────────────────────────────── */
const quickStart = [
  {
    icon: FlaskConical,
    iconClass: "text-indigo-400",
    iconBg: "bg-indigo-500/10",
    glowColor: "var(--primary-15)",
    title: "Text Classification",
    description: "Classify support tickets, reviews, or any text by category using LoRA fine-tuning.",
    tag: "Most popular",
    tagVariant: "default" as const,
  },
  {
    icon: Database,
    iconClass: "text-violet-400",
    iconBg: "bg-violet-500/10",
    glowColor: "color-mix(in srgb, var(--accent) 16%, transparent)",
    title: "NER · Entity Extraction",
    description: "Extract people, places, and products from unstructured text with token classification.",
    tag: "v0.2",
    tagVariant: "violet" as const,
  },
  {
    icon: Sparkles,
    iconClass: "text-cyan-400",
    iconBg: "bg-cyan-500/10",
    glowColor: "color-mix(in srgb, var(--cyan) 14%, transparent)",
    title: "LLM Fine-Tuning",
    description: "Teach Llama, Mistral, or Qwen your domain knowledge using QLoRA on 4-bit weights.",
    tag: "QLoRA",
    tagVariant: "running" as const,
  },
]

/* ── Page ───────────────────────────────────────────────────────── */
export default async function DashboardPage() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  const { data: runs } = await supabase
    .from("runs")
    .select("*")
    .eq("user_id", user!.id)
    .order("created_at", { ascending: false })
    .limit(8)

  const allRuns   = (runs ?? []) as Run[]
  const completed = allRuns.filter((r) => r.status === "completed").length
  const running   = allRuns.filter((r) => r.status === "running").length
  const pending   = allRuns.filter((r) => r.status === "pending").length

  const avgAcc = completed > 0
    ? allRuns.reduce((sum, r) => {
        const m = r.metrics as Record<string, unknown>
        return sum + (typeof m?.accuracy === "number" ? m.accuracy : 0)
      }, 0) / completed
    : 0

  // Query pipeline_summary events to compute total API spend + cache hit ratio
  const runIds = allRuns.map((r) => r.id)
  let totalApiSpend = 0
  let avgCacheRatio = 0
  if (runIds.length > 0) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { data: costEvents } = await (supabase as any)
      .from("run_events")
      .select("data")
      .in("run_id", runIds)
      .eq("event_type", "agent")

    const summaries = ((costEvents ?? []) as { data: Record<string, unknown> }[])
      .map((e) => e.data)
      .filter((d) => d.agent === "pipeline" && (d.output as Record<string, unknown>)?.type === "pipeline_summary")
      .map((d) => d.output as Record<string, unknown>)

    if (summaries.length > 0) {
      totalApiSpend = summaries.reduce((sum, s) => sum + (typeof s.total_cost_usd === "number" ? s.total_cost_usd : 0), 0)
      avgCacheRatio = Math.round(
        summaries.reduce((sum, s) => sum + (typeof s.overall_cache_hit_ratio === "number" ? s.overall_cache_hit_ratio * 100 : 0), 0) / summaries.length
      )
    }
  }
  const hasSpendData = totalApiSpend > 0

  const firstName = user?.email?.split("@")[0] ?? "there"
  const hour      = new Date().getHours()
  const greeting  = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening"

  return (
    <AppShell>
      {/* ── Page header ────────────────────────────────── */}
      <PageHeader
        icon={LayoutDashboard}
        iconColor="text-indigo-400"
        title="Dashboard"
        description={
          allRuns.length === 0
            ? "No runs yet — start your first training below."
            : `${allRuns.length} run${allRuns.length !== 1 ? "s" : ""} · ${completed} completed`
        }
        actions={
          <Button asChild>
            <Link href="/train">
              <Plus className="h-4 w-4" />
              New Run
            </Link>
          </Button>
        }
      />

      <div className="px-8 py-7 max-w-5xl mx-auto space-y-8 animate-fade-in">

        {/* ── Greeting ───────────────────────────────────── */}
        <div className="space-y-1">
          <h2 className="text-2xl font-bold tracking-tight">
            {greeting},{" "}
            <span className="text-gradient capitalize">{firstName}</span>
          </h2>
          <p className="text-sm text-muted-foreground font-mono">
            {new Date().toLocaleDateString("en-US", {
              weekday: "long", month: "long", day: "numeric",
            })}
            {running > 0 && (
              <span className="ml-3 text-cyan-400">
                · {running} active run{running !== 1 ? "s" : ""}
              </span>
            )}
          </p>
        </div>

        {/* ── Animated stat rings ─────────────────────────── */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          <StatRing
            value={allRuns.length}
            label="Total Runs"
            icon={Activity}
            color="indigo"
            fillPercent={Math.min(allRuns.length * 10, 100)}
            delay={0}
          />
          <StatRing
            value={completed}
            label="Completed"
            icon={CheckCircle2}
            color="emerald"
            fillPercent={allRuns.length > 0 ? (completed / allRuns.length) * 100 : 0}
            sub={allRuns.length > 0 ? `${Math.round((completed / allRuns.length) * 100)}% success rate` : undefined}
            delay={80}
          />
          <StatRing
            value={running + pending}
            label="In Progress"
            icon={Cpu}
            color="cyan"
            fillPercent={(running + pending) > 0 ? 70 : 0}
            sub={running > 0 ? `${running} training` : pending > 0 ? `${pending} queued` : undefined}
            delay={160}
          />
          <StatRing
            value={completed > 0 ? `${(avgAcc * 100).toFixed(1)}%` : "—"}
            label="Avg Accuracy"
            icon={BarChart3}
            color="violet"
            fillPercent={avgAcc * 100}
            sub={completed > 0 ? `over ${completed} run${completed !== 1 ? "s" : ""}` : "no data yet"}
            delay={240}
          />
          <StatRing
            value={hasSpendData ? `$${totalApiSpend.toFixed(2)}` : "$—"}
            label="API Spend"
            icon={DollarSign}
            color="amber"
            fillPercent={hasSpendData ? Math.min((totalApiSpend / 5) * 100, 100) : 0}
            sub={hasSpendData ? `${avgCacheRatio}% cache hits` : "no data yet"}
            delay={320}
          />
        </div>

        {/* ── Recent runs ────────────────────────────────── */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
              <h3 className="text-sm font-semibold tracking-tight">Recent Runs</h3>
            </div>
            {allRuns.length > 5 && (
              <Button variant="ghost" size="sm" asChild>
                <Link href="/runs">
                  View all <ArrowRight className="h-3 w-3" />
                </Link>
              </Button>
            )}
          </div>

          <div
            className="rounded-xl border border-border overflow-hidden"
            style={{ background: "color-mix(in srgb, var(--surface) 60%, transparent)" }}
          >
            {allRuns.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-20 gap-5 text-center px-8">
                <div
                  className="flex h-16 w-16 items-center justify-center rounded-2xl border border-border"
                  style={{
                    background: "linear-gradient(135deg, var(--primary-10), color-mix(in srgb, var(--accent) 6%, transparent))",
                  }}
                >
                  <Zap className="h-7 w-7 text-indigo-400" />
                </div>
                <div className="space-y-1.5">
                  <p className="font-semibold text-foreground">No training runs yet</p>
                  <p className="text-sm text-muted-foreground max-w-sm">
                    Describe your task in plain English — agents handle data cleaning, model selection, training, and evaluation.
                  </p>
                </div>
                <Button asChild>
                  <Link href="/train">
                    <Zap className="h-4 w-4" />
                    Start your first run
                  </Link>
                </Button>
              </div>
            ) : (
              allRuns.map((run, i) => <RunRow key={run.id} run={run} index={i} />)
            )}
          </div>
        </div>

        {/* ── Quick start (only before first run) ────────── */}
        {allRuns.length === 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-muted-foreground" />
              <h3 className="text-sm font-semibold tracking-tight">Task Templates</h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {quickStart.map((c) => (
                <Link
                  key={c.title}
                  href="/train"
                  className="group relative flex flex-col gap-4 p-5 rounded-xl border border-border overflow-hidden transition-all duration-300 hover:border-border-bright"
                  style={{ background: "color-mix(in srgb, var(--surface) 70%, transparent)" }}
                >
                  {/* Corner glow */}
                  <div
                    className="absolute -top-8 -right-8 h-24 w-24 rounded-full blur-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
                    style={{ background: c.glowColor }}
                  />

                  <div className="flex items-start justify-between relative">
                    <div className={`flex h-9 w-9 items-center justify-center rounded-lg ${c.iconBg}`}>
                      <c.icon className={`h-4 w-4 ${c.iconClass}`} />
                    </div>
                    <Badge variant={c.tagVariant} className="text-[10px]">{c.tag}</Badge>
                  </div>

                  <div className="relative">
                    <p className="font-semibold text-sm text-foreground group-hover:text-gradient transition-all duration-300">
                      {c.title}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1.5 leading-relaxed">{c.description}</p>
                  </div>

                  <div className="flex items-center gap-1.5 text-xs text-muted-foreground group-hover:text-indigo-400 transition-colors duration-200 mt-auto relative">
                    <span>Start training</span>
                    <ArrowRight className="h-3 w-3 group-hover:translate-x-0.5 transition-transform duration-200" />
                  </div>
                </Link>
              ))}
            </div>
          </div>
        )}
      </div>
    </AppShell>
  )
}
