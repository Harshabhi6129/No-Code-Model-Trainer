import { AppShell } from "@/components/layout/app-shell"
import { createClient } from "@/lib/supabase/server"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import Link from "next/link"
import { formatDistanceToNow } from "date-fns"
import {
  Plus, Activity, CheckCircle2, Clock, XCircle,
  Zap, ArrowRight, BarChart3, Cpu
} from "lucide-react"
import type { Run } from "@/lib/supabase/types"

const statusConfig: Record<string, { icon: React.ElementType; color: string; badge: "default" | "secondary" | "destructive" | "outline" }> = {
  completed: { icon: CheckCircle2, color: "text-emerald-400", badge: "default" },
  running:   { icon: Activity,     color: "text-blue-400",    badge: "secondary" },
  pending:   { icon: Clock,        color: "text-yellow-400",  badge: "outline" },
  failed:    { icon: XCircle,      color: "text-destructive", badge: "destructive" },
  cancelled: { icon: XCircle,      color: "text-muted-foreground", badge: "outline" },
}

function StatCard({ label, value, icon: Icon, sub }: { label: string; value: string | number; icon: React.ElementType; sub?: string }) {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm text-muted-foreground mb-1">{label}</p>
            <p className="text-3xl font-bold">{value}</p>
            {sub && <p className="text-xs text-muted-foreground mt-1">{sub}</p>}
          </div>
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
            <Icon className="h-4 w-4 text-primary" />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function RunRow({ run }: { run: Run }) {
  const cfg = statusConfig[run.status] ?? statusConfig.pending
  const StatusIcon = cfg.icon
  const metrics = run.metrics as Record<string, unknown>
  const accuracy = typeof metrics?.accuracy === "number"
    ? `${(metrics.accuracy * 100).toFixed(1)}%` : "—"

  return (
    <Link href={`/runs/${run.id}`} className="flex items-center gap-4 py-3 hover:bg-secondary/50 px-4 -mx-4 rounded-lg transition-colors group">
      <StatusIcon className={`h-4 w-4 shrink-0 ${cfg.color}`} />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">
          {run.task_type?.replace(/_/g, " ") ?? "Training run"}
        </p>
        <p className="text-xs text-muted-foreground truncate font-mono">
          {run.model_id ?? "—"} · {run.dataset_filename ?? "no file"}
        </p>
      </div>
      <div className="flex items-center gap-4 shrink-0">
        <span className="text-sm text-muted-foreground hidden md:block">{accuracy}</span>
        <Badge variant={cfg.badge} className="text-xs capitalize">{run.status}</Badge>
        <span className="text-xs text-muted-foreground hidden lg:block">
          {formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
        </span>
        <ArrowRight className="h-3.5 w-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
      </div>
    </Link>
  )
}

export default async function DashboardPage() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  const { data: runs } = await supabase
    .from("runs")
    .select("*")
    .eq("user_id", user!.id)
    .order("created_at", { ascending: false })
    .limit(10)

  const allRuns: Run[] = runs ?? []
  const completed = allRuns.filter((r) => r.status === "completed").length
  const running   = allRuns.filter((r) => r.status === "running").length
  const avgAcc    = allRuns.reduce((sum, r) => {
    const m = r.metrics as Record<string, unknown>
    return sum + (typeof m?.accuracy === "number" ? m.accuracy : 0)
  }, 0) / Math.max(completed, 1)

  const firstName = user?.email?.split("@")[0] ?? "there"

  return (
    <AppShell>
      <div className="p-8 max-w-5xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-bold">Good to see you, {firstName}</h1>
            <p className="text-muted-foreground text-sm mt-1">
              {allRuns.length === 0
                ? "Start your first training run to see results here."
                : `${allRuns.length} run${allRuns.length !== 1 ? "s" : ""} total`}
            </p>
          </div>
          <Button asChild>
            <Link href="/train">
              <Plus className="h-4 w-4 mr-2" /> New Run
            </Link>
          </Button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard label="Total Runs"  value={allRuns.length} icon={Activity} />
          <StatCard label="Completed"   value={completed}       icon={CheckCircle2} />
          <StatCard label="In Progress" value={running}         icon={Cpu} />
          <StatCard
            label="Avg Accuracy"
            value={completed > 0 ? `${(avgAcc * 100).toFixed(1)}%` : "—"}
            icon={BarChart3}
            sub={completed > 0 ? `across ${completed} run${completed !== 1 ? "s" : ""}` : undefined}
          />
        </div>

        {/* Runs list */}
        <Card>
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-base">Recent Runs</CardTitle>
            {allRuns.length > 5 && (
              <Button variant="ghost" size="sm" asChild>
                <Link href="/runs">View all <ArrowRight className="ml-1.5 h-3.5 w-3.5" /></Link>
              </Button>
            )}
          </CardHeader>
          <CardContent>
            {allRuns.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 gap-4 text-center">
                <div className="flex h-14 w-14 items-center justify-center rounded-full bg-secondary">
                  <Zap className="h-6 w-6 text-muted-foreground" />
                </div>
                <div>
                  <p className="font-medium">No training runs yet</p>
                  <p className="text-sm text-muted-foreground mt-1">Upload a dataset and describe your task to get started.</p>
                </div>
                <Button asChild>
                  <Link href="/train">Start your first run <ArrowRight className="ml-2 h-3.5 w-3.5" /></Link>
                </Button>
              </div>
            ) : (
              <div className="divide-y divide-border/50">
                {allRuns.map((run) => <RunRow key={run.id} run={run} />)}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Quick start cards */}
        {allRuns.length === 0 && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { title: "Text Classification", desc: "Classify support tickets, reviews, or any text by category.", href: "/train" },
              { title: "Named Entity Recognition", desc: "Extract people, places, products from unstructured text.", href: "/train" },
              { title: "LLM Fine-Tuning", desc: "Teach a language model your domain knowledge with LoRA.", href: "/train" },
            ].map((c) => (
              <Link key={c.title} href={c.href} className="p-5 rounded-xl border border-border bg-card hover:border-primary/30 hover:bg-primary/5 transition-colors group">
                <p className="font-semibold text-sm mb-1.5 group-hover:text-primary transition-colors">{c.title}</p>
                <p className="text-xs text-muted-foreground leading-relaxed">{c.desc}</p>
                <ArrowRight className="h-3.5 w-3.5 text-muted-foreground mt-3 group-hover:text-primary group-hover:translate-x-0.5 transition-all" />
              </Link>
            ))}
          </div>
        )}
      </div>
    </AppShell>
  )
}
