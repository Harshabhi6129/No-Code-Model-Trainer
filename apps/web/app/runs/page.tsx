import { AppShell } from "@/components/layout/app-shell"
import { createClient } from "@/lib/supabase/server"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import Link from "next/link"
import { formatDistanceToNow } from "date-fns"
import { Plus, Activity, CheckCircle2, Clock, XCircle, ArrowRight } from "lucide-react"
import type { Run } from "@/lib/supabase/types"

const statusConfig: Record<string, { icon: React.ElementType; color: string; badge: "default" | "secondary" | "destructive" | "outline" }> = {
  completed: { icon: CheckCircle2, color: "text-emerald-400", badge: "default" },
  running:   { icon: Activity,     color: "text-blue-400",    badge: "secondary" },
  pending:   { icon: Clock,        color: "text-yellow-400",  badge: "outline" },
  failed:    { icon: XCircle,      color: "text-destructive", badge: "destructive" },
  cancelled: { icon: XCircle,      color: "text-muted-foreground", badge: "outline" },
}

function RunRow({ run }: { run: Run }) {
  const cfg = statusConfig[run.status] ?? statusConfig.pending
  const StatusIcon = cfg.icon
  const metrics = run.metrics as Record<string, unknown>
  const accuracy = typeof metrics?.accuracy === "number" ? `${(metrics.accuracy * 100).toFixed(1)}%` : "—"
  const f1 = typeof metrics?.f1 === "number" ? `F1 ${metrics.f1.toFixed(3)}` : null

  return (
    <Link href={`/runs/${run.id}`}
      className="flex items-center gap-4 py-4 hover:bg-secondary/40 px-4 -mx-4 rounded-xl transition-colors group border-b border-border/40 last:border-0">
      <StatusIcon className={`h-4 w-4 shrink-0 ${cfg.color}`} />
      <div className="flex-1 min-w-0 space-y-0.5">
        <p className="text-sm font-medium capitalize">
          {run.task_type?.replace(/_/g, " ") ?? "Training run"}
        </p>
        <p className="text-xs text-muted-foreground font-mono">
          {run.model_id ?? "—"} · {run.dataset_filename ?? "no dataset"} · {run.dataset_rows?.toLocaleString() ?? "?"} rows
        </p>
      </div>
      <div className="flex items-center gap-4 shrink-0 text-sm">
        <div className="hidden md:flex flex-col items-end gap-0.5">
          <span className="font-medium">{accuracy}</span>
          {f1 && <span className="text-xs text-muted-foreground">{f1}</span>}
        </div>
        <Badge variant={cfg.badge} className="text-xs capitalize w-20 justify-center">{run.status}</Badge>
        <span className="text-xs text-muted-foreground hidden lg:block w-28 text-right">
          {formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
        </span>
        <ArrowRight className="h-3.5 w-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
      </div>
    </Link>
  )
}

export default async function RunsPage() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  const { data: runs } = await supabase
    .from("runs")
    .select("*")
    .eq("user_id", user!.id)
    .order("created_at", { ascending: false })

  const allRuns: Run[] = runs ?? []

  return (
    <AppShell>
      <div className="p-8 max-w-5xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">All Runs</h1>
            <p className="text-muted-foreground text-sm mt-1">{allRuns.length} training run{allRuns.length !== 1 ? "s" : ""}</p>
          </div>
          <Button asChild>
            <Link href="/train"><Plus className="h-4 w-4 mr-2" />New Run</Link>
          </Button>
        </div>

        <Card>
          <CardContent className="pt-2">
            {allRuns.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-20 gap-4 text-center">
                <p className="font-medium">No runs yet</p>
                <p className="text-sm text-muted-foreground">Start a training run to see it here.</p>
                <Button asChild><Link href="/train">Start first run <ArrowRight className="ml-2 h-3.5 w-3.5" /></Link></Button>
              </div>
            ) : (
              <div>{allRuns.map((run) => <RunRow key={run.id} run={run} />)}</div>
            )}
          </CardContent>
        </Card>
      </div>
    </AppShell>
  )
}
