import { AppShell } from "@/components/layout/app-shell"
import { createClient } from "@/lib/supabase/server"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { PageHeader } from "@/components/ui/page-header"
import Link from "next/link"
import {
  Plus, GitCompare, ListOrdered, Filter,
} from "lucide-react"
import type { Run } from "@/lib/supabase/types"
import { SweepGroup } from "./sweep-group"
import { RunsList } from "./runs-list"

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

        {/* Standalone runs — searchable, filterable, paginated client component */}
        {standaloneRuns.length > 0 || sweepGroups.length === 0 ? (
          <RunsList runs={standaloneRuns} />
        ) : (
          <div
            className="rounded-xl border border-border overflow-hidden"
            style={{ background: "color-mix(in srgb, var(--surface) 60%, transparent)" }}
          >
            <div className="flex items-center justify-center py-10 text-sm text-muted-foreground">
              All runs are part of sweeps above.
            </div>
          </div>
        )}
      </div>
    </AppShell>
  )
}
