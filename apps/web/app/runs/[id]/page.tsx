import { notFound } from "next/navigation"
import { AppShell } from "@/components/layout/app-shell"
import { createClient } from "@/lib/supabase/server"
import { RunDetailClient } from "./run-detail-client"
import type { Run, RunEvent } from "@/lib/supabase/types"

export default async function RunDetailPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  const supabase = await createClient()

  const { data: run } = await supabase
    .from("runs")
    .select("*")
    .eq("id", id)
    .single()

  if (!run) notFound()

  const { data: events } = await supabase
    .from("run_events")
    .select("*")
    .eq("run_id", id)
    .order("created_at", { ascending: true })

  return (
    <AppShell>
      <RunDetailClient run={run as Run} events={(events ?? []) as RunEvent[]} />
    </AppShell>
  )
}
