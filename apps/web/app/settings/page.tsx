import { AppShell } from "@/components/layout/app-shell"
import { createClient } from "@/lib/supabase/server"
import { SettingsClient } from "./settings-client"

export default async function SettingsPage() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  // Fetch run stats for the profile card
  let totalRuns = 0
  let completedRuns = 0
  if (user?.id) {
    const { data: runs } = await supabase
      .from("runs")
      .select("status")
      .eq("user_id", user.id)
    const runsArr = (runs ?? []) as { status: string }[]
    totalRuns = runsArr.length
    completedRuns = runsArr.filter(r => r.status === "completed").length
  }

  return (
    <AppShell>
      <SettingsClient
        email={user?.email ?? ""}
        createdAt={user?.created_at ?? ""}
        totalRuns={totalRuns}
        completedRuns={completedRuns}
      />
    </AppShell>
  )
}
