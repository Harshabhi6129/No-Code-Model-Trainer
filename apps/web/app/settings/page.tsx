import { AppShell } from "@/components/layout/app-shell"
import { createClient } from "@/lib/supabase/server"
import { SettingsClient } from "./settings-client"

export default async function SettingsPage() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  let totalRuns     = 0
  let completedRuns = 0
  let fullName      = ""

  if (user?.id) {
    const [runsResult, profileResult] = await Promise.all([
      supabase.from("runs").select("status").eq("user_id", user.id),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (supabase as any).from("profiles").select("full_name").eq("id", user.id).single(),
    ])
    const runsArr = (runsResult.data ?? []) as { status: string }[]
    totalRuns     = runsArr.length
    completedRuns = runsArr.filter(r => r.status === "completed").length
    fullName      = (profileResult.data as { full_name?: string } | null)?.full_name ?? ""
  }

  return (
    <AppShell>
      <SettingsClient
        email={user?.email ?? ""}
        createdAt={user?.created_at ?? ""}
        totalRuns={totalRuns}
        completedRuns={completedRuns}
        initialFullName={fullName}
      />
    </AppShell>
  )
}
