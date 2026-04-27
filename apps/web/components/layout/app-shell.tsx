import { createClient } from "@/lib/supabase/server"
import { Sidebar } from "./sidebar"

export async function AppShell({ children }: { children: React.ReactNode }) {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  return (
    <div className="flex min-h-screen">
      <Sidebar userEmail={user?.email} />
      <main className="flex-1 ml-60 min-h-screen bg-background">
        {children}
      </main>
    </div>
  )
}
