import { createClient } from "@/lib/supabase/server"
import { Sidebar } from "./sidebar"

export async function AppShell({ children }: { children: React.ReactNode }) {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  return (
    <div className="flex min-h-screen grain">
      <Sidebar userEmail={user?.email} />
      {/* Grid background on main content for lab-precision feel */}
      <main className="flex-1 ml-60 min-h-screen bg-grid relative">
        {/* Radial vignette so grid fades at edges */}
        <div
          className="pointer-events-none absolute inset-0 z-0"
          style={{
            background: "radial-gradient(ellipse 80% 60% at 50% 0%, transparent 40%, hsl(var(--bg) / 0.6) 100%)",
          }}
        />
        <div className="relative z-10">
          {children}
        </div>
      </main>
    </div>
  )
}
