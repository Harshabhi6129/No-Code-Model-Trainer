import { createClient } from "@/lib/supabase/server"
import { MobileNav } from "./mobile-nav"

const DEV_PREVIEW = process.env.NEXT_PUBLIC_DEV_PREVIEW === "true"

export async function AppShell({ children }: { children: React.ReactNode }) {
  let userEmail: string | undefined

  if (!DEV_PREVIEW) {
    const supabase = await createClient()
    const { data: { user } } = await supabase.auth.getUser()
    userEmail = user?.email
  } else {
    userEmail = "demo@modelforge.ai"
  }

  return (
    <div className="flex min-h-screen grain">
      {/* ── Aurora background — fixed, behind everything ── */}
      <div className="aurora-bg" aria-hidden>
        <div className="aurora-orb aurora-1" />
        <div className="aurora-orb aurora-2" />
        <div className="aurora-orb aurora-3" />
        <div className="aurora-orb aurora-4" />
      </div>

      <MobileNav userEmail={userEmail} />

      <main className="flex-1 ml-0 md:ml-60 min-h-screen relative z-10">
        {/* Dot grid */}
        <div className="pointer-events-none fixed inset-0 md:ml-60 bg-grid opacity-40 z-0" />
        {/* Vignette fade at edges */}
        <div
          className="pointer-events-none fixed inset-0 md:ml-60 z-0"
          style={{
            background:
              "radial-gradient(ellipse 90% 55% at 50% 0%, transparent 30%, rgba(6,10,16,0.7) 100%)",
          }}
        />

        {/* Dev preview banner */}
        {DEV_PREVIEW && (
          <div
            className="relative z-20 flex items-center justify-center gap-2 px-4 py-2 text-xs font-mono"
            style={{
              background: "rgba(245,158,11,0.1)",
              borderBottom: "1px solid rgba(245,158,11,0.25)",
              color: "rgba(251,191,36,0.85)",
            }}
          >
            <span
              className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse inline-block"
            />
            DEV PREVIEW — auth bypassed · Supabase queries return empty data · set real credentials to connect
          </div>
        )}

        {/* Page content */}
        <div className="relative z-10">
          {children}
        </div>
      </main>
    </div>
  )
}
