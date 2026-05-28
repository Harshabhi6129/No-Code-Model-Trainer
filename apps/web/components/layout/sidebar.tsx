"use client"

import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import {
  BrainCircuit, LayoutDashboard, Zap, ListOrdered,
  LogOut, Settings, Library, HardDrive, ChevronRight,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { createClient } from "@/lib/supabase/client"

const navItems = [
  { href: "/dashboard", label: "Dashboard",     icon: LayoutDashboard, activeColor: "text-indigo-400",  iconBg: "bg-indigo-500/12"  },
  { href: "/train",     label: "Train",         icon: Zap,             activeColor: "text-violet-400",  iconBg: "bg-violet-500/12"  },
  { href: "/models",    label: "Model Catalog", icon: Library,         activeColor: "text-cyan-400",    iconBg: "bg-cyan-500/12"    },
  { href: "/datasets",  label: "Datasets",      icon: HardDrive,       activeColor: "text-emerald-400", iconBg: "bg-emerald-500/12" },
  { href: "/runs",      label: "All Runs",      icon: ListOrdered,     activeColor: "text-amber-400",   iconBg: "bg-amber-500/12"   },
]

interface SidebarProps {
  userEmail?: string
}

export function Sidebar({ userEmail }: SidebarProps) {
  const pathname = usePathname()
  const router   = useRouter()
  const supabase = createClient()

  async function signOut() {
    await supabase.auth.signOut()
    router.push("/auth/login")
    router.refresh()
  }

  const initials = userEmail ? userEmail[0].toUpperCase() : "M"
  const handle   = userEmail?.split("@")[0] ?? "user"

  return (
    <aside
      className="fixed inset-y-0 left-0 z-50 flex w-60 flex-col border-r border-border/70"
      style={{
        background: "linear-gradient(180deg, hsl(222 42% 7% / 0.93) 0%, hsl(222 47% 5% / 0.97) 100%)",
        backdropFilter: "blur(20px) saturate(1.4)",
        WebkitBackdropFilter: "blur(20px) saturate(1.4)",
      }}
    >
      {/* ── Logo ──────────────────────────────────────────── */}
      <div className="flex h-[60px] items-center gap-3 px-5 shrink-0 border-b border-border/50">
        {/* Icon with breathing glow */}
        <div
          className="relative flex h-8 w-8 items-center justify-center rounded-lg shrink-0 animate-glow-breathe"
          style={{
            background: "linear-gradient(135deg, hsl(var(--indigo) / 0.18), hsl(var(--violet) / 0.12))",
            border: "1px solid hsl(var(--indigo) / 0.28)",
          }}
        >
          <BrainCircuit className="h-4 w-4" style={{ color: "hsl(var(--indigo))" }} />
          <span
            className="absolute inset-0 rounded-lg animate-neural-pulse pointer-events-none"
            style={{ border: "1px solid hsl(var(--indigo) / 0.35)" }}
          />
        </div>

        <div className="flex flex-col leading-none gap-0.5">
          <span className="text-sm font-bold tracking-tight text-gradient">ModelForge</span>
          <span className="text-[9px] font-mono tracking-[0.18em] uppercase text-muted-foreground/50">
            BETA
          </span>
        </div>
      </div>

      {/* ── Nav label ─────────────────────────────────────── */}
      <div className="px-5 pt-5 pb-1.5">
        <span className="text-[9px] font-semibold tracking-[0.16em] uppercase text-muted-foreground/40 font-mono">
          Platform
        </span>
      </div>

      {/* ── Nav items ─────────────────────────────────────── */}
      <nav className="flex-1 px-3 space-y-0.5 overflow-y-auto pb-2">
        {navItems.map(({ href, label, icon: Icon, activeColor, iconBg }) => {
          const active = pathname === href || pathname.startsWith(href + "/")
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "group relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
                active
                  ? "nav-active"
                  : "text-muted-foreground hover:bg-surface-high/50 hover:text-foreground"
              )}
            >
              {/* Icon wrapper */}
              <div
                className={cn(
                  "flex h-6 w-6 items-center justify-center rounded-md shrink-0 transition-all duration-200",
                  active ? iconBg : "group-hover:bg-surface-high"
                )}
              >
                <Icon className={cn(
                  "h-3.5 w-3.5 transition-colors duration-200",
                  active ? activeColor : "text-muted-foreground/70 group-hover:text-muted-foreground"
                )} />
              </div>

              <span className="flex-1 tracking-[-0.01em]">{label}</span>

              {active && (
                <ChevronRight className="h-3 w-3 text-muted-foreground/40 shrink-0" />
              )}
            </Link>
          )
        })}
      </nav>

      {/* ── Divider ───────────────────────────────────────── */}
      <div className="mx-4 h-px bg-gradient-to-r from-transparent via-border to-transparent" />

      {/* ── Bottom nav ────────────────────────────────────── */}
      <div className="px-3 pt-2 pb-2 space-y-0.5">
        <div className="px-5 pt-1 pb-1.5">
          <span className="text-[9px] font-semibold tracking-[0.16em] uppercase text-muted-foreground/40 font-mono">
            Account
          </span>
        </div>

        <Link
          href="/settings"
          className={cn(
            "group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
            pathname === "/settings"
              ? "nav-active"
              : "text-muted-foreground hover:bg-surface-high/50 hover:text-foreground"
          )}
        >
          <div className="flex h-6 w-6 items-center justify-center rounded-md group-hover:bg-surface-high shrink-0 transition-all duration-200">
            <Settings className="h-3.5 w-3.5 text-muted-foreground/70 group-hover:text-muted-foreground" />
          </div>
          Settings
        </Link>

        <button
          onClick={signOut}
          className="group flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-muted-foreground transition-all duration-200 hover:text-rose-400"
          style={{ ["--hover-bg" as string]: "hsl(var(--rose) / 0.06)" }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "hsl(var(--rose) / 0.07)")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
        >
          <div className="flex h-6 w-6 items-center justify-center rounded-md group-hover:bg-rose-500/10 shrink-0 transition-all duration-200">
            <LogOut className="h-3.5 w-3.5 group-hover:text-rose-400 transition-colors" />
          </div>
          Sign out
        </button>
      </div>

      {/* ── User card ─────────────────────────────────────── */}
      <div className="px-3 pb-3">
        <div
          className="flex items-center gap-3 rounded-xl px-3 py-2.5 border border-border/50 transition-all duration-200 hover:border-border-bright/60"
          style={{ background: "hsl(var(--surface-elevated) / 0.55)" }}
        >
          {/* Gradient avatar */}
          <div className="relative shrink-0">
            <div
              className="h-8 w-8 rounded-full flex items-center justify-center text-[11px] font-bold text-white"
              style={{ background: "linear-gradient(135deg, hsl(var(--indigo)), hsl(var(--violet)))" }}
            >
              {initials}
            </div>
            {/* Online dot */}
            <span
              className="absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border-2"
              style={{
                background: "hsl(var(--emerald))",
                borderColor: "hsl(var(--bg))",
              }}
            />
          </div>

          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-foreground truncate capitalize">{handle}</p>
            <p className="text-[10px] text-muted-foreground/60 truncate font-mono">{userEmail}</p>
          </div>
        </div>
      </div>
    </aside>
  )
}
