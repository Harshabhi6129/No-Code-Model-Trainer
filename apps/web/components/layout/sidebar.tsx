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
  { href: "/dashboard", label: "Dashboard",     icon: LayoutDashboard, activeColor: "text-indigo-400",  iconBg: "bg-indigo-500/15",  glow: "rgba(99,102,241,0.35)"  },
  { href: "/train",     label: "Train",         icon: Zap,             activeColor: "text-violet-400",  iconBg: "bg-violet-500/15",  glow: "rgba(139,92,246,0.35)"  },
  { href: "/models",    label: "Model Catalog", icon: Library,         activeColor: "text-cyan-400",    iconBg: "bg-cyan-500/15",    glow: "rgba(6,182,212,0.35)"   },
  { href: "/datasets",  label: "Datasets",      icon: HardDrive,       activeColor: "text-emerald-400", iconBg: "bg-emerald-500/15", glow: "rgba(16,185,129,0.35)"  },
  { href: "/runs",      label: "All Runs",      icon: ListOrdered,     activeColor: "text-amber-400",   iconBg: "bg-amber-500/15",   glow: "rgba(245,158,11,0.35)"  },
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
      className="fixed inset-y-0 left-0 z-50 flex w-60 flex-col border-r"
      style={{
        background: "rgba(6, 10, 16, 0.80)",
        backdropFilter: "blur(48px) saturate(2.2)",
        WebkitBackdropFilter: "blur(48px) saturate(2.2)",
        borderColor: "rgba(255,255,255,0.06)",
        boxShadow: "inset -1px 0 0 rgba(99,102,241,0.08), 2px 0 40px rgba(0,0,0,0.4)",
      }}
    >
      {/* ── Top gradient accent line ─────────────────────── */}
      <div
        className="absolute top-0 left-0 right-0 h-px pointer-events-none"
        style={{ background: "linear-gradient(90deg, transparent 0%, rgba(99,102,241,0.6) 40%, rgba(139,92,246,0.6) 70%, transparent 100%)" }}
      />

      {/* ── Logo ──────────────────────────────────────────── */}
      <div
        className="flex h-[60px] items-center gap-3 px-5 shrink-0"
        style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}
      >
        {/* Pulsing glow icon */}
        <div className="relative flex h-9 w-9 items-center justify-center rounded-xl shrink-0 animate-glow-breathe"
          style={{
            background: "linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15))",
            border: "1px solid rgba(99,102,241,0.35)",
            boxShadow: "0 0 20px -4px rgba(99,102,241,0.5), inset 0 1px 0 rgba(255,255,255,0.1)",
          }}
        >
          <BrainCircuit className="h-4.5 w-4.5" style={{ color: "#818CF8" }} />
          {/* Orbit ring */}
          <span
            className="absolute inset-[-4px] rounded-xl animate-neural-pulse pointer-events-none"
            style={{ border: "1px solid rgba(99,102,241,0.2)" }}
          />
        </div>

        <div className="flex flex-col leading-none gap-0.5">
          <span className="text-sm font-bold tracking-tight text-shimmer">ModelForge</span>
          <span className="text-[9px] font-mono tracking-[0.2em] uppercase"
            style={{ color: "rgba(99,102,241,0.6)" }}>
            BETA
          </span>
        </div>
      </div>

      {/* ── Nav section label ─────────────────────────────── */}
      <div className="px-5 pt-5 pb-1.5">
        <span className="text-[9px] font-semibold tracking-[0.18em] uppercase font-mono"
          style={{ color: "rgba(255,255,255,0.2)" }}>
          Platform
        </span>
      </div>

      {/* ── Nav items ─────────────────────────────────────── */}
      <nav className="flex-1 px-2.5 space-y-0.5 overflow-y-auto pb-2">
        {navItems.map(({ href, label, icon: Icon, activeColor, iconBg, glow }) => {
          const active = pathname === href || pathname.startsWith(href + "/")
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "group relative flex items-center gap-3 rounded-[10px] px-3 py-2.5 text-sm font-medium transition-all duration-200",
                active
                  ? "nav-active text-foreground"
                  : "hover:text-foreground"
              )}
              style={active ? {} : { color: "rgba(203,213,225,0.5)" }}
              onMouseEnter={(e) => {
                if (!active) {
                  e.currentTarget.style.background = "rgba(255,255,255,0.04)"
                  e.currentTarget.style.color = "rgba(241,245,249,0.9)"
                }
              }}
              onMouseLeave={(e) => {
                if (!active) {
                  e.currentTarget.style.background = "transparent"
                  e.currentTarget.style.color = "rgba(203,213,225,0.5)"
                }
              }}
            >
              {/* Icon */}
              <div className={cn(
                "relative flex h-7 w-7 items-center justify-center rounded-lg shrink-0 transition-all duration-200",
                active ? iconBg : "group-hover:bg-white/5"
              )}>
                <Icon className={cn(
                  "h-3.5 w-3.5 transition-colors duration-200",
                  active ? activeColor : "group-hover:text-muted-foreground"
                )}
                  style={!active ? { color: "rgba(203,213,225,0.4)" } : {}}
                />
                {active && (
                  <span
                    className="absolute inset-0 rounded-lg pointer-events-none"
                    style={{ boxShadow: `0 0 12px -2px ${glow}` }}
                  />
                )}
              </div>

              <span className="flex-1 tracking-[-0.01em]">{label}</span>

              {active && (
                <ChevronRight className="h-3 w-3 shrink-0" style={{ color: "rgba(255,255,255,0.2)" }} />
              )}
            </Link>
          )
        })}
      </nav>

      {/* ── Divider ───────────────────────────────────────── */}
      <div
        className="mx-4 h-px"
        style={{ background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent)" }}
      />

      {/* ── Bottom nav ────────────────────────────────────── */}
      <div className="px-2.5 pt-2 pb-2 space-y-0.5">
        <div className="px-5 pt-1 pb-1.5">
          <span className="text-[9px] font-semibold tracking-[0.18em] uppercase font-mono"
            style={{ color: "rgba(255,255,255,0.2)" }}>
            Account
          </span>
        </div>

        <Link
          href="/settings"
          className={cn(
            "group relative flex items-center gap-3 rounded-[10px] px-3 py-2.5 text-sm font-medium transition-all duration-200",
            pathname === "/settings" ? "nav-active text-foreground" : ""
          )}
          style={pathname !== "/settings" ? { color: "rgba(203,213,225,0.5)" } : {}}
          onMouseEnter={(e) => {
            if (pathname !== "/settings") {
              e.currentTarget.style.background = "rgba(255,255,255,0.04)"
              e.currentTarget.style.color = "rgba(241,245,249,0.9)"
            }
          }}
          onMouseLeave={(e) => {
            if (pathname !== "/settings") {
              e.currentTarget.style.background = "transparent"
              e.currentTarget.style.color = "rgba(203,213,225,0.5)"
            }
          }}
        >
          <div className="flex h-7 w-7 items-center justify-center rounded-lg group-hover:bg-white/5 shrink-0 transition-all duration-200">
            <Settings className="h-3.5 w-3.5" style={{ color: "rgba(203,213,225,0.4)" }} />
          </div>
          Settings
        </Link>

        <button
          onClick={signOut}
          className="group flex w-full items-center gap-3 rounded-[10px] px-3 py-2.5 text-sm font-medium transition-all duration-200"
          style={{ color: "rgba(203,213,225,0.5)" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "rgba(239,68,68,0.08)"
            e.currentTarget.style.color = "rgba(252,165,165,0.9)"
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent"
            e.currentTarget.style.color = "rgba(203,213,225,0.5)"
          }}
        >
          <div className="flex h-7 w-7 items-center justify-center rounded-lg group-hover:bg-rose-500/12 shrink-0 transition-all duration-200">
            <LogOut className="h-3.5 w-3.5 transition-colors" />
          </div>
          Sign out
        </button>
      </div>

      {/* ── User card ─────────────────────────────────────── */}
      <div className="px-3 pb-3">
        <div
          className="flex items-center gap-3 rounded-xl px-3 py-2.5 transition-all duration-200 cursor-default"
          style={{
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.06)",
            boxShadow: "inset 0 1px 0 rgba(255,255,255,0.04)",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = "rgba(99,102,241,0.2)"
            e.currentTarget.style.background = "rgba(99,102,241,0.06)"
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = "rgba(255,255,255,0.06)"
            e.currentTarget.style.background = "rgba(255,255,255,0.03)"
          }}
        >
          {/* Gradient avatar */}
          <div className="relative shrink-0">
            <div
              className="h-8 w-8 rounded-full flex items-center justify-center text-[11px] font-bold text-white"
              style={{
                background: "linear-gradient(135deg, #6366F1, #8B5CF6)",
                boxShadow: "0 0 14px -2px rgba(99,102,241,0.5)",
              }}
            >
              {initials}
            </div>
            {/* Online dot */}
            <span
              className="absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border-2"
              style={{
                background: "var(--success)",
                borderColor: "var(--bg)",
                boxShadow: "0 0 6px rgba(16,185,129,0.6)",
              }}
            />
          </div>

          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-foreground truncate capitalize">{handle}</p>
            <p className="text-[10px] truncate font-mono" style={{ color: "rgba(100,116,139,0.8)" }}>
              {userEmail}
            </p>
          </div>
        </div>
      </div>
    </aside>
  )
}
