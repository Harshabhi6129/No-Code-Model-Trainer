"use client"

import { useEffect, useState } from "react"
import { cn } from "@/lib/utils"

const RADIUS       = 26
const STROKE_WIDTH = 3
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

type SpectrumColor = "indigo" | "violet" | "cyan" | "emerald" | "amber"

const spectrum: Record<SpectrumColor, {
  stroke: string
  glow: string
  textClass: string
  bgClass: string
  glowRgba: string
  borderHover: string
}> = {
  indigo:  { stroke: "#818CF8", glow: "rgba(129,140,248,0.5)",  textClass: "text-indigo-300",  bgClass: "bg-indigo-500/12",  glowRgba: "rgba(99,102,241,0.4)",  borderHover: "99,102,241"  },
  violet:  { stroke: "#A78BFA", glow: "rgba(167,139,250,0.5)",  textClass: "text-violet-300",  bgClass: "bg-violet-500/12",  glowRgba: "rgba(139,92,246,0.4)",  borderHover: "139,92,246"  },
  cyan:    { stroke: "#22D3EE", glow: "rgba(34,211,238,0.45)",  textClass: "text-cyan-300",    bgClass: "bg-cyan-500/12",    glowRgba: "rgba(6,182,212,0.35)",  borderHover: "6,182,212"   },
  emerald: { stroke: "#34D399", glow: "rgba(52,211,153,0.45)",  textClass: "text-emerald-300", bgClass: "bg-emerald-500/12", glowRgba: "rgba(16,185,129,0.35)", borderHover: "16,185,129"  },
  amber:   { stroke: "#FBBF24", glow: "rgba(251,191,36,0.45)",  textClass: "text-amber-300",   bgClass: "bg-amber-500/12",   glowRgba: "rgba(245,158,11,0.35)", borderHover: "245,158,11"  },
}

interface StatRingProps {
  value: string | number
  label: string
  icon: React.ElementType
  color?: SpectrumColor
  fillPercent?: number
  sub?: string
  delay?: number
}

export function StatRing({
  value,
  label,
  icon: Icon,
  color = "indigo",
  fillPercent = 0,
  sub,
  delay = 0,
}: StatRingProps) {
  const [mounted, setMounted] = useState(false)
  const [visible, setVisible] = useState(false)
  const c                     = spectrum[color]
  const clampedFill           = Math.min(Math.max(fillPercent, 0), 100)
  const targetOffset          = CIRCUMFERENCE - (clampedFill / 100) * CIRCUMFERENCE

  useEffect(() => {
    setMounted(true)
    const t = setTimeout(() => setVisible(true), delay)
    return () => clearTimeout(t)
  }, [delay])

  return (
    <div
      className={cn(
        "group relative flex flex-col items-center text-center gap-2 rounded-xl p-4 overflow-hidden transition-all duration-300",
        visible ? "animate-slide-up opacity-100" : "opacity-0"
      )}
      style={{
        background: "rgba(12, 20, 32, 0.60)",
        backdropFilter: "blur(20px) saturate(1.6)",
        WebkitBackdropFilter: "blur(20px) saturate(1.6)",
        border: "1px solid rgba(255,255,255,0.07)",
        boxShadow: "0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06)",
        transitionDelay: `${delay}ms`,
        minHeight: 140,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = `rgba(${c.borderHover},0.35)`
        e.currentTarget.style.transform = "translateY(-2px)"
        e.currentTarget.style.boxShadow = `0 12px 40px rgba(0,0,0,0.4), 0 0 24px -8px ${c.glowRgba}, inset 0 1px 0 rgba(255,255,255,0.08)`
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = "rgba(255,255,255,0.07)"
        e.currentTarget.style.transform = "translateY(0)"
        e.currentTarget.style.boxShadow = "0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06)"
      }}
    >
      {/* Corner glow */}
      <div
        className="absolute -top-6 -right-6 h-20 w-20 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none blur-2xl"
        style={{ background: c.glow }}
      />

      {/* SVG ring with icon inside */}
      <div className="relative shrink-0" style={{ width: 64, height: 64 }}>
        <svg width={64} height={64} viewBox="0 0 64 64" className="-rotate-90" aria-hidden>
          <defs>
            <filter id={`ring-glow-${color}`}>
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          {/* Track */}
          <circle cx={32} cy={32} r={RADIUS} fill="none"
            stroke="rgba(255,255,255,0.07)" strokeWidth={STROKE_WIDTH} />
          {/* Fill arc */}
          <circle
            cx={32} cy={32} r={RADIUS}
            fill="none"
            stroke={c.stroke}
            strokeWidth={STROKE_WIDTH}
            strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={mounted ? targetOffset : CIRCUMFERENCE}
            filter={`url(#ring-glow-${color})`}
            style={{
              transition: `stroke-dashoffset 1.2s cubic-bezier(0.34, 1.56, 0.64, 1) ${delay + 100}ms`,
              filter: `drop-shadow(0 0 5px ${c.glow})`,
            }}
          />
        </svg>

        {/* Icon centered in ring */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className={cn("flex h-7 w-7 items-center justify-center rounded-lg", c.bgClass)}>
            <Icon className={cn("h-3.5 w-3.5", c.textClass)} />
          </div>
        </div>
      </div>

      {/* Value */}
      <p
        className={cn("text-2xl font-bold font-mono tracking-tight leading-none animate-count-up", c.textClass)}
        style={{ animationDelay: `${delay + 200}ms` }}
      >
        {value}
      </p>

      {/* Label + sub */}
      <div className="space-y-0.5">
        <p className="text-xs font-semibold text-foreground tracking-tight leading-tight">{label}</p>
        {sub && (
          <p className="text-[10px] text-muted-foreground font-mono leading-tight">{sub}</p>
        )}
      </div>
    </div>
  )
}
