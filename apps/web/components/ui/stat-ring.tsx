"use client"

import { useEffect, useRef, useState } from "react"
import { cn } from "@/lib/utils"

const RADIUS       = 30
const STROKE_WIDTH = 3
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

type SpectrumColor = "indigo" | "violet" | "cyan" | "emerald" | "amber"

const spectrum: Record<SpectrumColor, {
  stroke: string
  glow: string
  textClass: string
  bgClass: string
  trackClass: string
}> = {
  indigo:  { stroke: "hsl(239 84% 67%)", glow: "hsl(239 84% 67% / 0.5)",  textClass: "text-indigo-300",  bgClass: "bg-indigo-500/10",  trackClass: "stroke-indigo-950"  },
  violet:  { stroke: "hsl(262 84% 70%)", glow: "hsl(262 84% 70% / 0.5)",  textClass: "text-violet-300",  bgClass: "bg-violet-500/10",  trackClass: "stroke-violet-950"  },
  cyan:    { stroke: "hsl(187 96% 58%)", glow: "hsl(187 96% 58% / 0.45)", textClass: "text-cyan-300",    bgClass: "bg-cyan-500/10",    trackClass: "stroke-cyan-950"    },
  emerald: { stroke: "hsl(160 64% 52%)", glow: "hsl(160 64% 52% / 0.45)", textClass: "text-emerald-300", bgClass: "bg-emerald-500/10", trackClass: "stroke-emerald-950" },
  amber:   { stroke: "hsl(38 92% 58%)",  glow: "hsl(38 92% 58% / 0.45)",  textClass: "text-amber-300",   bgClass: "bg-amber-500/10",   trackClass: "stroke-amber-950"   },
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
  const [mounted, setMounted]     = useState(false)
  const [visible, setVisible]     = useState(false)
  const ringRef                   = useRef<SVGCircleElement>(null)
  const c                         = spectrum[color]
  const clampedFill               = Math.min(Math.max(fillPercent, 0), 100)
  const targetOffset              = CIRCUMFERENCE - (clampedFill / 100) * CIRCUMFERENCE

  useEffect(() => {
    setMounted(true)
    const t = setTimeout(() => setVisible(true), delay)
    return () => clearTimeout(t)
  }, [delay])

  return (
    <div
      className={cn(
        "group relative flex items-center gap-4 rounded-xl border border-border p-5 overflow-hidden transition-all duration-300",
        "hover:border-border-bright hover:shadow-lg",
        visible ? "animate-slide-up opacity-100" : "opacity-0"
      )}
      style={{
        background: "linear-gradient(135deg, hsl(var(--surface)) 0%, hsl(var(--surface-elevated) / 0.7) 100%)",
        transitionDelay: `${delay}ms`,
      }}
    >
      {/* Subtle corner glow on hover */}
      <div
        className="absolute -top-8 -right-8 h-24 w-24 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none blur-2xl"
        style={{ background: c.glow }}
      />

      {/* SVG ring */}
      <div className="relative shrink-0" style={{ width: 72, height: 72 }}>
        <svg
          width={72}
          height={72}
          viewBox="0 0 72 72"
          className="-rotate-90"
          aria-hidden
        >
          <defs>
            <filter id={`glow-${color}`}>
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Background track */}
          <circle
            cx={36} cy={36} r={RADIUS}
            fill="none"
            stroke="hsl(var(--border))"
            strokeWidth={STROKE_WIDTH}
          />

          {/* Filled arc */}
          <circle
            ref={ringRef}
            cx={36} cy={36} r={RADIUS}
            fill="none"
            stroke={c.stroke}
            strokeWidth={STROKE_WIDTH}
            strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={mounted ? targetOffset : CIRCUMFERENCE}
            filter={`url(#glow-${color})`}
            style={{
              transition: `stroke-dashoffset 1.1s cubic-bezier(0.34, 1.56, 0.64, 1) ${delay + 100}ms`,
              filter: `drop-shadow(0 0 5px ${c.glow})`,
            }}
          />
        </svg>

        {/* Icon centered in ring */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className={cn("flex h-8 w-8 items-center justify-center rounded-lg", c.bgClass)}>
            <Icon className={cn("h-4 w-4", c.textClass)} />
          </div>
        </div>
      </div>

      {/* Text */}
      <div className="flex-1 min-w-0">
        <p
          className={cn("text-3xl font-bold font-mono tracking-tight leading-none animate-count-up", c.textClass)}
          style={{ animationDelay: `${delay + 200}ms` }}
        >
          {value}
        </p>
        <p className="text-sm font-semibold text-foreground mt-1.5 tracking-tight">{label}</p>
        {sub && (
          <p className="text-xs text-muted-foreground mt-0.5 font-mono">{sub}</p>
        )}
      </div>
    </div>
  )
}
