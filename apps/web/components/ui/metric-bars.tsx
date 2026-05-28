"use client"

import { cn } from "@/lib/utils"

interface Metric {
  label: string
  value: number   // 0–1
  color?: string  // CSS color string; defaults to primary gradient
}

interface MetricBarsProps {
  metrics: Metric[]
  animateKey?: string | number
  className?: string
}

export function MetricBars({ metrics, animateKey, className }: MetricBarsProps) {
  return (
    <div className={cn("flex flex-col gap-5", className)}>
      {metrics.map((m, i) => (
        <div key={m.label}>
          <div className="flex justify-between items-baseline mb-1.5">
            <span className="text-[13px] text-muted-foreground">{m.label}</span>
            <span className="text-sm font-bold font-mono" style={{ color: "var(--heading)" }}>
              {(m.value * 100).toFixed(1)}%
            </span>
          </div>
          <div
            className="h-2 rounded-full overflow-hidden"
            style={{ background: "var(--border-c)" }}
          >
            <div
              key={animateKey ?? m.label}
              className="h-full rounded-full bar-fill"
              style={{
                width: `${Math.min(m.value * 100, 100)}%`,
                background: m.color ?? "linear-gradient(90deg, var(--primary), var(--primary-light))",
                animationDelay: `${i * 80}ms`,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}
