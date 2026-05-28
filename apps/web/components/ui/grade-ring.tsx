import { cn } from "@/lib/utils"

type Grade = "A" | "B" | "C" | "D" | "F"

const gradeConfig: Record<Grade, { color: string; pct: number }> = {
  A: { color: "var(--success)",  pct: 0.95 },
  B: { color: "var(--primary)",  pct: 0.82 },
  C: { color: "var(--pending)",  pct: 0.68 },
  D: { color: "#F59E0B",         pct: 0.50 },
  F: { color: "var(--error)",    pct: 0.30 },
}

interface GradeRingProps {
  grade: Grade
  size?: number
  className?: string
}

export function GradeRing({ grade, size = 80, className }: GradeRingProps) {
  const { color, pct } = gradeConfig[grade] ?? gradeConfig.C
  const r          = (size - 10) / 2
  const cx         = size / 2
  const circumference = 2 * Math.PI * r
  const offset     = circumference * (1 - pct)

  return (
    <div
      className={cn("relative shrink-0", className)}
      style={{ width: size, height: size }}
    >
      <svg
        width={size}
        height={size}
        style={{ transform: "rotate(-90deg)" }}
        aria-label={`Grade ${grade}`}
      >
        {/* Track */}
        <circle
          cx={cx} cy={cx} r={r}
          fill="none"
          stroke="var(--border-c)"
          strokeWidth={6}
        />
        {/* Fill arc */}
        <circle
          cx={cx} cy={cx} r={r}
          fill="none"
          stroke={color}
          strokeWidth={6}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{
            filter: `drop-shadow(0 0 6px ${color})`,
            transition: "stroke-dashoffset 1s cubic-bezier(0.16, 1, 0.3, 1)",
          }}
        />
      </svg>

      {/* Grade letter centered */}
      <div
        className="absolute inset-0 flex items-center justify-center font-bold"
        style={{
          fontSize: size * 0.38,
          color,
          lineHeight: 1,
        }}
      >
        {grade}
      </div>
    </div>
  )
}
