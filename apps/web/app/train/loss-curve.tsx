"use client"

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"

export interface EpochPoint {
  epoch: number
  step: number
  loss: number | null
  eval_loss: number | null
  learning_rate: number | null
}

interface LossCurveProps {
  data: EpochPoint[]
  isLive?: boolean
}

export function LossCurve({ data, isLive = false }: LossCurveProps) {
  if (data.length === 0) return null

  const hasEvalLoss = data.some((d) => d.eval_loss !== null)

  const formatted = data.map((d) => ({
    epoch: d.epoch + 1,
    "Train Loss": d.loss ?? undefined,
    "Eval Loss": d.eval_loss ?? undefined,
  }))

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
          Loss Curve
        </span>
        {isLive && (
          <span className="flex items-center gap-1 text-xs text-green-500">
            <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
            Live
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={formatted} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis
            dataKey="epoch"
            tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
            label={{ value: "Epoch", position: "insideBottom", offset: -2, fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
            height={32}
          />
          <YAxis
            tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
            width={44}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "6px",
              fontSize: 12,
            }}
            formatter={(v) => (typeof v === "number" ? v.toFixed(4) : String(v))}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line
            type="monotone"
            dataKey="Train Loss"
            stroke="hsl(var(--primary))"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3 }}
            connectNulls={false}
          />
          {hasEvalLoss && (
            <Line
              type="monotone"
              dataKey="Eval Loss"
              stroke="hsl(var(--chart-2, #f97316))"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 3 }}
              connectNulls={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
