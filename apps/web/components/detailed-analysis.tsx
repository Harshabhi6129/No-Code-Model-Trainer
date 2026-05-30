"use client"

/**
 * DetailedAnalysis — confusion matrix heatmap + per-class precision/recall/F1.
 * Rendered in a collapsible "Detailed Analysis" section on the run detail page.
 *
 * Design constraints:
 *  - >10 classes: confusion matrix is too dense to render → show per-class bars only
 *  - Binary classification: label rows/cols as TP/FP/FN/TN
 *  - Missing data (old runs): hide gracefully with "Not available" note
 */

import { useState } from "react"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartsTooltip, ResponsiveContainer, Legend,
} from "recharts"
import { ChevronDown, ChevronUp, Grid3X3, BarChart2 } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

interface PerClassEntry {
  precision: number
  recall:    number
  f1:        number
  support:   number
}

interface DetailedAnalysisProps {
  labelNames:      string[]
  confusionMatrix: number[][] | null | undefined
  perClassMetrics: Record<string, PerClassEntry> | null | undefined
}

// ──────────────────────────────────────────────────────────────────────────────
// Confusion Matrix
// ──────────────────────────────────────────────────────────────────────────────

const MAX_CM_CLASSES = 10   // above this, CM is too dense — show per-class bars instead
const CELL_SIZE_PX  = 52    // px per cell

function ConfusionMatrixHeatmap({
  cm, labels,
}: { cm: number[][]; labels: string[] }) {
  const n = labels.length
  const isBinary = n === 2

  // Max value for colour scale
  const maxVal = Math.max(...cm.flat(), 1)

  function cellBg(val: number): string {
    const intensity = val / maxVal   // 0 → 1
    // Map 0→white-ish, 1→brand blue (#7c6fcd)
    const r = Math.round(255 - intensity * (255 - 124))
    const g = Math.round(255 - intensity * (255 - 111))
    const b = Math.round(255 - intensity * (255 - 205))
    return `rgb(${r},${g},${b})`
  }

  function textColor(val: number): string {
    return val / maxVal > 0.55 ? "#fff" : "hsl(215 16% 20%)"
  }

  const rowLabels = isBinary ? ["Actual Positive", "Actual Negative"] : labels
  const colLabels = isBinary ? ["Pred Positive",  "Pred Negative"]   : labels

  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-0.5 mx-auto">
        <thead>
          <tr>
            {/* corner cell */}
            <th className="w-24 text-[10px] text-muted-foreground text-right pr-2 pb-1">Actual ↓ / Pred →</th>
            {colLabels.map(lbl => (
              <th
                key={lbl}
                className="text-[10px] font-medium text-muted-foreground pb-1 text-center max-w-[56px]"
                style={{ width: CELL_SIZE_PX }}
              >
                <span className="block truncate max-w-[52px]" title={lbl}>{lbl}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cm.map((row, i) => (
            <tr key={i}>
              <td className="text-[10px] font-medium text-muted-foreground pr-2 text-right max-w-[96px]">
                <span className="block truncate max-w-[88px]" title={rowLabels[i]}>{rowLabels[i]}</span>
              </td>
              {row.map((val, j) => (
                <td
                  key={j}
                  className="rounded-md text-center font-mono text-[11px] font-semibold"
                  style={{
                    width:     CELL_SIZE_PX,
                    height:    CELL_SIZE_PX,
                    background: cellBg(val),
                    color:      textColor(val),
                    outline:    i === j ? "2px solid #7c6fcd" : undefined,
                    outlineOffset: "-2px",
                  }}
                  title={`Actual: ${rowLabels[i]}\nPredicted: ${colLabels[j]}\nCount: ${val}`}
                >
                  {val}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[10px] text-muted-foreground text-center mt-2">
        Diagonal (outlined) = correct predictions · Off-diagonal = errors
      </p>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Per-Class Metrics Bar Chart
// ──────────────────────────────────────────────────────────────────────────────

function PerClassMetricsChart({
  labels, perClass,
}: { labels: string[]; perClass: Record<string, PerClassEntry> }) {
  const chartData = labels
    .filter(lbl => lbl in perClass)
    .map(lbl => ({
      name:      lbl.length > 14 ? lbl.slice(0, 13) + "…" : lbl,
      fullName:  lbl,
      Precision: +(perClass[lbl].precision * 100).toFixed(1),
      Recall:    +(perClass[lbl].recall    * 100).toFixed(1),
      F1:        +(perClass[lbl].f1        * 100).toFixed(1),
      support:   perClass[lbl].support,
    }))

  if (chartData.length === 0) return null

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 28)}>
      <BarChart
        data={chartData}
        layout="vertical"
        margin={{ top: 4, right: 16, left: 8, bottom: 4 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(224 18% 14%)" horizontal={false} />
        <XAxis
          type="number"
          domain={[0, 100]}
          tick={{ fontSize: 10, fill: "hsl(215 16% 55%)" }}
          axisLine={false}
          tickLine={false}
          unit="%"
        />
        <YAxis
          type="category"
          dataKey="name"
          tick={{ fontSize: 10, fill: "hsl(215 16% 55%)" }}
          axisLine={false}
          tickLine={false}
          width={80}
        />
        <RechartsTooltip
          formatter={(v, name, props) => [
            `${v}% (support: ${props.payload?.support ?? "?"})`,
            name,
          ]}
          labelFormatter={(label, payload) => payload?.[0]?.payload?.fullName ?? label}
          contentStyle={{
            background: "hsl(224 20% 9%)",
            border: "1px solid hsl(224 18% 14%)",
            borderRadius: "8px",
            fontSize: "12px",
          }}
        />
        <Legend
          iconType="circle"
          wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }}
        />
        <Bar dataKey="Precision" fill="#5ea5f8" radius={[0, 4, 4, 0]} maxBarSize={10} />
        <Bar dataKey="Recall"    fill="#4ade80" radius={[0, 4, 4, 0]} maxBarSize={10} />
        <Bar dataKey="F1"        fill="#7c6fcd" radius={[0, 4, 4, 0]} maxBarSize={10} />
      </BarChart>
    </ResponsiveContainer>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Main exported component
// ──────────────────────────────────────────────────────────────────────────────

export function DetailedAnalysis({
  labelNames,
  confusionMatrix,
  perClassMetrics,
}: DetailedAnalysisProps) {
  const [open, setOpen] = useState(false)

  const hasData = (
    (confusionMatrix && confusionMatrix.length > 0) ||
    (perClassMetrics && Object.keys(perClassMetrics).length > 0)
  )

  if (!hasData) return null

  const showCM = (
    confusionMatrix &&
    confusionMatrix.length > 0 &&
    labelNames.length <= MAX_CM_CLASSES
  )
  const showPerClass = perClassMetrics && Object.keys(perClassMetrics).length > 0
  const tooManyForCM = labelNames.length > MAX_CM_CLASSES

  return (
    <Card>
      <CardHeader
        className="pb-3 cursor-pointer select-none"
        onClick={() => setOpen(v => !v)}
      >
        <CardTitle className="text-sm flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Grid3X3 className="h-4 w-4 text-primary" />
            Detailed Analysis
          </div>
          {open ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </CardTitle>
      </CardHeader>

      {open && (
        <CardContent className="space-y-6">
          {/* Confusion matrix */}
          {showCM && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Grid3X3 className="h-3.5 w-3.5 text-primary" />
                <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Confusion Matrix
                </h3>
              </div>
              <ConfusionMatrixHeatmap cm={confusionMatrix!} labels={labelNames} />
            </div>
          )}

          {tooManyForCM && (
            <p className="text-xs text-muted-foreground text-center py-2">
              Confusion matrix hidden ({labelNames.length} classes — too dense to display). Per-class breakdown below.
            </p>
          )}

          {/* Per-class breakdown */}
          {showPerClass && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <BarChart2 className="h-3.5 w-3.5 text-primary" />
                <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Per-Class Metrics
                </h3>
              </div>
              <PerClassMetricsChart labels={labelNames} perClass={perClassMetrics!} />
            </div>
          )}

          {!hasData && (
            <p className="text-xs text-muted-foreground text-center py-4">
              Detailed analysis not available for this run.
            </p>
          )}
        </CardContent>
      )}
    </Card>
  )
}
