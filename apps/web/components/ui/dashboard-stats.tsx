"use client"

import { StatRing } from "./stat-ring"
import { Activity, CheckCircle2, Cpu, BarChart3, DollarSign } from "lucide-react"

interface DashboardStatsProps {
  totalRuns: number
  completed: number
  successRate: number
  inProgress: number
  running: number
  pending: number
  avgAccDisplay: string
  avgAccFill: number
  apiSpendDisplay: string
  apiSpendFill: number
  cacheRatio: number
  hasSpendData: boolean
}

export function DashboardStats({
  totalRuns, completed, successRate,
  inProgress, running, pending,
  avgAccDisplay, avgAccFill,
  apiSpendDisplay, apiSpendFill, cacheRatio, hasSpendData,
}: DashboardStatsProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
      <StatRing
        value={totalRuns}
        label="Total Runs"
        icon={Activity}
        color="indigo"
        fillPercent={Math.min(totalRuns * 10, 100)}
        delay={0}
      />
      <StatRing
        value={completed}
        label="Completed"
        icon={CheckCircle2}
        color="emerald"
        fillPercent={successRate}
        sub={totalRuns > 0 ? `${Math.round(successRate)}% success rate` : undefined}
        delay={80}
      />
      <StatRing
        value={inProgress}
        label="In Progress"
        icon={Cpu}
        color="cyan"
        fillPercent={inProgress > 0 ? 70 : 0}
        sub={running > 0 ? `${running} training` : pending > 0 ? `${pending} queued` : undefined}
        delay={160}
      />
      <StatRing
        value={avgAccDisplay}
        label="Avg Accuracy"
        icon={BarChart3}
        color="violet"
        fillPercent={avgAccFill}
        sub={completed > 0 ? `over ${completed} run${completed !== 1 ? "s" : ""}` : "no data yet"}
        delay={240}
      />
      <StatRing
        value={apiSpendDisplay}
        label="API Spend"
        icon={DollarSign}
        color="amber"
        fillPercent={apiSpendFill}
        sub={hasSpendData ? `${cacheRatio}% cache hits` : "no data yet"}
        delay={320}
      />
    </div>
  )
}
