import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { BrainCircuit, Plus, Activity } from "lucide-react"

const PLACEHOLDER_RUNS = [
  {
    id: "run-001",
    task: "Text Classification",
    model: "distilbert-base-uncased",
    status: "completed",
    accuracy: "91.4%",
    created: "2026-04-25",
  },
  {
    id: "run-002",
    task: "Token Classification",
    model: "roberta-base",
    status: "training",
    accuracy: "—",
    created: "2026-04-26",
  },
]

const statusColor: Record<string, "default" | "secondary" | "destructive"> = {
  completed: "default",
  training: "secondary",
  failed: "destructive",
}

export default function DashboardPage() {
  return (
    <main className="flex flex-col min-h-screen">
      <nav className="border-b px-6 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-semibold text-lg">
          <BrainCircuit className="h-5 w-5" />
          ModelForge
        </Link>
        <Button asChild>
          <Link href="/train">
            <Plus className="h-4 w-4 mr-2" />
            New Run
          </Link>
        </Button>
      </nav>

      <div className="p-6 max-w-5xl mx-auto w-full flex flex-col gap-6">
        <div>
          <h1 className="text-2xl font-semibold">Training Runs</h1>
          <p className="text-muted-foreground text-sm mt-1">All your model training experiments</p>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-4">
          {[
            { label: "Total Runs", value: "2", icon: Activity },
            { label: "Completed", value: "1", icon: Activity },
            { label: "In Progress", value: "1", icon: Activity },
          ].map((stat) => (
            <Card key={stat.label}>
              <CardContent className="pt-6">
                <p className="text-3xl font-bold">{stat.value}</p>
                <p className="text-sm text-muted-foreground">{stat.label}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Runs table */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Recent Runs</CardTitle>
            <CardDescription>Click a run to view details</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="divide-y">
              {PLACEHOLDER_RUNS.map((run) => (
                <div key={run.id} className="flex items-center justify-between py-3">
                  <div className="flex flex-col gap-0.5">
                    <p className="text-sm font-medium">{run.task}</p>
                    <p className="text-xs text-muted-foreground font-mono">{run.model}</p>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-sm text-muted-foreground">{run.accuracy}</span>
                    <Badge variant={statusColor[run.status] ?? "secondary"}>{run.status}</Badge>
                    <span className="text-xs text-muted-foreground">{run.created}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <p className="text-xs text-muted-foreground text-center">
          Persistent run history requires a connected Postgres database — see{" "}
          <code className="font-mono">backend/.env.example</code>
        </p>
      </div>
    </main>
  )
}
