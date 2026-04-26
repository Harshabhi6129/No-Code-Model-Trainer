import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { BrainCircuit, Upload, Zap, BarChart3, Rocket, ArrowRight } from "lucide-react"

const steps = [
  {
    icon: Upload,
    title: "Upload Dataset",
    description: "Drop a CSV or JSON file. Agents immediately profile your data — column types, class distribution, missing values.",
  },
  {
    icon: BrainCircuit,
    title: "Describe Your Task",
    description: "Tell us what you want in plain English. The Intent Agent translates your words into a precise ML specification.",
  },
  {
    icon: Zap,
    title: "Agents Train the Model",
    description: "Model Agent picks the best architecture and recipe. Training starts with real-time metric streaming.",
  },
  {
    icon: BarChart3,
    title: "Review Evaluation",
    description: "The Eval Agent runs a calibrated benchmark suite and surfaces actionable insights — not just accuracy.",
  },
  {
    icon: Rocket,
    title: "One-Click Deploy",
    description: "Push to HuggingFace Hub or get a live API endpoint via Modal in seconds.",
  },
]

const tasks = [
  { label: "Text Classification", status: "v0.1" },
  { label: "Token Classification (NER)", status: "v0.2" },
  { label: "LLM Fine-Tuning (LoRA/QLoRA)", status: "v0.3" },
  { label: "Embeddings", status: "v1.0" },
  { label: "Image Classification", status: "v1.0" },
  { label: "Audio (Whisper)", status: "v1.0" },
]

export default function HomePage() {
  return (
    <main className="flex flex-col min-h-screen">
      {/* Nav */}
      <nav className="border-b px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2 font-semibold text-lg">
          <BrainCircuit className="h-5 w-5" />
          ModelForge
        </div>
        <div className="flex gap-3">
          <Button variant="ghost" asChild>
            <Link href="/dashboard">Dashboard</Link>
          </Button>
          <Button asChild>
            <Link href="/train">Start Training</Link>
          </Button>
        </div>
      </nav>

      {/* Hero */}
      <section className="flex flex-col items-center text-center px-6 py-24 gap-6 max-w-3xl mx-auto">
        <Badge variant="secondary">Agent-Native AI Training</Badge>
        <h1 className="text-5xl font-bold tracking-tight leading-tight">
          Describe what you want.<br />We train the model.
        </h1>
        <p className="text-muted-foreground text-lg max-w-xl">
          ModelForge is an orchestrated team of Claude agents that turns a plain-English problem
          description into a deployed, production-ready AI model — no code required.
        </p>
        <div className="flex gap-3 mt-2">
          <Button size="lg" asChild>
            <Link href="/train">
              Get Started <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
          <Button size="lg" variant="outline" asChild>
            <Link href="/dashboard">View Runs</Link>
          </Button>
        </div>
      </section>

      {/* Steps */}
      <section className="px-6 py-16 bg-secondary/30">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-2xl font-semibold text-center mb-10">How it works</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {steps.map((step, i) => (
              <Card key={i} className="relative">
                <CardHeader className="pb-2">
                  <step.icon className="h-6 w-6 mb-1 text-primary" />
                  <CardTitle className="text-sm">{step.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground">{step.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Supported Tasks */}
      <section className="px-6 py-16 max-w-3xl mx-auto w-full">
        <h2 className="text-2xl font-semibold text-center mb-8">Supported ML Tasks</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {tasks.map((task) => (
            <Card key={task.label} className="flex items-center justify-between p-4">
              <span className="text-sm font-medium">{task.label}</span>
              <Badge variant={task.status === "v0.1" ? "default" : "secondary"}>
                {task.status}
              </Badge>
            </Card>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-auto border-t px-6 py-6 text-center text-sm text-muted-foreground">
        ModelForge — Powered by Claude Agent SDK + HuggingFace Transformers
      </footer>
    </main>
  )
}
