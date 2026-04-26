import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import {
  BrainCircuit, Upload, Zap, BarChart3, Rocket,
  ArrowRight, CheckCircle2, GitBranch, Cpu, Database, Layers
} from "lucide-react"

const pipeline = [
  { icon: BrainCircuit, label: "Intent", desc: "Understands your plain-English problem" },
  { icon: Database,     label: "Data",   desc: "Profiles, validates, and cleans your dataset" },
  { icon: Layers,       label: "Model",  desc: "Picks the best architecture and recipe" },
  { icon: Cpu,          label: "Train",  desc: "Runs fine-tuning with live metrics" },
  { icon: BarChart3,    label: "Eval",   desc: "Benchmarks accuracy, F1, confusion matrix" },
  { icon: Rocket,       label: "Deploy", desc: "Pushes to HuggingFace Hub or Modal endpoint" },
]

const features = [
  { icon: Upload,       title: "Drop a CSV",        body: "Upload any CSV or JSON dataset. Agents automatically detect columns, task type, and label distribution." },
  { icon: BrainCircuit, title: "Describe in English",body: "No forms. No config files. Say what you need — agents translate it into a precise ML specification." },
  { icon: Zap,          title: "Real-time Training", body: "Watch loss and accuracy curves update live via WebSocket as your model trains in the cloud." },
  { icon: BarChart3,    title: "Calibrated Eval",    body: "Get accuracy, per-class F1, and confusion matrices — not just a single headline number." },
  { icon: Rocket,       title: "One-click Deploy",   body: "Push your trained model to HuggingFace Hub or get a live API endpoint via Modal." },
  { icon: CheckCircle2, title: "Override Anything",  body: "Every agent decision can be overridden in natural language. You stay in control." },
]

const tasks = [
  { label: "Text Classification", badge: "Live",   color: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" },
  { label: "Token Classification (NER)", badge: "Soon", color: "bg-blue-500/10 text-blue-400 border-blue-500/20" },
  { label: "LLM Fine-Tuning (LoRA)", badge: "Soon", color: "bg-violet-500/10 text-violet-400 border-violet-500/20" },
  { label: "Text Generation",  badge: "Soon",   color: "bg-blue-500/10 text-blue-400 border-blue-500/20" },
  { label: "Embeddings",       badge: "v1.0",   color: "bg-secondary text-muted-foreground border-border" },
  { label: "Image Classification", badge: "v1.0", color: "bg-secondary text-muted-foreground border-border" },
]

export default function HomePage() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Nav */}
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/80 backdrop-blur-sm">
        <div className="mx-auto max-w-6xl flex h-14 items-center justify-between px-6">
          <Link href="/" className="flex items-center gap-2 font-semibold">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10">
              <BrainCircuit className="h-4 w-4 text-primary" />
            </div>
            <span className="text-sm">ModelForge</span>
            <Badge variant="secondary" className="text-[10px] px-1.5 h-4">BETA</Badge>
          </Link>
          <nav className="hidden md:flex items-center gap-6 text-sm text-muted-foreground">
            <Link href="#features" className="hover:text-foreground transition-colors">Features</Link>
            <Link href="#pipeline" className="hover:text-foreground transition-colors">How it works</Link>
          </nav>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" asChild>
              <Link href="/auth/login">Sign in</Link>
            </Button>
            <Button size="sm" asChild>
              <Link href="/auth/signup">Get started <ArrowRight className="ml-1.5 h-3.5 w-3.5" /></Link>
            </Button>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="relative hero-glow flex flex-col items-center text-center px-6 pt-24 pb-20 gap-6">
        <Badge variant="outline" className="text-xs border-primary/30 text-primary bg-primary/5">
          Powered by Claude Agents + HuggingFace
        </Badge>
        <h1 className="text-5xl md:text-6xl font-bold tracking-tight leading-tight max-w-3xl">
          Train AI models by{" "}
          <span className="gradient-text">describing what you want</span>
        </h1>
        <p className="text-muted-foreground text-lg max-w-xl leading-relaxed">
          An orchestrated team of Claude agents turns a plain-English problem description into
          a deployed, production-ready model — no code, no config, no ML expertise required.
        </p>
        <div className="flex flex-wrap gap-3 justify-center mt-2">
          <Button size="lg" asChild>
            <Link href="/auth/signup">
              Start training free <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
          <Button size="lg" variant="outline" asChild>
            <Link href="https://github.com/Harshabhi6129/No-Code-Model-Trainer" target="_blank">
              <GitBranch className="mr-2 h-4 w-4" /> View on GitHub
            </Link>
          </Button>
        </div>

        {/* Fake terminal / prompt preview */}
        <div className="mt-8 w-full max-w-2xl rounded-xl border border-border bg-card p-4 text-left text-sm font-mono">
          <div className="flex items-center gap-1.5 mb-3">
            <div className="h-2.5 w-2.5 rounded-full bg-destructive/70" />
            <div className="h-2.5 w-2.5 rounded-full bg-yellow-500/70" />
            <div className="h-2.5 w-2.5 rounded-full bg-emerald-500/70" />
          </div>
          <p className="text-muted-foreground">
            <span className="text-primary">you</span> → Classify customer support tickets by urgency
          </p>
          <p className="text-emerald-400 mt-1.5">✓ Intent: text_classification · base: distilbert-base-uncased</p>
          <p className="text-blue-400">✓ Data: 12,400 rows · 3 labels · avg 87 chars/input</p>
          <p className="text-violet-400">✓ Model: LoRA recipe · lr=2e-5 · 3 epochs · batch=32</p>
          <p className="text-muted-foreground animate-pulse mt-1">▊ Training… epoch 1/3 · loss 0.421 · acc 0.847</p>
        </div>
      </section>

      <Separator className="opacity-30" />

      {/* Pipeline */}
      <section id="pipeline" className="py-20 px-6">
        <div className="mx-auto max-w-5xl">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-3">Six agents, one goal</h2>
            <p className="text-muted-foreground">Each agent specialises in one part of the pipeline and hands off cleanly to the next.</p>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {pipeline.map((step, i) => (
              <div key={i} className="flex flex-col items-center text-center gap-2 p-4 rounded-xl border border-border bg-card hover:border-primary/30 hover:bg-primary/5 transition-colors group">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
                  <step.icon className="h-5 w-5 text-primary" />
                </div>
                <span className="text-sm font-semibold">{step.label}</span>
                <span className="text-[11px] text-muted-foreground leading-relaxed">{step.desc}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <Separator className="opacity-30" />

      {/* Features */}
      <section id="features" className="py-20 px-6 bg-secondary/20">
        <div className="mx-auto max-w-5xl">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-3">Everything you need</h2>
            <p className="text-muted-foreground">From raw CSV to deployed endpoint, every step is handled.</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {features.map((f, i) => (
              <div key={i} className="p-5 rounded-xl border border-border bg-card hover:border-primary/30 transition-colors group">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 mb-4 group-hover:bg-primary/20 transition-colors">
                  <f.icon className="h-4 w-4 text-primary" />
                </div>
                <h3 className="font-semibold mb-1.5">{f.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{f.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <Separator className="opacity-30" />

      {/* Supported tasks */}
      <section className="py-20 px-6">
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="text-3xl font-bold mb-3">Supported tasks</h2>
          <p className="text-muted-foreground mb-10">Current and upcoming ML task support.</p>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {tasks.map((t) => (
              <div key={t.label} className="flex items-center justify-between p-3.5 rounded-lg border border-border bg-card text-sm">
                <span className="font-medium text-left">{t.label}</span>
                <span className={`text-[10px] font-semibold border rounded px-1.5 py-0.5 ml-2 shrink-0 ${t.color}`}>
                  {t.badge}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-6 hero-glow">
        <div className="mx-auto max-w-xl text-center space-y-6">
          <h2 className="text-3xl font-bold">Ready to train your first model?</h2>
          <p className="text-muted-foreground">Free to start. No credit card required.</p>
          <Button size="lg" asChild>
            <Link href="/auth/signup">
              Get started free <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8 px-6 text-center text-sm text-muted-foreground">
        <p>ModelForge — Built with Claude Agent SDK + HuggingFace Transformers</p>
      </footer>
    </div>
  )
}
