"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollReveal } from "@/components/ui/scroll-reveal"
import {
  BrainCircuit, Upload, Zap, BarChart3, Rocket,
  ArrowRight, CheckCircle2, GitBranch, Cpu, Database,
  Layers, Sparkles, Shield, Code2,
} from "lucide-react"

const pipeline = [
  { icon: BrainCircuit, label: "Intent",  desc: "Translates plain English → formal ML spec",  color: "#6366F1", bg: "rgba(99,102,241,0.12)"  },
  { icon: Database,     label: "Data",    desc: "Profiles, validates, and cleans your dataset", color: "#06B6D4", bg: "rgba(6,182,212,0.12)"   },
  { icon: Shield,       label: "Clean",   desc: "Removes duplicates and normalises labels",      color: "#10B981", bg: "rgba(16,185,129,0.12)"  },
  { icon: Layers,       label: "Model",   desc: "Selects architecture and training recipe",       color: "#8B5CF6", bg: "rgba(139,92,246,0.12)"  },
  { icon: Cpu,          label: "Train",   desc: "Runs fine-tuning with live loss streaming",      color: "#F59E0B", bg: "rgba(245,158,11,0.12)"  },
  { icon: BarChart3,    label: "Eval",    desc: "Benchmarks F1, accuracy, confusion matrix",      color: "#EC4899", bg: "rgba(236,72,153,0.12)"  },
  { icon: Rocket,       label: "Deploy",  desc: "Pushes to HuggingFace Hub or Modal endpoint",   color: "#34D399", bg: "rgba(52,211,153,0.12)"  },
]

const features = [
  { icon: Upload,       color: "#6366F1", bg: "rgba(99,102,241,0.1)",  title: "Drop a CSV",          body: "Upload any CSV or JSON dataset. Agents automatically detect columns, task type, and label distribution." },
  { icon: BrainCircuit, color: "#8B5CF6", bg: "rgba(139,92,246,0.1)", title: "Describe in English",  body: "No forms. No config files. Say what you need — agents translate it into a precise ML specification." },
  { icon: Zap,          color: "#06B6D4", bg: "rgba(6,182,212,0.1)",  title: "Real-time Training",   body: "Watch loss and accuracy curves update live via SSE streaming as your model trains in the cloud." },
  { icon: BarChart3,    color: "#EC4899", bg: "rgba(236,72,153,0.1)", title: "Calibrated Eval",      body: "Accuracy, per-class F1, and confusion matrices — plus difficulty tier grading (A → F)." },
  { icon: Rocket,       color: "#10B981", bg: "rgba(16,185,129,0.1)", title: "One-click Deploy",     body: "Push your trained model to HuggingFace Hub or get a live API endpoint via Modal." },
  { icon: Code2,        color: "#F59E0B", bg: "rgba(245,158,11,0.1)", title: "Export & Download",    body: "Download ONNX or TorchScript format models ready for inference in any runtime." },
  { icon: CheckCircle2, color: "#34D399", bg: "rgba(52,211,153,0.1)", title: "Override Anything",    body: "Every agent decision can be overridden in natural language. You stay in control at every step." },
  { icon: Sparkles,     color: "#A78BFA", bg: "rgba(167,139,250,0.1)",title: "Hyperparameter Sweep", body: "Explore lr/batch/epoch combinations automatically. Best run surfaced with a star badge." },
]

const tasks = [
  { label: "Text Classification",      badge: "Live",   color: "#10B981", bg: "rgba(16,185,129,0.1)",  border: "rgba(16,185,129,0.25)"  },
  { label: "Token Classification (NER)",badge: "Live",   color: "#06B6D4", bg: "rgba(6,182,212,0.1)",   border: "rgba(6,182,212,0.25)"   },
  { label: "LLM Fine-Tuning (LoRA)",    badge: "v0.3",  color: "#8B5CF6", bg: "rgba(139,92,246,0.1)",  border: "rgba(139,92,246,0.25)"  },
  { label: "Text Generation",           badge: "Soon",   color: "#6366F1", bg: "rgba(99,102,241,0.07)", border: "rgba(99,102,241,0.15)"  },
  { label: "Embeddings",                badge: "v1.0",   color: "#64748B", bg: "rgba(100,116,139,0.07)",border: "rgba(100,116,139,0.15)" },
  { label: "Image Classification",      badge: "v1.0",   color: "#64748B", bg: "rgba(100,116,139,0.07)",border: "rgba(100,116,139,0.15)" },
]

export default function HomePage() {
  return (
    <div className="min-h-screen flex flex-col" style={{ background: "var(--bg)" }}>

      {/* ── Aurora background ──────────────────────────────────── */}
      <div className="aurora-bg" aria-hidden>
        <div className="aurora-orb aurora-1" />
        <div className="aurora-orb aurora-2" />
        <div className="aurora-orb aurora-3" />
        <div className="aurora-orb aurora-4" />
      </div>

      {/* ── Dot grid ──────────────────────────────────────────── */}
      <div className="fixed inset-0 bg-grid opacity-30 pointer-events-none" />

      {/* ══════════════════════════════════════════════════════════
          NAV
          ══════════════════════════════════════════════════════════ */}
      <header
        className="sticky top-0 z-40"
        style={{
          background: "rgba(6,10,16,0.7)",
          backdropFilter: "blur(24px) saturate(1.8)",
          WebkitBackdropFilter: "blur(24px) saturate(1.8)",
          borderBottom: "1px solid rgba(255,255,255,0.05)",
          boxShadow: "0 1px 0 rgba(99,102,241,0.08)",
        }}
      >
        {/* Gradient accent line */}
        <div className="absolute bottom-0 left-0 right-0 h-px"
          style={{ background: "linear-gradient(90deg, transparent, rgba(99,102,241,0.4), rgba(139,92,246,0.4), transparent)" }} />

        <div className="mx-auto max-w-6xl flex h-14 items-center justify-between px-6">
          <Link href="/" className="flex items-center gap-2.5 font-semibold group">
            <div
              className="flex h-8 w-8 items-center justify-center rounded-lg animate-glow-breathe"
              style={{
                background: "rgba(99,102,241,0.15)",
                border: "1px solid rgba(99,102,241,0.35)",
                boxShadow: "0 0 16px -4px rgba(99,102,241,0.5)",
              }}
            >
              <BrainCircuit className="h-4 w-4" style={{ color: "#818CF8" }} />
            </div>
            <span className="text-sm font-bold text-shimmer">ModelForge</span>
            <span
              className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded tracking-wider"
              style={{
                background: "rgba(99,102,241,0.1)",
                border: "1px solid rgba(99,102,241,0.25)",
                color: "rgba(129,140,248,0.8)",
              }}
            >
              BETA
            </span>
          </Link>

          <nav className="hidden md:flex items-center gap-6 text-sm"
            style={{ color: "rgba(203,213,225,0.55)" }}>
            <Link href="#features"
              className="transition-colors hover:text-foreground">Features</Link>
            <Link href="#pipeline"
              className="transition-colors hover:text-foreground">How it works</Link>
            <Link href="#tasks"
              className="transition-colors hover:text-foreground">Supported tasks</Link>
          </nav>

          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" asChild>
              <Link href="/auth/login">Sign in</Link>
            </Button>
            <Button size="sm" variant="glow" asChild>
              <Link href="/auth/signup">
                Get started <ArrowRight className="ml-0.5 h-3.5 w-3.5" />
              </Link>
            </Button>
          </div>
        </div>
      </header>

      {/* ══════════════════════════════════════════════════════════
          HERO
          ══════════════════════════════════════════════════════════ */}
      <section className="relative flex flex-col items-center text-center px-6 pt-28 pb-24 gap-7">
        {/* Mega hero glow */}
        <div className="absolute inset-0 hero-mega-glow pointer-events-none" />

        {/* Badge */}
        <div className="relative animate-fade-up" style={{ animationDelay: "0ms" }}>
          <div
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium"
            style={{
              background: "rgba(99,102,241,0.1)",
              border: "1px solid rgba(99,102,241,0.3)",
              color: "#A5B4FC",
              boxShadow: "0 0 20px -6px rgba(99,102,241,0.4)",
            }}
          >
            <span className="flex h-1.5 w-1.5 rounded-full bg-indigo-400 animate-pulse" />
            Powered by Claude Agents + HuggingFace Transformers
          </div>
        </div>

        {/* Headline */}
        <div className="animate-fade-up relative" style={{ animationDelay: "80ms" }}>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight leading-tight max-w-4xl">
            <span style={{ color: "#F1F5F9" }}>Train AI models by</span>
            <br />
            <span className="text-shimmer">describing what you want</span>
          </h1>
        </div>

        {/* Subheading */}
        <p
          className="text-lg max-w-xl leading-relaxed animate-fade-up"
          style={{ color: "rgba(203,213,225,0.65)", animationDelay: "160ms" }}
        >
          An orchestrated team of Claude agents turns a plain-English problem description
          into a deployed, production-ready model — no code, no config, no ML expertise required.
        </p>

        {/* CTA buttons */}
        <div className="flex flex-wrap gap-3 justify-center mt-1 animate-fade-up" style={{ animationDelay: "240ms" }}>
          <Button size="lg" variant="glow" asChild>
            <Link href="/auth/signup">
              Start training free <ArrowRight className="ml-1 h-4 w-4" />
            </Link>
          </Button>
          <Button size="lg" variant="outline" asChild
            style={{
              background: "rgba(255,255,255,0.03)",
              borderColor: "rgba(255,255,255,0.1)",
            }}
          >
            <Link href="https://github.com/Harshabhi6129/No-Code-Model-Trainer" target="_blank">
              <GitBranch className="mr-1.5 h-4 w-4" /> View on GitHub
            </Link>
          </Button>
        </div>

        {/* Stats row */}
        <div
          className="flex gap-8 mt-2 animate-fade-up"
          style={{ animationDelay: "320ms", color: "rgba(203,213,225,0.4)" }}
        >
          {[
            { n: "55+", label: "Models" },
            { n: "7",   label: "AI Agents" },
            { n: "∞",   label: "Experiments" },
          ].map(({ n, label }) => (
            <div key={label} className="flex flex-col items-center gap-0.5">
              <span className="text-2xl font-bold font-mono" style={{ color: "#818CF8" }}>{n}</span>
              <span className="text-xs tracking-wide uppercase">{label}</span>
            </div>
          ))}
        </div>

        {/* Terminal mockup */}
        <div
          className="relative mt-6 w-full max-w-2xl rounded-2xl text-left text-sm font-mono animate-fade-up terminal-window"
          style={{
            animationDelay: "400ms",
            background: "rgba(6,10,16,0.75)",
            backdropFilter: "blur(24px)",
            border: "1px solid rgba(255,255,255,0.07)",
            boxShadow: "0 24px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(99,102,241,0.08), inset 0 1px 0 rgba(255,255,255,0.05)",
          }}
        >
          {/* Window chrome */}
          <div
            className="flex items-center gap-2 px-5 py-3 border-b"
            style={{ borderColor: "rgba(255,255,255,0.05)" }}
          >
            <div className="h-3 w-3 rounded-full" style={{ background: "#EF4444", boxShadow: "0 0 6px rgba(239,68,68,0.5)" }} />
            <div className="h-3 w-3 rounded-full" style={{ background: "#F59E0B", boxShadow: "0 0 6px rgba(245,158,11,0.5)" }} />
            <div className="h-3 w-3 rounded-full" style={{ background: "#10B981", boxShadow: "0 0 6px rgba(16,185,129,0.5)" }} />
            <span className="ml-3 text-xs" style={{ color: "rgba(100,116,139,0.6)" }}>
              modelforge — training session
            </span>
          </div>

          {/* Terminal content */}
          <div className="p-5 space-y-2">
            <p className="terminal-line-1" style={{ color: "rgba(203,213,225,0.7)" }}>
              <span style={{ color: "#818CF8" }}>you</span>
              <span style={{ color: "rgba(99,102,241,0.5)" }}> → </span>
              Classify customer support tickets by urgency level
            </p>
            <p className="terminal-line-2" style={{ color: "#34D399" }}>
              <span>✓ Intent</span>
              <span style={{ color: "rgba(52,211,153,0.6)" }}> · text_classification · base: distilbert-base-uncased · confidence 0.94</span>
            </p>
            <p className="terminal-line-3" style={{ color: "#38BDF8" }}>
              <span>✓ Data</span>
              <span style={{ color: "rgba(56,189,248,0.6)" }}> · 12,400 rows · 3 labels · avg 87 chars/input · class balance 0.82</span>
            </p>
            <p className="terminal-line-4" style={{ color: "#A78BFA" }}>
              <span>✓ Model</span>
              <span style={{ color: "rgba(167,139,250,0.6)" }}> · LoRA recipe · lr=2e-5 · 3 epochs · batch=32 · grade-A target</span>
            </p>
            <p className="terminal-line-5 flex items-center gap-2">
              <span style={{ color: "#FBBF24" }}>⚡ Training</span>
              <span style={{ color: "rgba(251,191,36,0.6)" }}> epoch 2/3 · loss 0.284 · acc 0.912 · F1 0.908</span>
              <span className="terminal-cursor" />
            </p>
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════
          PIPELINE
          ══════════════════════════════════════════════════════════ */}
      <section id="pipeline" className="relative py-24 px-6">
        <div className="mx-auto max-w-5xl">
          <ScrollReveal className="text-center mb-16">
            <p className="text-xs font-mono tracking-[0.2em] uppercase mb-3"
              style={{ color: "rgba(99,102,241,0.7)" }}>
              Under the hood
            </p>
            <h2 className="text-4xl font-bold mb-4" style={{ color: "#F1F5F9" }}>
              Seven agents. One pipeline.
            </h2>
            <p style={{ color: "rgba(203,213,225,0.55)" }} className="text-lg max-w-lg mx-auto">
              Each agent specialises in one stage and hands off cleanly to the next.
              The whole pipeline takes minutes, not months.
            </p>
          </ScrollReveal>

          {/* Desktop: horizontal connected flow */}
          <div className="hidden lg:flex items-center gap-0">
            {pipeline.map((step, i) => (
              <div key={i} className="flex items-center" style={{ flex: i < pipeline.length - 1 ? "1 1 auto" : "0 0 auto" }}>
                <ScrollReveal delay={i * 80}>
                  <div
                    className="group flex flex-col items-center text-center gap-3 p-4 rounded-2xl transition-all duration-300 neon-border"
                    style={{
                      background: "rgba(12,20,32,0.6)",
                      backdropFilter: "blur(16px)",
                      border: `1px solid rgba(255,255,255,0.05)`,
                      width: 118,
                      boxShadow: "0 4px 20px rgba(0,0,0,0.3)",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = `rgba(12,20,32,0.8)`
                      e.currentTarget.style.borderColor = `${step.color}33`
                      e.currentTarget.style.transform = "translateY(-4px)"
                      e.currentTarget.style.boxShadow = `0 16px 40px rgba(0,0,0,0.4), 0 0 20px -6px ${step.color}55`
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = "rgba(12,20,32,0.6)"
                      e.currentTarget.style.borderColor = "rgba(255,255,255,0.05)"
                      e.currentTarget.style.transform = "translateY(0)"
                      e.currentTarget.style.boxShadow = "0 4px 20px rgba(0,0,0,0.3)"
                    }}
                  >
                    <div className="flex h-11 w-11 items-center justify-center rounded-xl"
                      style={{ background: step.bg, boxShadow: `0 0 16px -4px ${step.color}55` }}>
                      <step.icon className="h-5 w-5" style={{ color: step.color }} />
                    </div>
                    <span className="text-xs font-bold tracking-tight" style={{ color: "#F1F5F9" }}>{step.label}</span>
                    <span className="text-[10px] leading-relaxed" style={{ color: "rgba(203,213,225,0.45)" }}>{step.desc}</span>
                  </div>
                </ScrollReveal>
                {i < pipeline.length - 1 && (
                  <div className="pipeline-connector mx-1" style={{ minWidth: 20 }} />
                )}
              </div>
            ))}
          </div>

          {/* Mobile: grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 lg:hidden">
            {pipeline.map((step, i) => (
              <ScrollReveal key={i} delay={i * 60}>
                <div
                  className="flex flex-col items-center text-center gap-3 p-4 rounded-xl"
                  style={{
                    background: "rgba(12,20,32,0.6)",
                    backdropFilter: "blur(16px)",
                    border: "1px solid rgba(255,255,255,0.05)",
                  }}
                >
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl"
                    style={{ background: step.bg }}>
                    <step.icon className="h-4 w-4" style={{ color: step.color }} />
                  </div>
                  <span className="text-xs font-bold" style={{ color: "#F1F5F9" }}>{step.label}</span>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* Divider */}
      <div className="mx-auto max-w-4xl h-px"
        style={{ background: "linear-gradient(90deg, transparent, rgba(99,102,241,0.2), rgba(139,92,246,0.2), transparent)" }} />

      {/* ══════════════════════════════════════════════════════════
          FEATURES
          ══════════════════════════════════════════════════════════ */}
      <section id="features" className="py-24 px-6">
        <div className="mx-auto max-w-5xl">
          <ScrollReveal className="text-center mb-16">
            <p className="text-xs font-mono tracking-[0.2em] uppercase mb-3"
              style={{ color: "rgba(139,92,246,0.7)" }}>
              Everything included
            </p>
            <h2 className="text-4xl font-bold mb-4" style={{ color: "#F1F5F9" }}>
              Built for serious builders
            </h2>
            <p style={{ color: "rgba(203,213,225,0.55)" }} className="text-lg max-w-lg mx-auto">
              From raw CSV to deployed endpoint — every step is handled with production-grade quality.
            </p>
          </ScrollReveal>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {features.map((f, i) => (
              <ScrollReveal key={i} delay={i * 50}>
                <div
                  className="group flex flex-col gap-4 p-5 rounded-2xl cursor-default neon-border"
                  style={{
                    background: "rgba(12,20,32,0.55)",
                    backdropFilter: "blur(20px)",
                    border: "1px solid rgba(255,255,255,0.05)",
                    boxShadow: "0 4px 20px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.04)",
                    transition: "all 0.3s cubic-bezier(0.16,1,0.3,1)",
                    minHeight: 180,
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = "translateY(-4px)"
                    e.currentTarget.style.borderColor = `${f.color}25`
                    e.currentTarget.style.boxShadow = `0 20px 48px rgba(0,0,0,0.4), 0 0 24px -8px ${f.color}44`
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "translateY(0)"
                    e.currentTarget.style.borderColor = "rgba(255,255,255,0.05)"
                    e.currentTarget.style.boxShadow = "0 4px 20px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.04)"
                  }}
                >
                  {/* Corner glow */}
                  <div
                    className="absolute -top-6 -right-6 h-20 w-20 rounded-full blur-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
                    style={{ background: f.color + "33", position: "absolute" }}
                  />

                  <div className="flex h-10 w-10 items-center justify-center rounded-xl shrink-0"
                    style={{ background: f.bg, boxShadow: `0 0 16px -4px ${f.color}44` }}>
                    <f.icon className="h-4.5 w-4.5" style={{ color: f.color }} />
                  </div>
                  <div>
                    <h3 className="font-semibold text-sm mb-1.5" style={{ color: "#F1F5F9" }}>
                      {f.title}
                    </h3>
                    <p className="text-xs leading-relaxed" style={{ color: "rgba(203,213,225,0.5)" }}>
                      {f.body}
                    </p>
                  </div>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* Divider */}
      <div className="mx-auto max-w-4xl h-px"
        style={{ background: "linear-gradient(90deg, transparent, rgba(6,182,212,0.2), rgba(16,185,129,0.2), transparent)" }} />

      {/* ══════════════════════════════════════════════════════════
          TASKS
          ══════════════════════════════════════════════════════════ */}
      <section id="tasks" className="py-24 px-6">
        <div className="mx-auto max-w-3xl text-center">
          <ScrollReveal>
            <p className="text-xs font-mono tracking-[0.2em] uppercase mb-3"
              style={{ color: "rgba(16,185,129,0.7)" }}>
              Task support
            </p>
            <h2 className="text-4xl font-bold mb-4" style={{ color: "#F1F5F9" }}>
              Supported ML tasks
            </h2>
            <p style={{ color: "rgba(203,213,225,0.55)" }} className="text-lg max-w-lg mx-auto mb-12">
              Current capabilities and what&apos;s coming next.
            </p>
          </ScrollReveal>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {tasks.map((t, i) => (
              <ScrollReveal key={t.label} delay={i * 60}>
                <div
                  className="flex items-center justify-between p-4 rounded-xl"
                  style={{
                    background: t.bg,
                    border: `1px solid ${t.border}`,
                    boxShadow: `0 4px 16px rgba(0,0,0,0.2), 0 0 0 0 ${t.color}00`,
                  }}
                >
                  <span className="text-sm font-medium text-left" style={{ color: "#E2E8F0" }}>
                    {t.label}
                  </span>
                  <span
                    className="text-[10px] font-bold font-mono ml-3 shrink-0 px-2 py-0.5 rounded-md"
                    style={{
                      background: t.color + "20",
                      color: t.color,
                      border: `1px solid ${t.color}40`,
                    }}
                  >
                    {t.badge}
                  </span>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════
          CTA
          ══════════════════════════════════════════════════════════ */}
      <section className="py-28 px-6 relative">
        {/* Massive glow behind CTA */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: "radial-gradient(ellipse 80% 60% at 50% 50%, rgba(99,102,241,0.18) 0%, transparent 70%)",
          }}
        />
        {/* Animated ring */}
        <div
          className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 h-96 w-96 rounded-full pointer-events-none animate-neural-pulse"
          style={{
            background: "transparent",
            border: "1px solid rgba(99,102,241,0.1)",
          }}
        />
        <div
          className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 h-[32rem] w-[32rem] rounded-full pointer-events-none animate-neural-pulse"
          style={{
            background: "transparent",
            border: "1px solid rgba(139,92,246,0.06)",
            animationDelay: "0.5s",
          }}
        />

        <div className="mx-auto max-w-2xl text-center space-y-7 relative">
          <ScrollReveal>
            <div
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium mb-2"
              style={{
                background: "rgba(16,185,129,0.1)",
                border: "1px solid rgba(16,185,129,0.25)",
                color: "#34D399",
              }}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse inline-block" />
              Free to start · No credit card required
            </div>
            <h2 className="text-5xl font-bold leading-tight" style={{ color: "#F1F5F9" }}>
              Ready to train your<br />
              <span className="text-shimmer">first model?</span>
            </h2>
            <p className="text-lg mt-4" style={{ color: "rgba(203,213,225,0.55)" }}>
              Upload a dataset. Describe the task. Get a deployed model.
            </p>
            <div className="flex gap-3 justify-center mt-8">
              <Button size="xl" variant="glow" asChild>
                <Link href="/auth/signup">
                  Get started free <ArrowRight className="ml-1.5 h-4 w-4" />
                </Link>
              </Button>
            </div>
          </ScrollReveal>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════
          FOOTER
          ══════════════════════════════════════════════════════════ */}
      <footer
        className="py-8 px-6 text-center text-sm"
        style={{
          borderTop: "1px solid rgba(255,255,255,0.04)",
          color: "rgba(100,116,139,0.6)",
        }}
      >
        <div className="flex items-center justify-center gap-2 mb-1">
          <BrainCircuit className="h-3.5 w-3.5" style={{ color: "rgba(99,102,241,0.5)" }} />
          <span className="font-medium">ModelForge</span>
        </div>
        <p>Built with Claude Agent SDK + HuggingFace Transformers</p>
      </footer>
    </div>
  )
}
