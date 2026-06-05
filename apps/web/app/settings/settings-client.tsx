"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import {
  Key, Cpu, Trash2, Eye, EyeOff, Check, AlertTriangle,
  ExternalLink, BrainCircuit, Zap, BarChart3, Calendar,
  ShieldCheck, Sparkles, ChevronRight,
} from "lucide-react"
import { toast } from "sonner"
import { formatDistanceToNow, format } from "date-fns"

const LS_HF_TOKEN = "modelforge_hf_token"
const LS_SESSIONS = "modelforge_sessions_v2"
const LS_DEFAULTS = "modelforge_train_defaults"

interface TrainDefaults {
  training_approach: string
  num_epochs: number
  batch_size: number
}

const DEFAULT_TRAIN_DEFAULTS: TrainDefaults = {
  training_approach: "full_finetune",
  num_epochs: 3,
  batch_size: 16,
}

interface Props {
  email: string
  createdAt: string
  totalRuns: number
  completedRuns: number
}

function Section({ title, icon: Icon, iconColor = "text-primary", children }: {
  title: string
  icon: React.ElementType
  iconColor?: string
  children: React.ReactNode
}) {
  return (
    <div
      className="rounded-2xl overflow-hidden"
      style={{
        background: "rgba(12,20,32,0.55)",
        backdropFilter: "blur(20px)",
        border: "1px solid rgba(255,255,255,0.06)",
        boxShadow: "0 4px 20px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05)",
      }}
    >
      <div
        className="flex items-center gap-2.5 px-5 py-3.5"
        style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}
      >
        <Icon className={`h-4 w-4 ${iconColor}`} />
        <span className="text-sm font-semibold text-foreground">{title}</span>
      </div>
      <div className="p-5">{children}</div>
    </div>
  )
}

export function SettingsClient({ email, createdAt, totalRuns, completedRuns }: Props) {
  const [hfToken, setHfToken]       = useState("")
  const [showToken, setShowToken]   = useState(false)
  const [tokenSaved, setTokenSaved] = useState(false)
  const [tokenStatus, setTokenStatus] = useState<"none" | "local">("none")
  const [defaults, setDefaults]     = useState<TrainDefaults>(DEFAULT_TRAIN_DEFAULTS)
  const [defaultsSaved, setDefaultsSaved] = useState(false)
  const [sessionCount, setSessionCount] = useState(0)
  const [datasetCount, setDatasetCount] = useState(0)

  useEffect(() => {
    const stored = localStorage.getItem(LS_HF_TOKEN) ?? ""
    setHfToken(stored)
    setTokenStatus(stored ? "local" : "none")
    try {
      const raw = localStorage.getItem(LS_DEFAULTS)
      if (raw) setDefaults({ ...DEFAULT_TRAIN_DEFAULTS, ...JSON.parse(raw) })
    } catch {}
    try {
      const raw = localStorage.getItem(LS_SESSIONS)
      if (raw) {
        const sessions = JSON.parse(raw)
        setSessionCount(sessions.length)
        const seen = new Set<string>()
        sessions.forEach((s: { uploadResult?: { file_id?: string } }) => {
          if (s.uploadResult?.file_id) seen.add(s.uploadResult.file_id)
        })
        setDatasetCount(seen.size)
      }
    } catch {}
  }, [])

  function saveToken() {
    const t = hfToken.trim()
    if (t) { localStorage.setItem(LS_HF_TOKEN, t); setTokenStatus("local") }
    else   { localStorage.removeItem(LS_HF_TOKEN); setTokenStatus("none") }
    setTokenSaved(true)
    setTimeout(() => setTokenSaved(false), 2500)
    toast.success(t ? "HuggingFace token saved" : "HuggingFace token cleared")
  }

  function saveDefaults() {
    localStorage.setItem(LS_DEFAULTS, JSON.stringify(defaults))
    setDefaultsSaved(true)
    setTimeout(() => setDefaultsSaved(false), 2500)
    toast.success("Training defaults saved")
  }

  function clearSessions() {
    if (!confirm(`Delete all ${sessionCount} training session${sessionCount !== 1 ? "s" : ""} from history? This cannot be undone.`)) return
    localStorage.removeItem(LS_SESSIONS)
    setSessionCount(0); setDatasetCount(0)
    toast.success("Session history cleared")
  }

  const handle    = email?.split("@")[0] ?? "user"
  const initials  = email ? email[0].toUpperCase() : "M"
  const memberSince = createdAt
    ? format(new Date(createdAt), "MMMM yyyy")
    : "Recently"
  const memberAge = createdAt
    ? formatDistanceToNow(new Date(createdAt), { addSuffix: false })
    : null
  const successRate = totalRuns > 0
    ? Math.round((completedRuns / totalRuns) * 100)
    : 0

  return (
    <div className="px-8 py-7 max-w-2xl mx-auto space-y-6 animate-fade-in">

      {/* ── Page title ─────────────────────────────────────────── */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-foreground">Settings</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Manage your profile, API keys, and training preferences.
        </p>
      </div>

      {/* ══════════════════════════════════════════════════════════
          PROFILE CARD — full-width hero
          ══════════════════════════════════════════════════════════ */}
      <div
        className="relative rounded-2xl overflow-hidden"
        style={{
          background: "rgba(12,20,32,0.65)",
          backdropFilter: "blur(24px)",
          border: "1px solid rgba(255,255,255,0.07)",
          boxShadow: "0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.07)",
        }}
      >
        {/* Gradient bar at top */}
        <div
          className="absolute top-0 left-0 right-0 h-0.5"
          style={{ background: "linear-gradient(90deg, #6366F1, #8B5CF6, #06B6D4)" }}
        />

        {/* Ambient glow behind avatar */}
        <div
          className="absolute top-0 left-0 w-48 h-48 rounded-full blur-3xl pointer-events-none"
          style={{ background: "rgba(99,102,241,0.12)" }}
        />

        <div className="relative p-6">
          {/* Top row: avatar + info + badge */}
          <div className="flex items-start gap-5">
            {/* Avatar */}
            <div className="relative shrink-0">
              <div
                className="h-16 w-16 rounded-2xl flex items-center justify-center text-2xl font-bold text-white"
                style={{
                  background: "linear-gradient(135deg, #6366F1, #8B5CF6)",
                  boxShadow: "0 0 24px -4px rgba(99,102,241,0.6)",
                }}
              >
                {initials}
              </div>
              {/* Online dot */}
              <span
                className="absolute -bottom-0.5 -right-0.5 h-3.5 w-3.5 rounded-full border-2"
                style={{ background: "#10B981", borderColor: "rgba(6,10,16,0.9)", boxShadow: "0 0 6px rgba(16,185,129,0.6)" }}
              />
            </div>

            {/* Name + email + meta */}
            <div className="flex-1 min-w-0 pt-0.5">
              <div className="flex items-center gap-2 flex-wrap">
                <h2 className="text-xl font-bold text-foreground capitalize tracking-tight">
                  {handle}
                </h2>
                {/* Free plan badge */}
                <span
                  className="text-[10px] font-bold px-2 py-0.5 rounded-full"
                  style={{
                    background: "rgba(99,102,241,0.12)",
                    border: "1px solid rgba(99,102,241,0.3)",
                    color: "#A5B4FC",
                  }}
                >
                  FREE PLAN
                </span>
              </div>
              <p className="text-sm text-muted-foreground mt-0.5 font-mono truncate">{email}</p>
              <div className="flex items-center gap-1.5 mt-1.5 text-xs text-muted-foreground">
                <Calendar className="h-3 w-3" />
                <span>Member since {memberSince}</span>
                {memberAge && (
                  <>
                    <span className="opacity-30">·</span>
                    <span className="opacity-60">{memberAge} ago</span>
                  </>
                )}
              </div>
            </div>

            {/* Verified badge */}
            <div
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg shrink-0"
              style={{
                background: "rgba(16,185,129,0.08)",
                border: "1px solid rgba(16,185,129,0.2)",
              }}
            >
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
              <span className="text-[11px] font-medium text-emerald-400">Verified</span>
            </div>
          </div>

          {/* Divider */}
          <div
            className="my-5 h-px"
            style={{ background: "rgba(255,255,255,0.05)" }}
          />

          {/* Stats row */}
          <div className="grid grid-cols-3 gap-4">
            {[
              {
                icon: Zap,
                color: "text-indigo-400",
                bg: "rgba(99,102,241,0.1)",
                value: String(totalRuns),
                label: "Total Runs",
              },
              {
                icon: BarChart3,
                color: "text-emerald-400",
                bg: "rgba(16,185,129,0.1)",
                value: String(completedRuns),
                label: "Completed",
              },
              {
                icon: Sparkles,
                color: "text-violet-400",
                bg: "rgba(139,92,246,0.1)",
                value: totalRuns > 0 ? `${successRate}%` : "—",
                label: "Success Rate",
              },
            ].map(({ icon: Icon, color, bg, value, label }) => (
              <div
                key={label}
                className="flex items-center gap-3 rounded-xl p-3"
                style={{ background: bg, border: "1px solid rgba(255,255,255,0.04)" }}
              >
                <Icon className={`h-4 w-4 shrink-0 ${color}`} />
                <div>
                  <p className={`text-lg font-bold font-mono leading-none ${color}`}>{value}</p>
                  <p className="text-[10px] text-muted-foreground mt-0.5">{label}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Platform pills */}
          <div className="flex flex-wrap gap-2 mt-4">
            {[
              { label: "Claude Agent SDK", color: "rgba(99,102,241,0.15)", border: "rgba(99,102,241,0.25)", text: "#A5B4FC" },
              { label: "HuggingFace Transformers", color: "rgba(245,158,11,0.1)", border: "rgba(245,158,11,0.2)", text: "#FCD34D" },
              { label: "LoRA / QLoRA", color: "rgba(139,92,246,0.1)", border: "rgba(139,92,246,0.2)", text: "#C4B5FD" },
              { label: "Supabase", color: "rgba(16,185,129,0.1)", border: "rgba(16,185,129,0.2)", text: "#6EE7B7" },
            ].map(({ label, color, border, text }) => (
              <span
                key={label}
                className="text-[10px] font-medium px-2 py-1 rounded-md"
                style={{ background: color, border: `1px solid ${border}`, color: text }}
              >
                {label}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ── HuggingFace Token ──────────────────────────────────── */}
      <Section title="HuggingFace API Token" icon={Key} iconColor="text-amber-400">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground leading-relaxed max-w-sm">
              Required to push trained models to HuggingFace Hub. Stored locally in your browser only.
            </p>
            <span
              className="text-[10px] font-semibold px-2 py-1 rounded-full shrink-0"
              style={tokenStatus === "local"
                ? { background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.25)", color: "#34D399" }
                : { background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", color: "rgba(100,116,139,0.8)" }
              }
            >
              {tokenStatus === "local" ? "✓ Connected" : "Not set"}
            </span>
          </div>

          <div className="flex gap-2">
            <div className="relative flex-1">
              <Input
                type={showToken ? "text" : "password"}
                value={hfToken}
                onChange={e => setHfToken(e.target.value)}
                placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
                className="pr-10 font-mono text-sm"
              />
              <button
                type="button"
                onClick={() => setShowToken(v => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
              >
                {showToken ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
            <Button onClick={saveToken} size="sm" className="shrink-0">
              {tokenSaved ? <><Check className="h-3.5 w-3.5 mr-1" /> Saved</> : "Save"}
            </Button>
          </div>

          <div
            className="flex items-start gap-2 p-3 rounded-xl text-xs text-muted-foreground"
            style={{ background: "rgba(245,158,11,0.06)", border: "1px solid rgba(245,158,11,0.15)" }}
          >
            <AlertTriangle className="h-3.5 w-3.5 text-amber-400 mt-0.5 shrink-0" />
            <span>
              Stored in browser localStorage only — never sent to our servers.{" "}
              <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer"
                className="text-amber-400 hover:underline inline-flex items-center gap-0.5">
                Get a token <ExternalLink className="h-2.5 w-2.5" />
              </a>
            </span>
          </div>
        </div>
      </Section>

      {/* ── Training Defaults ──────────────────────────────────── */}
      <Section title="Training Defaults" icon={Cpu} iconColor="text-cyan-400">
        <div className="space-y-4">
          <p className="text-xs text-muted-foreground">
            Applied when creating a new training session. Override per-session in the workspace.
          </p>
          <div className="grid grid-cols-3 gap-4">
            {[
              {
                label: "Default Approach",
                key: "training_approach" as const,
                options: [
                  { value: "full_finetune", label: "Full Fine-tune" },
                  { value: "lora",          label: "LoRA" },
                  { value: "qlora",         label: "QLoRA" },
                ],
              },
              {
                label: "Default Epochs",
                key: "num_epochs" as const,
                options: [1, 2, 3, 5, 10, 15, 20].map(v => ({ value: v, label: String(v) })),
              },
              {
                label: "Default Batch",
                key: "batch_size" as const,
                options: [4, 8, 16, 32, 64].map(v => ({ value: v, label: String(v) })),
              },
            ].map(({ label, key, options }) => (
              <div key={key} className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">{label}</label>
                <select
                  value={defaults[key]}
                  onChange={e => setDefaults(d => ({ ...d, [key]: key === "training_approach" ? e.target.value : Number(e.target.value) }))}
                  className="w-full h-9 rounded-lg border text-sm px-2 focus:outline-none focus:ring-1 focus:ring-primary"
                  style={{ background: "rgba(255,255,255,0.04)", borderColor: "rgba(255,255,255,0.08)", color: "var(--body)" }}
                >
                  {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
            ))}
          </div>
          <Button onClick={saveDefaults} size="sm" className="gap-1.5">
            {defaultsSaved ? <><Check className="h-3.5 w-3.5" /> Saved</> : "Save Defaults"}
          </Button>
        </div>
      </Section>

      {/* ── Data Management ────────────────────────────────────── */}
      <Section title="Data Management" icon={Trash2} iconColor="text-rose-400">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-foreground">Local Session History</p>
            <p className="text-xs text-muted-foreground mt-0.5">
              {sessionCount} session{sessionCount !== 1 ? "s" : ""} · {datasetCount} dataset{datasetCount !== 1 ? "s" : ""} stored in your browser
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={clearSessions}
            disabled={sessionCount === 0}
            className="gap-1.5 text-rose-400 hover:text-rose-300 border-rose-500/25 hover:bg-rose-500/8"
          >
            <Trash2 className="h-3.5 w-3.5" /> Clear history
          </Button>
        </div>
      </Section>

      {/* ── About ──────────────────────────────────────────────── */}
      <div
        className="flex items-center gap-3 rounded-2xl px-5 py-4"
        style={{
          background: "rgba(99,102,241,0.06)",
          border: "1px solid rgba(99,102,241,0.12)",
        }}
      >
        <div
          className="h-9 w-9 rounded-xl flex items-center justify-center shrink-0"
          style={{ background: "rgba(99,102,241,0.12)", border: "1px solid rgba(99,102,241,0.25)" }}
        >
          <BrainCircuit className="h-4.5 w-4.5 text-indigo-400" />
        </div>
        <div className="flex-1">
          <p className="text-sm font-semibold text-foreground">ModelForge</p>
          <p className="text-xs text-muted-foreground">v0.4.0 · No-code AI model trainer · Claude Agent SDK</p>
        </div>
        <a
          href="https://github.com/Harshabhi6129/No-Code-Model-Trainer"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-indigo-400 transition-colors"
        >
          GitHub <ChevronRight className="h-3 w-3" />
        </a>
      </div>

    </div>
  )
}
