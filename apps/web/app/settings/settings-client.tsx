"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import {
  User, Key, Cpu, Trash2, Eye, EyeOff, Check,
  AlertTriangle, ExternalLink, BrainCircuit,
} from "lucide-react"
import { toast } from "sonner"

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

export function SettingsClient({ email }: { email: string }) {
  const router = useRouter()

  // HF Token
  const [hfToken, setHfToken]   = useState("")
  const [showToken, setShowToken] = useState(false)
  const [tokenSaved, setTokenSaved] = useState(false)
  const [tokenStatus, setTokenStatus] = useState<"none" | "local" | "env">("none")

  // Training defaults
  const [defaults, setDefaults] = useState<TrainDefaults>(DEFAULT_TRAIN_DEFAULTS)
  const [defaultsSaved, setDefaultsSaved] = useState(false)

  // Stats
  const [sessionCount, setSessionCount] = useState(0)
  const [datasetCount, setDatasetCount]  = useState(0)

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
    if (t) {
      localStorage.setItem(LS_HF_TOKEN, t)
      setTokenStatus("local")
    } else {
      localStorage.removeItem(LS_HF_TOKEN)
      setTokenStatus("none")
    }
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
    setSessionCount(0)
    setDatasetCount(0)
    toast.success("Session history cleared")
  }

  const initials = email ? email[0].toUpperCase() : "M"

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-sm text-muted-foreground mt-1">Manage your account, API keys, and training preferences.</p>
      </div>

      {/* ── Profile ────────────────────────────────────────────────────────── */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <User className="h-4 w-4 text-primary" /> Profile
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center gap-4">
            <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center text-lg font-bold text-primary">
              {initials}
            </div>
            <div>
              <p className="font-medium">{email || "—"}</p>
              <p className="text-xs text-muted-foreground">ModelForge account</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ── HuggingFace Token ───────────────────────────────────────────────── */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Key className="h-4 w-4 text-primary" /> HuggingFace API Token
            {tokenStatus === "local" && (
              <Badge variant="outline" className="ml-auto text-emerald-500 border-emerald-500/30 bg-emerald-500/5 text-[10px]">
                <Check className="h-2.5 w-2.5 mr-1" /> Connected
              </Badge>
            )}
            {tokenStatus === "none" && (
              <Badge variant="outline" className="ml-auto text-muted-foreground text-[10px]">
                Not set
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-xs text-muted-foreground leading-relaxed">
            Required to push trained models to the HuggingFace Hub. Stored locally in your browser.
          </p>

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
            <Button onClick={saveToken} size="sm" className="gap-1.5 shrink-0">
              {tokenSaved ? <Check className="h-3.5 w-3.5" /> : null}
              {tokenSaved ? "Saved" : "Save"}
            </Button>
          </div>

          <div className="flex items-start gap-2 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20 text-xs text-muted-foreground">
            <AlertTriangle className="h-3.5 w-3.5 text-amber-400 mt-0.5 shrink-0" />
            <span>
              Token is stored in browser localStorage. Never share your browser profile with others.{" "}
              <a
                href="https://huggingface.co/settings/tokens"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary inline-flex items-center gap-0.5 hover:underline"
              >
                Get a token <ExternalLink className="h-2.5 w-2.5" />
              </a>
            </span>
          </div>
        </CardContent>
      </Card>

      {/* ── Training Defaults ───────────────────────────────────────────────── */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Cpu className="h-4 w-4 text-primary" /> Training Defaults
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-xs text-muted-foreground">
            Applied when creating a new training session. Override per-session in the workspace.
          </p>

          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-1.5">
              <label className="text-xs font-medium">Default Approach</label>
              <select
                value={defaults.training_approach}
                onChange={e => setDefaults(d => ({ ...d, training_approach: e.target.value }))}
                className="w-full h-8 rounded-md border border-input bg-card px-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary"
              >
                <option value="full_finetune">Full Fine-tune</option>
                <option value="lora">LoRA</option>
                <option value="qlora">QLoRA</option>
              </select>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium">Default Epochs</label>
              <select
                value={defaults.num_epochs}
                onChange={e => setDefaults(d => ({ ...d, num_epochs: Number(e.target.value) }))}
                className="w-full h-8 rounded-md border border-input bg-card px-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary"
              >
                {[1, 2, 3, 5, 10, 15, 20].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium">Default Batch Size</label>
              <select
                value={defaults.batch_size}
                onChange={e => setDefaults(d => ({ ...d, batch_size: Number(e.target.value) }))}
                className="w-full h-8 rounded-md border border-input bg-card px-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary"
              >
                {[4, 8, 16, 32, 64].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
          </div>

          <Button onClick={saveDefaults} size="sm" className="gap-1.5">
            {defaultsSaved ? <Check className="h-3.5 w-3.5" /> : null}
            {defaultsSaved ? "Saved" : "Save Defaults"}
          </Button>
        </CardContent>
      </Card>

      {/* ── Data Management ─────────────────────────────────────────────────── */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Trash2 className="h-4 w-4 text-primary" /> Data Management
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between py-2">
            <div>
              <p className="text-sm font-medium">Session History</p>
              <p className="text-xs text-muted-foreground mt-0.5">
                {sessionCount} session{sessionCount !== 1 ? "s" : ""} · {datasetCount} dataset{datasetCount !== 1 ? "s" : ""} stored locally
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={clearSessions}
              disabled={sessionCount === 0}
              className="gap-1.5 text-destructive hover:text-destructive border-destructive/30 hover:bg-destructive/5"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Clear history
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* ── About ───────────────────────────────────────────────────────────── */}
      <Card className="bg-secondary/20">
        <CardContent className="pt-4 pb-4">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center">
              <BrainCircuit className="h-4 w-4 text-primary" />
            </div>
            <div>
              <p className="text-sm font-semibold">ModelForge</p>
              <p className="text-xs text-muted-foreground">v0.4.0 · No-code AI model trainer</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
