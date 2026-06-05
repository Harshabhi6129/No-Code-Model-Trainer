"use client"

import { useState, useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Key, Cpu, Trash2, Eye, EyeOff, Check, AlertTriangle,
  ExternalLink, BrainCircuit, Zap, BarChart3, Calendar,
  ShieldCheck, Sparkles, ChevronRight, Pencil, Mail,
  Lock, ChevronDown, X,
} from "lucide-react"
import { toast } from "sonner"
import { formatDistanceToNow, format } from "date-fns"
import { createClient } from "@/lib/supabase/client"

const LS_HF_TOKEN = "modelforge_hf_token"
const LS_SESSIONS = "modelforge_sessions_v2"
const LS_DEFAULTS = "modelforge_train_defaults"

interface TrainDefaults {
  training_approach: string
  num_epochs: number
  batch_size: number
}
const DEFAULT_TRAIN_DEFAULTS: TrainDefaults = { training_approach: "full_finetune", num_epochs: 3, batch_size: 16 }

interface Props {
  email: string
  createdAt: string
  totalRuns: number
  completedRuns: number
  initialFullName: string
}

function Section({ title, icon: Icon, iconColor = "text-primary", children, defaultOpen = true }: {
  title: string; icon: React.ElementType; iconColor?: string; children: React.ReactNode; defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="rounded-2xl overflow-hidden"
      style={{ background: "rgba(12,20,32,0.55)", backdropFilter: "blur(20px)", border: "1px solid rgba(255,255,255,0.06)", boxShadow: "0 4px 20px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05)" }}>
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center gap-2.5 px-5 py-3.5 transition-colors hover:bg-white/2"
        style={{ borderBottom: open ? "1px solid rgba(255,255,255,0.05)" : "none" }}
      >
        <Icon className={`h-4 w-4 ${iconColor}`} />
        <span className="text-sm font-semibold text-foreground flex-1 text-left">{title}</span>
        <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
      </button>
      {open && <div className="p-5">{children}</div>}
    </div>
  )
}

export function SettingsClient({ email, createdAt, totalRuns, completedRuns, initialFullName }: Props) {
  const router   = useRouter()
  const supabase = createClient()

  // Profile
  const [fullName,      setFullName]      = useState(initialFullName)
  const [editingName,   setEditingName]   = useState(false)
  const [savingName,    setSavingName]    = useState(false)
  const nameInputRef = useRef<HTMLInputElement>(null)

  // HF Token
  const [hfToken,     setHfToken]     = useState("")
  const [showToken,   setShowToken]   = useState(false)
  const [tokenSaved,  setTokenSaved]  = useState(false)
  const [tokenStatus, setTokenStatus] = useState<"none" | "local">("none")

  // Training defaults
  const [defaults,      setDefaults]      = useState<TrainDefaults>(DEFAULT_TRAIN_DEFAULTS)
  const [defaultsSaved, setDefaultsSaved] = useState(false)

  // Session stats
  const [sessionCount, setSessionCount] = useState(0)
  const [datasetCount, setDatasetCount] = useState(0)

  // Password reset
  const [resetSent, setResetSent] = useState(false)
  const [sendingReset, setSendingReset] = useState(false)

  // Email change
  const [newEmail,       setNewEmail]       = useState("")
  const [changingEmail,  setChangingEmail]  = useState(false)
  const [emailSent,      setEmailSent]      = useState(false)

  // Account deletion
  const [deleteInput,   setDeleteInput]   = useState("")
  const [deleting,      setDeleting]      = useState(false)

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

  useEffect(() => {
    if (editingName && nameInputRef.current) nameInputRef.current.focus()
  }, [editingName])

  async function saveFullName() {
    const name = fullName.trim()
    setSavingName(true)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const { error } = await (supabase as any).from("profiles").update({ full_name: name || null }).eq("id", (await supabase.auth.getUser()).data.user?.id)
    setSavingName(false)
    setEditingName(false)
    if (error) toast.error("Failed to save name")
    else toast.success(name ? "Display name updated" : "Display name cleared")
  }

  function saveToken() {
    const t = hfToken.trim()
    if (t) { localStorage.setItem(LS_HF_TOKEN, t); setTokenStatus("local") }
    else   { localStorage.removeItem(LS_HF_TOKEN); setTokenStatus("none") }
    setTokenSaved(true); setTimeout(() => setTokenSaved(false), 2500)
    toast.success(t ? "HuggingFace token saved" : "HuggingFace token cleared")
  }

  function saveDefaults() {
    localStorage.setItem(LS_DEFAULTS, JSON.stringify(defaults))
    setDefaultsSaved(true); setTimeout(() => setDefaultsSaved(false), 2500)
    toast.success("Training defaults saved")
  }

  function clearSessions() {
    if (!confirm(`Delete all ${sessionCount} session${sessionCount !== 1 ? "s" : ""}? Cannot be undone.`)) return
    localStorage.removeItem(LS_SESSIONS)
    setSessionCount(0); setDatasetCount(0)
    toast.success("Session history cleared")
  }

  async function sendPasswordReset() {
    setSendingReset(true)
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/auth/reset-password`,
    })
    setSendingReset(false)
    if (error) toast.error(error.message)
    else { setResetSent(true); toast.success("Password reset email sent — check your inbox") }
  }

  async function sendEmailChange() {
    if (!newEmail.trim() || !newEmail.includes("@")) { toast.error("Enter a valid email address"); return }
    if (newEmail.trim() === email) { toast.error("That's already your current email"); return }
    setChangingEmail(true)
    const { error } = await supabase.auth.updateUser({ email: newEmail.trim() })
    setChangingEmail(false)
    if (error) toast.error(error.message)
    else { setEmailSent(true); toast.success("Confirmation emails sent — check both inboxes") }
  }

  async function deleteAccount() {
    if (deleteInput !== "DELETE") return
    setDeleting(true)
    try {
      const res = await fetch("/api/account/delete", { method: "POST" })
      const body = await res.json() as { error?: string }
      if (!res.ok) throw new Error(body.error ?? "Delete failed")
      await supabase.auth.signOut()
      router.push("/")
      toast.success("Account deleted")
    } catch (err) {
      toast.error(String(err))
      setDeleting(false)
    }
  }

  const handle      = fullName?.split(" ")[0] || email?.split("@")[0] || "user"
  const initials    = (fullName || email || "M")[0].toUpperCase()
  const memberSince = createdAt ? format(new Date(createdAt), "MMMM yyyy") : "Recently"
  const memberAge   = createdAt ? formatDistanceToNow(new Date(createdAt), { addSuffix: false }) : null
  const successRate = totalRuns > 0 ? Math.round((completedRuns / totalRuns) * 100) : 0

  return (
    <div className="px-8 py-7 max-w-2xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-foreground">Settings</h1>
        <p className="text-sm text-muted-foreground mt-1">Manage your profile, API keys, and training preferences.</p>
      </div>

      {/* ── Profile card ────────────────────────────────────── */}
      <div className="relative rounded-2xl overflow-hidden"
        style={{ background: "rgba(12,20,32,0.65)", backdropFilter: "blur(24px)", border: "1px solid rgba(255,255,255,0.07)", boxShadow: "0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.07)" }}>
        <div className="absolute top-0 left-0 right-0 h-0.5"
          style={{ background: "linear-gradient(90deg, #6366F1, #8B5CF6, #06B6D4)" }} />
        <div className="absolute top-0 left-0 w-48 h-48 rounded-full blur-3xl pointer-events-none"
          style={{ background: "rgba(99,102,241,0.12)" }} />
        <div className="relative p-6">
          <div className="flex items-start gap-5">
            <div className="relative shrink-0">
              <div className="h-16 w-16 rounded-2xl flex items-center justify-center text-2xl font-bold text-white"
                style={{ background: "linear-gradient(135deg, #6366F1, #8B5CF6)", boxShadow: "0 0 24px -4px rgba(99,102,241,0.6)" }}>
                {initials}
              </div>
              <span className="absolute -bottom-0.5 -right-0.5 h-3.5 w-3.5 rounded-full border-2"
                style={{ background: "#10B981", borderColor: "rgba(6,10,16,0.9)", boxShadow: "0 0 6px rgba(16,185,129,0.6)" }} />
            </div>

            <div className="flex-1 min-w-0 pt-0.5">
              {/* Editable display name */}
              <div className="flex items-center gap-2 mb-0.5">
                {editingName ? (
                  <div className="flex items-center gap-2">
                    <input
                      ref={nameInputRef}
                      value={fullName}
                      onChange={e => setFullName(e.target.value)}
                      onKeyDown={e => { if (e.key === "Enter") saveFullName(); if (e.key === "Escape") { setEditingName(false); setFullName(initialFullName) } }}
                      onBlur={saveFullName}
                      placeholder="Your display name"
                      className="text-xl font-bold bg-transparent border-b outline-none pb-0.5"
                      style={{ borderColor: "rgba(99,102,241,0.5)", color: "#F1F5F9", width: 180 }}
                    />
                    {savingName && <span className="text-xs text-muted-foreground">Saving…</span>}
                  </div>
                ) : (
                  <>
                    <h2 className="text-xl font-bold text-foreground capitalize tracking-tight">{handle}</h2>
                    <button onClick={() => setEditingName(true)}
                      className="text-muted-foreground hover:text-indigo-400 transition-colors opacity-0 group-hover:opacity-100"
                      title="Edit display name">
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                    <button onClick={() => setEditingName(true)}
                      className="p-1 rounded hover:bg-white/5 transition-colors"
                      title="Edit display name">
                      <Pencil className="h-3 w-3 text-muted-foreground hover:text-indigo-400 transition-colors" />
                    </button>
                  </>
                )}
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full ml-1"
                  style={{ background: "rgba(99,102,241,0.12)", border: "1px solid rgba(99,102,241,0.3)", color: "#A5B4FC" }}>
                  FREE PLAN
                </span>
              </div>
              <p className="text-sm text-muted-foreground font-mono truncate">{email}</p>
              <div className="flex items-center gap-1.5 mt-1.5 text-xs text-muted-foreground">
                <Calendar className="h-3 w-3" />
                <span>Member since {memberSince}</span>
                {memberAge && <><span className="opacity-30">·</span><span className="opacity-60">{memberAge} ago</span></>}
              </div>
            </div>

            <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg shrink-0"
              style={{ background: "rgba(16,185,129,0.08)", border: "1px solid rgba(16,185,129,0.2)" }}>
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
              <span className="text-[11px] font-medium text-emerald-400">Verified</span>
            </div>
          </div>

          <div className="my-5 h-px" style={{ background: "rgba(255,255,255,0.05)" }} />

          <div className="grid grid-cols-3 gap-4">
            {[
              { icon: Zap,       color: "text-indigo-400",  bg: "rgba(99,102,241,0.1)",  value: String(totalRuns),     label: "Total Runs"    },
              { icon: BarChart3, color: "text-emerald-400", bg: "rgba(16,185,129,0.1)", value: String(completedRuns), label: "Completed"     },
              { icon: Sparkles,  color: "text-violet-400",  bg: "rgba(139,92,246,0.1)", value: totalRuns > 0 ? `${successRate}%` : "—", label: "Success Rate" },
            ].map(({ icon: Icon, color, bg, value, label }) => (
              <div key={label} className="flex items-center gap-3 rounded-xl p-3"
                style={{ background: bg, border: "1px solid rgba(255,255,255,0.04)" }}>
                <Icon className={`h-4 w-4 shrink-0 ${color}`} />
                <div>
                  <p className={`text-lg font-bold font-mono leading-none ${color}`}>{value}</p>
                  <p className="text-[10px] text-muted-foreground mt-0.5">{label}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="flex flex-wrap gap-2 mt-4">
            {[
              { label: "Claude Agent SDK",        color: "rgba(99,102,241,0.15)",  border: "rgba(99,102,241,0.25)",  text: "#A5B4FC" },
              { label: "HuggingFace Transformers", color: "rgba(245,158,11,0.1)",  border: "rgba(245,158,11,0.2)",   text: "#FCD34D" },
              { label: "LoRA / QLoRA",             color: "rgba(139,92,246,0.1)",  border: "rgba(139,92,246,0.2)",   text: "#C4B5FD" },
              { label: "Supabase",                 color: "rgba(16,185,129,0.1)",  border: "rgba(16,185,129,0.2)",   text: "#6EE7B7" },
            ].map(({ label, color, border, text }) => (
              <span key={label} className="text-[10px] font-medium px-2 py-1 rounded-md"
                style={{ background: color, border: `1px solid ${border}`, color: text }}>{label}</span>
            ))}
          </div>
        </div>
      </div>

      {/* ── Security ────────────────────────────────────────── */}
      <Section title="Security" icon={Lock} iconColor="text-indigo-400">
        <div className="space-y-5">
          {/* Password reset */}
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-medium text-foreground">Password</p>
              <p className="text-xs text-muted-foreground mt-0.5">Send a reset link to your email address</p>
            </div>
            {resetSent ? (
              <div className="flex items-center gap-1.5 text-xs text-emerald-400">
                <Check className="h-3.5 w-3.5" /> Sent — check your inbox
              </div>
            ) : (
              <Button variant="outline" size="sm" onClick={sendPasswordReset} disabled={sendingReset} className="shrink-0">
                {sendingReset ? "Sending…" : "Reset password"}
              </Button>
            )}
          </div>

          <div className="h-px" style={{ background: "rgba(255,255,255,0.05)" }} />

          {/* Email change */}
          <div>
            <p className="text-sm font-medium text-foreground mb-1">Email address</p>
            <p className="text-xs text-muted-foreground mb-3">Currently: <span className="font-mono text-foreground/70">{email}</span></p>
            {emailSent ? (
              <div className="flex items-center gap-2 text-sm text-emerald-400">
                <Mail className="h-4 w-4" /> Check both inboxes to confirm the change
              </div>
            ) : (
              <div className="flex gap-2">
                <Input value={newEmail} onChange={e => setNewEmail(e.target.value)}
                  placeholder="New email address" type="email" className="flex-1 h-9 text-sm" />
                <Button size="sm" variant="outline" onClick={sendEmailChange} disabled={changingEmail || !newEmail.trim()} className="shrink-0">
                  {changingEmail ? "Sending…" : "Change email"}
                </Button>
              </div>
            )}
          </div>
        </div>
      </Section>

      {/* ── HuggingFace Token ────────────────────────────────── */}
      <Section title="HuggingFace API Token" icon={Key} iconColor="text-amber-400">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground max-w-sm">Required to push trained models to HuggingFace Hub. Stored locally in your browser.</p>
            <span className="text-[10px] font-semibold px-2 py-1 rounded-full shrink-0 ml-3"
              style={tokenStatus === "local"
                ? { background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.25)", color: "#34D399" }
                : { background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", color: "rgba(100,116,139,0.8)" }}>
              {tokenStatus === "local" ? "✓ Connected" : "Not set"}
            </span>
          </div>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Input type={showToken ? "text" : "password"} value={hfToken} onChange={e => setHfToken(e.target.value)}
                placeholder="hf_xxxxxxxxxxxxxxxxxxxx" className="pr-10 font-mono text-sm" />
              <button type="button" onClick={() => setShowToken(v => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors">
                {showToken ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
            <Button onClick={saveToken} size="sm" className="shrink-0">
              {tokenSaved ? <><Check className="h-3.5 w-3.5 mr-1" />Saved</> : "Save"}
            </Button>
          </div>
          <div className="flex items-start gap-2 p-3 rounded-xl text-xs text-muted-foreground"
            style={{ background: "rgba(245,158,11,0.06)", border: "1px solid rgba(245,158,11,0.15)" }}>
            <AlertTriangle className="h-3.5 w-3.5 text-amber-400 mt-0.5 shrink-0" />
            <span>Stored in browser localStorage — never sent to our servers.{" "}
              <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer"
                className="text-amber-400 hover:underline inline-flex items-center gap-0.5">
                Get a token <ExternalLink className="h-2.5 w-2.5" />
              </a>
            </span>
          </div>
        </div>
      </Section>

      {/* ── Training Defaults ────────────────────────────────── */}
      <Section title="Training Defaults" icon={Cpu} iconColor="text-cyan-400">
        <div className="space-y-4">
          <p className="text-xs text-muted-foreground">Applied when creating a new training session. Override per-session in the workspace.</p>
          <div className="grid grid-cols-3 gap-4">
            {[
              { label: "Default Approach", key: "training_approach" as const,
                options: [{ value: "full_finetune", label: "Full Fine-tune" }, { value: "lora", label: "LoRA" }, { value: "qlora", label: "QLoRA" }] },
              { label: "Default Epochs", key: "num_epochs" as const,
                options: [1, 2, 3, 5, 10, 15, 20].map(v => ({ value: v, label: String(v) })) },
              { label: "Default Batch", key: "batch_size" as const,
                options: [4, 8, 16, 32, 64].map(v => ({ value: v, label: String(v) })) },
            ].map(({ label, key, options }) => (
              <div key={key} className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">{label}</label>
                <select value={defaults[key]}
                  onChange={e => setDefaults(d => ({ ...d, [key]: key === "training_approach" ? e.target.value : Number(e.target.value) }))}
                  className="w-full h-9 rounded-lg border text-sm px-2 focus:outline-none focus:ring-1 focus:ring-primary"
                  style={{ background: "rgba(255,255,255,0.04)", borderColor: "rgba(255,255,255,0.08)", color: "var(--body)" }}>
                  {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
            ))}
          </div>
          <Button onClick={saveDefaults} size="sm">
            {defaultsSaved ? <><Check className="h-3.5 w-3.5 mr-1" />Saved</> : "Save Defaults"}
          </Button>
        </div>
      </Section>

      {/* ── Data Management ──────────────────────────────────── */}
      <Section title="Data Management" icon={Trash2} iconColor="text-rose-400">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-foreground">Local Session History</p>
            <p className="text-xs text-muted-foreground mt-0.5">
              {sessionCount} session{sessionCount !== 1 ? "s" : ""} · {datasetCount} dataset{datasetCount !== 1 ? "s" : ""} in browser storage
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={clearSessions} disabled={sessionCount === 0}
            className="gap-1.5 text-rose-400 hover:text-rose-300 border-rose-500/25 hover:bg-rose-500/8">
            <Trash2 className="h-3.5 w-3.5" /> Clear history
          </Button>
        </div>
      </Section>

      {/* ── About ────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 rounded-2xl px-5 py-4"
        style={{ background: "rgba(99,102,241,0.06)", border: "1px solid rgba(99,102,241,0.12)" }}>
        <div className="h-9 w-9 rounded-xl flex items-center justify-center shrink-0"
          style={{ background: "rgba(99,102,241,0.12)", border: "1px solid rgba(99,102,241,0.25)" }}>
          <BrainCircuit className="h-4 w-4 text-indigo-400" />
        </div>
        <div className="flex-1">
          <p className="text-sm font-semibold text-foreground">ModelForge</p>
          <p className="text-xs text-muted-foreground">v0.4.0 · No-code AI model trainer · Claude Agent SDK</p>
        </div>
        <a href="https://github.com/Harshabhi6129/No-Code-Model-Trainer" target="_blank" rel="noopener noreferrer"
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-indigo-400 transition-colors">
          GitHub <ChevronRight className="h-3 w-3" />
        </a>
      </div>

      {/* ── Danger Zone ──────────────────────────────────────── */}
      <Section title="Danger Zone" icon={X} iconColor="text-rose-500" defaultOpen={false}>
        <div className="space-y-4">
          <div className="p-4 rounded-xl" style={{ background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.2)" }}>
            <p className="text-sm font-semibold text-rose-400 mb-1">Delete account</p>
            <p className="text-xs text-muted-foreground mb-4">
              Permanently deletes your account, all training runs, and all associated data. This cannot be undone.
            </p>
            <p className="text-xs text-muted-foreground mb-2">
              Type <span className="font-mono font-bold text-rose-400">DELETE</span> to confirm:
            </p>
            <div className="flex gap-2">
              <input value={deleteInput} onChange={e => setDeleteInput(e.target.value)}
                placeholder="DELETE"
                className="flex-1 h-9 rounded-lg px-3 text-sm font-mono outline-none"
                style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(239,68,68,0.3)", color: "#F1F5F9" }} />
              <Button variant="destructive" size="sm" onClick={deleteAccount}
                disabled={deleteInput !== "DELETE" || deleting} className="shrink-0">
                {deleting ? "Deleting…" : "Delete my account"}
              </Button>
            </div>
          </div>
        </div>
      </Section>
    </div>
  )
}
