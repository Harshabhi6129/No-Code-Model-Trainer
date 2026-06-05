"use client"

import { useState, useEffect, Suspense } from "react"
import { useRouter } from "next/navigation"
import { BrainCircuit, Loader2, Eye, EyeOff, CheckCircle2 } from "lucide-react"
import { createClient } from "@/lib/supabase/client"
import { toast } from "sonner"

function ResetForm() {
  const router    = useRouter()
  const supabase  = createClient()
  const [password,  setPassword]  = useState("")
  const [confirm,   setConfirm]   = useState("")
  const [show,      setShow]      = useState(false)
  const [loading,   setLoading]   = useState(false)
  const [done,      setDone]      = useState(false)
  const [hasHash,   setHasHash]   = useState(false)

  useEffect(() => {
    // Supabase puts the recovery token in the URL hash — check it's present
    setHasHash(window.location.hash.includes("access_token"))
  }, [])

  async function handleReset(e: React.FormEvent) {
    e.preventDefault()
    if (password.length < 8) { toast.error("Password must be at least 8 characters"); return }
    if (password !== confirm)  { toast.error("Passwords do not match"); return }
    setLoading(true)
    const { error } = await supabase.auth.updateUser({ password })
    setLoading(false)
    if (error) { toast.error(error.message); return }
    setDone(true)
    setTimeout(() => router.push("/dashboard"), 2500)
  }

  if (done) {
    return (
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl animate-float"
          style={{ background: "rgba(16,185,129,0.12)", border: "1px solid rgba(16,185,129,0.3)", boxShadow: "0 0 24px -4px rgba(16,185,129,0.4)" }}>
          <CheckCircle2 size={26} style={{ color: "#34D399" }} />
        </div>
        <p className="text-lg font-bold" style={{ color: "#F1F5F9" }}>Password updated!</p>
        <p className="text-sm" style={{ color: "rgba(203,213,225,0.6)" }}>Redirecting to dashboard…</p>
      </div>
    )
  }

  if (!hasHash) {
    return (
      <div className="text-center space-y-3">
        <p className="text-sm" style={{ color: "rgba(203,213,225,0.6)" }}>
          This link is invalid or has expired.
        </p>
        <button onClick={() => router.push("/auth/login")}
          className="text-sm underline underline-offset-3" style={{ color: "#818CF8" }}>
          Back to sign in
        </button>
      </div>
    )
  }

  return (
    <form onSubmit={handleReset} className="flex flex-col gap-4">
      <p className="text-sm font-semibold" style={{ color: "#F1F5F9" }}>Set new password</p>
      {[
        { id: "pw",  value: password, onChange: setPassword, placeholder: "New password (min 8 chars)" },
        { id: "cpw", value: confirm,  onChange: setConfirm,  placeholder: "Confirm new password"       },
      ].map(({ id, value, onChange, placeholder }) => (
        <div key={id} className="relative">
          <input
            type={show ? "text" : "password"}
            value={value}
            onChange={e => onChange(e.target.value)}
            placeholder={placeholder}
            required
            className="w-full h-11 rounded-xl px-4 pr-10 text-sm outline-none transition-all duration-200"
            style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", color: "#F1F5F9" }}
            onFocus={e => { e.currentTarget.style.borderColor = "rgba(99,102,241,0.5)"; e.currentTarget.style.boxShadow = "0 0 0 3px rgba(99,102,241,0.12)" }}
            onBlur={e  => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)"; e.currentTarget.style.boxShadow = "none" }}
          />
          {id === "pw" && (
            <button type="button" onClick={() => setShow(v => !v)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors">
              {show ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          )}
        </div>
      ))}
      <button
        type="submit" disabled={loading}
        className="relative overflow-hidden w-full h-11 rounded-xl text-sm font-semibold text-white btn-shimmer"
        style={{ background: "linear-gradient(135deg, #6366F1, #8B5CF6)", boxShadow: "0 0 28px -4px rgba(99,102,241,0.6)", opacity: loading ? 0.75 : 1 }}
      >
        {loading ? <Loader2 size={15} className="inline animate-spin mr-2" /> : null}
        Update password
      </button>
    </form>
  )
}

export default function ResetPasswordPage() {
  return (
    <div className="min-h-screen flex items-center justify-center px-6 relative">
      <div className="aurora-bg" aria-hidden>
        <div className="aurora-orb aurora-1" /><div className="aurora-orb aurora-2" />
      </div>
      <div className="fixed inset-0 bg-grid opacity-25 pointer-events-none" />
      <div className="fixed inset-0 pointer-events-none z-0"
        style={{ background: "radial-gradient(ellipse 70% 50% at 50% 10%, rgba(99,102,241,0.15) 0%, transparent 70%)" }} />

      <div className="relative z-10 w-full max-w-sm animate-fade-up">
        <div className="flex flex-col items-center gap-3 mb-10">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl animate-glow-breathe"
            style={{ background: "rgba(99,102,241,0.12)", border: "1px solid rgba(99,102,241,0.3)", boxShadow: "0 0 30px -4px rgba(99,102,241,0.5)" }}>
            <BrainCircuit size={26} style={{ color: "#818CF8" }} />
          </div>
          <div className="text-center">
            <h1 className="text-2xl font-bold text-shimmer">ModelForge</h1>
            <p className="text-sm mt-1" style={{ color: "rgba(203,213,225,0.5)" }}>Reset your password</p>
          </div>
        </div>

        <div className="rounded-2xl p-7"
          style={{ background: "rgba(8,14,24,0.75)", backdropFilter: "blur(40px)", border: "1px solid rgba(255,255,255,0.07)", boxShadow: "0 24px 80px rgba(0,0,0,0.5)" }}>
          <Suspense><ResetForm /></Suspense>
        </div>
      </div>
    </div>
  )
}
