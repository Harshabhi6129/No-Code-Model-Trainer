"use client"

export const dynamic = 'force-dynamic'

import { useState } from "react"
import Link from "next/link"
import { BrainCircuit, Loader2, ArrowRight, CheckCircle2 } from "lucide-react"
import { createClient } from "@/lib/supabase/client"
import { toast } from "sonner"

function Field({
  label, id, type = "text", placeholder, value, onChange, required,
}: {
  label: string; id: string; type?: string; placeholder?: string
  value: string; onChange: (v: string) => void; required?: boolean
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} className="text-xs font-medium"
        style={{ color: "rgba(203,213,225,0.7)" }}>
        {label}
      </label>
      <input
        id={id} type={type} placeholder={placeholder}
        value={value} onChange={(e) => onChange(e.target.value)} required={required}
        className="w-full h-11 rounded-xl px-4 text-sm outline-none transition-all duration-200"
        style={{
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.08)",
          color: "#F1F5F9",
          boxShadow: "inset 0 1px 0 rgba(255,255,255,0.04)",
        }}
        onFocus={(e) => {
          e.currentTarget.style.borderColor = "rgba(99,102,241,0.5)"
          e.currentTarget.style.boxShadow = "0 0 0 3px rgba(99,102,241,0.12), inset 0 1px 0 rgba(255,255,255,0.04)"
          e.currentTarget.style.background = "rgba(99,102,241,0.06)"
        }}
        onBlur={(e) => {
          e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)"
          e.currentTarget.style.boxShadow = "inset 0 1px 0 rgba(255,255,255,0.04)"
          e.currentTarget.style.background = "rgba(255,255,255,0.04)"
        }}
      />
    </div>
  )
}

export default function SignupPage() {
  const [email,    setEmail]    = useState("")
  const [password, setPassword] = useState("")
  const [loading,  setLoading]  = useState(false)
  const [done,     setDone]     = useState(false)
  const supabase = createClient()

  async function handleSignup(e: React.FormEvent) {
    e.preventDefault()
    if (password.length < 8) { toast.error("Password must be at least 8 characters"); return }
    setLoading(true)
    const { error } = await supabase.auth.signUp({
      email, password,
      options: { emailRedirectTo: `${window.location.origin}/auth/callback` },
    })
    setLoading(false)
    if (error) toast.error(error.message)
    else setDone(true)
  }

  if (done) {
    return (
      <div className="min-h-screen flex items-center justify-center px-6 relative">
        <div className="aurora-bg" aria-hidden>
          <div className="aurora-orb aurora-1" />
          <div className="aurora-orb aurora-2" />
        </div>
        <div className="fixed inset-0 bg-grid opacity-25 pointer-events-none" />
        <div className="relative z-10 text-center space-y-5 animate-fade-up max-w-sm">
          <div
            className="flex h-16 w-16 items-center justify-center rounded-2xl mx-auto animate-float"
            style={{
              background: "rgba(16,185,129,0.12)",
              border: "1px solid rgba(16,185,129,0.3)",
              boxShadow: "0 0 30px -6px rgba(16,185,129,0.4)",
            }}
          >
            <CheckCircle2 size={26} style={{ color: "#34D399" }} />
          </div>
          <h2 className="text-2xl font-bold" style={{ color: "#F1F5F9" }}>Check your email</h2>
          <p className="text-sm leading-relaxed" style={{ color: "rgba(203,213,225,0.6)" }}>
            We sent a confirmation link to{" "}
            <span style={{ color: "#A5B4FC" }}>{email}</span>.<br />
            Click it to activate your account.
          </p>
          <Link href="/auth/login"
            className="inline-block text-sm underline underline-offset-3"
            style={{ color: "#818CF8" }}>
            Back to sign in
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-6 relative">
      {/* Aurora background */}
      <div className="aurora-bg" aria-hidden>
        <div className="aurora-orb aurora-1" />
        <div className="aurora-orb aurora-2" />
        <div className="aurora-orb aurora-3" />
      </div>
      <div className="fixed inset-0 bg-grid opacity-25 pointer-events-none" />

      {/* Hero glow */}
      <div
        className="pointer-events-none fixed inset-0 z-0"
        style={{
          background: "radial-gradient(ellipse 70% 50% at 50% 10%, rgba(139,92,246,0.15) 0%, transparent 70%)",
        }}
      />

      <div className="relative z-10 w-full max-w-sm animate-fade-up">
        {/* Logo */}
        <div className="flex flex-col items-center gap-3 mb-10">
          <div
            className="flex h-14 w-14 items-center justify-center rounded-2xl animate-glow-breathe"
            style={{
              background: "rgba(139,92,246,0.12)",
              border: "1px solid rgba(139,92,246,0.3)",
              boxShadow: "0 0 30px -4px rgba(139,92,246,0.5), inset 0 1px 0 rgba(255,255,255,0.08)",
            }}
          >
            <BrainCircuit size={26} style={{ color: "#A78BFA" }} />
          </div>
          <div className="text-center">
            <h1 className="text-2xl font-bold text-shimmer">ModelForge</h1>
            <p className="text-sm mt-1" style={{ color: "rgba(203,213,225,0.5)" }}>
              Create your free account
            </p>
          </div>
        </div>

        {/* Card */}
        <div
          className="rounded-2xl p-7"
          style={{
            background: "rgba(8,14,24,0.75)",
            backdropFilter: "blur(40px) saturate(2)",
            WebkitBackdropFilter: "blur(40px) saturate(2)",
            border: "1px solid rgba(255,255,255,0.07)",
            boxShadow: "0 24px 80px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06)",
          }}
        >
          <form onSubmit={handleSignup} className="flex flex-col gap-5">
            <Field label="Email" id="email" type="email" placeholder="you@example.com"
              value={email} onChange={setEmail} required />
            <Field label="Password" id="password" type="password" placeholder="Min. 8 characters"
              value={password} onChange={setPassword} required />

            <button
              type="submit"
              disabled={loading}
              className="relative overflow-hidden w-full h-11 rounded-xl text-sm font-semibold text-white transition-all duration-200 btn-shimmer"
              style={{
                background: "linear-gradient(135deg, #6366F1, #8B5CF6)",
                boxShadow: "0 0 28px -4px rgba(99,102,241,0.6)",
                opacity: loading ? 0.75 : 1,
                cursor: loading ? "not-allowed" : "pointer",
              }}
              onMouseEnter={(e) => {
                if (!loading) e.currentTarget.style.boxShadow = "0 0 40px -2px rgba(99,102,241,0.8)"
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.boxShadow = "0 0 28px -4px rgba(99,102,241,0.6)"
              }}
            >
              <span className="flex items-center justify-center gap-2">
                {loading ? <Loader2 size={15} className="animate-spin" /> : null}
                Create account
                {!loading && <ArrowRight size={14} />}
              </span>
            </button>

            <p className="text-center text-xs" style={{ color: "rgba(100,116,139,0.7)" }}>
              Already have an account?{" "}
              <Link href="/auth/login"
                style={{ color: "#818CF8", textDecoration: "underline", textUnderlineOffset: 3 }}>
                Sign in
              </Link>
            </p>
          </form>
        </div>
      </div>
    </div>
  )
}
