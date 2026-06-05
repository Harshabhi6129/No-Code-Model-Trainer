"use client"

export const dynamic = 'force-dynamic'

import { Suspense, useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import Link from "next/link"
import { BrainCircuit, Loader2, Sparkles, Mail, ArrowRight } from "lucide-react"
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

function ConfirmView({ email, onBack }: { email: string; onBack: () => void }) {
  return (
    <div className="flex flex-col items-center text-center gap-5">
      <div
        className="flex h-16 w-16 items-center justify-center rounded-2xl animate-float"
        style={{
          background: "rgba(99,102,241,0.12)",
          border: "1px solid rgba(99,102,241,0.3)",
          boxShadow: "0 0 30px -6px rgba(99,102,241,0.4)",
        }}
      >
        <Mail size={24} style={{ color: "#818CF8" }} />
      </div>
      <div>
        <p className="text-xl font-bold" style={{ color: "#F1F5F9" }}>
          Check your email
        </p>
        <p className="text-sm mt-2 leading-relaxed" style={{ color: "rgba(203,213,225,0.6)" }}>
          We sent a magic link to{" "}
          <span style={{ color: "#A5B4FC" }}>{email}</span>.
          <br />Click it to sign in instantly.
        </p>
      </div>
      <button
        onClick={onBack}
        className="w-full h-11 rounded-xl text-sm font-medium transition-all duration-200"
        style={{
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.08)",
          color: "rgba(203,213,225,0.8)",
        }}
        onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.07)" }}
        onMouseLeave={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.04)" }}
      >
        Back to sign in
      </button>
    </div>
  )
}

function LoginForm() {
  const router       = useRouter()
  const searchParams = useSearchParams()
  const next         = searchParams.get("next") ?? "/dashboard"
  const supabase     = createClient()

  const [email,    setEmail]    = useState("")
  const [password, setPassword] = useState("")
  const [loading,  setLoading]  = useState(false)
  const [view,     setView]     = useState<"login" | "confirm">("login")

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    const { error } = await supabase.auth.signInWithPassword({ email, password })
    if (error) { toast.error(error.message); setLoading(false) }
    else { router.push(next); router.refresh() }
  }

  async function handleMagicLink() {
    if (!email) { toast.error("Enter your email first"); return }
    setLoading(true)
    const { error } = await supabase.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: `${window.location.origin}/auth/callback?next=${next}` },
    })
    setLoading(false)
    if (error) toast.error(error.message)
    else setView("confirm")
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-6 relative">
      {/* Aurora background */}
      <div className="aurora-bg" aria-hidden>
        <div className="aurora-orb aurora-1" />
        <div className="aurora-orb aurora-2" />
        <div className="aurora-orb aurora-3" />
      </div>
      {/* Dot grid */}
      <div className="fixed inset-0 bg-grid opacity-25 pointer-events-none" />

      {/* Hero glow */}
      <div
        className="pointer-events-none fixed inset-0 z-0"
        style={{
          background: "radial-gradient(ellipse 70% 50% at 50% 10%, rgba(99,102,241,0.15) 0%, transparent 70%)",
        }}
      />

      <div className="relative z-10 w-full max-w-sm animate-fade-up">
        {/* Logo */}
        <div className="flex flex-col items-center gap-3 mb-10">
          <div
            className="flex h-14 w-14 items-center justify-center rounded-2xl animate-glow-breathe"
            style={{
              background: "rgba(99,102,241,0.12)",
              border: "1px solid rgba(99,102,241,0.3)",
              boxShadow: "0 0 30px -4px rgba(99,102,241,0.5), inset 0 1px 0 rgba(255,255,255,0.08)",
            }}
          >
            <BrainCircuit size={26} style={{ color: "#818CF8" }} />
          </div>
          <div className="text-center">
            <h1 className="text-2xl font-bold text-shimmer">ModelForge</h1>
            <p className="text-sm mt-1" style={{ color: "rgba(203,213,225,0.5)" }}>
              Train AI models with plain English
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
          {view === "confirm" ? (
            <ConfirmView email={email} onBack={() => setView("login")} />
          ) : (
            <div className="flex flex-col gap-5">
              <form onSubmit={handleLogin} className="flex flex-col gap-4">
                <Field label="Email" id="email" type="email" placeholder="you@example.com"
                  value={email} onChange={setEmail} required />
                <Field label="Password" id="password" type="password" placeholder="••••••••"
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
                    Sign in
                    {!loading && <ArrowRight size={14} />}
                  </span>
                </button>
              </form>

              {/* Divider */}
              <div className="flex items-center gap-3">
                <div className="flex-1 h-px" style={{ background: "rgba(255,255,255,0.06)" }} />
                <span className="text-xs" style={{ color: "rgba(100,116,139,0.7)" }}>or</span>
                <div className="flex-1 h-px" style={{ background: "rgba(255,255,255,0.06)" }} />
              </div>

              {/* Magic link */}
              <button
                onClick={handleMagicLink}
                disabled={loading}
                className="w-full h-11 rounded-xl text-sm font-medium transition-all duration-200 flex items-center justify-center gap-2"
                style={{
                  background: "rgba(255,255,255,0.04)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  color: "rgba(203,213,225,0.8)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "rgba(99,102,241,0.08)"
                  e.currentTarget.style.borderColor = "rgba(99,102,241,0.25)"
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "rgba(255,255,255,0.04)"
                  e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)"
                }}
              >
                <Sparkles size={14} style={{ color: "#818CF8" }} />
                Send me a magic link
              </button>

              {/* Sign up link */}
              <p className="text-center text-xs" style={{ color: "rgba(100,116,139,0.7)" }}>
                Don&apos;t have an account?{" "}
                <Link href="/auth/signup"
                  style={{ color: "#818CF8", textDecoration: "underline", textUnderlineOffset: 3 }}>
                  Sign up free
                </Link>
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default function LoginPage() {
  return (
    <Suspense>
      <LoginForm />
    </Suspense>
  )
}
