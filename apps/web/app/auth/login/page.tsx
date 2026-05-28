"use client"

export const dynamic = 'force-dynamic'

import { Suspense, useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import Link from "next/link"
import { BrainCircuit, Loader2, Sparkles, Mail } from "lucide-react"
import { createClient } from "@/lib/supabase/client"
import { toast } from "sonner"

/* ── Field helper ────────────────────────────────────────────────────── */
function Field({
  label,
  id,
  type = "text",
  placeholder,
  value,
  onChange,
  required,
}: {
  label: string
  id: string
  type?: string
  placeholder?: string
  value: string
  onChange: (v: string) => void
  required?: boolean
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} style={{ fontSize: 13, color: "#94a3b8", fontWeight: 500 }}>
        {label}
      </label>
      <input
        id={id}
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        required={required}
        style={{
          width: "100%",
          height: 40,
          background: "var(--elevated)",
          border: "1px solid var(--border-c)",
          borderRadius: "var(--radius-btn)",
          padding: "0 12px",
          color: "var(--body)",
          fontSize: 14,
          outline: "none",
          transition: "border-color .15s, box-shadow .15s",
        }}
        onFocus={(e) => {
          e.currentTarget.style.borderColor = "var(--primary)"
          e.currentTarget.style.boxShadow = "0 0 0 2px var(--primary-10)"
        }}
        onBlur={(e) => {
          e.currentTarget.style.borderColor = "var(--border-c)"
          e.currentTarget.style.boxShadow = "none"
        }}
      />
    </div>
  )
}

/* ── Confirm (magic link sent) state ─────────────────────────────────── */
function ConfirmView({ email, onBack }: { email: string; onBack: () => void }) {
  return (
    <div className="flex flex-col items-center text-center gap-4">
      <div
        className="flex h-14 w-14 items-center justify-center rounded-full"
        style={{ background: "var(--primary-10)" }}
      >
        <Mail size={22} style={{ color: "var(--primary-light)" }} />
      </div>
      <div>
        <p style={{ fontSize: 18, fontWeight: 600, color: "var(--heading)" }}>
          Check your email
        </p>
        <p style={{ fontSize: 13, color: "var(--muted)", marginTop: 6, lineHeight: 1.6 }}>
          We sent a link to{" "}
          <span style={{ color: "var(--primary-lighter)" }}>{email}</span>.
          <br />Click it to sign in instantly.
        </p>
      </div>
      <button
        onClick={onBack}
        style={{
          marginTop: 8,
          width: "100%",
          height: 40,
          borderRadius: "var(--radius-btn)",
          border: "1px solid var(--border-c)",
          background: "transparent",
          color: "var(--body)",
          fontSize: 14,
          fontWeight: 500,
          cursor: "pointer",
          transition: "background .15s",
        }}
        onMouseEnter={(e) => (e.currentTarget.style.background = "var(--elevated)")}
        onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
      >
        Back to sign in
      </button>
    </div>
  )
}

/* ── Main form ───────────────────────────────────────────────────────── */
function LoginForm() {
  const router       = useRouter()
  const searchParams = useSearchParams()
  const next         = searchParams.get("next") ?? "/dashboard"
  const supabase     = createClient()

  const [email,     setEmail]     = useState("")
  const [password,  setPassword]  = useState("")
  const [loading,   setLoading]   = useState(false)
  const [view,      setView]      = useState<"login" | "confirm">("login")

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    const { error } = await supabase.auth.signInWithPassword({ email, password })
    if (error) {
      toast.error(error.message)
      setLoading(false)
    } else {
      router.push(next)
      router.refresh()
    }
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
    <div
      className="min-h-screen flex items-center justify-center px-6"
      style={{ background: "var(--bg)" }}
    >
      {/* Radial hero glow behind the card */}
      <div
        className="pointer-events-none fixed inset-0"
        style={{
          background: "radial-gradient(ellipse 70% 50% at 50% 10%, var(--primary-05) 0%, transparent 70%)",
          zIndex: 0,
        }}
      />

      <div
        className="relative z-10 w-full animate-fade-up"
        style={{ maxWidth: 384 }}
      >
        {/* Logo */}
        <div className="flex flex-col items-center gap-2 mb-8">
          <div
            className="flex h-12 w-12 items-center justify-center rounded-xl mb-1 animate-glow-breathe"
            style={{
              background: "var(--primary-10)",
              border: "1px solid var(--primary-30)",
            }}
          >
            <BrainCircuit size={22} style={{ color: "var(--primary)" }} />
          </div>
          <h1
            style={{
              fontSize: 24,
              fontWeight: 700,
              color: "var(--heading)",
              letterSpacing: "-0.02em",
            }}
          >
            ModelForge
          </h1>
          <p style={{ fontSize: 14, color: "var(--muted)" }}>
            Train AI models with plain English
          </p>
        </div>

        {/* Card */}
        {view === "confirm" ? (
          <ConfirmView email={email} onBack={() => setView("login")} />
        ) : (
          <div className="flex flex-col gap-4">
            <form onSubmit={handleLogin} className="flex flex-col gap-4">
              <Field
                label="Email"
                id="email"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={setEmail}
                required
              />
              <Field
                label="Password"
                id="password"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={setPassword}
                required
              />

              <button
                type="submit"
                disabled={loading}
                style={{
                  width: "100%",
                  height: 40,
                  borderRadius: "var(--radius-btn)",
                  background: loading ? "var(--primary-light)" : "var(--primary)",
                  color: "#fff",
                  fontSize: 14,
                  fontWeight: 500,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 8,
                  cursor: loading ? "not-allowed" : "pointer",
                  opacity: loading ? 0.8 : 1,
                  boxShadow: "0 0 20px -4px var(--primary-50)",
                  transition: "opacity .15s, box-shadow .15s",
                  border: "none",
                }}
                onMouseEnter={(e) => {
                  if (!loading) e.currentTarget.style.boxShadow = "0 0 28px -2px var(--primary-50)"
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.boxShadow = "0 0 20px -4px var(--primary-50)"
                }}
              >
                {loading ? <Loader2 size={16} className="animate-spin" /> : null}
                Sign in
              </button>
            </form>

            {/* Divider */}
            <div className="flex items-center gap-3">
              <div style={{ flex: 1, height: 1, background: "var(--border-c)" }} />
              <span style={{ fontSize: 12, color: "var(--muted)" }}>or</span>
              <div style={{ flex: 1, height: 1, background: "var(--border-c)" }} />
            </div>

            {/* Magic link */}
            <button
              onClick={handleMagicLink}
              disabled={loading}
              style={{
                width: "100%",
                height: 40,
                borderRadius: "var(--radius-btn)",
                background: "transparent",
                border: "1px solid var(--border-c)",
                color: "var(--body)",
                fontSize: 14,
                fontWeight: 500,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 8,
                cursor: "pointer",
                transition: "background .15s, border-color .15s",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "var(--elevated)"
                e.currentTarget.style.borderColor = "var(--elevated-2)"
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "transparent"
                e.currentTarget.style.borderColor = "var(--border-c)"
              }}
            >
              <Sparkles size={15} style={{ color: "var(--primary-light)" }} />
              Send me a magic link
            </button>

            {/* Sign up link */}
            <p
              className="text-center"
              style={{ fontSize: 13, color: "var(--muted)", marginTop: 4 }}
            >
              Don&apos;t have an account?{" "}
              <Link
                href="/auth/signup"
                style={{
                  color: "var(--primary-light)",
                  textDecoration: "underline",
                  textUnderlineOffset: 3,
                }}
              >
                Sign up free
              </Link>
            </p>
          </div>
        )}
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
