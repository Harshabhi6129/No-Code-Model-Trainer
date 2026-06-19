import { createClient } from "@/lib/supabase/client"

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

/**
 * Authorization header for the FastAPI backend, derived from the current
 * Supabase session. Returns an empty object when signed out so callers can
 * always spread it into a `headers` block.
 */
export async function authHeader(): Promise<Record<string, string>> {
  try {
    const supabase = createClient()
    const { data: { session } } = await supabase.auth.getSession()
    const token = session?.access_token
    return token ? { Authorization: `Bearer ${token}` } : {}
  } catch {
    return {}
  }
}
