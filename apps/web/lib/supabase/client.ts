import { createBrowserClient } from "@supabase/ssr"
import type { Database } from "./types"

export function createClient() {
  return createBrowserClient<Database>(
    process.env.NEXT_PUBLIC_SUPABASE_URL ?? "http://localhost:54321",
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "placeholder-build-time-key"
  )
}
