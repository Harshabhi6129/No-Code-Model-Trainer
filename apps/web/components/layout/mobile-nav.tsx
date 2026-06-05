"use client"

import { useState, useEffect } from "react"
import { Menu, X } from "lucide-react"
import { usePathname } from "next/navigation"
import { Sidebar } from "./sidebar"

interface MobileNavProps {
  userEmail?: string
}

export function MobileNav({ userEmail }: MobileNavProps) {
  const [open, setOpen] = useState(false)
  const pathname = usePathname()

  // Close sidebar on navigation
  useEffect(() => { setOpen(false) }, [pathname])

  return (
    <>
      {/* Hamburger — only on mobile */}
      <button
        onClick={() => setOpen(v => !v)}
        className="fixed top-4 left-4 z-[60] flex h-9 w-9 items-center justify-center rounded-xl md:hidden transition-colors"
        style={{
          background: "rgba(12, 20, 32, 0.85)",
          backdropFilter: "blur(20px)",
          border: "1px solid rgba(255,255,255,0.08)",
          boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
        }}
        aria-label="Toggle navigation"
      >
        {open
          ? <X className="h-4 w-4 text-muted-foreground" />
          : <Menu className="h-4 w-4 text-muted-foreground" />
        }
      </button>

      {/* Backdrop overlay — mobile only */}
      {open && (
        <div
          className="fixed inset-0 z-[55] bg-black/60 backdrop-blur-sm md:hidden"
          onClick={() => setOpen(false)}
        />
      )}

      {/* Sidebar with mobile slide-in */}
      <div
        className={`md:translate-x-0 transition-transform duration-300 ease-out ${open ? "translate-x-0" : "-translate-x-full"}`}
      >
        <Sidebar userEmail={userEmail} />
      </div>
    </>
  )
}
