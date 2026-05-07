import { AppShell } from "@/components/layout/app-shell"
import { DatasetsClient } from "./datasets-client"

export default function DatasetsPage() {
  return (
    <AppShell>
      <DatasetsClient />
    </AppShell>
  )
}
