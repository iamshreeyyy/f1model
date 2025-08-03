"use client"

import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"
import { ModelStats } from "@/components/model-stats"
import { RacePredictions } from "@/components/race-predictions"
import { RecentActivity } from "@/components/recent-activity"
import { ProfessionalOverview } from "@/components/professional-overview"
import { Separator } from "@/components/ui/separator"

export default function Dashboard() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-2 h-4" />
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold">F1 ML Dashboard</h1>
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-sm text-muted-foreground">Live</span>
          </div>
        </header>

        <div className="flex flex-1 flex-col gap-4 p-4">
          <div className="min-h-[100vh] flex-1 rounded-xl bg-muted/50 p-4">
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold tracking-tight mb-2">Welcome back</h2>
                <p className="text-muted-foreground">Here's what's happening with your F1 prediction model today.</p>
              </div>

              <ModelStats />

              <div className="grid gap-6 lg:grid-cols-3">
                <div className="lg:col-span-2">
                  <RacePredictions />
                </div>
                <div className="space-y-6">
                  <RecentActivity />
                  <ProfessionalOverview />
                </div>
              </div>
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
