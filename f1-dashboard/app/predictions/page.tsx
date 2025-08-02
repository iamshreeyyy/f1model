"use client"

import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { Calendar, Clock, MapPin } from "lucide-react"

const upcomingRaces = [
  {
    name: "Monaco Grand Prix",
    date: "May 26, 2024",
    location: "Monte Carlo",
    predictions: [
      { driver: "Max Verstappen", position: 1, confidence: 94 },
      { driver: "Charles Leclerc", position: 2, confidence: 89 },
      { driver: "Lewis Hamilton", position: 3, confidence: 82 },
      { driver: "Lando Norris", position: 4, confidence: 78 },
      { driver: "George Russell", position: 5, confidence: 74 },
    ],
  },
  {
    name: "Canadian Grand Prix",
    date: "June 9, 2024",
    location: "Montreal",
    predictions: [
      { driver: "Max Verstappen", position: 1, confidence: 91 },
      { driver: "Lewis Hamilton", position: 2, confidence: 87 },
      { driver: "George Russell", position: 3, confidence: 83 },
      { driver: "Charles Leclerc", position: 4, confidence: 79 },
      { driver: "Lando Norris", position: 5, confidence: 75 },
    ],
  },
]

export default function PredictionsPage() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-2 h-4" />
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold">Race Predictions</h1>
          </div>
        </header>

        <div className="flex flex-1 flex-col gap-4 p-4">
          <div className="min-h-[100vh] flex-1 rounded-xl bg-muted/50 p-4">
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold tracking-tight mb-2">Upcoming Race Predictions</h2>
                <p className="text-muted-foreground">AI-powered predictions for upcoming Formula 1 races</p>
              </div>

              <div className="space-y-6">
                {upcomingRaces.map((race, index) => (
                  <Card key={index}>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="flex items-center gap-2">
                          <Calendar className="size-5" />
                          {race.name}
                        </CardTitle>
                        <div className="flex items-center gap-4 text-sm text-muted-foreground">
                          <div className="flex items-center gap-1">
                            <Clock className="size-4" />
                            {race.date}
                          </div>
                          <div className="flex items-center gap-1">
                            <MapPin className="size-4" />
                            {race.location}
                          </div>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
                        {race.predictions.map((prediction) => (
                          <Card key={prediction.driver} className="p-4">
                            <div className="flex items-center justify-between mb-2">
                              <Badge variant={prediction.position <= 3 ? "default" : "secondary"}>
                                P{prediction.position}
                              </Badge>
                              <span className="text-sm font-medium">{prediction.confidence}%</span>
                            </div>
                            <h3 className="font-semibold text-sm mb-2">{prediction.driver}</h3>
                            <Progress value={prediction.confidence} className="h-2" />
                          </Card>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
