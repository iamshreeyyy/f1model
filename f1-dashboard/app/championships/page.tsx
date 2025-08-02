"use client"

import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { Trophy, Users, Building } from "lucide-react"

const driversChampionship = [
  { position: 1, driver: "Max Verstappen", team: "Red Bull Racing", points: 575, gap: 0 },
  { position: 2, driver: "Charles Leclerc", team: "Ferrari", points: 308, gap: 267 },
  { position: 3, driver: "Lewis Hamilton", team: "Mercedes", points: 234, gap: 341 },
  { position: 4, driver: "Lando Norris", team: "McLaren", points: 205, gap: 370 },
  { position: 5, driver: "Carlos Sainz", team: "Ferrari", points: 190, gap: 385 },
  { position: 6, driver: "George Russell", team: "Mercedes", points: 175, gap: 400 },
]

const constructorsChampionship = [
  { position: 1, team: "Red Bull Racing", points: 860, gap: 0 },
  { position: 2, team: "Ferrari", points: 498, gap: 362 },
  { position: 3, team: "Mercedes", points: 409, gap: 451 },
  { position: 4, team: "McLaren", points: 302, gap: 558 },
  { position: 5, team: "Aston Martin", points: 280, gap: 580 },
  { position: 6, team: "Alpine", points: 120, gap: 740 },
]

const predictions = {
  driversChampion: { driver: "Max Verstappen", confidence: 99.8 },
  constructorsChampion: { team: "Red Bull Racing", confidence: 99.5 },
  remainingRaces: 2,
}

export default function ChampionshipsPage() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-2 h-4" />
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold">Championships</h1>
          </div>
        </header>

        <div className="flex flex-1 flex-col gap-4 p-4">
          <div className="min-h-[100vh] flex-1 rounded-xl bg-muted/50 p-4">
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold tracking-tight mb-2">2024 Championship Standings</h2>
                <p className="text-muted-foreground">Current standings and championship predictions</p>
              </div>

              {/* Championship Predictions */}
              <div className="grid gap-4 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Trophy className="size-5 text-yellow-500" />
                      Drivers' Championship Prediction
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="text-center">
                        <div className="text-2xl font-bold">{predictions.driversChampion.driver}</div>
                        <div className="text-lg text-green-600">
                          {predictions.driversChampion.confidence}% Confidence
                        </div>
                      </div>
                      <Progress value={predictions.driversChampion.confidence} className="h-3" />
                      <p className="text-sm text-muted-foreground text-center">
                        {predictions.remainingRaces} races remaining
                      </p>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Building className="size-5 text-blue-500" />
                      Constructors' Championship Prediction
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="text-center">
                        <div className="text-2xl font-bold">{predictions.constructorsChampion.team}</div>
                        <div className="text-lg text-green-600">
                          {predictions.constructorsChampion.confidence}% Confidence
                        </div>
                      </div>
                      <Progress value={predictions.constructorsChampion.confidence} className="h-3" />
                      <p className="text-sm text-muted-foreground text-center">
                        {predictions.remainingRaces} races remaining
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Drivers Championship */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Users className="size-5" />
                    Drivers' Championship Standings
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left p-3">Position</th>
                          <th className="text-left p-3">Driver</th>
                          <th className="text-left p-3">Team</th>
                          <th className="text-left p-3">Points</th>
                          <th className="text-left p-3">Gap</th>
                        </tr>
                      </thead>
                      <tbody>
                        {driversChampionship.map((driver) => (
                          <tr key={driver.driver} className="border-b hover:bg-muted/50">
                            <td className="p-3">
                              <Badge variant={driver.position <= 3 ? "default" : "secondary"}>{driver.position}</Badge>
                            </td>
                            <td className="p-3 font-medium">{driver.driver}</td>
                            <td className="p-3 text-muted-foreground">{driver.team}</td>
                            <td className="p-3 font-semibold">{driver.points}</td>
                            <td className="p-3 text-muted-foreground">
                              {driver.gap === 0 ? "Leader" : `+${driver.gap}`}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>

              {/* Constructors Championship */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Building className="size-5" />
                    Constructors' Championship Standings
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left p-3">Position</th>
                          <th className="text-left p-3">Team</th>
                          <th className="text-left p-3">Points</th>
                          <th className="text-left p-3">Gap</th>
                        </tr>
                      </thead>
                      <tbody>
                        {constructorsChampionship.map((team) => (
                          <tr key={team.team} className="border-b hover:bg-muted/50">
                            <td className="p-3">
                              <Badge variant={team.position <= 3 ? "default" : "secondary"}>{team.position}</Badge>
                            </td>
                            <td className="p-3 font-medium">{team.team}</td>
                            <td className="p-3 font-semibold">{team.points}</td>
                            <td className="p-3 text-muted-foreground">{team.gap === 0 ? "Leader" : `+${team.gap}`}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
