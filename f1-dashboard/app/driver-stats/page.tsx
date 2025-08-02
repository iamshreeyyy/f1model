"use client"

import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"

const driverStats = [
  {
    name: "Max Verstappen",
    team: "Red Bull Racing",
    points: 575,
    wins: 19,
    podiums: 21,
    poles: 9,
    predictedWinRate: 94,
    actualWinRate: 90,
    avatar: "/placeholder.svg?height=60&width=60&text=MV",
  },
  {
    name: "Charles Leclerc",
    team: "Ferrari",
    points: 308,
    wins: 2,
    podiums: 8,
    poles: 9,
    predictedWinRate: 78,
    actualWinRate: 75,
    avatar: "/placeholder.svg?height=60&width=60&text=CL",
  },
  {
    name: "Lewis Hamilton",
    team: "Mercedes",
    points: 234,
    wins: 1,
    podiums: 5,
    poles: 1,
    predictedWinRate: 65,
    actualWinRate: 62,
    avatar: "/placeholder.svg?height=60&width=60&text=LH",
  },
  {
    name: "George Russell",
    team: "Mercedes",
    points: 175,
    wins: 0,
    podiums: 3,
    poles: 1,
    predictedWinRate: 45,
    actualWinRate: 48,
    avatar: "/placeholder.svg?height=60&width=60&text=GR",
  },
  {
    name: "Lando Norris",
    team: "McLaren",
    points: 205,
    wins: 1,
    podiums: 7,
    poles: 0,
    predictedWinRate: 52,
    actualWinRate: 55,
    avatar: "/placeholder.svg?height=60&width=60&text=LN",
  },
  {
    name: "Carlos Sainz",
    team: "Ferrari",
    points: 190,
    wins: 1,
    podiums: 4,
    poles: 0,
    predictedWinRate: 48,
    actualWinRate: 45,
    avatar: "/placeholder.svg?height=60&width=60&text=CS",
  },
]

export default function DriverStatsPage() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-2 h-4" />
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold">Driver Statistics</h1>
          </div>
        </header>

        <div className="flex flex-1 flex-col gap-4 p-4">
          <div className="min-h-[100vh] flex-1 rounded-xl bg-muted/50 p-4">
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold tracking-tight mb-2">2024 Season Driver Performance</h2>
                <p className="text-muted-foreground">
                  Comprehensive statistics and prediction accuracy for each driver
                </p>
              </div>

              <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                {driverStats.map((driver) => (
                  <Card key={driver.name} className="hover:shadow-lg transition-shadow">
                    <CardHeader>
                      <div className="flex items-center gap-3">
                        <img
                          src={driver.avatar || "/placeholder.svg"}
                          alt={driver.name}
                          className="size-12 rounded-full"
                        />
                        <div>
                          <CardTitle className="text-lg">{driver.name}</CardTitle>
                          <p className="text-sm text-muted-foreground">{driver.team}</p>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-600">{driver.points}</div>
                          <p className="text-xs text-muted-foreground">Points</p>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-600">{driver.wins}</div>
                          <p className="text-xs text-muted-foreground">Wins</p>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="text-center">
                          <div className="text-lg font-semibold">{driver.podiums}</div>
                          <p className="text-xs text-muted-foreground">Podiums</p>
                        </div>
                        <div className="text-center">
                          <div className="text-lg font-semibold">{driver.poles}</div>
                          <p className="text-xs text-muted-foreground">Pole Positions</p>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Predicted Win Rate</span>
                          <span>{driver.predictedWinRate}%</span>
                        </div>
                        <Progress value={driver.predictedWinRate} className="h-2" />
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Actual Win Rate</span>
                          <span>{driver.actualWinRate}%</span>
                        </div>
                        <Progress value={driver.actualWinRate} className="h-2" />
                      </div>

                      <div className="flex justify-center">
                        <div
                          className={`text-sm font-medium ${
                            Math.abs(driver.predictedWinRate - driver.actualWinRate) <= 5
                              ? "text-green-600"
                              : "text-yellow-600"
                          }`}
                        >
                          Prediction Accuracy: {100 - Math.abs(driver.predictedWinRate - driver.actualWinRate)}%
                        </div>
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
