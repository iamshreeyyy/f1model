"use client"

import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Trophy, CheckCircle, XCircle } from "lucide-react"

const raceResults = [
  {
    race: "Bahrain Grand Prix",
    date: "March 2, 2024",
    results: [
      { position: 1, driver: "Max Verstappen", team: "Red Bull Racing", predicted: 1, correct: true },
      { position: 2, driver: "Sergio Perez", team: "Red Bull Racing", predicted: 3, correct: false },
      { position: 3, driver: "Charles Leclerc", team: "Ferrari", predicted: 2, correct: false },
      { position: 4, driver: "Carlos Sainz", team: "Ferrari", predicted: 4, correct: true },
      { position: 5, driver: "Lewis Hamilton", team: "Mercedes", predicted: 5, correct: true },
    ],
    accuracy: 60,
  },
  {
    race: "Saudi Arabian Grand Prix",
    date: "March 9, 2024",
    results: [
      { position: 1, driver: "Max Verstappen", team: "Red Bull Racing", predicted: 1, correct: true },
      { position: 2, driver: "Charles Leclerc", team: "Ferrari", predicted: 2, correct: true },
      { position: 3, driver: "George Russell", team: "Mercedes", predicted: 4, correct: false },
      { position: 4, driver: "Sergio Perez", team: "Red Bull Racing", predicted: 3, correct: false },
      { position: 5, driver: "Lewis Hamilton", team: "Mercedes", predicted: 5, correct: true },
    ],
    accuracy: 60,
  },
]

export default function RaceResultsPage() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-2 h-4" />
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold">Race Results</h1>
          </div>
        </header>

        <div className="flex flex-1 flex-col gap-4 p-4">
          <div className="min-h-[100vh] flex-1 rounded-xl bg-muted/50 p-4">
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold tracking-tight mb-2">Recent Race Results</h2>
                <p className="text-muted-foreground">Compare predictions vs actual race outcomes</p>
              </div>

              <div className="space-y-6">
                {raceResults.map((race, index) => (
                  <Card key={index}>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="flex items-center gap-2">
                          <Trophy className="size-5" />
                          {race.race}
                        </CardTitle>
                        <div className="flex items-center gap-4">
                          <span className="text-sm text-muted-foreground">{race.date}</span>
                          <Badge
                            variant={
                              race.accuracy >= 70 ? "default" : race.accuracy >= 50 ? "secondary" : "destructive"
                            }
                          >
                            {race.accuracy}% Accuracy
                          </Badge>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b">
                              <th className="text-left p-2">Position</th>
                              <th className="text-left p-2">Driver</th>
                              <th className="text-left p-2">Team</th>
                              <th className="text-left p-2">Predicted</th>
                              <th className="text-left p-2">Accuracy</th>
                            </tr>
                          </thead>
                          <tbody>
                            {race.results.map((result) => (
                              <tr key={result.driver} className="border-b">
                                <td className="p-2">
                                  <Badge variant={result.position <= 3 ? "default" : "secondary"}>
                                    P{result.position}
                                  </Badge>
                                </td>
                                <td className="p-2 font-medium">{result.driver}</td>
                                <td className="p-2 text-muted-foreground">{result.team}</td>
                                <td className="p-2">P{result.predicted}</td>
                                <td className="p-2">
                                  {result.correct ? (
                                    <CheckCircle className="size-5 text-green-600" />
                                  ) : (
                                    <XCircle className="size-5 text-red-600" />
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
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
