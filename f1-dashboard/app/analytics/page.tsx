"use client"

import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { TrendingUp, Target, Zap, Activity, Brain } from "lucide-react"

const modelMetrics = [
  { name: "Overall Accuracy", value: 87.3, change: 2.1, trend: "up" },
  { name: "Podium Predictions", value: 92.1, change: 1.8, trend: "up" },
  { name: "Winner Predictions", value: 84.5, change: -0.5, trend: "down" },
  { name: "Qualifying Accuracy", value: 89.7, change: 3.2, trend: "up" },
]

const performanceByTrack = [
  { track: "Monaco", accuracy: 95.2, predictions: 45 },
  { track: "Silverstone", accuracy: 91.8, predictions: 52 },
  { track: "Spa-Francorchamps", accuracy: 88.4, predictions: 38 },
  { track: "Monza", accuracy: 86.7, predictions: 41 },
  { track: "Suzuka", accuracy: 84.3, predictions: 35 },
  { track: "Interlagos", accuracy: 82.1, predictions: 29 },
]

const featureImportance = [
  { feature: "Qualifying Position", importance: 92 },
  { feature: "Weather Conditions", importance: 78 },
  { feature: "Track Temperature", importance: 65 },
  { feature: "Driver Form", importance: 58 },
  { feature: "Car Performance", importance: 54 },
  { feature: "Tire Strategy", importance: 47 },
]

export default function AnalyticsPage() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-2 h-4" />
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold">Analytics</h1>
          </div>
        </header>

        <div className="flex flex-1 flex-col gap-4 p-4">
          <div className="min-h-[100vh] flex-1 rounded-xl bg-muted/50 p-4">
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold tracking-tight mb-2">Model Analytics</h2>
                <p className="text-muted-foreground">Deep insights into ML model performance and predictions</p>
              </div>

              {/* Model Performance Metrics */}
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                {modelMetrics.map((metric) => (
                  <Card key={metric.name}>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">{metric.name}</CardTitle>
                      <Activity className="size-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{metric.value}%</div>
                      <div
                        className={`flex items-center gap-1 text-xs ${
                          metric.trend === "up" ? "text-green-600" : "text-red-600"
                        }`}
                      >
                        <TrendingUp className={`size-3 ${metric.trend === "down" ? "rotate-180" : ""}`} />
                        {metric.change > 0 ? "+" : ""}
                        {metric.change}% from last month
                      </div>
                      <Progress value={metric.value} className="mt-2 h-2" />
                    </CardContent>
                  </Card>
                ))}
              </div>

              <div className="grid gap-6 lg:grid-cols-2">
                {/* Performance by Track */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Target className="size-5" />
                      Performance by Track
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {performanceByTrack.map((track) => (
                        <div key={track.track} className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="font-medium">{track.track}</span>
                            <div className="flex items-center gap-2">
                              <span className="text-sm text-muted-foreground">{track.predictions} predictions</span>
                              <span className="font-semibold">{track.accuracy}%</span>
                            </div>
                          </div>
                          <Progress value={track.accuracy} className="h-2" />
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Feature Importance */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Brain className="size-5" />
                      Feature Importance
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {featureImportance.map((feature) => (
                        <div key={feature.feature} className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="font-medium">{feature.feature}</span>
                            <span className="font-semibold">{feature.importance}%</span>
                          </div>
                          <Progress value={feature.importance} className="h-2" />
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Model Training History */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="size-5" />
                    Model Training History
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-3">
                      <div className="text-center p-4 border rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">v2.4.1</div>
                        <p className="text-sm text-muted-foreground">Current Version</p>
                        <p className="text-xs text-muted-foreground mt-1">Updated 2 days ago</p>
                      </div>
                      <div className="text-center p-4 border rounded-lg">
                        <div className="text-2xl font-bold">15,847</div>
                        <p className="text-sm text-muted-foreground">Training Samples</p>
                        <p className="text-xs text-muted-foreground mt-1">Last 5 seasons</p>
                      </div>
                      <div className="text-center p-4 border rounded-lg">
                        <div className="text-2xl font-bold">2.3h</div>
                        <p className="text-sm text-muted-foreground">Training Time</p>
                        <p className="text-xs text-muted-foreground mt-1">Latest update</p>
                      </div>
                    </div>
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
