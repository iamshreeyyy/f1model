"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { useEffect, useState } from "react"
import { apiClient, ModelStats as ModelStatsType } from "@/lib/api"

export function ModelStats() {
  const [stats, setStats] = useState<ModelStatsType | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchStats = async () => {
      try {
        setLoading(true)
        setError(null)
        console.log('🔄 Fetching real model stats from API...')
        const data = await apiClient.getModelStats()
        console.log('📊 Received model stats:', data)
        setStats(data)
      } catch (error) {
        console.error('❌ Failed to fetch model stats:', error)
        setError('Failed to connect to ML model')
      } finally {
        setLoading(false)
      }
    }

    fetchStats()
    
    // Refresh stats every 10 seconds for real-time updates
    const interval = setInterval(fetchStats, 10000)
    return () => clearInterval(interval)
  }, [])

  if (loading && !stats) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <Card key={i}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <div className="h-4 bg-muted animate-pulse rounded w-24" />
            </CardHeader>
            <CardContent>
              <div className="h-8 bg-muted animate-pulse rounded w-16 mb-2" />
              <div className="h-2 bg-muted animate-pulse rounded w-full mb-2" />
              <div className="h-3 bg-muted animate-pulse rounded w-20" />
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  if (error) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="md:col-span-2 lg:col-span-4">
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-red-500 font-medium">⚠️ {error}</p>
              <p className="text-sm text-muted-foreground mt-1">
                Make sure the backend API is running on http://localhost:8000
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!stats) return null

  const accuracyValue = parseFloat(stats.model_accuracy.replace('%', ''))
  const successRate = parseFloat(stats.success_rate.replace('%', ''))

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Model Accuracy</CardTitle>
          <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-green-600">
            {stats.model_accuracy}
          </div>
          <Progress value={accuracyValue} className="mt-2" />
          <p className="text-xs text-muted-foreground mt-2">
            🤖 Live ML Model Performance
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Predictions Made</CardTitle>
          <div className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {stats.predictions_made.toLocaleString()}
          </div>
          <p className="text-xs text-muted-foreground">
            📈 Total ML Predictions
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
          <div className="h-2 w-2 rounded-full bg-purple-500 animate-pulse" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-blue-600">
            {stats.success_rate}
          </div>
          <Progress value={successRate} className="mt-2" />
          <p className="text-xs text-muted-foreground mt-2">
            🎯 Top 3 Prediction Accuracy
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Model Version</CardTitle>
          <div className="h-2 w-2 rounded-full bg-yellow-500 animate-pulse" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {stats.model_version}
          </div>
          <p className="text-xs text-muted-foreground">
            🔄 Updated {new Date(stats.last_updated).toLocaleDateString()}
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
