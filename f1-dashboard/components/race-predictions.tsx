"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { useEffect, useState } from "react"
import { apiClient, PredictionsResponse } from "@/lib/api"

export function RacePredictions() {
  const [predictionsData, setPredictionsData] = useState<PredictionsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        setLoading(true)
        setError(null)
        console.log('🔄 Fetching real race predictions from ML model...')
        const data = await apiClient.getPredictions()
        console.log('🏎️ Received predictions:', data)
        setPredictionsData(data)
      } catch (error) {
        console.error('❌ Failed to fetch predictions:', error)
        setError('Failed to load ML predictions')
      } finally {
        setLoading(false)
      }
    }

    fetchPredictions()
    
    // Refresh predictions every 30 seconds for real-time updates
    const interval = setInterval(fetchPredictions, 30000)
    return () => clearInterval(interval)
  }, [])

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 85) return "bg-green-500"
    if (confidence >= 70) return "bg-yellow-500"
    return "bg-red-500"
  }

  const getPositionBadgeColor = (position: number) => {
    if (position === 1) return "bg-yellow-500 text-black" // Gold
    if (position === 2) return "bg-gray-400 text-black"   // Silver
    if (position === 3) return "bg-amber-600 text-white"  // Bronze
    return "bg-blue-500 text-white"
  }

  if (loading && !predictionsData) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>🏁 Race Predictions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="flex items-center space-x-4">
                <div className="h-12 w-12 bg-muted animate-pulse rounded-full" />
                <div className="flex-1">
                  <div className="h-4 bg-muted animate-pulse rounded w-32 mb-2" />
                  <div className="h-3 bg-muted animate-pulse rounded w-24" />
                </div>
                <div className="h-6 bg-muted animate-pulse rounded w-16" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>🏁 Race Predictions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <p className="text-red-500 font-medium">⚠️ {error}</p>
            <p className="text-sm text-muted-foreground mt-2">
              Make sure the backend API is running on http://localhost:8000
            </p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!predictionsData) return null

  const predictions = predictionsData.next_race.predictions

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          🏁 {predictionsData.next_race.race_name}
          <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
          <span className="text-sm text-muted-foreground font-normal">
            Live ML Model
          </span>
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          📍 {predictionsData.next_race.circuit} • 📅 {new Date(predictionsData.next_race.date).toLocaleDateString()}
        </p>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {predictions.slice(0, 8).map((prediction) => (
            <div key={prediction.driver} className="flex items-center space-x-4 p-3 rounded-lg bg-muted/50">
              <Badge className={getPositionBadgeColor(prediction.position)}>
                P{prediction.position}
              </Badge>
              
              <Avatar className="h-10 w-10">
                <AvatarImage 
                  src={`/placeholder-user.jpg`} 
                  alt={prediction.driver}
                />
                <AvatarFallback>
                  {prediction.driver.split(' ').map(n => n[0]).join('')}
                </AvatarFallback>
              </Avatar>
              
              <div className="flex-1">
                <p className="font-medium">{prediction.driver}</p>
                <p className="text-sm text-muted-foreground">
                  Position {prediction.position}
                  {prediction.change !== 0 && (
                    <span className={`ml-2 font-medium ${
                      prediction.change > 0 
                        ? 'text-green-600' 
                        : 'text-red-600'
                    }`}>
                      {prediction.change > 0 ? '↗️' : '↘️'}
                      {Math.abs(prediction.change)} from last prediction
                    </span>
                  )}
                </p>
              </div>
              
              <div className="text-right">
                <div className="flex items-center space-x-2">
                  <div className={`h-2 w-2 rounded-full ${getConfidenceColor(prediction.confidence)}`} />
                  <span className="text-sm font-medium">
                    {prediction.confidence}%
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">
                  confidence
                </p>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-950/30 rounded-lg">
          <p className="text-xs text-blue-600 dark:text-blue-400">
            🤖 Predictions generated by ML ensemble model with {
              predictions.length > 0 
                ? (predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length).toFixed(1)
                : '0'
            }% average confidence
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Last updated: {new Date(predictionsData.last_updated).toLocaleTimeString()}
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
