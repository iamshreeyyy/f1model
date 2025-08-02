"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { CheckCircle, XCircle, Clock, Brain, TrendingUp, Settings } from "lucide-react"
import { useEffect, useState } from "react"
import { apiClient, PredictionsResponse } from "@/lib/api"

interface Activity {
  id: number
  type: string
  description: string
  status: string
  time: string
}

export function RecentActivity() {
  const [activities, setActivities] = useState<Activity[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchActivity = async () => {
      try {
        setLoading(true)
        setError(null)
        console.log('🔄 Fetching real recent activity from ML model...')
        const data = await apiClient.getPredictions()
        console.log('📈 Received activity data:', data.recent_activity)
        
        // Convert API data to component format
        const convertedActivities = (data.recent_activity || []).map((activity, index) => ({
          id: index + 1,
          type: activity.type,
          description: activity.message,
          status: activity.type === "prediction" ? "completed" : 
                  activity.type === "training" ? "completed" :
                  activity.type === "analysis" ? "completed" : "pending",
          time: formatTimeAgo(activity.timestamp)
        }))
        
        setActivities(convertedActivities)
      } catch (error) {
        console.error('❌ Failed to fetch recent activity:', error)
        setError('Failed to load activity data')
      } finally {
        setLoading(false)
      }
    }

    fetchActivity()
    
    // Refresh activity every 2 minutes
    const interval = setInterval(fetchActivity, 120000)
    return () => clearInterval(interval)
  }, [])

  const formatTimeAgo = (timestamp: string) => {
    const now = new Date()
    const time = new Date(timestamp)
    const diffInHours = Math.floor((now.getTime() - time.getTime()) / (1000 * 60 * 60))
    
    if (diffInHours < 1) return "Just now"
    if (diffInHours === 1) return "1 hour ago"
    if (diffInHours < 24) return `${diffInHours} hours ago`
    
    const diffInDays = Math.floor(diffInHours / 24)
    if (diffInDays === 1) return "1 day ago"
    return `${diffInDays} days ago`
  }

  const getActivityIcon = (type: string) => {
    switch (type) {
      case "prediction":
        return <Brain className="size-5 text-blue-600" />
      case "training":
        return <TrendingUp className="size-5 text-green-600" />
      case "analysis":
        return <Settings className="size-5 text-purple-600" />
      default:
        return <Clock className="size-5 text-gray-600" />
    }
  }

  if (loading && activities.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            📈 Recent Activity
            <div className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="flex items-center gap-3 p-3 rounded-lg border">
                <div className="w-5 h-5 bg-muted animate-pulse rounded-full" />
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-muted animate-pulse rounded w-3/4" />
                  <div className="h-3 bg-muted animate-pulse rounded w-1/2" />
                </div>
                <div className="w-16 h-6 bg-muted animate-pulse rounded" />
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
          <CardTitle>📈 Recent Activity</CardTitle>
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

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          📈 Recent Activity
          <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
          <span className="text-sm text-muted-foreground font-normal">
            Live ML System
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {activities.length > 0 ? (
            activities.map((activity) => (
              <div key={activity.id} className="flex items-center gap-3 p-3 rounded-lg border bg-muted/30">
                <div className="flex-shrink-0">
                  {getActivityIcon(activity.type)}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium">{activity.description}</p>
                  <p className="text-xs text-muted-foreground flex items-center gap-1">
                    <span className="capitalize">{activity.type}</span> • {activity.time}
                  </p>
                </div>
                <Badge
                  variant={
                    activity.status === "completed"
                      ? "default"
                      : activity.status === "error"
                        ? "destructive"
                        : "secondary"
                  }
                  className="text-xs"
                >
                  {activity.status}
                </Badge>
              </div>
            ))
          ) : (
            <div className="text-center py-8">
              <Clock className="size-8 text-muted-foreground mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">
                No recent activity available
              </p>
            </div>
          )}
        </div>
        
        {activities.length > 0 && (
          <div className="mt-4 p-3 bg-green-50 dark:bg-green-950/30 rounded-lg">
            <p className="text-xs text-green-600 dark:text-green-400">
              🤖 ML system is actively generating predictions and analyzing F1 data
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
