"use client"

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Activity, CheckCircle, AlertCircle, Database, TrendingUp, Cpu } from 'lucide-react'
import Link from 'next/link'

interface PipelineStatus {
  pipeline_status: {
    data_acquisition: { status: string }
    feature_engineering: { status: string }
    model_training: { status: string }
    continuous_update: { status: string }
  }
  professional_features_available: boolean
}

export function ProfessionalOverview() {
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchPipelineStatus()
  }, [])

  const fetchPipelineStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/professional/status')
      if (response.ok) {
        const data = await response.json()
        setPipelineStatus(data)
      }
    } catch (error) {
      console.error('Failed to fetch pipeline status:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'ready':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'running':
        return <Activity className="w-4 h-4 text-blue-500" />
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-600" />
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />
      default:
        return <Activity className="w-4 h-4 text-gray-500" />
    }
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            Professional ML Pipeline
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-3">
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Cpu className="w-5 h-5" />
          Professional ML Pipeline
          {pipelineStatus?.professional_features_available ? (
            <Badge variant="outline" className="text-green-600">Active</Badge>
          ) : (
            <Badge variant="secondary">Basic Mode</Badge>
          )}
        </CardTitle>
        <CardDescription>
          Enterprise-grade machine learning and strategic analysis system
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {pipelineStatus ? (
          <>
            <div className="grid grid-cols-2 gap-3">
              <div className="flex items-center gap-2 text-sm">
                {getStatusIcon(pipelineStatus.pipeline_status.data_acquisition.status)}
                <span>Data Acquisition</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                {getStatusIcon(pipelineStatus.pipeline_status.feature_engineering.status)}
                <span>Feature Engineering</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                {getStatusIcon(pipelineStatus.pipeline_status.model_training.status)}
                <span>Model Training</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                {getStatusIcon(pipelineStatus.pipeline_status.continuous_update.status)}
                <span>Continuous Updates</span>
              </div>
            </div>
            
            <div className="pt-2 border-t">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Professional Features</p>
                  <p className="text-xs text-muted-foreground">
                    Strategy analysis, pit optimization, and automated updates
                  </p>
                </div>
                <Link href="/professional">
                  <Button variant="outline" size="sm">
                    <Database className="w-4 h-4 mr-2" />
                    Open Pipeline
                  </Button>
                </Link>
              </div>
            </div>
          </>
        ) : (
          <div className="text-center py-4">
            <p className="text-sm text-muted-foreground">Pipeline status unavailable</p>
            <Button onClick={fetchPipelineStatus} variant="outline" size="sm" className="mt-2">
              Retry
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
