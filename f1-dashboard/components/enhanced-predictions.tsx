"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Trophy, Target, Award, AlertTriangle, TrendingUp, BarChart3 } from "lucide-react"

interface EnhancedPrediction {
  driver: string
  podium_probability: number
  points_probability: number
  winner_probability: number
  dnf_probability: number
  confidence_score: number
}

interface ModelComparison {
  model_comparison: {
    [key: string]: {
      strengths: string[]
      f1_racing_applications: string[]
      optimization_features: string[]
    }
  }
  ensemble_advantages: {
    why_ensemble_works: string[]
    f1_racing_benefits: string[]
  }
}

export default function EnhancedPredictions() {
  const [predictions, setPredictions] = useState<EnhancedPrediction[]>([])
  const [modelInfo, setModelInfo] = useState<ModelComparison | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchEnhancedPredictions = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('http://localhost:8000/api/predict-race-enhanced', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          drivers: [
            'Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc', 'George Russell',
            'Sergio Perez', 'Carlos Sainz'
          ],
          circuit: 'Monaco',
          weather: {
            AirTemp: 25,
            TrackTemp: 35,
            Humidity: 60,
            WindSpeed: 5,
            Rainfall: 0
          }
        })
      })

      if (!response.ok) {
        const errorData = await response.text()
        console.error('API Error Response:', errorData)
        throw new Error(`HTTP error! status: ${response.status} - ${errorData}`)
      }

      const data = await response.json()
      setPredictions(data.enhanced_predictions || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch predictions')
      console.error('Frontend Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const fetchModelComparison = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/model-comparison')
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setModelInfo(data)
    } catch (err) {
      console.error('Failed to fetch model comparison:', err)
    }
  }

  useEffect(() => {
    fetchModelComparison()
  }, [])

  const getProbabilityColor = (probability: number) => {
    if (probability >= 0.7) return "text-green-600 bg-green-50"
    if (probability >= 0.4) return "text-yellow-600 bg-yellow-50"
    return "text-red-600 bg-red-50"
  }

  const getProbabilityBadgeVariant = (probability: number) => {
    if (probability >= 0.7) return "default"
    if (probability >= 0.4) return "secondary"
    return "outline"
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Enhanced F1 Predictions</h2>
          <p className="text-muted-foreground">
            Advanced ML models optimized for F1-Score metrics
          </p>
        </div>
        <Button onClick={fetchEnhancedPredictions} disabled={loading}>
          {loading ? "Generating..." : "Generate Enhanced Predictions"}
        </Button>
      </div>

      {error && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="predictions" className="space-y-4">
        <TabsList>
          <TabsTrigger value="predictions">Race Predictions</TabsTrigger>
          <TabsTrigger value="models">Model Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="predictions" className="space-y-4">
          {predictions.length > 0 && (
            <div className="grid gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    F1-Score Optimized Predictions
                  </CardTitle>
                  <CardDescription>
                    Ensemble model predictions using XGBoost, Random Forest, Gradient Boosting, Logistic Regression, and SVM
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {predictions.map((prediction, index) => (
                      <div key={index} className="border rounded-lg p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <h3 className="font-semibold text-lg">{prediction.driver}</h3>
                          <Badge variant={getProbabilityBadgeVariant(prediction.confidence_score)}>
                            Confidence: {(prediction.confidence_score * 100).toFixed(1)}%
                          </Badge>
                        </div>
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <Trophy className="h-4 w-4 text-yellow-500" />
                              <span className="text-sm font-medium">Podium</span>
                            </div>
                            <div className="space-y-1">
                              <Progress value={prediction.podium_probability * 100} className="h-2" />
                              <Badge variant="outline" className={getProbabilityColor(prediction.podium_probability)}>
                                {(prediction.podium_probability * 100).toFixed(1)}%
                              </Badge>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <Target className="h-4 w-4 text-blue-500" />
                              <span className="text-sm font-medium">Points</span>
                            </div>
                            <div className="space-y-1">
                              <Progress value={prediction.points_probability * 100} className="h-2" />
                              <Badge variant="outline" className={getProbabilityColor(prediction.points_probability)}>
                                {(prediction.points_probability * 100).toFixed(1)}%
                              </Badge>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <Award className="h-4 w-4 text-gold-500" />
                              <span className="text-sm font-medium">Winner</span>
                            </div>
                            <div className="space-y-1">
                              <Progress value={prediction.winner_probability * 100} className="h-2" />
                              <Badge variant="outline" className={getProbabilityColor(prediction.winner_probability)}>
                                {(prediction.winner_probability * 100).toFixed(1)}%
                              </Badge>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <AlertTriangle className="h-4 w-4 text-red-500" />
                              <span className="text-sm font-medium">DNF Risk</span>
                            </div>
                            <div className="space-y-1">
                              <Progress value={prediction.dnf_probability * 100} className="h-2" />
                              <Badge variant="outline" className={getProbabilityColor(1 - prediction.dnf_probability)}>
                                {(prediction.dnf_probability * 100).toFixed(1)}%
                              </Badge>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="models" className="space-y-4">
          {modelInfo && (
            <div className="grid gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    ML Model Analysis
                  </CardTitle>
                  <CardDescription>
                    Detailed breakdown of F1-Score optimized models for Formula 1 predictions
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    {Object.entries(modelInfo.model_comparison).map(([modelName, modelData]) => (
                      <div key={modelName} className="border rounded-lg p-4">
                        <h3 className="font-semibold text-lg mb-3 capitalize">
                          {modelName.replace('_', ' ')}
                        </h3>
                        
                        <div className="grid md:grid-cols-3 gap-4">
                          <div>
                            <h4 className="font-medium text-sm text-muted-foreground mb-2">Strengths</h4>
                            <ul className="space-y-1 text-sm">
                              {modelData.strengths.map((strength, idx) => (
                                <li key={idx} className="flex items-start gap-2">
                                  <span className="text-green-500 mt-1">•</span>
                                  {strength}
                                </li>
                              ))}
                            </ul>
                          </div>
                          
                          <div>
                            <h4 className="font-medium text-sm text-muted-foreground mb-2">F1 Applications</h4>
                            <ul className="space-y-1 text-sm">
                              {modelData.f1_racing_applications.map((app, idx) => (
                                <li key={idx} className="flex items-start gap-2">
                                  <span className="text-blue-500 mt-1">•</span>
                                  {app}
                                </li>
                              ))}
                            </ul>
                          </div>
                          
                          <div>
                            <h4 className="font-medium text-sm text-muted-foreground mb-2">Optimization Features</h4>
                            <ul className="space-y-1 text-sm">
                              {modelData.optimization_features.map((feature, idx) => (
                                <li key={idx} className="flex items-start gap-2">
                                  <span className="text-purple-500 mt-1">•</span>
                                  {feature}
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    ))}
                    
                    <div className="border rounded-lg p-4 bg-gradient-to-r from-blue-50 to-purple-50">
                      <h3 className="font-semibold text-lg mb-3">Ensemble Advantages</h3>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div>
                          <h4 className="font-medium text-sm text-muted-foreground mb-2">Why Ensemble Works</h4>
                          <ul className="space-y-1 text-sm">
                            {modelInfo.ensemble_advantages.why_ensemble_works.map((reason, idx) => (
                              <li key={idx} className="flex items-start gap-2">
                                <span className="text-green-500 mt-1">✓</span>
                                {reason}
                              </li>
                            ))}
                          </ul>
                        </div>
                        
                        <div>
                          <h4 className="font-medium text-sm text-muted-foreground mb-2">F1 Racing Benefits</h4>
                          <ul className="space-y-1 text-sm">
                            {modelInfo.ensemble_advantages.f1_racing_benefits.map((benefit, idx) => (
                              <li key={idx} className="flex items-start gap-2">
                                <span className="text-blue-500 mt-1">✓</span>
                                {benefit}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
