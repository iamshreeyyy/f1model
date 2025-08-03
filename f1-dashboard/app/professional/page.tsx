"use client"

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { AlertCircle, CheckCircle, Activity, TrendingUp, Settings, Database } from 'lucide-react'

interface PipelineStatus {
  pipeline_status: {
    data_acquisition: { status: string; last_update: string | null }
    feature_engineering: { status: string; last_update: string | null }
    model_training: { status: string; last_update: string | null }
    continuous_update: { status: string; last_update: string | null }
  }
  timestamp: string
  professional_features_available: boolean
}

interface StrategyAnalysis {
  driver: string
  circuit: string
  historical_performance: {
    avg_position: number
    podium_rate: number
    performance_trend: string
    recommendations: string[]
  }
  qualifying_strategy: {
    qualifying_importance_score: number
    recommendations: string[]
    session_approach: string
  }
  pit_stop_strategy: {
    adjusted_pit_window: number[]
    strategy_note: string
    safety_car_strategy: string
  }
  key_insights: string[]
  strategic_priorities: string[]
  risk_factors: string[]
  opportunities: string[]
}

interface PitOptimization {
  circuit: string
  adjusted_pit_window: number[]
  strategy_note: string
  safety_car_strategy: string
  driver_strategies: Array<{
    driver: string
    strategy: string
    pit_window: number[]
    notes: string
  }>
  weather_impact: {
    temperature: number
    rain_risk: number
    degradation_multiplier: number
  }
}

export default function ProfessionalPipeline() {
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null)
  const [strategyAnalysis, setStrategyAnalysis] = useState<StrategyAnalysis | null>(null)
  const [pitOptimization, setPitOptimization] = useState<PitOptimization | null>(null)
  const [loading, setLoading] = useState(false)
  const [selectedDriver, setSelectedDriver] = useState('Max Verstappen')
  const [selectedCircuit, setSelectedCircuit] = useState('Monaco')
  const [dataAcquisitionYears, setDataAcquisitionYears] = useState('2023,2024')
  const [dataAcquisitionStatus, setDataAcquisitionStatus] = useState<string | null>(null)
  const [dataAcquisitionResult, setDataAcquisitionResult] = useState<any | null>(null)

  const drivers = [
    'Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc', 'Lando Norris',
    'Carlos Sainz', 'George Russell', 'Fernando Alonso', 'Oscar Piastri'
  ]

  const circuits = [
    'Monaco', 'Silverstone', 'Spa', 'Monza', 'Suzuka', 'Imola', 
    'Barcelona', 'Zandvoort', 'Austin', 'Interlagos'
  ]

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
    }
  }

  const analyzeStrategy = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/professional/strategy/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          driver: selectedDriver,
          circuit: selectedCircuit,
          weather_forecast: {
            rain_probability: 0.3,
            temperature: 28,
            humidity: 65
          },
          race_predictions: [
            { driver: selectedDriver, predicted_position: 1, grid_position: 1 },
            { driver: 'Charles Leclerc', predicted_position: 2, grid_position: 2 }
          ]
        })
      })
      
      if (response.ok) {
        const data = await response.json()
        setStrategyAnalysis(data)
      }
    } catch (error) {
      console.error('Strategy analysis failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const optimizePitStrategy = async () => {
    setLoading(true)
    try {
      const response = await fetch(
        `http://localhost:8000/api/professional/strategy/pit-optimization/${selectedCircuit.toLowerCase()}?track_temp=30&rain_prob=0.2`
      )
      
      if (response.ok) {
        const data = await response.json()
        setPitOptimization(data)
      }
    } catch (error) {
      console.error('Pit optimization failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const triggerDataAcquisition = async () => {
    setLoading(true)
    setDataAcquisitionStatus('Starting data acquisition...')
    setDataAcquisitionResult(null)
    
    try {
      const years = dataAcquisitionYears.split(',').map(y => parseInt(y.trim()))
      const response = await fetch('http://localhost:8000/api/professional/data/acquire', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          years: years,
          force_refresh: false
        })
      })
      
      if (response.ok) {
        const data = await response.json()
        setDataAcquisitionStatus('Data acquisition in progress...')
        
        // Poll for completion status
        let attempts = 0
        const maxAttempts = 10
        const pollStatus = async () => {
          try {
            const statusResponse = await fetch('http://localhost:8000/api/professional/status')
            const statusData = await statusResponse.json()
            const dataAcqStatus = statusData.pipeline_status.data_acquisition.status
            
            if (dataAcqStatus === 'completed') {
              setDataAcquisitionStatus('✅ Data acquisition completed successfully!')
              setDataAcquisitionResult({
                message: 'Data successfully acquired and processed',
                years: years,
                lastUpdate: statusData.pipeline_status.data_acquisition.last_update,
                status: 'success'
              })
              fetchPipelineStatus() // Refresh overall status
            } else if (dataAcqStatus === 'error') {
              setDataAcquisitionStatus('❌ Data acquisition failed')
              setDataAcquisitionResult({
                message: 'Failed to acquire data. Using existing data or demo data.',
                error: statusData.pipeline_status.data_acquisition.error,
                status: 'error'
              })
            } else if (attempts < maxAttempts) {
              attempts++
              setTimeout(pollStatus, 1000) // Poll every second
            } else {
              setDataAcquisitionStatus('⏱️ Data acquisition taking longer than expected')
            }
          } catch (error) {
            setDataAcquisitionStatus('❌ Failed to check acquisition status')
          }
        }
        
        // Start polling after a short delay
        setTimeout(pollStatus, 1000)
      } else {
        setDataAcquisitionStatus('❌ Failed to start data acquisition')
        setDataAcquisitionResult({
          message: 'Server responded with error',
          status: 'error'
        })
      }
    } catch (error) {
      console.error('Data acquisition failed:', error)
      setDataAcquisitionStatus('❌ Connection error')
      setDataAcquisitionResult({
        message: 'Failed to connect to backend server',
        error: error instanceof Error ? error.message : 'Unknown error',
        status: 'error'
      })
    } finally {
      setLoading(false)
    }
  }

  const triggerContinuousUpdate = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/professional/continuous/update', {
        method: 'POST'
      })
      
      if (response.ok) {
        const data = await response.json()
        alert('Continuous update process started!')
        setTimeout(fetchPipelineStatus, 2000)
      }
    } catch (error) {
      console.error('Continuous update failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'ready':
        return <Badge variant="secondary"><CheckCircle className="w-3 h-3 mr-1" />Ready</Badge>
      case 'running':
        return <Badge variant="default"><Activity className="w-3 h-3 mr-1" />Running</Badge>
      case 'completed':
        return <Badge variant="outline" className="text-green-600"><CheckCircle className="w-3 h-3 mr-1" />Completed</Badge>
      case 'error':
        return <Badge variant="destructive"><AlertCircle className="w-3 h-3 mr-1" />Error</Badge>
      default:
        return <Badge variant="secondary">{status}</Badge>
    }
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">F1 Professional ML Pipeline</h1>
          <p className="text-muted-foreground">Enterprise-grade machine learning and strategic analysis</p>
        </div>
        <Button onClick={fetchPipelineStatus} variant="outline">
          <Activity className="w-4 h-4 mr-2" />
          Refresh Status
        </Button>
      </div>

      {pipelineStatus && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5" />
              Pipeline Status
              {pipelineStatus.professional_features_available ? (
                <Badge variant="outline" className="text-green-600">Professional Active</Badge>
              ) : (
                <Badge variant="destructive">Basic Mode</Badge>
              )}
            </CardTitle>
            <CardDescription>
              Last updated: {new Date(pipelineStatus.timestamp).toLocaleString()}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="space-y-2">
                <p className="text-sm font-medium">Data Acquisition</p>
                {getStatusBadge(pipelineStatus.pipeline_status.data_acquisition.status)}
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium">Feature Engineering</p>
                {getStatusBadge(pipelineStatus.pipeline_status.feature_engineering.status)}
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium">Model Training</p>
                {getStatusBadge(pipelineStatus.pipeline_status.model_training.status)}
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium">Continuous Update</p>
                {getStatusBadge(pipelineStatus.pipeline_status.continuous_update.status)}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="strategy" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="strategy">Strategy Analysis</TabsTrigger>
          <TabsTrigger value="pit">Pit Optimization</TabsTrigger>
          <TabsTrigger value="data">Data Management</TabsTrigger>
          <TabsTrigger value="updates">Continuous Updates</TabsTrigger>
        </TabsList>

        <TabsContent value="strategy" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Race Strategy Analysis</CardTitle>
              <CardDescription>
                Generate comprehensive strategic analysis for specific driver-circuit combinations
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Driver</label>
                  <Select value={selectedDriver} onValueChange={setSelectedDriver}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {drivers.map(driver => (
                        <SelectItem key={driver} value={driver}>{driver}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Circuit</label>
                  <Select value={selectedCircuit} onValueChange={setSelectedCircuit}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {circuits.map(circuit => (
                        <SelectItem key={circuit} value={circuit}>{circuit}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <Button onClick={analyzeStrategy} disabled={loading} className="w-full">
                {loading ? 'Analyzing...' : 'Analyze Strategy'}
              </Button>
              
              {strategyAnalysis && (
                <div className="space-y-4 mt-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg">Historical Performance</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm">Average Position:</span>
                          <span className="font-medium">P{strategyAnalysis.historical_performance.avg_position}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Podium Rate:</span>
                          <span className="font-medium">{(strategyAnalysis.historical_performance.podium_rate * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Trend:</span>
                          <Badge variant={strategyAnalysis.historical_performance.performance_trend === 'improving' ? 'default' : 'secondary'}>
                            {strategyAnalysis.historical_performance.performance_trend}
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg">Qualifying Strategy</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm">Importance Score:</span>
                          <span className="font-medium">{strategyAnalysis.qualifying_strategy.qualifying_importance_score}/10</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Approach:</span>
                          <Badge variant="outline">{strategyAnalysis.qualifying_strategy.session_approach}</Badge>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg">Pit Strategy</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm">Window:</span>
                          <span className="font-medium">Laps {strategyAnalysis.pit_stop_strategy.adjusted_pit_window[0]}-{strategyAnalysis.pit_stop_strategy.adjusted_pit_window[1]}</span>
                        </div>
                        <p className="text-xs text-muted-foreground">{strategyAnalysis.pit_stop_strategy.strategy_note}</p>
                      </CardContent>
                    </Card>
                  </div>

                  {strategyAnalysis.strategic_priorities.length > 0 && (
                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg text-red-600">Strategic Priorities</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="space-y-1">
                          {strategyAnalysis.strategic_priorities.map((priority, index) => (
                            <li key={index} className="text-sm flex items-start gap-2">
                              <AlertCircle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                              {priority}
                            </li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>
                  )}

                  {strategyAnalysis.opportunities.length > 0 && (
                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg text-green-600">Opportunities</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="space-y-1">
                          {strategyAnalysis.opportunities.map((opportunity, index) => (
                            <li key={index} className="text-sm flex items-start gap-2">
                              <TrendingUp className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                              {opportunity}
                            </li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="pit" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Pit Stop Optimization</CardTitle>
              <CardDescription>
                Circuit-specific pit window analysis with weather and strategy considerations
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button onClick={optimizePitStrategy} disabled={loading} className="w-full">
                {loading ? 'Optimizing...' : `Optimize Pit Strategy for ${selectedCircuit}`}
              </Button>
              
              {pitOptimization && (
                <div className="space-y-4 mt-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg">Pit Window</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        <div className="text-center">
                          <span className="text-2xl font-bold">Laps {pitOptimization.adjusted_pit_window[0]}-{pitOptimization.adjusted_pit_window[1]}</span>
                        </div>
                        <p className="text-sm text-muted-foreground text-center">{pitOptimization.strategy_note}</p>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg">Weather Impact</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm">Track Temperature:</span>
                          <span className="font-medium">{pitOptimization.weather_impact.temperature}°C</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Rain Risk:</span>
                          <span className="font-medium">{(pitOptimization.weather_impact.rain_risk * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Degradation Factor:</span>
                          <span className="font-medium">{pitOptimization.weather_impact.degradation_multiplier.toFixed(2)}x</span>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg">Driver Strategies</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {pitOptimization.driver_strategies.map((strategy, index) => (
                          <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                            <div>
                              <p className="font-medium">{strategy.driver}</p>
                              <p className="text-sm text-muted-foreground">{strategy.notes}</p>
                            </div>
                            <div className="text-right">
                              <Badge variant="outline">{strategy.strategy}</Badge>
                              <p className="text-sm text-muted-foreground mt-1">
                                Laps {strategy.pit_window[0]}-{strategy.pit_window[1]}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="data" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Data Acquisition</CardTitle>
              <CardDescription>
                Acquire F1 data from Ergast Motor Racing API for training and analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Years (comma-separated)</label>
                <Input
                  value={dataAcquisitionYears}
                  onChange={(e) => setDataAcquisitionYears(e.target.value)}
                  placeholder="2023,2024"
                />
                <p className="text-xs text-muted-foreground">
                  Enter years to acquire data for (e.g., 2020,2021,2022,2023,2024)
                </p>
              </div>
              
              <Button onClick={triggerDataAcquisition} disabled={loading} className="w-full">
                {loading ? 'Acquiring Data...' : 'Start Data Acquisition'}
              </Button>
              
              {/* Status Display */}
              {dataAcquisitionStatus && (
                <div className="mt-4 p-3 border rounded-md">
                  <p className="text-sm font-medium mb-2">Status:</p>
                  <p className="text-sm">{dataAcquisitionStatus}</p>
                </div>
              )}
              
              {/* Result Display */}
              {dataAcquisitionResult && (
                <div className={`mt-4 p-3 border rounded-md ${
                  dataAcquisitionResult.status === 'success' 
                    ? 'border-green-200 bg-green-50' 
                    : 'border-red-200 bg-red-50'
                }`}>
                  <p className="text-sm font-medium mb-2">Result:</p>
                  <p className="text-sm mb-2">{dataAcquisitionResult.message}</p>
                  
                  {dataAcquisitionResult.years && (
                    <p className="text-xs text-muted-foreground">
                      Years processed: {dataAcquisitionResult.years.join(', ')}
                    </p>
                  )}
                  
                  {dataAcquisitionResult.lastUpdate && (
                    <p className="text-xs text-muted-foreground">
                      Last updated: {new Date(dataAcquisitionResult.lastUpdate).toLocaleString()}
                    </p>
                  )}
                  
                  {dataAcquisitionResult.error && (
                    <p className="text-xs text-red-600 mt-1">
                      Error: {dataAcquisitionResult.error}
                    </p>
                  )}
                </div>
              )}
              
              {/* Current Pipeline Status */}
              {pipelineStatus && (
                <div className="mt-4 p-3 border rounded-md bg-gray-50">
                  <p className="text-sm font-medium mb-2">Current Pipeline Status:</p>
                  <div className="text-xs space-y-1">
                    <div className="flex justify-between">
                      <span>Data Acquisition:</span>
                      <span className={`font-medium ${
                        pipelineStatus.pipeline_status.data_acquisition.status === 'completed' 
                          ? 'text-green-600' 
                          : pipelineStatus.pipeline_status.data_acquisition.status === 'error'
                          ? 'text-red-600'
                          : 'text-yellow-600'
                      }`}>
                        {pipelineStatus.pipeline_status.data_acquisition.status}
                      </span>
                    </div>
                    {pipelineStatus.pipeline_status.data_acquisition.last_update && (
                      <div className="text-xs text-muted-foreground">
                        Last update: {new Date(pipelineStatus.pipeline_status.data_acquisition.last_update).toLocaleString()}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="updates" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Continuous Updates</CardTitle>
              <CardDescription>
                Manage automated model updates and performance monitoring
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button onClick={triggerContinuousUpdate} disabled={loading} className="w-full">
                {loading ? 'Starting Update...' : 'Trigger Continuous Update'}
              </Button>
              <p className="text-sm text-muted-foreground">
                This will check for new data, detect model drift, and retrain models if necessary.
                The process runs in the background and may take several minutes to complete.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
