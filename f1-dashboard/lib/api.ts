// lib/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ModelStats {
  model_accuracy: string;
  predictions_made: number;
  success_rate: string;
  model_version: string;
  last_updated: string;
}

export interface RacePrediction {
  driver: string;
  position: number;
  confidence: number;
  change: number;
}

export interface PredictionsResponse {
  last_updated: string;
  next_race: {
    race_name: string;
    date: string;
    circuit: string;
    predictions: RacePrediction[];
  };
  recent_activity: Array<{
    type: string;
    message: string;
    timestamp: string;
  }>;
}

export interface RaceResult {
  race_name: string;
  date: string;
  circuit: string;
  winner: string;
  podium: string[];
  fastest_lap: string;
  results: Array<{
    position: number;
    driver: string;
    team: string;
    time: string;
    points: number;
  }>;
}

export interface DriverStats {
  driver: string;
  team: string;
  points: number;
  wins: number;
  podiums: number;
  pole_positions: number;
  fastest_laps: number;
  dnfs: number;
  position: number;
}

export interface Championships {
  drivers_championship: {
    year: number;
    standings: Array<{
      position: number;
      driver: string;
      team: string;
      points: number;
    }>;
  };
  constructors_championship: {
    year: number;
    standings: Array<{
      position: number;
      team: string;
      points: number;
    }>;
  };
}

export interface AnalyticsData {
  model_performance: {
    accuracy_trend: Array<{
      race: string;
      accuracy: number;
    }>;
    prediction_confidence: Array<{
      driver: string;
      avg_confidence: number;
    }>;
  };
  driver_performance: {
    consistency_scores: Array<{
      driver: string;
      score: number;
    }>;
  };
}

class ApiClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  private async request<T>(endpoint: string): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    console.log(`🔄 Making API request to: ${url}`);
    
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ API request failed: ${response.status} - ${errorText}`);
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    console.log(`✅ API response received:`, data);
    return data;
  }

  async getModelStats(): Promise<ModelStats> {
    return this.request<ModelStats>('/api/model-stats');
  }

  async getPredictions(): Promise<PredictionsResponse> {
    return this.request<PredictionsResponse>('/api/predictions');
  }

  async getRaceResults(): Promise<RaceResult[]> {
    return this.request<RaceResult[]>('/api/race-results');
  }

  async getDriverStats(): Promise<DriverStats[]> {
    return this.request<DriverStats[]>('/api/driver-stats');
  }

  async getChampionships(): Promise<Championships> {
    return this.request<Championships>('/api/championships');
  }

  async getAnalytics(): Promise<AnalyticsData> {
    return this.request<AnalyticsData>('/api/analytics');
  }
}

export const apiClient = new ApiClient();

// React Query helpers (if you want to add react-query later)
export const queryKeys = {
  modelStats: ['modelStats'],
  predictions: ['predictions'],
  raceResults: ['raceResults'],
  driverStats: ['driverStats'],
  championships: ['championships'],
  analytics: ['analytics'],
} as const;
