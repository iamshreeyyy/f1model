import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { TrendingUp, TrendingDown } from "lucide-react"

interface PredictionCardProps {
  driver: string
  position: number
  confidence: number
  change: number
}

export function PredictionCard({ driver, position, confidence, change }: PredictionCardProps) {
  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">{driver}</CardTitle>
          <Badge variant={position <= 3 ? "default" : "secondary"}>P{position}</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between">
          <div className="text-2xl font-bold">{confidence}%</div>
          <div className={`flex items-center gap-1 text-sm ${change >= 0 ? "text-green-600" : "text-red-600"}`}>
            {change >= 0 ? <TrendingUp className="size-4" /> : <TrendingDown className="size-4" />}
            {Math.abs(change)}%
          </div>
        </div>
        <p className="text-xs text-muted-foreground mt-1">Confidence Level</p>
      </CardContent>
    </Card>
  )
}
