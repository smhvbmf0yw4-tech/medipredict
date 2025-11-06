import { Target, TrendingUp, Activity, Zap } from "lucide-react"

interface MetricsDisplayProps {
  metrics: {
    accuracy: number
    precision: number
    recall: number
    f1Score: number
  }
}

export function MetricsDisplay({ metrics }: MetricsDisplayProps) {
  const metricsList = [
    {
      key: "accuracy",
      label: "Exactitud",
      description: "Porcentaje de predicciones correctas",
      value: metrics.accuracy,
      icon: Target,
      iconColor: "text-blue-600",
      barColor: "bg-blue-600",
    },
    {
      key: "precision",
      label: "Precisión",
      description: "Proporción de positivos correctos",
      value: metrics.precision,
      icon: TrendingUp,
      iconColor: "text-teal-600",
      barColor: "bg-teal-600",
    },
    {
      key: "recall",
      label: "Sensibilidad",
      description: "Capacidad de detectar casos positivos",
      value: metrics.recall,
      icon: Activity,
      iconColor: "text-teal-600",
      barColor: "bg-teal-600",
    },
    {
      key: "f1Score",
      label: "F1-Score",
      description: "Media armónica de precisión y sensibilidad",
      value: metrics.f1Score,
      icon: Zap,
      iconColor: "text-orange-600",
      barColor: "bg-orange-600",
    },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-foreground mb-2">Métricas de Desempeño</h2>
        <p className="text-sm text-muted-foreground">Evaluación del modelo con datos reales</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metricsList.map((metric) => {
          const percentage = metric.value * 100
          const Icon = metric.icon

          return (
            <div key={metric.key} className="bg-muted/30 p-6 rounded-xl border border-border">
              {/* Icon */}
              <div className="mb-4">
                <Icon className={`h-8 w-8 ${metric.iconColor}`} />
              </div>

              {/* Percentage */}
              <div className="mb-3">
                <span className="text-4xl font-bold text-foreground">{percentage.toFixed(1)}%</span>
              </div>

              {/* Label */}
              <h3 className="font-semibold text-foreground text-base mb-1">{metric.label}</h3>

              {/* Description */}
              <p className="text-xs text-muted-foreground mb-4 leading-relaxed">{metric.description}</p>

              {/* Progress bar */}
              <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                <div
                  className={`h-full ${metric.barColor} transition-all duration-500 rounded-full`}
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
