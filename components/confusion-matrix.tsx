interface ConfusionMatrixProps {
  matrix: number[][]
}

export function ConfusionMatrix({ matrix }: ConfusionMatrixProps) {
  const classLabels = ["Dengue", "Malaria", "Leptospirosis"]

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-foreground mb-2">Matriz de Confusión</h2>
        <p className="text-sm text-muted-foreground">Comparación entre predicciones y diagnósticos reales</p>
      </div>

      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          {/* Header section */}
          <div className="flex mb-4">
            <div className="w-32"></div>
            <div className="flex-1 text-center">
              <p className="text-sm font-semibold text-foreground mb-4">Predicción</p>
            </div>
          </div>

          {/* Column labels */}
          <div className="flex mb-3">
            <div className="w-32 flex items-end pb-2">
              <span className="text-sm font-semibold text-foreground">Real</span>
            </div>
            <div className="flex-1 grid grid-cols-3 gap-3">
              {classLabels.map((label) => (
                <div key={label} className="text-center text-sm font-semibold text-foreground pb-2">
                  {label}
                </div>
              ))}
            </div>
          </div>

          {/* Matrix rows */}
          <div className="space-y-3">
            {matrix.map((row, realIdx) => (
              <div key={realIdx} className="flex gap-3">
                <div className="w-32 flex items-center">
                  <span className="text-sm font-semibold text-foreground">{classLabels[realIdx]}</span>
                </div>
                <div className="flex-1 grid grid-cols-3 gap-3">
                  {row.map((value, predIdx) => {
                    const isCorrect = realIdx === predIdx
                    return (
                      <div
                        key={predIdx}
                        className={`text-center py-6 px-4 font-bold text-lg rounded-xl transition-colors ${
                          isCorrect ? "bg-blue-600 text-white shadow-sm" : "bg-muted/50 text-muted-foreground"
                        }`}
                      >
                        {value}
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
