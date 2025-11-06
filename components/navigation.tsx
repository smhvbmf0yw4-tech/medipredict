"use client"

import Link from "next/link"
import { useState } from "react"
import { Activity, Brain, Zap, Loader2, CheckCircle2, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { trainLogisticRegression, trainNeuralNetwork } from "@/lib/ml-models"
import { generateSampleData } from "@/lib/data-parser"
import type * as tf from "@tensorflow/tfjs"

interface NavigationProps {
  selectedModel?: string
  onModelChange?: (model: string) => void
  onModelTrained?: (model: tf.LayersModel, normalization: { min: number[]; max: number[] }) => void
}

export function Navigation({ selectedModel = "neural", onModelChange, onModelTrained }: NavigationProps) {
  const [trainingOpen, setTrainingOpen] = useState(false)
  const [trainingLoading, setTrainingLoading] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [currentEpoch, setCurrentEpoch] = useState(0)
  const [totalEpochs, setTotalEpochs] = useState(0)
  const [trainingResult, setTrainingResult] = useState<any>(null)
  const [trainingError, setTrainingError] = useState<string | null>(null)

  const handleTrain = async (model: "logistic" | "neural") => {
    setTrainingLoading(true)
    setTrainingError(null)
    setTrainingResult(null)
    setTrainingProgress(0)
    setCurrentEpoch(0)

    try {
      const data = generateSampleData(500) // Increased sample size for better training
      const epochs = model === "logistic" ? 50 : 75
      setTotalEpochs(epochs)

      const onProgress = (epoch: number) => {
        setCurrentEpoch(epoch + 1)
        setTrainingProgress(((epoch + 1) / epochs) * 100)
      }

      let result
      if (model === "logistic") {
        result = await trainLogisticRegression(data, onProgress)
      } else {
        result = await trainNeuralNetwork(data, onProgress)
      }

      setTrainingResult({
        modelType: model === "logistic" ? "Regresión Logística" : "Red Neuronal Artificial",
        metrics: result.metrics,
        status: "success",
      })

      onModelTrained?.(result.model, result.normalization)
      onModelChange?.(model)
    } catch (err) {
      setTrainingError(err instanceof Error ? err.message : "Error al entrenar el modelo")
    } finally {
      setTrainingLoading(false)
    }
  }

  return (
    <nav className="border-b border-border bg-card sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between gap-4">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 font-semibold text-lg flex-shrink-0">
            <Activity className="h-6 w-6 text-primary" />
            <span className="text-foreground hidden sm:inline">MediPredict</span>
          </Link>

          {/* Navigation Links and Training Button */}
          <div className="flex items-center gap-4 ml-auto">
            <div className="hidden sm:flex items-center gap-2 px-3 py-2 rounded-lg bg-muted">
              <span className="text-xs font-medium text-muted-foreground">Modelo:</span>
              <span className="text-sm font-semibold text-foreground">
                {selectedModel === "logistic" ? "Regresión Logística" : "Red Neuronal"}
              </span>
            </div>

            <Dialog open={trainingOpen} onOpenChange={setTrainingOpen}>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm">
                  <Brain className="mr-2 h-4 w-4" />
                  Entrenar
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Entrenar Modelo</DialogTitle>
                  <DialogDescription>Selecciona el modelo que deseas entrenar</DialogDescription>
                </DialogHeader>

                <div className="grid md:grid-cols-2 gap-4">
                  {/* Logistic Regression */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-base">
                        <Zap className="h-4 w-4 text-primary" />
                        Regresión Logística
                      </CardTitle>
                      <CardDescription className="text-xs">Modelo lineal rápido</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <p className="text-xs text-muted-foreground">
                        Entrenamiento en 50 épocas. Ideal para relaciones lineales.
                      </p>
                      <Button
                        onClick={() => {
                          handleTrain("logistic")
                        }}
                        disabled={trainingLoading}
                        className="w-full"
                        size="sm"
                      >
                        {trainingLoading ? (
                          <>
                            <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                            Entrenando...
                          </>
                        ) : (
                          <>
                            <Zap className="mr-2 h-3 w-3" />
                            Entrenar
                          </>
                        )}
                      </Button>
                    </CardContent>
                  </Card>

                  {/* Neural Network */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-base">
                        <Brain className="h-4 w-4 text-secondary" />
                        Red Neuronal
                      </CardTitle>
                      <CardDescription className="text-xs">Modelo profundo</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <p className="text-xs text-muted-foreground">Entrenamiento en 75 épocas. Mayor precisión.</p>
                      <Button
                        onClick={() => {
                          handleTrain("neural")
                        }}
                        disabled={trainingLoading}
                        className="w-full"
                        size="sm"
                      >
                        {trainingLoading ? (
                          <>
                            <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                            Entrenando...
                          </>
                        ) : (
                          <>
                            <Brain className="mr-2 h-3 w-3" />
                            Entrenar
                          </>
                        )}
                      </Button>
                    </CardContent>
                  </Card>
                </div>

                {/* Training Progress */}
                {trainingLoading && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">
                        Época {currentEpoch} de {totalEpochs}
                      </span>
                      <span className="font-semibold">{Math.round(trainingProgress)}%</span>
                    </div>
                    <Progress value={trainingProgress} />
                  </div>
                )}

                {/* Training Results */}
                {trainingResult && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 p-3 bg-primary/10 rounded-lg">
                      <CheckCircle2 className="h-5 w-5 text-primary flex-shrink-0" />
                      <div>
                        <p className="text-sm font-medium">{trainingResult.modelType}</p>
                      </div>
                    </div>
                    <Button onClick={() => setTrainingOpen(false)} className="w-full" size="sm">
                      Cerrar
                    </Button>
                  </div>
                )}

                {trainingError && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription className="text-xs">{trainingError}</AlertDescription>
                  </Alert>
                )}
              </DialogContent>
            </Dialog>
          </div>
        </div>
      </div>
    </nav>
  )
}
