import * as tf from "@tensorflow/tfjs"

export interface TrainingData {
  features: number[][]
  labels: number[]
}

export interface ModelMetrics {
  accuracy: number
  loss: number
  trainingTime: number
}

export interface PredictionResult {
  disease: string
  confidence: number
  probabilities: { [key: string]: number }
}

const DISEASE_LABELS = ["Dengue", "Malaria", "Leptospirosis"]

/**
 * Normalize features using min-max scaling
 */
function normalizeFeatures(features: number[][]): { normalized: number[][]; min: number[]; max: number[] } {
  const numFeatures = features[0].length
  const min: number[] = []
  const max: number[] = []

  // Calculate min and max for each feature
  for (let i = 0; i < numFeatures; i++) {
    const column = features.map((row) => row[i])
    min.push(Math.min(...column))
    max.push(Math.max(...column))
  }

  // Normalize
  const normalized = features.map((row) =>
    row.map((value, i) => {
      const range = max[i] - min[i]
      return range === 0 ? 0 : (value - min[i]) / range
    }),
  )

  return { normalized, min, max }
}

/**
 * Train a Logistic Regression model using TensorFlow.js
 */
export async function trainLogisticRegression(
  data: TrainingData,
  onProgress?: (epoch: number, logs: tf.Logs) => void,
): Promise<{ model: tf.LayersModel; metrics: ModelMetrics; normalization: { min: number[]; max: number[] } }> {
  const startTime = Date.now()

  // Normalize features
  const { normalized, min, max } = normalizeFeatures(data.features)

  // Convert to tensors
  const xs = tf.tensor2d(normalized)
  const ys = tf.oneHot(tf.tensor1d(data.labels, "int32"), DISEASE_LABELS.length)

  // Create logistic regression model
  const model = tf.sequential({
    layers: [
      tf.layers.dense({
        inputShape: [data.features[0].length],
        units: DISEASE_LABELS.length,
        activation: "softmax",
      }),
    ],
  })

  // Compile model
  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  })

  // Train model with improved parameters
  const history = await model.fit(xs, ys, {
    epochs: 200, // Increased from 100 for better training
    batchSize: 16, // Smaller batch size for better gradient updates
    validationSplit: 0.15, // Reduced validation split to use more training data
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (onProgress && logs) {
          onProgress(epoch, logs)
        }
      },
    },
  })

  // Calculate metrics
  const finalLogs = history.history
  const accuracy = finalLogs.acc[finalLogs.acc.length - 1] as number
  const loss = finalLogs.loss[finalLogs.loss.length - 1] as number
  const trainingTime = Date.now() - startTime

  // Cleanup tensors
  xs.dispose()
  ys.dispose()

  return {
    model,
    metrics: { accuracy, loss, trainingTime },
    normalization: { min, max },
  }
}

/**
 * Train a Neural Network model using TensorFlow.js
 */
export async function trainNeuralNetwork(
  data: TrainingData,
  onProgress?: (epoch: number, logs: tf.Logs) => void,
): Promise<{ model: tf.LayersModel; metrics: ModelMetrics; normalization: { min: number[]; max: number[] } }> {
  const startTime = Date.now()

  // Normalize features
  const { normalized, min, max } = normalizeFeatures(data.features)

  // Convert to tensors
  const xs = tf.tensor2d(normalized)
  const ys = tf.oneHot(tf.tensor1d(data.labels, "int32"), DISEASE_LABELS.length)

  // Create neural network model with improved architecture
  const model = tf.sequential({
    layers: [
      tf.layers.dense({
        inputShape: [data.features[0].length],
        units: 128, // Increased from 64
        activation: "relu",
        kernelInitializer: "heNormal",
      }),
      tf.layers.batchNormalization(),
      tf.layers.dropout({ rate: 0.3 }),
      tf.layers.dense({
        units: 64, // Increased from 32
        activation: "relu",
        kernelInitializer: "heNormal",
      }),
      tf.layers.batchNormalization(),
      tf.layers.dropout({ rate: 0.3 }),
      tf.layers.dense({
        units: 32,
        activation: "relu",
      }),
      tf.layers.dropout({ rate: 0.2 }),
      tf.layers.dense({
        units: DISEASE_LABELS.length,
        activation: "softmax",
      }),
    ],
  })

  // Compile model with improved optimizer
  model.compile({
    optimizer: tf.train.adam(0.0005), // Lower learning rate for more stable training
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  })

  // Train model with improved parameters
  const history = await model.fit(xs, ys, {
    epochs: 250, // Increased from 150 for better training
    batchSize: 16, // Smaller batch size for better gradient updates
    validationSplit: 0.15, // Reduced validation split to use more training data
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (onProgress && logs) {
          onProgress(epoch, logs)
        }
      },
    },
  })

  // Calculate metrics
  const finalLogs = history.history
  const accuracy = finalLogs.acc[finalLogs.acc.length - 1] as number
  const loss = finalLogs.loss[finalLogs.loss.length - 1] as number
  const trainingTime = Date.now() - startTime

  // Cleanup tensors
  xs.dispose()
  ys.dispose()

  return {
    model,
    metrics: { accuracy, loss, trainingTime },
    normalization: { min, max },
  }
}

/**
 * Make a prediction using a trained model
 */
export function predict(
  model: tf.LayersModel,
  features: number[],
  normalization: { min: number[]; max: number[] },
): PredictionResult {
  // Normalize input features
  const normalized = features.map((value, i) => {
    const range = normalization.max[i] - normalization.min[i]
    return range === 0 ? 0 : (value - normalization.min[i]) / range
  })

  // Make prediction
  const input = tf.tensor2d([normalized])
  const prediction = model.predict(input) as tf.Tensor
  const probabilities = prediction.dataSync()

  // Apply temperature scaling for diversity
  const temperature = 0.8
  const scaledProbs = Array.from(probabilities).map((p) => Math.pow(p, 1 / temperature))
  const sum = scaledProbs.reduce((a, b) => a + b, 0)
  const normalizedProbs = scaledProbs.map((p) => p / sum)

  // Find the class with highest probability
  let maxProb = 0
  let predictedClass = 0
  for (let i = 0; i < normalizedProbs.length; i++) {
    if (normalizedProbs[i] > maxProb) {
      maxProb = normalizedProbs[i]
      predictedClass = i
    }
  }

  // Create probabilities object
  const probabilitiesObj: { [key: string]: number } = {}
  DISEASE_LABELS.forEach((label, i) => {
    probabilitiesObj[label] = normalizedProbs[i]
  })

  // Cleanup tensors
  input.dispose()
  prediction.dispose()

  return {
    disease: DISEASE_LABELS[predictedClass],
    confidence: maxProb,
    probabilities: probabilitiesObj,
  }
}

/**
 * Make batch predictions
 */
export function predictBatch(
  model: tf.LayersModel,
  featuresArray: number[][],
  normalization: { min: number[]; max: number[] },
): PredictionResult[] {
  // Normalize all features
  const normalized = featuresArray.map((features) =>
    features.map((value, i) => {
      const range = normalization.max[i] - normalization.min[i]
      return range === 0 ? 0 : (value - normalization.min[i]) / range
    }),
  )

  // Make predictions
  const input = tf.tensor2d(normalized)
  const predictions = model.predict(input) as tf.Tensor
  const probabilitiesArray = predictions.arraySync() as number[][]

  // Process results with diversity enhancement
  // Apply temperature scaling to make predictions more diverse
  const temperature = 0.8 // Lower temperature = more diverse predictions
  const results: PredictionResult[] = probabilitiesArray.map((probabilities) => {
    // Apply temperature scaling
    const scaledProbs = probabilities.map((p) => Math.pow(p, 1 / temperature))
    const sum = scaledProbs.reduce((a, b) => a + b, 0)
    const normalizedProbs = scaledProbs.map((p) => p / sum)
    
    let maxProb = 0
    let predictedClass = 0
    for (let i = 0; i < normalizedProbs.length; i++) {
      if (normalizedProbs[i] > maxProb) {
        maxProb = normalizedProbs[i]
        predictedClass = i
      }
    }

    const probabilitiesObj: { [key: string]: number } = {}
    DISEASE_LABELS.forEach((label, i) => {
      probabilitiesObj[label] = normalizedProbs[i]
    })

    return {
      disease: DISEASE_LABELS[predictedClass],
      confidence: maxProb,
      probabilities: probabilitiesObj,
    }
  })

  // Cleanup tensors
  input.dispose()
  predictions.dispose()

  return results
}

/**
 * Calculate confusion matrix
 */
export function calculateConfusionMatrix(predictions: number[], actuals: number[]): number[][] {
  const numClasses = DISEASE_LABELS.length
  const matrix: number[][] = Array(numClasses)
    .fill(0)
    .map(() => Array(numClasses).fill(0))

  for (let i = 0; i < predictions.length; i++) {
    matrix[actuals[i]][predictions[i]]++
  }

  return matrix
}

/**
 * Calculate evaluation metrics
 */
export function calculateMetrics(confusionMatrix: number[][]): {
  accuracy: number
  precision: number
  recall: number
  f1Score: number
} {
  const numClasses = confusionMatrix.length
  let totalCorrect = 0
  let totalSamples = 0
  let totalPrecision = 0
  let totalRecall = 0

  for (let i = 0; i < numClasses; i++) {
    const truePositive = confusionMatrix[i][i]
    let falsePositive = 0
    let falseNegative = 0

    for (let j = 0; j < numClasses; j++) {
      if (j !== i) {
        falsePositive += confusionMatrix[j][i]
        falseNegative += confusionMatrix[i][j]
      }
      totalSamples += confusionMatrix[i][j]
    }

    totalCorrect += truePositive

    const precision = truePositive / (truePositive + falsePositive) || 0
    const recall = truePositive / (truePositive + falseNegative) || 0

    totalPrecision += precision
    totalRecall += recall
  }

  const accuracy = totalCorrect / totalSamples
  const avgPrecision = totalPrecision / numClasses
  const avgRecall = totalRecall / numClasses
  const f1Score = (2 * avgPrecision * avgRecall) / (avgPrecision + avgRecall) || 0

  return {
    accuracy,
    precision: avgPrecision,
    recall: avgRecall,
    f1Score,
  }
}
