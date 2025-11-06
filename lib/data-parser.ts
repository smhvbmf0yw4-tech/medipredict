/**
 * Parse CSV file content - FIXED version with proper row counting
 */
export function parseCSV(content: string): { headers: string[]; rows: string[][] } {
  // Split by newline and filter empty lines
  const lines = content
    .trim()
    .split(/\r?\n/) // Support both \n and \r\n
    .filter((line) => line.trim().length > 0) // Remove completely empty lines

  if (lines.length < 2) {
    throw new Error("El archivo CSV debe tener encabezados y al menos una fila de datos")
  }

  const headers = lines[0].split(",").map((h) => h.trim())

  const rows = lines.slice(1).map((line) => {
    // Handle quoted values in CSV
    const regex = /("(?:[^"]*(?:""[^"]*)*)")|([^,]+)/g
    const matches = line.matchAll(regex)
    const cells: string[] = []

    for (const match of matches) {
      let cell = match[1] || match[2] || ""
      // Remove surrounding quotes if present
      if (cell.startsWith('"') && cell.endsWith('"')) {
        cell = cell.slice(1, -1).replace(/""/g, '"')
      }
      cells.push(cell.trim())
    }

    return cells
  })

  return { headers, rows }
}

export async function parseXLSX(file: File): Promise<{
  headers: string[]
  rows: string[][]
}> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.onload = async (e) => {
      try {
        const data = e.target?.result as ArrayBuffer
        // Dynamic import of xlsx library
        const XLSX = await import("xlsx")

        const workbook = XLSX.read(data, { type: "array" })
        const worksheet = workbook.Sheets[workbook.SheetNames[0]]
        const rows = XLSX.utils.sheet_to_json(worksheet, { header: 1 }) as string[][]

        if (rows.length < 2) {
          reject(new Error("El archivo XLSX debe tener encabezados y al menos una fila de datos"))
          return
        }

        const headers = (rows[0] || []).map((h) => String(h).trim())
        const dataRows = rows.slice(1).map((row) => row.map((cell) => String(cell || "").trim()))

        resolve({ headers, rows: dataRows })
      } catch (error) {
        reject(new Error("Error al procesar archivo XLSX: " + (error instanceof Error ? error.message : String(error))))
      }
    }

    reader.onerror = () => {
      reject(new Error("Error al leer el archivo"))
    }

    reader.readAsArrayBuffer(file)
  })
}

/**
 * Convert parsed data to training format
 */
export function convertToTrainingData(
  headers: string[],
  rows: string[][],
): {
  features: number[][]
  labels: number[]
  hasLabels: boolean
  rowCount: number
} {
  const featureColumns = [
    "age",
    "gender",
    "residence", // unified residence field
    "homemaker",
    "student",
    "professional",
    "merchant",
    "agriculture_livestock",
    "various_jobs",
    "unemployed",
    "hospitalization_days",
    "body_temperature",
    "fever",
    "headache",
    "dizziness",
    "loss_of_appetite",
    "weakness",
    "myalgias",
    "arthralgias",
    "eye_pain",
    "hemorrhages",
    "vomiting",
    "abdominal_pain",
    "chills",
    "hemoptysis",
    "edema",
    "jaundice",
    "bruises",
    "petechiae",
    "rash",
    "diarrhea",
    "respiratory_difficulty",
    "itching",
    "hematocrit",
    "hemoglobin",
    "red_blood_cells",
    "white_blood_cells",
    "neutrophils",
    "eosinophils",
    "basophils",
    "monocytes",
    "lymphocytes",
    "platelets",
    "AST",
    "ALT",
    "ALP",
    "total_bilirubin",
    "direct_bilirubin",
    "indirect_bilirubin",
    "total_proteins",
    "albumin",
    "creatinine",
    "urea",
  ]

  const labelColumn = "diagnosis"
  const diseaseMap: { [key: string]: number } = {
    dengue: 0,
    malaria: 1,
    leptospirosis: 2,
  }

  // Find column indices
  const featureIndices = featureColumns.map((col) => headers.findIndex((h) => h.toLowerCase() === col.toLowerCase()))

  const labelIndex = headers.findIndex((h) => h.toLowerCase() === labelColumn.toLowerCase())
  const hasLabels = labelIndex !== -1

  // Extract features and labels
  const features: number[][] = []
  const labels: number[] = []

  for (const row of rows) {
    // Skip empty rows or rows with fewer cells than headers
    if (row.length === 0 || row.every((cell) => cell === "") || row.length < headers.length) {
      continue
    }

    // Extract features
    const featureRow = featureIndices.map((idx, featureIdx) => {
      if (idx === -1) return 0
      const value = row[idx]
      const columnName = featureColumns[featureIdx].toLowerCase()

      if (columnName === "gender") {
        return value.toLowerCase() === "m" ? 1 : 0
      }

      if (columnName === "residence") {
        return value.toLowerCase() === "urban" || value.toLowerCase() === "urbana" ? 1 : 0
      }

      const num = Number.parseFloat(value)
      return Number.isNaN(num) ? 0 : num
    })

    features.push(featureRow)

    // Extract label if available
    if (hasLabels && labelIndex !== -1) {
      const diagnosis = row[labelIndex]?.toLowerCase() || ""
      labels.push(diseaseMap[diagnosis] ?? 0)
    }
  }

  return { features, labels, hasLabels, rowCount: features.length }
}

/**
 * Generate sample training data for demonstration
 */
export function generateSampleData(numSamples = 300): {
  features: number[][]
  labels: number[]
} {
  const features: number[][] = []
  const labels: number[] = []

  // Generate balanced data - ensure equal distribution
  const samplesPerClass = Math.floor(numSamples / 3)
  const diseases: number[] = []
  for (let d = 0; d < 3; d++) {
    for (let j = 0; j < samplesPerClass; j++) {
      diseases.push(d)
    }
  }
  // Shuffle the diseases array
  for (let i = diseases.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [diseases[i], diseases[j]] = [diseases[j], diseases[i]]
  }
  
  for (let i = 0; i < numSamples; i++) {
    const disease = diseases[i] || Math.floor(Math.random() * 3) // 0: Dengue, 1: Malaria, 2: Leptospirosis

    const age = 20 + Math.floor(Math.random() * 50)
    const gender = Math.random() > 0.5 ? 1 : 0
    const residence = Math.random() > 0.5 ? 1 : 0 // urban=1, rural=0

    // Occupation (7 fields)
    const homemaker = Math.random() > 0.8 ? 1 : 0
    const student = Math.random() > 0.7 ? 1 : 0
    const professional = Math.random() > 0.6 ? 1 : 0
    const merchant = Math.random() > 0.8 ? 1 : 0
    const agriculture_livestock = Math.random() > 0.85 ? 1 : 0
    const various_jobs = Math.random() > 0.75 ? 1 : 0
    const unemployed = Math.random() > 0.9 ? 1 : 0

    const hospitalization_days = 1 + Math.floor(Math.random() * 10)
    const body_temperature = 36 + Math.random() * 5 // 36-41°C

    // Symptoms (21 fields)
    let fever = 0
    let headache = 0
    let dizziness = 0
    let loss_of_appetite = 0
    let weakness = 0
    let myalgias = 0
    let arthralgias = 0
    let eye_pain = 0
    let hemorrhages = 0
    let vomiting = 0
    let abdominal_pain = 0
    let chills = 0
    const hemoptysis = 0
    const edema = 0
    let jaundice = 0
    const bruises = 0
    let petechiae = 0
    let rash = 0
    const diarrhea = 0
    const respiratory_difficulty = 0
    const itching = 0

    // Initialize laboratory values first (some may be overridden in disease-specific blocks)
    let platelets = 150 + Math.random() * 250 // 150-400 K/μL (default, may be overridden)
    let AST = 10 + Math.random() * 40 // 10-50 U/L (default, may be overridden)
    let ALT = 10 + Math.random() * 40 // 10-50 U/L (default, may be overridden)
    let total_bilirubin = 0.3 + Math.random() * 1.5 // 0.3-1.8 mg/dL (default, may be overridden)

    if (disease === 0) {
      // Dengue characteristics - make more distinctive
      fever = Math.random() > 0.05 ? 1 : 0 // Very high fever probability
      headache = Math.random() > 0.1 ? 1 : 0 // Very common
      weakness = Math.random() > 0.15 ? 1 : 0 // Very common
      rash = Math.random() > 0.3 ? 1 : 0 // Distinctive for Dengue
      arthralgias = Math.random() > 0.2 ? 1 : 0 // Very common
      myalgias = Math.random() > 0.15 ? 1 : 0 // Very common
      eye_pain = Math.random() > 0.25 ? 1 : 0 // Common
      hemorrhages = Math.random() > 0.5 ? 1 : 0 // Distinctive
      petechiae = Math.random() > 0.4 ? 1 : 0 // Distinctive
      loss_of_appetite = Math.random() > 0.3 ? 1 : 0
      // Dengue: Lower platelets, normal liver function
      platelets = Math.max(50, 50 + Math.random() * 150) // Lower platelets for Dengue
    } else if (disease === 1) {
      // Malaria characteristics - make more distinctive
      fever = Math.random() > 0.02 ? 1 : 0 // Almost always present
      chills = Math.random() > 0.1 ? 1 : 0 // Very common and distinctive
      headache = Math.random() > 0.15 ? 1 : 0
      weakness = Math.random() > 0.1 ? 1 : 0
      vomiting = Math.random() > 0.4 ? 1 : 0
      myalgias = Math.random() > 0.3 ? 1 : 0
      dizziness = Math.random() > 0.35 ? 1 : 0
      abdominal_pain = Math.random() > 0.45 ? 1 : 0
      // Malaria: Lower platelets, higher bilirubin
      platelets = Math.max(80, 80 + Math.random() * 120) // Lower platelets
      total_bilirubin = 0.5 + Math.random() * 2.0 // Higher bilirubin
    } else {
      // Leptospirosis characteristics - make more distinctive
      fever = Math.random() > 0.05 ? 1 : 0
      headache = Math.random() > 0.1 ? 1 : 0
      myalgias = Math.random() > 0.1 ? 1 : 0 // Very common
      chills = Math.random() > 0.2 ? 1 : 0
      vomiting = Math.random() > 0.35 ? 1 : 0
      jaundice = Math.random() > 0.4 ? 1 : 0 // Distinctive for Leptospirosis
      weakness = Math.random() > 0.2 ? 1 : 0
      abdominal_pain = Math.random() > 0.3 ? 1 : 0
      // Leptospirosis: Higher liver enzymes, higher bilirubin
      AST = 30 + Math.random() * 100 // Higher AST
      ALT = 30 + Math.random() * 100 // Higher ALT
      total_bilirubin = 1.0 + Math.random() * 3.0 // Higher bilirubin
    }

    // Laboratory values (19 fields) - remaining values
    let hematocrit = 35 + Math.random() * 15 // 35-50%
    let hemoglobin = 11 + Math.random() * 6 // 11-17 g/dL
    let red_blood_cells = 3.5 + Math.random() * 2 // 3.5-5.5 M/μL
    let white_blood_cells = 4 + Math.random() * 8 // 4-12 K/μL
    let neutrophils = 40 + Math.random() * 30 // 40-70%
    let eosinophils = Math.random() * 5 // 0-5%
    let basophils = Math.random() * 2 // 0-2%
    let monocytes = 2 + Math.random() * 8 // 2-10%
    let lymphocytes = 20 + Math.random() * 25 // 20-45%
    let ALP = 30 + Math.random() * 100 // 30-130 U/L
    let direct_bilirubin = 0.1 + Math.random() * 0.4 // 0.1-0.5 mg/dL
    let indirect_bilirubin = total_bilirubin - direct_bilirubin
    let total_proteins = 6 + Math.random() * 2 // 6-8 g/dL
    let albumin = 3.5 + Math.random() * 1.5 // 3.5-5 g/dL
    let creatinine = 0.6 + Math.random() * 0.8 // 0.6-1.4 mg/dL
    let urea = 10 + Math.random() * 30 // 10-40 mg/dL

    features.push([
      age,
      gender,
      residence,
      homemaker,
      student,
      professional,
      merchant,
      agriculture_livestock,
      various_jobs,
      unemployed,
      hospitalization_days,
      body_temperature,
      fever,
      headache,
      dizziness,
      loss_of_appetite,
      weakness,
      myalgias,
      arthralgias,
      eye_pain,
      hemorrhages,
      vomiting,
      abdominal_pain,
      chills,
      hemoptysis,
      edema,
      jaundice,
      bruises,
      petechiae,
      rash,
      diarrhea,
      respiratory_difficulty,
      itching,
      hematocrit,
      hemoglobin,
      red_blood_cells,
      white_blood_cells,
      neutrophils,
      eosinophils,
      basophils,
      monocytes,
      lymphocytes,
      platelets,
      AST,
      ALT,
      ALP,
      total_bilirubin,
      direct_bilirubin,
      indirect_bilirubin,
      total_proteins,
      albumin,
      creatinine,
      urea,
    ])
    labels.push(disease)
  }

  return { features, labels }
}
