import type {
  ExperimentIndexItem,
  NormalizedExperiment,
  RawBatchingConfig,
  RawExperimentSummary,
  RawRegularizationConfig,
} from "../types/summary";

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function toNumberOrNull(value: unknown): number | null {
  if (isFiniteNumber(value)) {
    return value;
  }

  return null;
}

function toNumberArray(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => Number(item))
    .filter((item) => Number.isFinite(item));
}

function getLastNumber(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }

  return values[values.length - 1];
}

function formatCanonicalNumberKey(value: number | null): string {
  if (value === null) {
    return "unknown";
  }

  return Number(value.toPrecision(12)).toString();
}

function formatNeuronsProfileLabel(neuronsProfile: number[]): string {
  if (neuronsProfile.length === 0) {
    return "Unknown";
  }

  return `[${neuronsProfile.join(", ")}]`;
}

function inferNeuronsProfile(summary: RawExperimentSummary): number[] {
  const configuredProfile = toNumberArray(summary.config?.model?.neurons_profile);

  if (configuredProfile.length > 0) {
    return configuredProfile;
  }

  const outputSize = toNumberOrNull(summary.config?.model?.output_size);

  if (outputSize !== null) {
    return [outputSize];
  }

  return [];
}

function getValidationErrorPercent(finalValidationAccuracy: number | null): number | null {
  if (finalValidationAccuracy === null) {
    return null;
  }

  return (1 - finalValidationAccuracy) * 100;
}

function getTrainValidationGapPercent(
  finalTrainAccuracy: number | null,
  finalValidationAccuracy: number | null,
): number | null {
  if (finalTrainAccuracy === null || finalValidationAccuracy === null) {
    return null;
  }

  return (finalTrainAccuracy - finalValidationAccuracy) * 100;
}

function getRegularizationConfig(
  summary: RawExperimentSummary,
): RawRegularizationConfig | undefined {
  return summary.metadata?.regularization ?? summary.config?.training?.regularization;
}

function getRegularizationEnabled(
  regularizationConfig: RawRegularizationConfig | undefined,
): boolean | null {
  if (typeof regularizationConfig?.enabled === "boolean") {
    return regularizationConfig.enabled;
  }

  return null;
}

function getRegularizationType(
  regularizationConfig: RawRegularizationConfig | undefined,
  regularizationEnabled: boolean | null,
): string {
  if (regularizationEnabled === false) {
    return "none";
  }

  const regularizationType = regularizationConfig?.type?.trim();

  if (regularizationType !== undefined && regularizationType.length > 0) {
    return regularizationType;
  }

  return "unknown";
}

function getRegularizationLabel(
  regularizationEnabled: boolean | null,
  regularizationType: string,
): string {
  if (regularizationEnabled === false) {
    return "No regularization";
  }

  if (regularizationEnabled === true) {
    if (regularizationType === "l2") {
      return "L2";
    }

    if (regularizationType === "unknown") {
      return "Regularized";
    }

    return regularizationType.toUpperCase();
  }

  return "Unknown";
}

function getBatchingConfig(
  summary: RawExperimentSummary,
): RawBatchingConfig | undefined {
  return summary.metadata?.batching ?? summary.config?.training?.batching;
}

function getBatchingStrategy(
  batchingConfig: RawBatchingConfig | undefined,
): string {
  const batchingStrategy = batchingConfig?.strategy?.trim();

  if (batchingStrategy !== undefined && batchingStrategy.length > 0) {
    return batchingStrategy;
  }

  return "full_batch";
}

function getBatchingLabel(batchingStrategy: string): string {
  if (batchingStrategy === "full_batch") {
    return "Full batch";
  }

  if (batchingStrategy === "mini_batch") {
    return "Mini-batch";
  }

  return batchingStrategy;
}

function getShuffleBatches(
  batchingConfig: RawBatchingConfig | undefined,
): boolean | null {
  if (typeof batchingConfig?.shuffle === "boolean") {
    return batchingConfig.shuffle;
  }

  return null;
}

export function normalizeSummary(
  summary: RawExperimentSummary,
  indexItem?: ExperimentIndexItem,
): NormalizedExperiment {
  const experimentName =
    summary.experiment_name?.trim() ||
    indexItem?.experimentName ||
    "unknown_experiment";

  const modelName = summary.config?.model?.name?.trim() || "unknown_model";
  const neuronsProfile = inferNeuronsProfile(summary);

  const normalizePixels =
    typeof summary.config?.preprocessing?.normalize_pixels === "boolean"
      ? summary.config.preprocessing.normalize_pixels
      : null;

  const optimizer = summary.config?.training?.optimizer?.trim() || "unknown";

  const learningRate = toNumberOrNull(summary.config?.training?.learning_rate);
  const numIterations = toNumberOrNull(summary.config?.training?.num_iterations);
  const numEpochs = toNumberOrNull(summary.config?.training?.num_epochs);

  const batchingConfig = getBatchingConfig(summary);
  const batchingStrategy = getBatchingStrategy(batchingConfig);
  const batchSize = toNumberOrNull(batchingConfig?.batch_size);
  const shuffleBatches = getShuffleBatches(batchingConfig);
  const batchRandomSeed = toNumberOrNull(batchingConfig?.random_seed);

  const regularizationConfig = getRegularizationConfig(summary);
  const regularizationEnabled = getRegularizationEnabled(regularizationConfig);
  const regularizationType = getRegularizationType(
    regularizationConfig,
    regularizationEnabled,
  );
  const regularizationLambda = toNumberOrNull(regularizationConfig?.lambda);

  const trainLoss = toNumberArray(summary.metrics?.train_loss);
  const validationLoss = toNumberArray(summary.metrics?.validation_loss);
  const trainAccuracy = toNumberArray(summary.metrics?.train_accuracy);
  const validationAccuracy = toNumberArray(summary.metrics?.validation_accuracy);

  const finalTrainLoss = getLastNumber(trainLoss);
  const finalValidationLoss = getLastNumber(validationLoss);
  const finalTrainAccuracy = getLastNumber(trainAccuracy);
  const finalValidationAccuracy = getLastNumber(validationAccuracy);

  return {
    experimentName,
    modelName,
    neuronsProfile,
    neuronsProfileLabel: formatNeuronsProfileLabel(neuronsProfile),
    normalizePixels,
    optimizer,
    learningRate,
    learningRateKey: formatCanonicalNumberKey(learningRate),
    numIterations,
    numIterationsKey: formatCanonicalNumberKey(numIterations),

    batchingStrategy,
    batchingLabel: getBatchingLabel(batchingStrategy),
    batchSize,
    batchSizeKey: formatCanonicalNumberKey(batchSize),
    shuffleBatches,
    batchRandomSeed,
    numEpochs,
    numEpochsKey: formatCanonicalNumberKey(numEpochs),

    regularizationEnabled,
    regularizationType,
    regularizationLambda,
    regularizationLambdaKey: formatCanonicalNumberKey(regularizationLambda),
    regularizationLabel: getRegularizationLabel(
      regularizationEnabled,
      regularizationType,
    ),

    trainLoss,
    validationLoss,
    trainAccuracy,
    validationAccuracy,

    finalTrainLoss,
    finalValidationLoss,
    finalTrainAccuracy,
    finalValidationAccuracy,
    validationErrorPercent: getValidationErrorPercent(finalValidationAccuracy),
    trainValidationGapPercent: getTrainValidationGapPercent(
      finalTrainAccuracy,
      finalValidationAccuracy,
    ),

    modifiedAt: indexItem?.modifiedAt ?? null,
    raw: summary,
  };
}