export type RawExperimentSummary = {
  experiment_name?: string;

  config?: {
    model?: {
      name?: string;
      neurons_profile?: number[];
      input_size?: number;
      output_size?: number;
    };
    preprocessing?: {
      normalize_pixels?: boolean;
    };
    training?: {
      learning_rate?: number;
      num_iterations?: number;
      optimizer?: string;
    };
  };

  metrics?: {
    train_loss?: number[];
    validation_loss?: number[];
    train_accuracy?: number[];
    validation_accuracy?: number[];
  };

  metadata?: {
    model?: {
      final_parameter_shapes?: Record<string, number[] | string>;
    };
    data_shapes?: Record<string, number[] | string>;
  };

  data_distribution?: Record<string, unknown>;

  outputs?: {
    validation_predictions_preview?: number[];
    validation_labels_preview?: number[];
  };

  summary_path?: string;

  [key: string]: unknown;
};

export type ExperimentIndexItem = {
  experimentName: string;
  modifiedAt: number;
};

export type NormalizedExperiment = {
  experimentName: string;
  modelName: string;
  neuronsProfile: number[];
  neuronsProfileLabel: string;
  normalizePixels: boolean | null;
  optimizer: string;
  learningRate: number | null;
  learningRateKey: string;
  numIterations: number | null;
  numIterationsKey: string;

  trainLoss: number[];
  validationLoss: number[];
  trainAccuracy: number[];
  validationAccuracy: number[];

  finalTrainLoss: number | null;
  finalValidationLoss: number | null;
  finalTrainAccuracy: number | null;
  finalValidationAccuracy: number | null;
  validationErrorPercent: number | null;
  trainValidationGapPercent: number | null;

  modifiedAt: number | null;
  raw: RawExperimentSummary;
};

export type LoadedExperiments = {
  experiments: NormalizedExperiment[];
  failedExperimentNames: string[];
};
