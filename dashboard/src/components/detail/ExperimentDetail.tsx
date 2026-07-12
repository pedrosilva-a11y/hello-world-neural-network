import {
  formatBoolean,
  formatLoss,
  formatNumber,
  formatPercent,
} from "../../lib/formatters";
import type { NormalizedExperiment } from "../../types/summary";

type ExperimentDetailProps = {
  experiments: NormalizedExperiment[];
  selectedExperiment: NormalizedExperiment | null;
  onSelectedExperimentNameChange: (experimentName: string) => void;
};

function formatPercentValue(value: number | null): string {
  return formatPercent(value);
}

function formatValidationErrorPercent(value: number | null): string {
  if (value === null || Number.isNaN(value)) {
    return "Unknown";
  }

  return `${value.toFixed(2)}%`;
}

function formatModifiedAt(modifiedAt: number | null): string {
  if (modifiedAt === null) {
    return "Unknown";
  }

  return new Date(modifiedAt).toLocaleString();
}

function getPrettyJson(value: unknown): string {
  return JSON.stringify(value ?? {}, null, 2);
}

function DetailMetric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="detail-metric">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function JsonBlock({
  title,
  value,
}: {
  title: string;
  value: unknown;
}) {
  return (
    <details className="json-details">
      <summary>{title}</summary>
      <pre>{getPrettyJson(value)}</pre>
    </details>
  );
}

export function ExperimentDetail({
  experiments,
  selectedExperiment,
  onSelectedExperimentNameChange,
}: ExperimentDetailProps) {
  if (experiments.length === 0 || selectedExperiment === null) {
    return null;
  }

  const rawSummary = selectedExperiment.raw;

  return (
    <section className="dashboard-section">
      <div className="section-header section-header-with-controls">
        <div>
          <h2>Experiment detail</h2>
          <p className="muted">
            Inspect one filtered experiment in more depth.
          </p>
        </div>

        <label className="detail-select-label">
          Selected experiment
          <select
            value={selectedExperiment.experimentName}
            onChange={(event) =>
              onSelectedExperimentNameChange(event.target.value)
            }
          >
            {experiments.map((experiment) => (
              <option
                key={experiment.experimentName}
                value={experiment.experimentName}
              >
                {experiment.experimentName}
              </option>
            ))}
          </select>
        </label>
      </div>

      <article className="detail-card">
        <div>
          <h3>{selectedExperiment.experimentName}</h3>
          <p className="muted">{selectedExperiment.modelName}</p>
        </div>

        <dl className="detail-metric-grid">
          <DetailMetric
            label="Architecture"
            value={selectedExperiment.neuronsProfileLabel}
          />
          <DetailMetric
            label="Normalized"
            value={formatBoolean(selectedExperiment.normalizePixels)}
          />
          <DetailMetric
            label="Optimizer"
            value={selectedExperiment.optimizer}
          />
          <DetailMetric
            label="Learning rate"
            value={formatNumber(selectedExperiment.learningRate)}
          />
          <DetailMetric
            label="Iterations"
            value={formatNumber(selectedExperiment.numIterations)}
          />
          <DetailMetric
            label="Validation accuracy"
            value={formatPercentValue(selectedExperiment.finalValidationAccuracy)}
          />
          <DetailMetric
            label="Validation error"
            value={formatValidationErrorPercent(
              selectedExperiment.validationErrorPercent,
            )}
          />
          <DetailMetric
            label="Train-validation gap"
            value={
              selectedExperiment.trainValidationGapPercent === null
                ? "Unknown"
                : `${selectedExperiment.trainValidationGapPercent.toFixed(2)} pp`
            }
          />
          <DetailMetric
            label="Train loss"
            value={formatLoss(selectedExperiment.finalTrainLoss)}
          />
          <DetailMetric
            label="Validation loss"
            value={formatLoss(selectedExperiment.finalValidationLoss)}
          />
          <DetailMetric
            label="Modified"
            value={formatModifiedAt(selectedExperiment.modifiedAt)}
          />
        </dl>

        <div className="json-grid">
          <JsonBlock title="Config" value={rawSummary.config} />
          <JsonBlock
            title="Parameter shapes"
            value={rawSummary.metadata?.model?.final_parameter_shapes}
          />
          <JsonBlock
            title="Data shapes"
            value={rawSummary.metadata?.data_shapes}
          />
          <JsonBlock
            title="Data distribution"
            value={rawSummary.data_distribution}
          />
          <JsonBlock
            title="Prediction preview"
            value={rawSummary.outputs}
          />
          <JsonBlock title="Raw summary JSON" value={rawSummary} />
        </div>
      </article>
    </section>
  );
}
