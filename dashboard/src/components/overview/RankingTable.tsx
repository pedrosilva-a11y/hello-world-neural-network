import type { NormalizedExperiment } from "../../types/summary";
import { buildCsv, downloadCsv } from "../../lib/csv";
import {
  formatBoolean,
  formatLoss,
  formatNumber,
  formatPercent,
  formatPercentPoints,
} from "../../lib/formatters";

type RankingTableProps = {
  experiments: NormalizedExperiment[];
};

function compareValidationAccuracy(
  left: NormalizedExperiment,
  right: NormalizedExperiment,
): number {
  const leftAccuracy = left.finalValidationAccuracy ?? Number.NEGATIVE_INFINITY;
  const rightAccuracy = right.finalValidationAccuracy ?? Number.NEGATIVE_INFINITY;

  return rightAccuracy - leftAccuracy;
}

function buildRankingCsvRows(experiments: NormalizedExperiment[]) {
  return experiments.map((experiment) => ({
    experiment_name: experiment.experimentName,
    model_name: experiment.modelName,
    neurons_profile: experiment.neuronsProfileLabel,
    normalize_pixels: formatBoolean(experiment.normalizePixels),
    optimizer: experiment.optimizer,
    batching_strategy: experiment.batchingLabel,
    batch_size: experiment.batchSize,
    num_epochs: experiment.numEpochs,
    shuffle_batches: formatBoolean(experiment.shuffleBatches),
    regularization: experiment.regularizationLabel ?? "Unknown",
    regularization_lambda: experiment.regularizationLambda ?? null,
    learning_rate: experiment.learningRate,
    num_iterations: experiment.numIterations,
    final_train_accuracy: experiment.finalTrainAccuracy,
    final_validation_accuracy: experiment.finalValidationAccuracy,
    validation_error_percent: experiment.validationErrorPercent,
    train_validation_gap_percent: experiment.trainValidationGapPercent,
    final_train_loss: experiment.finalTrainLoss,
    final_validation_loss: experiment.finalValidationLoss,
  }));
}

export function RankingTable({ experiments }: RankingTableProps) {
  const rankedExperiments = [...experiments].sort(compareValidationAccuracy);

  function handleDownloadCsv() {
    const csvContent = buildCsv(buildRankingCsvRows(rankedExperiments));
    downloadCsv("digit_recognizer_experiment_ranking.csv", csvContent);
  }

  if (rankedExperiments.length === 0) {
    return <p className="muted">No experiments available for ranking.</p>;
  }

  return (
    <section className="dashboard-section">
      <div className="section-header">
        <div>
          <h2>Experiment ranking</h2>
          <p className="muted">
            Sorted by final validation accuracy, highest first.
          </p>
        </div>

        <button type="button" onClick={handleDownloadCsv}>
          Download CSV
        </button>
      </div>

      <div className="table-scroll">
        <table className="ranking-table">
          <thead>
            <tr>
              <th>Experiment</th>
              <th>Model</th>
              <th>Architecture</th>
              <th>Normalized</th>
              <th>Optimizer</th>
              <th>Batching</th>
              <th>Batch size</th>
              <th>Epochs</th>
              <th>Shuffle</th>
              <th>Regularization</th>
              <th>Lambda</th>
              <th>Learning rate</th>
              <th>Iterations</th>
              <th>Train accuracy</th>
              <th>Validation accuracy</th>
              <th>Validation error</th>
              <th>Train-val gap</th>
              <th>Train loss</th>
              <th>Validation loss</th>
            </tr>
          </thead>

          <tbody>
            {rankedExperiments.map((experiment) => (
              <tr key={experiment.experimentName}>
                <td className="strong-cell">{experiment.experimentName}</td>
                <td>{experiment.modelName}</td>
                <td>{experiment.neuronsProfileLabel}</td>
                <td>{formatBoolean(experiment.normalizePixels)}</td>
                <td>{experiment.optimizer}</td>
                <td>{experiment.batchingLabel}</td>
                <td>{formatNumber(experiment.batchSize)}</td>
                <td>{formatNumber(experiment.numEpochs)}</td>
                <td>{formatBoolean(experiment.shuffleBatches)}</td>
                <td>{experiment.regularizationLabel ?? "Unknown"}</td>
                <td>{formatNumber(experiment.regularizationLambda ?? null)}</td>
                <td>{formatNumber(experiment.learningRate)}</td>
                <td>{formatNumber(experiment.numIterations)}</td>
                <td>{formatPercent(experiment.finalTrainAccuracy)}</td>
                <td>{formatPercent(experiment.finalValidationAccuracy)}</td>
                <td>
                  {experiment.validationErrorPercent === null
                    ? "Unknown"
                    : `${experiment.validationErrorPercent.toFixed(2)}%`}
                </td>
                <td>{formatPercentPoints(experiment.trainValidationGapPercent)}</td>
                <td>{formatLoss(experiment.finalTrainLoss)}</td>
                <td>{formatLoss(experiment.finalValidationLoss)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
