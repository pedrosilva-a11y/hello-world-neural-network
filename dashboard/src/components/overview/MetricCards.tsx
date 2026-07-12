import { getDashboardMetrics } from "../../lib/dashboardMetrics";
import { formatLoss, formatPercent } from "../../lib/formatters";
import type { NormalizedExperiment } from "../../types/summary";

type MetricCardsProps = {
  experiments: NormalizedExperiment[];
};

function formatValidationErrorPercent(value: number | null): string {
  if (value === null || Number.isNaN(value)) {
    return "Unknown";
  }

  return `${value.toFixed(2)}%`;
}

function MetricCard({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: string | null;
}) {
  return (
    <article className="metric-card">
      <p className="metric-label">{label}</p>
      <p className="metric-value">{value}</p>
      {detail !== null && detail !== undefined && (
        <p className="metric-detail">{detail}</p>
      )}
    </article>
  );
}

export function MetricCards({ experiments }: MetricCardsProps) {
  const metrics = getDashboardMetrics(experiments);

  return (
    <section className="metric-grid" aria-label="Experiment summary metrics">
      <MetricCard
        label="Experiments shown"
        value={String(metrics.experimentCount)}
        detail="After current filters"
      />

      <MetricCard
        label="Best validation accuracy"
        value={formatPercent(metrics.bestValidationAccuracy)}
        detail={metrics.bestValidationAccuracyExperimentName}
      />

      <MetricCard
        label="Average validation accuracy"
        value={formatPercent(metrics.averageValidationAccuracy)}
        detail="Known values only"
      />

      <MetricCard
        label="Best validation loss"
        value={formatLoss(metrics.bestValidationLoss)}
        detail={metrics.bestValidationLossExperimentName}
      />

      <MetricCard
        label="Lowest validation error"
        value={formatValidationErrorPercent(metrics.lowestValidationErrorPercent)}
        detail={metrics.lowestValidationErrorExperimentName}
      />
    </section>
  );
}
