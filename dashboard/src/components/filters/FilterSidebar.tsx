import {
  emptyDashboardFilters,
  getFilterOptions,
  hasActiveFilters,
  type DashboardFilters,
} from "../../lib/filterExperiments";
import type { NormalizedExperiment } from "../../types/summary";
import { FilterGroup } from "./FilterGroup";

type FilterSidebarProps = {
  experiments: NormalizedExperiment[];
  filteredExperimentCount: number;
  filters: DashboardFilters;
  onFiltersChange: (filters: DashboardFilters) => void;
};

export function FilterSidebar({
  experiments,
  filteredExperimentCount,
  filters,
  onFiltersChange,
}: FilterSidebarProps) {
  function updateFilter<Key extends keyof DashboardFilters>(
    key: Key,
    selectedValues: DashboardFilters[Key],
  ) {
    onFiltersChange({
      ...filters,
      [key]: selectedValues,
    });
  }

  return (
    <aside className="filter-sidebar">
      <div className="filter-sidebar-header">
        <div>
          <h2>Filters</h2>
          <p className="muted">Leave empty to include all values.</p>
        </div>

        {hasActiveFilters(filters) && (
          <button
            className="secondary-button"
            type="button"
            onClick={() => onFiltersChange(emptyDashboardFilters)}
          >
            Reset
          </button>
        )}
      </div>

      <p className="filter-summary">
        {filteredExperimentCount} of {experiments.length} experiments shown
      </p>

      <FilterGroup
        title="Model"
        options={getFilterOptions(experiments, "modelNames")}
        selectedValues={filters.modelNames}
        onChange={(selectedValues) => updateFilter("modelNames", selectedValues)}
      />

      <FilterGroup
        title="Architecture"
        options={getFilterOptions(experiments, "architectures")}
        selectedValues={filters.architectures}
        onChange={(selectedValues) => updateFilter("architectures", selectedValues)}
      />

      <FilterGroup
        title="Pixel normalization"
        options={getFilterOptions(experiments, "normalizationOptions")}
        selectedValues={filters.normalizationOptions}
        onChange={(selectedValues) =>
          updateFilter("normalizationOptions", selectedValues)
        }
      />

      <FilterGroup
        title="Optimizer"
        options={getFilterOptions(experiments, "optimizers")}
        selectedValues={filters.optimizers}
        onChange={(selectedValues) => updateFilter("optimizers", selectedValues)}
      />

      <FilterGroup
        title="Regularization"
        options={getFilterOptions(experiments, "regularizationOptions")}
        selectedValues={filters.regularizationOptions}
        onChange={(selectedValues) =>
          updateFilter("regularizationOptions", selectedValues)
        }
      />

      <FilterGroup
        title="Regularization lambda"
        options={getFilterOptions(experiments, "regularizationLambdaKeys")}
        selectedValues={filters.regularizationLambdaKeys}
        onChange={(selectedValues) =>
          updateFilter("regularizationLambdaKeys", selectedValues)
        }
      />

      <FilterGroup
        title="Learning rate"
        options={getFilterOptions(experiments, "learningRateKeys")}
        selectedValues={filters.learningRateKeys}
        onChange={(selectedValues) => updateFilter("learningRateKeys", selectedValues)}
      />

      <FilterGroup
        title="Iterations"
        options={getFilterOptions(experiments, "iterationCountKeys")}
        selectedValues={filters.iterationCountKeys}
        onChange={(selectedValues) =>
          updateFilter("iterationCountKeys", selectedValues)
        }
      />

      <FilterGroup
        title="Experiment"
        options={getFilterOptions(experiments, "experimentNames")}
        selectedValues={filters.experimentNames}
        onChange={(selectedValues) => updateFilter("experimentNames", selectedValues)}
      />
    </aside>
  );
}
