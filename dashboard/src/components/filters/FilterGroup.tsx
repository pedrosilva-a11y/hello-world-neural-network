import type { FilterOption } from "../../lib/filterExperiments";

type FilterGroupProps = {
  title: string;
  options: FilterOption[];
  selectedValues: string[];
  onChange: (selectedValues: string[]) => void;
};

export function FilterGroup({
  title,
  options,
  selectedValues,
  onChange,
}: FilterGroupProps) {
  function handleOptionChange(value: string, isChecked: boolean) {
    if (isChecked) {
      onChange([...selectedValues, value]);
      return;
    }

    onChange(selectedValues.filter((selectedValue) => selectedValue !== value));
  }

  return (
    <fieldset className="filter-group">
      <legend>{title}</legend>

      {options.length === 0 ? (
        <p className="muted">No options</p>
      ) : (
        <div className="filter-options">
          {options.map((option) => (
            <label className="filter-option" key={option.value}>
              <input
                type="checkbox"
                checked={selectedValues.includes(option.value)}
                onChange={(event) =>
                  handleOptionChange(option.value, event.target.checked)
                }
              />
              <span>{option.label}</span>
              <span className="filter-count">{option.count}</span>
            </label>
          ))}
        </div>
      )}
    </fieldset>
  );
}
