import type { Style } from "../types";

interface Props {
  styles: Style[];
  styleA: number | null;
  styleB: number | null;
  alpha: number;
  onStyleAChange: (id: number | null) => void;
  onStyleBChange: (id: number | null) => void;
  onAlphaChange: (v: number) => void;
}

export function StyleMixSlider({
  styles,
  styleA,
  styleB,
  alpha,
  onStyleAChange,
  onStyleBChange,
  onAlphaChange,
}: Props) {
  return (
    <div className="panel">
      <div className="panel-title">Style Mix</div>

      <div className="slider-group">
        <label>Style A</label>
        <select
          value={styleA ?? ""}
          onChange={(e) => onStyleAChange(e.target.value ? Number(e.target.value) : null)}
          style={{
            width: "100%",
            padding: "8px",
            background: "var(--c-bg)",
            border: "1px solid var(--c-border)",
            borderRadius: "var(--radius)",
            color: "var(--c-text)",
            fontFamily: "var(--font-body)",
          }}
        >
          <option value="">None</option>
          {styles.map((s) => (
            <option key={s.id} value={s.id}>
              {s.display_name}
            </option>
          ))}
        </select>
      </div>

      <div className="slider-group" style={{ marginTop: "var(--space-3)" }}>
        <label>Style B</label>
        <select
          value={styleB ?? ""}
          onChange={(e) => onStyleBChange(e.target.value ? Number(e.target.value) : null)}
          style={{
            width: "100%",
            padding: "8px",
            background: "var(--c-bg)",
            border: "1px solid var(--c-border)",
            borderRadius: "var(--radius)",
            color: "var(--c-text)",
            fontFamily: "var(--font-body)",
          }}
        >
          <option value="">None</option>
          {styles.map((s) => (
            <option key={s.id} value={s.id}>
              {s.display_name}
            </option>
          ))}
        </select>
      </div>

      {styleA !== null && styleB !== null && (
        <div className="slider-group" style={{ marginTop: "var(--space-3)" }}>
          <label>
            Blend <span>{Math.round(alpha * 100)}%</span>
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={alpha}
            onChange={(e) => onAlphaChange(Number(e.target.value))}
          />
        </div>
      )}
    </div>
  );
}
