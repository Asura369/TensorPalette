import type { Style } from "../types";

interface Props {
  styles: Style[];
  selected: number | null;
  onSelect: (id: number) => void;
}

export function StyleGallery({ styles, selected, onSelect }: Props) {
  if (styles.length === 0) {
    return (
      <div className="panel">
        <div className="panel-title">Styles</div>
        <p style={{ color: "var(--c-text-muted)", fontSize: "0.85rem" }}>
          No styles available. Connect to the API server.
        </p>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-title">Curated Styles</div>
      <div className="style-grid">
        {styles.map((s) => (
          <div
            key={s.id}
            className={`style-card ${selected === s.id ? "active" : ""}`}
            onClick={() => onSelect(s.id)}
          >
            <div className="name">{s.display_name}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
