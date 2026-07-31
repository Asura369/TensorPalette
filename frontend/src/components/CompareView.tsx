import { useCallback, useRef, useState } from "react";

interface Props {
  original: string | null;
  result: string | null;
  loading: boolean;
}

export function CompareView({ original, result, loading }: Props) {
  const [sliderPos, setSliderPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const handlePointerDown = useCallback(() => {
    dragging.current = true;
  }, []);

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 100;
      setSliderPos(Math.max(0, Math.min(100, x)));
    },
    []
  );

  const handlePointerUp = useCallback(() => {
    dragging.current = false;
  }, []);

  return (
    <div
      className="compare-view"
      ref={containerRef}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
    >
      {original && <img src={original} alt="Original" />}
      {result && (
        <div className="overlay" style={{ width: `${sliderPos}%` }}>
          <img src={result} alt="Stylized" />
        </div>
      )}
      {result && (
        <div
          className="compare-slider"
          style={{ left: `${sliderPos}%` }}
          onPointerDown={handlePointerDown}
        />
      )}
      {loading && (
        <div className="loading-overlay">
          <div className="spinner" />
        </div>
      )}
    </div>
  );
}
