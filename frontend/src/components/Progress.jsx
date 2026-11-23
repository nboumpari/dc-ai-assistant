import { useEffect, useState } from "react";

// The live list of graph steps while an answer is being made.
export default function Progress({ steps, startedAt }) {
  const [now, setNow] = useState(performance.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(performance.now()), 100);
    return () => clearInterval(timer);
  }, []);
  const shown = steps.length ? steps : [{ label: "Sending", at: 0 }];

  return (
    <div className="row assistant">
      <div className="card working">
        <div className="card-meta">
          <span className="source">Working</span>
          <span className="muted">{((now - startedAt) / 1000).toFixed(1)} s</span>
        </div>
        <ol className="steps">
          {shown.map((step, i) => {
            const current = i === shown.length - 1;
            return (
              <li key={i} className={current ? "current" : "done"}>
                <span className={current ? "spinner" : "check"} aria-hidden="true" />
                {step.label}
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
