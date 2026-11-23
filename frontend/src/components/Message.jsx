import { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const SOURCE = {
  data_agent: "Data agent · SQL",
  rules_agent: "Rules agent · documents",
  fallback: "Assistant",
  error: "Error",
};

const DOCUMENTS = {
  adr_2025: "ADR 2025",
  cmr_convention: "CMR Convention",
  eu_561_2006: "Regulation (EC) 561/2006",
  eu_gpsr: "Regulation (EU) 2023/988 (GPSR)",
  gs1_general_specifications: "GS1 General Specifications",
  gs1_logistic_label: "GS1 Logistic Label Guideline",
  hse_hsg76: "HSE HSG76",
};

function pages(list) {
  if (!list?.length) return "";
  return list.length === 1 ? `p. ${list[0]}` : `pp. ${list[0]}–${list[list.length - 1]}`;
}

// "[2]" in the answer becomes a link "#cite-2", rendered as a small numbered button
const withCitationLinks = (text) => text.replace(/\[(\d+)\]/g, "[$1](#cite-$1)");

function Sources({ citations, chunks, open, setOpen, active, setActive, listRef }) {
  const documents = [...new Set(chunks.map((c) => DOCUMENTS[c.document] || c.document))];
  return (
    <div className="sources" ref={listRef}>
      <button className="sources-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className={`chevron ${open ? "open" : ""}`} aria-hidden="true" />
        Sources · {chunks.length} passage{chunks.length > 1 ? "s" : ""} from {documents.join(", ")}
      </button>
      {open && (
        <ol className="source-list">
          {chunks.map((chunk, i) => {
            const n = citations[i];
            return (
              <li key={chunk.chunk_id} className={active === n ? "active" : ""}>
                <button className="source-head" onClick={() => setActive(active === n ? null : n)}>
                  <span className="cite static">{n}</span>
                  <span>
                    <strong>{DOCUMENTS[chunk.document] || chunk.document}</strong>
                    {chunk.section_title && <span className="muted"> · {chunk.section_title}</span>}
                    {chunk.pages?.length > 0 && <span className="muted"> · {pages(chunk.pages)}</span>}
                  </span>
                </button>
                {active === n && chunk.text && <blockquote className="passage">{chunk.text}</blockquote>}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

export default function Message({ message }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(null);
  const listRef = useRef(null);

  if (message.role === "user") {
    return (
      <div className="row user">
        <div className="bubble">{message.content}</div>
      </div>
    );
  }

  const details = message.details || {};
  const chunks = details.cited_chunks || [];
  const citations = details.citations || [];

  function cite(n) {
    setOpen(true);
    setActive(n);
    requestAnimationFrame(() => listRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }));
  }

  const components = {
    a({ href, children }) {
      if (href?.startsWith("#cite-")) {
        const n = Number(href.slice(6));
        return (
          <button className="cite" onClick={() => cite(n)} title="Show the passage">
            {children}
          </button>
        );
      }
      return (
        <a href={href} target="_blank" rel="noreferrer">
          {children}
        </a>
      );
    },
  };

  return (
    <div className={`row assistant ${message.route}`}>
      <div className="card">
        <div className="card-meta">
          <span className="source">{SOURCE[message.route] || "Assistant"}</span>
          {message.seconds != null && <span className="muted">{message.seconds.toFixed(1)} s</span>}
        </div>
        <div className="answer">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
            {chunks.length ? withCitationLinks(message.content) : message.content}
          </ReactMarkdown>
        </div>
        {chunks.length > 0 && (
          <Sources {...{ citations, chunks, open, setOpen, active, setActive, listRef }} />
        )}
        {(details.sql || details.assumptions?.length > 0 || message.steps?.length > 0) && (
          <details className="more">
            <summary>Details</summary>
            {details.sql && (
              <>
                <h4>SQL query</h4>
                <pre>{details.sql}</pre>
              </>
            )}
            {details.assumptions?.length > 0 && (
              <>
                <h4>Assumptions</h4>
                <ul>
                  {details.assumptions.map((a) => (
                    <li key={a}>{a}</li>
                  ))}
                </ul>
              </>
            )}
            {message.steps?.length > 0 && (
              <>
                <h4>Steps</h4>
                <ol className="steps done">
                  {message.steps.map((step, i) => (
                    <li key={i}>
                      <span className="t">{step.at.toFixed(1)} s</span>
                      {step.label}
                    </li>
                  ))}
                </ol>
              </>
            )}
          </details>
        )}
      </div>
    </div>
  );
}
