import { useEffect, useRef, useState } from "react";
import { streamChat } from "./api.js";
import Composer from "./components/Composer.jsx";
import Message from "./components/Message.jsx";
import Progress from "./components/Progress.jsx";

const SUGGESTIONS = [
  "What was our OTIF by month in 2025?",
  "Which carrier had the most late deliveries in Q3?",
  "How tall must the word OVERPACK be on a limited-quantity overpack?",
  "Can the 45-minute driver break be split?",
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [pending, setPending] = useState(null); // { steps, startedAt } while an answer is streaming
  const bottom = useRef(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages, pending]);

  async function send(text) {
    const history = messages.map(({ role, content, route }) => ({ role, content, route }));
    setMessages((m) => [...m, { role: "user", content: text }]);
    const startedAt = performance.now();
    let steps = [];
    setPending({ steps, startedAt });
    const add = (message) => setMessages((m) => [...m, { role: "assistant", ...message }]);
    try {
      await streamChat(text, history, (event, data) => {
        const at = (performance.now() - startedAt) / 1000;
        if (event === "step") {
          steps = [...steps, { ...data, at }];
          setPending({ steps, startedAt });
        } else if (event === "answer") {
          add({ ...data, content: data.answer, steps, seconds: at });
        } else if (event === "error") {
          add({ route: "error", content: data.message, steps, seconds: at });
        }
      });
    } catch {
      add({ route: "error", content: "The assistant could not be reached." });
    } finally {
      setPending(null);
    }
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <svg className="mark" viewBox="0 0 24 24" aria-hidden="true">
            <rect x="3" y="11" width="5" height="9" rx="1" />
            <rect x="9.5" y="6" width="5" height="14" rx="1" />
            <rect x="16" y="3" width="5" height="17" rx="1" />
          </svg>
          <span>DC Assistant</span>
        </div>
      </header>

      <main className="panel">
        <div className="panel-head">
          <div>
            <h1>Control tower assistant</h1>
            <p>Warehouse data from the 2025 database, and logistics rules from official documents.</p>
          </div>
          {messages.length > 0 && (
            <button className="text-button" onClick={() => setMessages([])} disabled={!!pending}>
              New chat
            </button>
          )}
        </div>

        <div className="thread">
          {messages.length === 0 && !pending && (
            <div className="empty">
              <p>
                Ask about orders, shipments, pallets, carriers and KPIs, or about GS1 labelling, ADR, CMR, drivers'
                hours, product safety and warehouse safety.
              </p>
              <div className="suggestions">
                {SUGGESTIONS.map((s) => (
                  <button key={s} className="suggestion" onClick={() => send(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((message, i) => (
            <Message key={i} message={message} />
          ))}
          {pending && <Progress steps={pending.steps} startedAt={pending.startedAt} />}
          <div ref={bottom} />
        </div>

        <Composer onSend={send} disabled={!!pending} />
      </main>
    </div>
  );
}
