import { useRef, useState } from "react";

// Message box: Enter sends, Shift+Enter adds a line.
export default function Composer({ onSend, disabled }) {
  const [text, setText] = useState("");
  const box = useRef(null);

  function submit(event) {
    event?.preventDefault();
    const message = text.trim();
    if (!message || disabled) return;
    onSend(message);
    setText("");
    box.current?.focus();
  }

  return (
    <form className="composer" onSubmit={submit}>
      <textarea
        ref={box}
        rows={1}
        value={text}
        placeholder="Ask about warehouse data or logistics rules"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) submit(e);
        }}
        autoFocus
      />
      <button type="submit" disabled={disabled || !text.trim()} aria-label="Send">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M5 12h13M13 6l6 6-6 6" />
        </svg>
      </button>
    </form>
  );
}
