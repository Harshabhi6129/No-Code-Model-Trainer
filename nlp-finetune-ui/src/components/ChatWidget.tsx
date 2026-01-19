import React, { useRef } from "react";
import { useWizard }     from "../store/wizardStore";

export default function ChatWidget({
  disabled,
  send,
}: {
  disabled?: boolean;
  send: (txt: string) => void;
}) {
  const { chat, pushChat } = useWizard();
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSend = () => {
    const txt = inputRef.current?.value?.trim();
    if (!txt) return;
    pushChat({ from: "user", text: txt });
    send(txt);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="w-80 border-l flex flex-col">
      {/* messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2 text-sm">
        {chat.map((m, i) =>
          m.from === "user" ? (
            <p key={i} className="text-right">
              <span className="bg-blue-600 text-white px-2 py-1 rounded inline-block">
                {m.text}
              </span>
            </p>
          ) : (
            <p key={i} dangerouslySetInnerHTML={{ __html:
              `<span class="bg-gray-100 px-2 py-1 rounded inline-block">${m.text}</span>` }} />
          )
        )}
      </div>

      {/* input */}
      <div className="border-t p-2 flex gap-2">
        <input
          ref={inputRef}
          placeholder="Ask something…"
          className="flex-1 border px-2 py-1 rounded text-sm disabled:opacity-50"
          disabled={disabled}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
        />
        <button
          onClick={handleSend}
          disabled={disabled}
          className="bg-blue-600 text-white px-3 rounded text-sm disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  );
}
