import React, { useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "@/index.css";
import {
  CHAT_HISTORY_SCROLL_CLASS,
  PROMPT_INPUT_CONTAINER_ID,
  useChatHistoryBottomPadding,
} from "@/components/WorkspaceChat/ChatContainer/chatScrollLayout";

/**
 * Faithful, dependency-free reproduction of the production chat-area layout
 * that causes the "assistant answer bottom gets clipped" bug.
 *
 * Imports the REAL chatScrollLayout module so this harness exercises the exact
 * production layout + scroll-stickiness contract:
 *   - a `relative` + `overflow-hidden` chat area of a fixed height
 *   - a scroll viewport using CHAT_HISTORY_SCROLL_CLASS + the padding hook
 *   - an absolutely-positioned PromptInput overlay whose height can change at
 *     runtime (the real-world trigger across turns: textarea collapses,
 *     attachments clear, menus open/close)
 *
 * Scaffold box/overlay sizes use inline styles so the reproduction is
 * deterministic and independent of Tailwind purging in the tests/ dir.
 */
function Harness() {
  const params = new URLSearchParams(window.location.search);
  const areaHeight = Number(params.get("areaHeight") || 600);
  const initialOverlay = Number(params.get("overlayHeight") || 240);
  const growBy = Number(params.get("growBy") || 160);

  const [overlayHeight, setOverlayHeight] = useState(initialOverlay);
  const scrollRef = useRef(null);

  // The harness always simulates a user sitting at the bottom of the chat.
  const bottomPadding = useChatHistoryBottomPadding({
    scrollRef,
    pinned: true,
  });

  return (
    <div
      data-testid="chat-area"
      style={{
        position: "relative",
        overflow: "hidden",
        height: `${areaHeight}px`,
        width: "900px",
        background: "#1b1b1e",
      }}
    >
      <div className="flex flex-col h-full w-full">
        <div className="contents">
          <div
            id="chat-history"
            data-testid="chat-history"
            ref={scrollRef}
            className={`${CHAT_HISTORY_SCROLL_CLASS} flex flex-col items-center justify-start`}
            style={
              bottomPadding ? { paddingBottom: bottomPadding } : undefined
            }
          >
            <div style={{ width: "100%", maxWidth: "750px" }}>
              {Array.from({ length: 60 }).map((_, i) => (
                <p key={i} style={{ margin: "0 0 12px", color: "#fff" }}>
                  Assistant answer line {i + 1} — lorem ipsum dolor sit amet
                  consectetur adipiscing elit sed do eiusmod tempor.
                </p>
              ))}
              <p
                data-testid="last-line"
                style={{ margin: "0 0 4px", color: "#fff", fontWeight: 700 }}
              >
                FINAL LINE OF THE ASSISTANT ANSWER
              </p>
              <div
                data-testid="message-footer"
                style={{ fontSize: "12px", color: "rgba(255,255,255,0.6)" }}
              >
                12:34 PM
              </div>
            </div>
          </div>

          <div
            id={PROMPT_INPUT_CONTAINER_ID}
            data-testid="prompt-input"
            className="w-full fixed md:absolute bottom-0 left-0 z-10 flex justify-center items-center"
            style={{
              position: "absolute",
              bottom: 0,
              left: 0,
              height: `${overlayHeight}px`,
              width: "100%",
              background: "#27282a",
              borderTop: "1px solid #444",
            }}
          >
            <button
              data-testid="grow-overlay"
              onClick={() => setOverlayHeight((h) => h + growBy)}
              style={{ color: "#fff" }}
            >
              [ PromptInput overlay — grow ]
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<Harness />);
