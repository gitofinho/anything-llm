import { useLayoutEffect, useState } from "react";

/**
 * Stable id placed on the PromptInput outer wrapper while in chat mode
 * (centered === false), where the input is an absolutely-positioned overlay
 * pinned to the bottom of the chat area. ChatHistory observes this element to
 * reserve enough bottom space so the last assistant message is never hidden
 * behind the overlay.
 */
export const PROMPT_INPUT_CONTAINER_ID = "primary-prompt-input-container";

/**
 * Layout-relevant classes for the ChatHistory scroll viewport.
 *
 * The chat area is `relative` with an absolutely-positioned PromptInput
 * overlay of *variable* height. The viewport therefore fills the whole area
 * (`h-full`) and the space that clears the overlay is reserved dynamically by
 * `useChatHistoryBottomPadding` (an inline `padding-bottom` that tracks the
 * overlay's real height) instead of via fixed magic numbers. `pb-[100px]` is
 * kept only as a pre-measurement fallback for the first paint.
 */
export const CHAT_HISTORY_SCROLL_CLASS =
  "h-full pb-[100px] pt-6 md:pt-0 md:mx-0 overflow-y-scroll";

/**
 * Breathing room (px) kept between the last message footer (timestamp /
 * actions) and the top of the PromptInput overlay.
 */
export const CHAT_HISTORY_BOTTOM_GAP = 72;

// Cap on the mount-race retry so a never-appearing overlay can't leave a
// per-frame getElementById loop running for the page lifetime (~1s at 60fps).
const ATTACH_MAX_FRAMES = 60;

/**
 * Reserves bottom padding so the last message always clears the
 * absolutely-positioned PromptInput overlay (which varies in height with
 * multi-line input, attachments, menus), and re-pins the view to the bottom
 * when that reserved space changes.
 *
 * @param {object}  [opts]
 * @param {object}  [opts.scrollRef] ref to the scroll viewport element
 * @param {boolean} [opts.pinned] whether the view is currently stuck to the
 *   bottom (so it should re-stick when the reserved padding changes)
 * @param {number}  [gap] breathing room below the last message, in px; must be
 *   referentially stable (the measurement effect is keyed on it)
 * @returns {number} pixels of bottom padding to apply inline (0 until measured)
 */
export function useChatHistoryBottomPadding(
  { scrollRef, pinned } = {},
  gap = CHAT_HISTORY_BOTTOM_GAP
) {
  const [bottomPadding, setBottomPadding] = useState(0);

  useLayoutEffect(() => {
    let frame = 0;
    let frames = 0;
    let resizeObserver = null;

    const attach = () => {
      const el = document.getElementById(PROMPT_INPUT_CONTAINER_ID);
      if (!el) {
        // PromptInput may mount a frame or two later (empty -> chat transition)
        if (frames++ > ATTACH_MAX_FRAMES) return;
        frame = requestAnimationFrame(attach);
        return;
      }
      const measure = () => {
        const next = el.offsetHeight + gap;
        setBottomPadding((prev) => (prev === next ? prev : next));
      };
      measure();
      resizeObserver = new ResizeObserver(measure);
      resizeObserver.observe(el);
    };

    attach();
    return () => {
      if (frame) cancelAnimationFrame(frame);
      if (resizeObserver) resizeObserver.disconnect();
    };
  }, [gap]);

  // Keep the view pinned to the bottom when the reserved padding changes.
  // Across turns the PromptInput height changes (textarea collapses,
  // attachments clear, menus toggle); without this the freshly rendered
  // answer would sit clipped under the overlay until the next manual scroll.
  useLayoutEffect(() => {
    if (!pinned) return;
    const el = scrollRef?.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [bottomPadding, pinned, scrollRef]);

  return bottomPadding;
}
