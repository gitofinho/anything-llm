import { test, expect } from "@playwright/test";

/**
 * Bug: when the Chat Assistant answer is long and the PromptInput overlay is
 * tall (multi-line input, attachments, menus), the bottom of the last message
 * — including its timestamp/footer — is hidden behind, or uncomfortably close
 * to, the absolutely-positioned PromptInput.
 *
 * Contract:
 *  1. After scrolling to the bottom, the message footer must keep a
 *     comfortable gap above the PromptInput overlay (not "too tight").
 *  2. When the overlay height changes mid-conversation (next turn: textarea
 *     collapses, attachments clear, menus toggle), the view must re-stick to
 *     the bottom so the footer stays visible without any manual scroll.
 */

// Minimum comfortable breathing room between the message footer and the
// PromptInput overlay, in px.
const MIN_GAP = 48;

async function scrollChatToBottom(page) {
  await page.locator("#chat-history").evaluate((el) => {
    el.scrollTop = el.scrollHeight;
  });
  await page.waitForTimeout(150);
}

async function footerVsOverlay(page) {
  const footerBottom = await page
    .getByTestId("message-footer")
    .evaluate((el) => el.getBoundingClientRect().bottom);
  const overlayTop = await page
    .getByTestId("prompt-input")
    .evaluate((el) => el.getBoundingClientRect().top);
  return { footerBottom, overlayTop, gap: overlayTop - footerBottom };
}

for (const overlayHeight of [240, 400]) {
  test(`message footer keeps a comfortable gap above a ${overlayHeight}px overlay`, async ({
    page,
  }) => {
    await page.goto(
      `/harness.html?areaHeight=600&overlayHeight=${overlayHeight}`
    );
    await page.getByTestId("message-footer").waitFor();
    await scrollChatToBottom(page);

    const { footerBottom, overlayTop, gap } = await footerVsOverlay(page);

    expect(
      gap,
      `footer bottom (${footerBottom}px) -> overlay top (${overlayTop}px): gap ${gap}px must be >= ${MIN_GAP}px`
    ).toBeGreaterThanOrEqual(MIN_GAP);
  });
}

test("view re-sticks to bottom when the overlay grows mid-conversation", async ({
  page,
}) => {
  await page.goto("/harness.html?areaHeight=600&overlayHeight=200&growBy=160");
  await page.getByTestId("message-footer").waitFor();
  await scrollChatToBottom(page);

  // Sanity: footer not occluded before the overlay changes.
  let state = await footerVsOverlay(page);
  expect(state.gap).toBeGreaterThanOrEqual(0);

  // Simulate the next turn making the PromptInput taller (no manual scroll).
  await page.getByTestId("grow-overlay").click();
  await page.waitForTimeout(200);

  state = await footerVsOverlay(page);
  expect(
    state.gap,
    `after overlay grew: footer bottom -> overlay top gap ${state.gap}px must stay >= 0 (view should re-stick to bottom, footer not occluded)`
  ).toBeGreaterThanOrEqual(0);
});
