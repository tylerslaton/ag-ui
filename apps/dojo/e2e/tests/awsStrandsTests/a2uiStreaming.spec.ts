import { test, expect } from "../../test-isolation-helper";
import { A2UIPage } from "../../featurePages/A2UIPage";

// A2UI progressive-streaming regression net (AWS Strands TS).
//
// The visible symptom this guards: surfaces must paint progressively (cards
// appearing one by one) instead of in one bulk paint after a long wait. The
// load-bearing mechanism is on the wire — the sub-agent's render_a2ui call
// must stream MANY incremental TOOL_CALL_ARGS deltas (aimock chunks tool-call
// arguments, mirroring the OpenAI chat-completions API), and the middleware
// must emit its "building" lifecycle before the surface paints.
//
// Two historical regressions this catches (both shipped green through the
// surface-only specs):
//  1. Sub-agent ran hidden inside the tool (`invoke()`), no inner events on
//     the wire at all → 0 render_a2ui frames.
//  2. Demo model used the OpenAI Responses API, whose Strands adapter buffers
//     `function_call_arguments.delta` and emits one blob at the end → exactly
//     1 ARGS frame.
// Healthy streaming = many small ARGS frames. Asserting on the COMPLETED
// response body keeps this flake-free (no live timing involved).

test("[AWS Strands] A2UI streams render_a2ui args incrementally (no bulk paint)", async ({
  page,
}) => {
  // Capture the runtime's SSE body for the chat run.
  const ssePromise = new Promise<string>((resolve, reject) => {
    page.on("response", async (response) => {
      if (
        response.url().includes("/api/copilotkit/aws-strands") &&
        response.request().method() === "POST" &&
        (response.headers()["content-type"] ?? "").includes("text/event-stream")
      ) {
        try {
          resolve(await response.text());
        } catch (e) {
          reject(e);
        }
      }
    });
  });

  await page.goto("/aws-strands/feature/a2ui_dynamic_schema");
  const a2ui = new A2UIPage(page);
  await a2ui.openChat();
  await a2ui.sendMessage(
    "Use the generate_a2ui tool to create a comparison of 3 hotels with name, location, price per night, and star rating using the StarRating component.",
  );
  await a2ui.assertSurfaceWithIdVisible("hotel-comparison");

  const sse = await ssePromise;

  // The inner render_a2ui call started on the wire…
  const startMatches = sse.match(
    /"type":"TOOL_CALL_START"[^\n]*"toolCallName":"render_a2ui"/g,
  );
  expect(
    startMatches,
    "inner render_a2ui TOOL_CALL_START must reach the wire (sub-agent streaming)",
  ).not.toBeNull();

  // …and its args arrived as MANY incremental deltas, not one blob. The
  // hotel-comparison envelope is ~700 chars; aimock chunks it into well over
  // 3 frames. 1 frame = provider buffering; 0 = sub-agent not streamed.
  const renderStart = startMatches![0];
  const renderCallId = renderStart.match(/"toolCallId":"([^"]+)"/)?.[1];
  expect(renderCallId).toBeTruthy();
  const argFrames = sse.match(
    new RegExp(`"type":"TOOL_CALL_ARGS"[^\\n]*"toolCallId":"${renderCallId}"`, "g"),
  );
  expect(
    argFrames?.length ?? 0,
    "render_a2ui args must stream as multiple incremental deltas",
  ).toBeGreaterThanOrEqual(3);

  // The middleware's pre-paint lifecycle fired (the "Building interface"
  // skeleton's data source) before the surface painted.
  expect(
    sse.includes('"status":"building"') || sse.includes('\\"status\\":\\"building\\"'),
    "middleware must emit the building lifecycle on the wire",
  ).toBe(true);
});
