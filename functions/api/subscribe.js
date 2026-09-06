/**
 * POST /api/subscribe
 *
 * Cloudflare Pages Function. If docs/ is deployed as a Cloudflare Pages
 * site (or this file is copied into a project deployed that way), this
 * route is live automatically — no separate server to run.
 *
 * Simple signup notification: each request emails the submitted address
 * straight to wes@squish.run via Resend (https://resend.com) — a single
 * authenticated POST, no ESP subscriber list to manage. Swap the
 * `notifySignup` body for a different transactional email provider if
 * needed — the request/response contract with the page's
 * fetch("/api/subscribe") call (JSON in: { email, src }, 2xx = success)
 * is what matters, not which provider sends the mail.
 *
 * Required Cloudflare Pages environment variable:
 *   RESEND_API_KEY  — set in the Pages project settings, never
 *                      committed to the repo.
 *
 * Konjo constraint: no silent failures — every non-2xx path returns a
 * real error status and logs the cause server-side.
 */

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const NOTIFY_TO = "wes@squish.run";

export async function onRequestPost(context) {
  const { request, env } = context;

  let body;
  try {
    body = await request.json();
  } catch (err) {
    return jsonResponse({ error: "Invalid JSON body" }, 400);
  }

  const email = typeof body.email === "string" ? body.email.trim() : "";
  const src = typeof body.src === "string" && body.src ? body.src : undefined;

  if (!EMAIL_PATTERN.test(email)) {
    return jsonResponse({ error: "A valid email address is required" }, 400);
  }

  if (!env.RESEND_API_KEY) {
    // Fail loudly server-side rather than silently swallowing the
    // signup — matches the "no silent failures" rule in CLAUDE.md.
    console.error("[api/subscribe] RESEND_API_KEY is not configured");
    return jsonResponse(
      { error: "Signup is not configured yet. Please try again later." },
      500
    );
  }

  try {
    const result = await notifySignup({ email, src, apiKey: env.RESEND_API_KEY });
    if (!result.ok) {
      console.error("[api/subscribe] Signup notification failed:", result.status, result.detail);
      return jsonResponse({ error: "Could not complete signup" }, 502);
    }
    return jsonResponse({ ok: true });
  } catch (err) {
    console.error("[api/subscribe] Signup notification request failed:", err);
    return jsonResponse({ error: "Could not complete signup" }, 502);
  }
}

async function notifySignup({ email, src, apiKey }) {
  const response = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: "Learning with Squish <signups@squish.run>",
      to: [NOTIFY_TO],
      reply_to: email,
      subject: "New Learning with Squish signup",
      text:
        `New printables signup:\n\n` +
        `Email: ${email}\n` +
        `Source: ${src || "(none)"}\n`,
    }),
  });

  if (response.ok) {
    return { ok: true };
  }

  const detail = await response.text().catch(() => "");
  return { ok: false, status: response.status, detail };
}

function jsonResponse(data, status) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
