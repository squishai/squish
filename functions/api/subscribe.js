/**
 * POST /api/subscribe
 *
 * Cloudflare Pages Function. If docs/ is deployed as a Cloudflare Pages
 * site (or this file is copied into a project deployed that way), this
 * route is live automatically — no separate server to run.
 *
 * Wired to Buttondown (https://buttondown.email) by default because its
 * API is a single authenticated POST with no extra setup. Swap the
 * `subscribeToESP` body for Mailchimp / ConvertKit / anything else —
 * the request/response contract with the page's fetch("/api/subscribe")
 * call (JSON in: { email, src }, 2xx = success) is what matters, not
 * which ESP is behind it.
 *
 * Required Cloudflare Pages environment variable:
 *   BUTTONDOWN_API_KEY  — set in the Pages project settings, never
 *                          committed to the repo.
 *
 * Konjo constraint: no silent failures — every non-2xx path returns a
 * real error status and logs the cause server-side.
 */

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

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

  if (!env.BUTTONDOWN_API_KEY) {
    // Fail loudly server-side rather than silently swallowing the
    // signup — matches the "no silent failures" rule in CLAUDE.md.
    console.error("[api/subscribe] BUTTONDOWN_API_KEY is not configured");
    return jsonResponse(
      { error: "Signup is not configured yet. Please try again later." },
      500
    );
  }

  try {
    const result = await subscribeToESP({ email, src, apiKey: env.BUTTONDOWN_API_KEY });
    if (!result.ok) {
      console.error("[api/subscribe] ESP rejected signup:", result.status, result.detail);
      return jsonResponse({ error: "Could not complete signup" }, 502);
    }
    return jsonResponse({ ok: true });
  } catch (err) {
    console.error("[api/subscribe] ESP request failed:", err);
    return jsonResponse({ error: "Could not complete signup" }, 502);
  }
}

async function subscribeToESP({ email, src, apiKey }) {
  const response = await fetch("https://api.buttondown.email/v1/subscribers", {
    method: "POST",
    headers: {
      Authorization: `Token ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      email_address: email,
      // Tag the subscriber with which book/QR code drove the signup so
      // it stays visible in the ESP dashboard, matching the ?src=
      // attribution already captured client-side.
      tags: src ? [`squish-books:${src}`] : ["squish-books"],
      referrer_url: "https://squish.run/books",
    }),
  });

  if (response.ok || response.status === 201) {
    return { ok: true };
  }

  // Buttondown returns 400 for "already subscribed" — treat that as a
  // success from the user's point of view, they're on the list either
  // way, but still log it for visibility.
  const detail = await response.text().catch(() => "");
  if (response.status === 400 && /already/i.test(detail)) {
    return { ok: true };
  }

  return { ok: false, status: response.status, detail };
}

function jsonResponse(data, status) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
