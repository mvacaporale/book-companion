# Google Drive Re-Auth Runbook

Fixes `{"error": "Google Drive not configured"}` from `find_book_in_drive`,
`load_book_from_drive`, and `ingest_book_from_drive` on the deployed MCP server.

Run this on the Mac that has `~/.bookrc/google_credentials.json`.

## Why it breaks

The deploy workflow injects a refresh token from the `GOOGLE_DRIVE_TOKEN` GitHub
secret. Google expires refresh tokens after **7 days** while the OAuth consent
screen is in *Testing* status, so every deploy re-injects an already-dead token.

Step 4 stops the recurrence. Skip it and you will be back here next week.

## 1. Confirm the cause

```bash
gcloud run services describe book-companion-mcp \
  --project=general-477905 --region=us-central1 \
  --format='value(spec.template.spec.containers[0].env)' | tr ',' '\n' | grep -i drive
```

Then read the server's own diagnosis (requires the logging fix to be deployed):

```bash
gcloud run services logs read book-companion-mcp \
  --project=general-477905 --region=us-central1 --limit=100 | grep -i drive
```

You'll get one of:
- `neither GOOGLE_DRIVE_TOKEN_B64 nor GOOGLE_DRIVE_TOKEN is set` → env var missing, go to step 3
- `Failed to load credentials from ...` → token malformed
- `did not yield valid credentials (expired=True, has_refresh_token=True)` → refresh rejected, token revoked/expired

## 2. Mint a fresh token

```bash
rm -f ~/.bookrc/google_token.json
cd ~/Documents/Projects/claude/book-companion
uv run bookrc setup-drive          # opens a browser; approve Drive read-only

uv run bookrc find-book ""         # verify locally before deploying
```

## 3. Push the token to GitHub + Cloud Run

```bash
gh secret set GOOGLE_DRIVE_TOKEN \
  --repo mvacaporale/book-companion \
  < ~/.bookrc/google_token.json

# Update the live service now (avoids waiting on a deploy)
gcloud run services update book-companion-mcp \
  --project=general-477905 --region=us-central1 \
  --update-env-vars="GOOGLE_DRIVE_TOKEN_B64=$(base64 -i ~/.bookrc/google_token.json)"
```

Note: macOS `base64` has no `-w` flag and does not wrap by default; the
workflow's Linux runner uses `base64 -w 0`. Both produce a single line.

## 4. Stop the 7-day expiry (do not skip)

1. Open https://console.cloud.google.com/auth/audience?project=general-477905
2. Under **Audience**, find the "GTD Client" app in *Testing*
3. Click **Publish app** → confirm

The app only uses the `drive.readonly` scope, which is
[sensitive but not restricted](https://developers.google.com/identity/protocols/oauth2/scopes#drive),
so publishing does not require Google's security assessment for personal use.
Refresh tokens then stop expiring on a 7-day clock.

Alternative if you prefer not to publish: swap to a **service account** with the
Books folder shared to it. No refresh-token expiry at all, no consent screen.
Larger change — needs `google_drive/auth.py` to support service-account creds.

## 5. Verify end to end

```bash
curl -s -X POST https://book-companion-mcp-526741643129.us-central1.run.app/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"find_book_in_drive","arguments":{}}}' | grep '^data:'
```

Expect a JSON list of Drive files. If you still see `Google Drive not configured`,
go back to step 1 — the logs now say exactly which branch failed.
