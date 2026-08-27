# Next Steps — Claude App Integration

Handoff notes from the 2026-08-26 session. Start here in a new session.

## Where things stand

The MCP server is deployed on Cloud Run and healthy. The Claude app connector
was showing **zero tools**; root cause found and fixed in `8f70288` (see below).

**The fix is committed and pushed but NOT deployed** — the CI deploy job is
failing. See item A, which is the first thing to pick up.

**Service URLs** (both aliases for the same Cloud Run service):
- `https://book-companion-mcp-526741643129.us-central1.run.app/mcp` ← what the connector uses
- `https://book-companion-mcp-zpgj4t2a3a-uc.a.run.app/mcp`

**Project:** `general-477905` / `us-central1` / service `book-companion-mcp`

## What was fixed

### 1. Empty tool list in the Claude app (the main bug)

FastMCP ran in its default **stateful** mode. Claude.ai connectors call
`tools/list` without an `mcp-session-id`, which stateful mode rejects:

```
HTTP 400  {"code":-32600,"message":"Bad Request: Missing session ID"}
```

The connector then shows no tools, with no error and no auth prompt — it looks
like the server was never registered. Fixed with `stateless_http=True` in
`book_companion/mcp/server.py`.

Contributing factor: `--max-instances 2` with in-memory session state and no
Cloud Run session affinity, so `initialize` and `tools/list` could land on
different instances even for a well-behaved stateful client.

### 2. Drive auth diagnosability

`_get_credentials_from_env()` in `book_companion/google_drive/auth.py` swallowed
every exception with a bare `except: pass`, and all three Drive tools returned an
identical "not configured" message. It now logs which branch failed.

## Open items

### A. THE DEPLOY IS FAILING — the fix is not live yet

Commit `9093709` was pushed to `main` on 2026-08-27. CI ran:

- **Tests job: passed**
- **Deploy job: FAILED** at the `Deploy to Cloud Run` step
  (run `33025327160`; GCP auth and Cloud SDK setup both succeeded, the
  `gcloud run deploy --source .` command itself failed)

Production is still serving the **old** revision. It is healthy and not
regressed — Cloud Run keeps the last good revision when a deploy fails — but
stateless `tools/list` still returns 400, so the Claude app still shows no tools.

This reproduced on a second push (`788ecc9`, run `33027182380`) — same job, same
step, same result. It is a persistent break, not a transient blip.

The failure logs need repo admin rights, which were unavailable in the session
that pushed this. The check-run annotations expose only
`Process completed with exit code 1`, with no gcloud detail. **Get the real logs
first:**

```bash
gh run view 33027182380 --repo mvacaporale/book-companion --log-failed
```

The last successful deploy was `a7e8d5c` on 2026-05-18 — over three months
earlier. Since the only code delta is two Python files that were verified to
start and serve correctly locally, and `docs/` isn't even copied into the image
(see `.gcloudignore` and the Dockerfile's `COPY` lines), suspect an
infrastructure change during that gap rather than the code. Things to check in
the logs, roughly in order of likelihood:

- Cloud Build service account permissions (Google deprecated the legacy
  `PROJECT_NUMBER@cloudbuild.gserviceaccount.com` account)
- The `GCP_SA_KEY` secret expiring or its service account being disabled
- Artifact Registry repo missing/cleaned up
- Billing on `general-477905`

**Fastest path to getting the fix live** — deploy straight from the Mac,
bypassing CI entirely:

```bash
cd ~/Documents/Projects/claude/book-companion
git pull
gcloud run deploy book-companion-mcp \
  --project=general-477905 --region=us-central1 --source=.
```

Then verify:

```bash
curl -s -X POST https://book-companion-mcp-526741643129.us-central1.run.app/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' -i | head -3
```

- **200 + tool list** → fix is live. Reconnect the connector in the Claude app
  and confirm all 10 tools appear.
- **400 Missing session ID** → still the old revision.

Note that a manual `gcloud run deploy` does **not** set the env vars the workflow
passes (`GEMINI_API_KEY`, `GOOGLE_DRIVE_TOKEN_B64`, etc.). Deploying from source
preserves the existing revision's env vars, so this is normally fine — but
confirm with `gcloud run services describe` afterward if chat starts failing.

### B. Google Drive tools are still broken

`find_book_in_drive`, `load_book_from_drive`, `ingest_book_from_drive` all return
`{"error": "Google Drive not configured"}`. The other 7 tools work.

Follow `docs/drive-reauth-runbook.md`. Likely cause: the refresh token in the
`GOOGLE_DRIVE_TOKEN` secret expired because the OAuth consent screen is in
*Testing* status (Google expires those after 7 days), so every deploy re-injects
a dead token. **Publish the consent screen** or it recurs weekly.

With the logging fix deployed, the logs now name the exact failure:

```bash
gcloud run services logs read book-companion-mcp \
  --project=general-477905 --region=us-central1 --limit=100 | grep -i drive
```

### C. Re-test OAuth (possibly misdiagnosed)

CLAUDE.md blames Claude.ai OAuth bugs ([#11814], [#82]) and OAuth is disabled
because of them. That attribution looks wrong: the OAuth discovery endpoints all
return clean 404s, so the server correctly presents as no-auth, and the stateless
400 alone fully explains the empty tool list. Worth re-testing `MCP_OAUTH_ENABLED=true`
now that stateless mode is on — the original problem may not have been OAuth at all.

### D. Postgres backend is committed but dormant

`storage/database.py`, `pg_session_store.py`, `pg_vector_store.py` and
`scripts/migrate_to_postgres.py` all shipped in `57e340b`, but the deploy workflow
never sets `DATABASE_URL`, so `is_postgres_configured()` is False and production
still runs ChromaDB + GCS. Safe (psycopg2-binary is a main dep, no import crash),
just inactive. **Decision deferred — intentionally left dormant.**

Activating it needs: a Cloud SQL instance with pgvector, env vars in
`.github/workflows/deploy-simple.yml`, and running the migration script against
the 13 existing books.

### E. Test coverage gap

Suite is 41 tests, all passing, but 17% coverage. The new Postgres modules have
**0%** (252 statements), as does `mcp/server.py` (419). CLAUDE.md §4 requires
integration tests for anything crossing a DB/API/file boundary — worth closing
before any Postgres cutover.

### F. Housekeeping

Two stray files at the repo root look like accidental prompt spillover:
`Great, now take a minute to reflect on t.md` and `Here's a prompt you could use:.md`.
Left in place (untracked) rather than deleted unilaterally.

## Verified working

Called against the live deployment: `list_books` (13 books), `get_stats`,
`search_books`, `chat` (returned a correct cited answer from INSPIRED),
`get_book_index`, `get_chapter_summary`, `get_narratives`. RAG, embeddings, the
Gemini key, and the GCS-mounted store are all healthy.

## Environment note

The Linux box this ran on has **no `gcloud` and no `gh`**, so Cloud Run env vars
and CI status could not be inspected directly. `uv` is at `~/.local/bin/uv`, not
on PATH. Anything in the runbook using gcloud/gh needs to run on the Mac.
