# Drive watcher

A Google Apps Script that checks the Drive books folder every 5 minutes. When
its contents change, it starts the `drive-sync.yml` workflow, which ingests any
new books into Neon. It runs on Google's servers under your account, free.

The script isn't deployed from this repo automatically. If `Code.gs` changes,
paste it into the Apps Script project again.

## Setup (once)

1. **GitHub token:** create a fine-grained personal access token at
   <https://github.com/settings/personal-access-tokens/new>. Limit it to the
   `mvacaporale/book-companion` repository, with **Actions: Read and write**
   permission (and nothing else).
2. **Apps Script project:** go to <https://script.google.com>, choose
   **New project**, and name it "book-companion drive watcher".
   - Replace the contents of `Code.gs` with this folder's `Code.gs`.
   - Optional, for the narrowest permissions: in **Project Settings**, tick
     "Show appsscript.json manifest file in editor", then replace that file's
     contents with this folder's `appsscript.json`.
   - In **Project Settings → Script properties**, add `GITHUB_TOKEN` and set it
     to the token from step 1.
3. **Start it:** in the editor, pick the `install` function and click **Run**,
   then approve the permission prompt. This creates the 5-minute trigger and
   runs a first check, which starts one workflow run.

## Monitoring

- Apps Script emails you when a trigger run fails, for example if the token
  expires. Runs are listed under **Executions** in the script editor.
- GitHub emails you when the ingest workflow fails. Runs are in the repo's
  **Actions** tab.
