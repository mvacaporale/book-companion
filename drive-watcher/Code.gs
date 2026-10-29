/**
 * Drive watcher for book-companion auto-ingestion.
 *
 * Runs in Google Apps Script under your Google account. Every 5 minutes it
 * fingerprints the list of files in the books folder; when the fingerprint
 * changes (a book was added, removed or replaced) it starts the
 * `drive-sync.yml` GitHub Actions workflow, which ingests whatever is new.
 * The workflow skips books that are already ingested, so an extra start is
 * harmless.
 *
 * Setup (once): see drive-watcher/README.md. In short, paste this file into a
 * new project at script.google.com, set the GITHUB_TOKEN script property, and
 * run `install` once.
 */

const FOLDER_ID = '1ip13wKCBGznT2S2HSESUseiNciJBMFCm';
const REPO = 'mvacaporale/book-companion';
const WORKFLOW = 'drive-sync.yml';
const FINGERPRINT_KEY = 'folderFingerprint';

/** Run once by hand: creates the 5-minute trigger (replacing any old one). */
function install() {
  ScriptApp.getProjectTriggers()
    .filter((t) => t.getHandlerFunction() === 'checkFolder')
    .forEach((t) => ScriptApp.deleteTrigger(t));
  ScriptApp.newTrigger('checkFolder').timeBased().everyMinutes(5).create();
  checkFolder();
}

/** Trigger handler: start the ingest workflow when the folder's contents change. */
function checkFolder() {
  const props = PropertiesService.getScriptProperties();
  const fingerprint = folderFingerprint_();
  if (fingerprint === props.getProperty(FINGERPRINT_KEY)) return;

  startWorkflow_(props.getProperty('GITHUB_TOKEN'));
  // Saved only after a successful start, so a failed start retries next tick.
  props.setProperty(FINGERPRINT_KEY, fingerprint);
}

function folderFingerprint_() {
  const ids = [];
  const files = DriveApp.getFolderById(FOLDER_ID).getFiles();
  while (files.hasNext()) ids.push(files.next().getId());
  ids.sort();
  const digest = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, ids.join(','));
  return Utilities.base64Encode(digest);
}

function startWorkflow_(token) {
  if (!token) throw new Error('Set the GITHUB_TOKEN script property (see drive-watcher/README.md)');
  const res = UrlFetchApp.fetch(
    `https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW}/dispatches`,
    {
      method: 'post',
      contentType: 'application/json',
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: 'application/vnd.github+json',
      },
      payload: JSON.stringify({ ref: 'main' }),
      muteHttpExceptions: true,
    },
  );
  if (res.getResponseCode() !== 204) {
    throw new Error(`GitHub dispatch failed: ${res.getResponseCode()} ${res.getContentText()}`);
  }
}
