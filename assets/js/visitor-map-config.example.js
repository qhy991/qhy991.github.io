/**
 * Visitor Map — JSONBin.io global aggregation config
 *
 * Local dev: fill in jsonBinId + jsonBinAccessKey below for testing.
 * Production: GitHub Actions injects secrets on deploy (see .github/workflows/pages.yml).
 *
 * Setup (one-time, ~3 min):
 *   1. Register at https://jsonbin.io (free)
 *   2. Create a bin with content: { "visitors": [] }
 *   3. Copy Bin ID + Access Key (X-Master-Key)
 *   4. Add GitHub repo Secrets:
 *        JSONBIN_BIN_ID      → Bin ID
 *        JSONBIN_ACCESS_KEY  → X-Master-Key
 *      (Settings → Secrets and variables → Actions)
 *   5. Push to master — next deploy enables global sync automatically
 */
window.VISITOR_MAP_CONFIG = {
    jsonBinId: '',
    jsonBinAccessKey: '',
};
