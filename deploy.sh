#!/bin/bash
# Deploy book-companion MCP server to Cloud Run
set -e

# Configuration - UPDATE THESE
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="book-companion-mcp"
BUCKET_NAME="${PROJECT_ID}-bookrc-data"

echo "=== Book Companion MCP Server - Cloud Run Deployment ==="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo ""

# Check if gcloud is configured
if ! gcloud config get-value project &>/dev/null; then
    echo "Error: gcloud not configured. Run: gcloud auth login && gcloud config set project YOUR_PROJECT"
    exit 1
fi

# Step 1: Create GCS bucket for data persistence (if not exists)
echo "==> Step 1: Setting up Cloud Storage bucket..."
if ! gsutil ls "gs://${BUCKET_NAME}" &>/dev/null; then
    echo "Creating bucket: gs://${BUCKET_NAME}"
    gsutil mb -l "$REGION" "gs://${BUCKET_NAME}"
else
    echo "Bucket already exists: gs://${BUCKET_NAME}"
fi

# Step 2: Sync local bookrc data to GCS
echo ""
echo "==> Step 2: Syncing local data to Cloud Storage..."
LOCAL_BOOKRC="${BOOKRC_DB_PATH:-$HOME/.bookrc}"
if [ -d "$LOCAL_BOOKRC" ]; then
    echo "Syncing $LOCAL_BOOKRC to gs://${BUCKET_NAME}/"
    gsutil -m rsync -r "$LOCAL_BOOKRC" "gs://${BUCKET_NAME}/"
    echo "Data synced successfully!"
else
    echo "Warning: No local data found at $LOCAL_BOOKRC"
    echo "You'll need to ingest books after deployment."
fi

# Step 3: Build and deploy to Cloud Run
echo ""
echo "==> Step 3: Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 2 \
    --set-env-vars "GEMINI_API_KEY=${GEMINI_API_KEY}" \
    --set-env-vars "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}" \
    --set-env-vars "BOOKRC_DB_PATH=/data/bookrc" \
    --execution-environment gen2 \
    --add-volume "name=bookrc-data,type=cloud-storage,bucket=${BUCKET_NAME}" \
    --add-volume-mount "volume=bookrc-data,mount-path=/data/bookrc"

# Step 4: Get the service URL
echo ""
echo "==> Step 4: Getting service URL..."
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format 'value(status.url)')

echo ""
echo "=========================================="
echo "Deployment complete!"
echo ""
echo "MCP Server URL: ${SERVICE_URL}/sse"
echo ""
echo "To add to Claude.ai:"
echo "1. Go to Settings → Integrations → Add more"
echo "2. Name: book-companion"
echo "3. URL: ${SERVICE_URL}/sse"
echo "=========================================="
