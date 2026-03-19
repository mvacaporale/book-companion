#!/bin/bash
# Set up GitHub secrets for Cloud Run deployment
# Run this after creating the GitHub repo

set -e

REPO="mvacaporale/book-companion"  # Update if different

echo "=== Setting up GitHub Secrets for Cloud Run Deployment ==="
echo ""

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) not installed. Install with: brew install gh"
    exit 1
fi

# Check if logged in
if ! gh auth status &> /dev/null; then
    echo "Please log in to GitHub CLI first: gh auth login"
    exit 1
fi

echo "This script will set up the following secrets in $REPO:"
echo "  - GEMINI_API_KEY"
echo "  - ANTHROPIC_API_KEY"
echo "  - GCP_SA_KEY (service account key for deployment)"
echo ""

# Set API keys from environment
if [ -n "$GEMINI_API_KEY" ]; then
    echo "Setting GEMINI_API_KEY..."
    echo "$GEMINI_API_KEY" | gh secret set GEMINI_API_KEY --repo "$REPO"
else
    echo "Warning: GEMINI_API_KEY not set in environment. Skipping."
fi

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "Setting ANTHROPIC_API_KEY..."
    echo "$ANTHROPIC_API_KEY" | gh secret set ANTHROPIC_API_KEY --repo "$REPO"
else
    echo "Warning: ANTHROPIC_API_KEY not set in environment. Skipping."
fi

echo ""
echo "=== Creating GCP Service Account for GitHub Actions ==="

PROJECT_ID="general-477905"
SA_NAME="github-actions-deploy"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Create service account (if not exists)
if ! gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" &> /dev/null; then
    echo "Creating service account: $SA_EMAIL"
    gcloud iam service-accounts create "$SA_NAME" \
        --project "$PROJECT_ID" \
        --display-name "GitHub Actions Deploy"
fi

# Grant necessary roles
echo "Granting roles to service account..."
for role in "roles/run.admin" "roles/storage.admin" "roles/iam.serviceAccountUser" "roles/cloudbuild.builds.builder"; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member "serviceAccount:$SA_EMAIL" \
        --role "$role" \
        --quiet
done

# Create and download key
echo "Creating service account key..."
KEY_FILE="/tmp/gcp-sa-key.json"
gcloud iam service-accounts keys create "$KEY_FILE" \
    --iam-account "$SA_EMAIL" \
    --project "$PROJECT_ID"

# Set as GitHub secret
echo "Setting GCP_SA_KEY secret..."
cat "$KEY_FILE" | gh secret set GCP_SA_KEY --repo "$REPO"

# Clean up
rm "$KEY_FILE"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Secrets configured for $REPO:"
echo "  ✓ GEMINI_API_KEY"
echo "  ✓ ANTHROPIC_API_KEY"
echo "  ✓ GCP_SA_KEY"
echo ""
echo "Push to main branch to trigger deployment!"
