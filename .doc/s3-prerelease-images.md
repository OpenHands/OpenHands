# GitHub Workflow to Publish Prerelease Images

- [ALL-3467](https://linear.app/all-hands-ai/issue/ALL-3467)
- [ALL-2610](https://linear.app/all-hands-ai/issue/ALL-2610)

## 1. Introduction

### 1.1 Problem Statement

Large banks, insurance companies, etc. often cannot use images from public
internet and must instead consume them from a local image repo where images are
quarantined before approval and use. When supporting such clients, we often need
to build custom images and share them via an s3 bucket with presigned url. Our
internal process for this is currently long multistep process that could easily
be automated.

### 1.2 Proposed Solution

Two scenarios:

**Regular SaaS Release** - A new job is added to the release workflow so that
for every SaaS release tarballs are pushed to S3. Presigned URLs need to be
published for our customers.

**Special Troubleshooting Builds** - Add GitHub a workflow to the `deploy` repo
which is manually triggered to build and push to S3 from a named feature branch.
The FDE would then make changes on the feature branch, run the special workflow
and copy and paste the URL from the GitHub action to share with the client.

## 2. Additional Context

### 2.1 Images

Three images are required for self-hosted, in early V1 from two repos:

   1. `enterprise-server` built from
      ([All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands))
   2. `runtime:{version}-nikolaik` built from
      ([All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands))
   3. `runtime-api` built from
      ([All-Hands-AI/runtime-api](https://github.com/All-Hands-AI/runtime-api))

We expect that eventually the runtime microservice will be deprecated but for
now it means that we will have workflows in these two repos for building the
complete image set.

### 2.2 Current Process

- In the `OpenHands` repository images are built for feature branches that have
  an associated PR. Images are tagged with feature branch name or or number:
  e.g. `pr-123`.
- In the `runtime-api` repository images are built on any feature branch.
- Production SaaS releases are kicked off when a tag using semver format is
  pushed to the OpenHands repo. Deployment of SaaS is delegated to workflows in
  the `deploy` repo.
- When a release includes changes to the `runtime-api` image, the `runtime-api`
  repo needs to be tagged with the release version as well.

## 3. Developer Experience

### 3.1 Future State Process - Typical Releases

S3 is now automatically updated when the image repos are on tagging of the
release. No process change for the release engineer or developer.

### 3.2 Future State Process - Troubleshooting Releases

Engineers will go to a new Github Workflow in `OpenHands` and/or `runtime-api`
to manually trigger a build supplying the target branch name. The image(s) are
built and published to S3. An S3 link in the build log may be shared with the
customer.

### 3.3 HTML Report Generation

For both typical releases and troubleshooting releases, the workflow will
automatically generate a comprehensive HTML report containing all download links
and usage instructions. This report will be:

- **Attached as a build artifact** - Available for download from the GitHub Actions
  run page
- **Self-contained** - Includes all necessary download links with presigned URLs
- **Customer-ready** - Professional formatting with clear instructions for loading
  Docker images
- **Comprehensive** - Contains build information, file sizes, expiration times,
  and step-by-step usage instructions

The HTML report serves as a complete package that can be downloaded and shared
directly with customers, eliminating the need to manually copy individual URLs
from build logs or S3 console.

## 4. Solution Design

### 4.1 S3 Bucket Structure

Images in the S3 bucket will be namespaced by branch name and commit SHA. If
engineers use the same branch name across repos, images for a particular issue
can be stored together. Since multiple builds could be produced from the same
branch, the commit SHA is used within the branch folder to ensure uniqueness and
traceability to the exact code being built.

This approach aligns with the standard GHCR workflow which uses commit SHAs for
image tagging, ensuring consistency across our build processes.

Example:

```plaintext
s3://prerelease-bucket/
├── jps/publish-prerelease-images/
│   ├── abc1234/
│   │   ├── openhands:latest.tar
│   │   ├── openhands:main-abc1234.tar
│   │   ├── enterprise:latest.tar
│   │   ├── enterprise:main-abc1234.tar
│   │   ├── runtime-api:latest.tar
│   │   └── runtime-api:main-abc1234.tar
│   └── def5678/
│       ├── openhands:latest.tar
│       ├── enterprise:latest.tar
│       └── runtime-api:latest.tar
└── mj/security-patch-123/
    └── ghi9012/
        ├── openhands:hotfix-security.tar
        └── enterprise:hotfix-security.tar
```

### 4.2 Security Architecture

The S3 publishing workflow implements a two-tier security model using AWS IAM roles
to minimize credential exposure while maintaining functionality.

#### 4.2.1 Service Account and Read-Only Role

**Service Account**: The primary `service` IAM user with write permissions for S3
operations (upload, delete, etc.). This account is used for uploading Docker images
to S3.

**Read-Only Role**: A separate IAM role (`S3PresignedURLRole`) with minimal read-only
permissions used exclusively for generating presigned URLs.

#### 4.2.2 Security Challenge and Solution

**Initial Problem**: When using a single service account for both uploads and presigned
URL generation, the `AWS_ACCESS_KEY_ID` was automatically redacted from GitHub Actions
logs as a secret. This made it impossible to emit presigned URLs in build logs since
the access key ID is embedded in the URL structure.

**Solution**: Role assumption pattern where:

1. Service account uploads images to S3 (write operations)
2. Service account assumes the read-only role for presigned URL generation
3. Presigned URLs are generated with temporary read-only credentials
4. The temporary access key ID can be safely emitted in logs since it has no write permissions

#### 4.2.3 Implementation Details

The workflow follows this secure sequence:

```bash
# Step 1: Upload using service account (write permissions)
aws s3 cp image.tar.gz s3://bucket/path/

# Step 2: Assume read-only role
aws sts assume-role --role-arn "arn:aws:iam::ACCOUNT:role/S3PresignedURLRole"

# Step 3: Generate presigned URL with temporary read-only credentials
aws s3 presign s3://bucket/path/image.tar.gz --expires-in 604800
```

**Security Benefits**:

- Presigned URLs contain only temporary read-only access keys
- No additional secrets or variables required beyond the service account
- Temporary credentials expire automatically (typically 1 hour)
- If presigned URLs are exposed in logs, only read-only access is compromised

#### 4.2.4 Infrastructure Documentation

The complete AWS infrastructure configuration is documented in the `infra` repository
under `doc/manually-managed.md`, including:

- IAM user and policy definitions
- Read-only role configuration
- S3 bucket setup and permissions
- Complete Terraform resource definitions

### 4.3 Composite Actions Architecture

The S3 publishing workflow is built using GitHub Actions composite actions to promote
reusability, maintainability, and consistency across different workflow types. The
composite actions are organized as follows:

#### 4.3.1 Core Actions

**`publish-image-to-s3`** - Main action for publishing individual Docker images:

- Pulls images from GHCR
- Exports to compressed tarballs
- Uploads to S3 with proper folder structure
- Generates presigned URLs using read-only IAM role
- Returns tab-delimited data for report generation

**`generate-presigned-url`** - Security-focused URL generation:

- Assumes read-only IAM role for minimal permissions
- Generates presigned URLs with configurable expiration
- Validates URL format and detects encoding issues
- Provides comprehensive error handling and logging

**`s3-publish-report`** - HTML report generation:

- Processes tab-delimited image data
- Uses Python template engine for reliable processing
- Generates professional HTML reports with styling
- Handles file size formatting and usage instructions

#### 4.3.2 Data Flow

```text
Workflow → publish-image-to-s3 → generate-presigned-url
    ↓
Tab-delimited data → s3-publish-report → HTML artifact
```

This architecture ensures that both manual troubleshooting builds and automated
release workflows can leverage the same underlying actions while maintaining
security and reliability.

### 4.4 Manual Prerelease Build Workflow

#### 4.4.1 Prerequisites

The following infrastructure and configuration must be in place before
implementing the manual prerelease build workflow:

**AWS S3 Configuration:**

- S3 bucket created and configured for prerelease image storage
- Bucket name stored as GitHub repository secret: `S3_PRERELEASE_BUCKET`
- AWS credentials configured with appropriate permissions:
  - `s3:PutObject` - Upload tarball files
  - `s3:PutObjectAcl` - Set object permissions
  - `s3:GetObject` - Generate presigned URLs
  - `s3:ListBucket` - List objects for S3 management
- AWS credentials stored as GitHub repository secrets:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_REGION` (optional, defaults to us-east-1)

**GitHub Repository Configuration:**

- Workflow permissions configured to allow:
  - `contents: read` - Checkout repository code
  - `packages: write` - Build and push Docker images to GHCR
  - `actions: write` - Update workflow status
- Manual workflow dispatch enabled
- Branch protection rules configured to allow workflow runs from feature
  branches

**Docker Registry Access:**

- GitHub Container Registry (GHCR) access configured
- `GITHUB_TOKEN` secret available for GHCR authentication
- Docker images built and available in GHCR before S3 export

#### 4.4.2 Workflow Implementation

The manual prerelease build workflow will be implemented as a new GitHub Actions
workflow file: `.github/workflows/s3-prerelease-build.yml`

**Workflow Triggers:**

- Manual dispatch only (`workflow_dispatch`)
- Required input: `branch_name` (target branch to build from)
- Optional input: `pr` (PR number to build from, uses PR's branch if provided)

**Workflow Jobs:**

1. **Build Images Job** (`build-images`)
   - Determine target branch (from PR or branch_name input)
   - Checkout specified branch
   - Set up Docker Buildx and QEMU for multi-platform builds
   - Login to GHCR
   - Build enterprise-server image
   - Build runtime images (both nikolaik and ubuntu flavors)
   - Export all images to tarball files

2. **Publish to S3 Job** (`publish-to-s3`)
   - Configure AWS CLI with provided credentials
   - Create S3 folder structure: `{branch-name}/{commit-sha}/`
   - Upload tarball files to S3
   - Generate presigned URLs for each image
   - Generate build information for reporting
   - Output presigned URLs for easy sharing

**Sample Workflow Structure:**

```yaml
name: S3 Prerelease Build

on:
  workflow_dispatch:
    inputs:
      branch_name:
        description: 'Branch name to build from'
        required: true
        type: string
      pr:
        description: 'PR number to build from (uses PR branch if provided)'
        required: false
        type: string

jobs:
  build-images:
    runs-on: ubuntu-latest
    steps:
      - name: Determine target branch
        id: determine-branch
        run: |
          if [[ -n "${{ inputs.pr }}" ]]; then
            # Get PR branch name from GitHub API
            BRANCH_NAME=$(gh pr view ${{ inputs.pr }} --json headRefName --jq '.headRefName')
            echo "branch_name=$BRANCH_NAME" >> $GITHUB_OUTPUT
            echo "Using PR ${{ inputs.pr }} branch: $BRANCH_NAME"
          else
            echo "branch_name=${{ inputs.branch_name }}" >> $GITHUB_OUTPUT
            echo "Using specified branch: ${{ inputs.branch_name }}"
          fi
      
      - name: Checkout branch
        uses: actions/checkout@v4
        with:
          ref: ${{ steps.determine-branch.outputs.branch_name }}
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.repository_owner }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build enterprise-server image
        run: |
          # Build and export enterprise-server image
          docker buildx build --platform linux/amd64 \
            -t enterprise-server:latest \
            -f enterprise/Dockerfile . \
            --load
          docker save enterprise-server:latest > enterprise-server-latest.tar
      
      - name: Build runtime images
        run: |
          # Build nikolaik and ubuntu runtime images
          # Export to tarballs for S3 upload
          # ... (detailed implementation)

  publish-to-s3:
    needs: build-images
    runs-on: ubuntu-latest
    steps:
      - name: Configure AWS CLI
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION || 'us-east-1' }}
      
      - name: Upload images to S3
        run: |
          # Upload tarballs to S3 with proper structure
          # Generate presigned URLs
          # Generate build information
          # ... (detailed implementation)
      
      - name: Output presigned URLs
        run: |
          # Display presigned URLs for easy copying
          echo "## Prerelease Images Ready"
          echo "Branch: ${{ steps.determine-branch.outputs.branch_name }}"
          echo "Commit: ${{ github.sha }}"
          echo ""
          echo "### Download Links:"
          # ... (output presigned URLs)
```

### 4.5 Enhanced Release Workflow with S3 Publishing

#### 4.5.1 Prerequisites

The following infrastructure and configuration must be in place before
implementing the enhanced release workflow:

**AWS S3 Configuration:**

- S3 bucket created and configured for prerelease image storage (same as 4.4.1)
- Bucket name stored as GitHub repository secret: `S3_PRERELEASE_BUCKET`
- AWS credentials configured with appropriate permissions (same as 4.4.1)
- AWS credentials stored as GitHub repository secrets:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_REGION` (optional, defaults to us-east-1)

**GitHub Repository Configuration:**

- Workflow permissions configured to allow:
  - `contents: read` - Checkout repository code
  - `packages: write` - Build and push Docker images to GHCR
  - `actions: write` - Update workflow status
- Existing `ghcr-build.yml` workflow must be functional
- Version tagging process must be in place for triggering releases

**Docker Registry Access:**

- GitHub Container Registry (GHCR) access configured
- `GITHUB_TOKEN` secret available for GHCR authentication
- Existing build jobs (`ghcr_build_enterprise` and `ghcr_build_runtime`) must be
  working

#### 4.5.2 Workflow Implementation

The enhanced release workflow will extend the existing `ghcr-build.yml` workflow
by adding S3 publishing steps to the existing build jobs.

**Workflow Modifications:**

1. **Extend `ghcr_build_enterprise` Job**
   - Add S3 publishing steps after successful GHCR push
   - Only trigger S3 upload on version tag pushes (not PRs or feature branches)
   - Export enterprise-server image to tarball
   - Upload to S3 under `releases/{version}/` structure

2. **Extend `ghcr_build_runtime` Job**
   - Add S3 publishing steps after successful GHCR push
   - Only trigger S3 upload on version tag pushes
   - Export both nikolaik and ubuntu runtime images to tarballs
   - Upload to S3 under `releases/{version}/` structure

3. **Add New Job: `s3-publish-release`**
   - Depends on both `ghcr_build_enterprise` and `ghcr_build_runtime` jobs
   - Only runs on version tag pushes
   - Generate release information
   - Output release information and S3 links

**Sample Workflow Extensions:**

```yaml
# Add to existing ghcr_build_enterprise job
- name: Export enterprise-server image for S3
  if: startsWith(github.ref, 'refs/tags/v')
  run: |
    # Load the image that was just built and pushed to GHCR
    docker pull ghcr.io/${{ env.REPO_OWNER }}/enterprise-server:${{ github.ref_name }}
    docker tag ghcr.io/${{ env.REPO_OWNER }}/enterprise-server:${{ github.ref_name }} enterprise-server:${{ github.ref_name }}
    docker save enterprise-server:${{ github.ref_name }} > enterprise-server-${{ github.ref_name }}.tar
    docker save enterprise-server:latest > enterprise-server-latest.tar

- name: Upload enterprise-server to S3
  if: startsWith(github.ref, 'refs/tags/v')
  uses: aws-actions/configure-aws-credentials@v4
  with:
    aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    aws-region: ${{ secrets.AWS_REGION || 'us-east-1' }}
  run: |
    aws s3 cp enterprise-server-${{ github.ref_name }}.tar s3://${{ secrets.S3_PRERELEASE_BUCKET }}/releases/${{ github.ref_name }}/enterprise-server:${{ github.ref_name }}.tar
    aws s3 cp enterprise-server-latest.tar s3://${{ secrets.S3_PRERELEASE_BUCKET }}/releases/${{ github.ref_name }}/enterprise-server:latest.tar

# Add to existing ghcr_build_runtime job
- name: Export runtime images for S3
  if: startsWith(github.ref, 'refs/tags/v')
  run: |
    # Load and export both nikolaik and ubuntu runtime images
    docker pull ghcr.io/${{ env.REPO_OWNER }}/runtime:${{ github.ref_name }}-${{ matrix.base_image.tag }}
    docker tag ghcr.io/${{ env.REPO_OWNER }}/runtime:${{ github.ref_name }}-${{ matrix.base_image.tag }} runtime-${{ matrix.base_image.tag }}:${{ github.ref_name }}
    docker save runtime-${{ matrix.base_image.tag }}:${{ github.ref_name }} > runtime-${{ matrix.base_image.tag }}-${{ github.ref_name }}.tar
    docker save runtime-${{ matrix.base_image.tag }}:latest > runtime-${{ matrix.base_image.tag }}-latest.tar

- name: Upload runtime images to S3
  if: startsWith(github.ref, 'refs/tags/v')
  uses: aws-actions/configure-aws-credentials@v4
  with:
    aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    aws-region: ${{ secrets.AWS_REGION || 'us-east-1' }}
  run: |
    aws s3 cp runtime-${{ matrix.base_image.tag }}-${{ github.ref_name }}.tar s3://${{ secrets.S3_PRERELEASE_BUCKET }}/releases/${{ github.ref_name }}/runtime-${{ matrix.base_image.tag }}:${{ github.ref_name }}.tar
    aws s3 cp runtime-${{ matrix.base_image.tag }}-latest.tar s3://${{ secrets.S3_PRERELEASE_BUCKET }}/releases/${{ github.ref_name }}/runtime-${{ matrix.base_image.tag }}:latest.tar

# New job for release management
s3-publish-release:
  name: Publish Release to S3
  runs-on: ubuntu-latest
  needs: [ghcr_build_enterprise, ghcr_build_runtime]
  if: startsWith(github.ref, 'refs/tags/v')
  steps:
    - name: Configure AWS CLI
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ secrets.AWS_REGION || 'us-east-1' }}
    
    - name: Generate release information
      run: |
        # Generate release information
        echo "## Release ${{ github.ref_name }}"
        echo "**Release Date**: $(date -u +"%Y-%m-%d %H:%M:%SZ")"
        echo "**Commit**: ${{ github.sha }}"
        echo "**Images**: enterprise-server, runtime-nikolaik, runtime-ubuntu"
    
    - name: Output release information
      run: |
        echo "## Release ${{ github.ref_name }} Published to S3"
        echo "S3 Location: s3://${{ secrets.S3_PRERELEASE_BUCKET }}/releases/${{ github.ref_name }}/"
        echo "Images available for download from S3"
```

**Key Features:**

- **Conditional S3 Upload**: Only uploads to S3 on version tag pushes using `if:
  startsWith(github.ref, 'refs/tags/v')`
- **Maintains GHCR Publishing**: Existing GHCR publishing remains unchanged
- **Version-based Structure**: Uses version tags for S3 folder organization
- **Release Information**: Automatically generates release information
- **Backward Compatibility**: No changes to existing workflow behavior

## 5. Open Questions

### 5.1 Where does the push to S3 live?

- Delegate to `deploy` repo GitHub action or do S3 push directly from OpenHands
  repo?

### 5.2 Secrets

- How do we secure AWS S3 secret? In OpenHands repo GitHub Actions Secrets or
  via another mechanism?

### 5.3 White List

- How do we secure who can trigger these builds?

## 5. Appendix

### 5.1 Existing Jobs

- [.github/workflows/ghcr-build.yml](https://github.com/All-Hands-AI/OpenHands/blob/main/.github/workflows/ghcr-build.yml#L169)

And the runtime build job (not to be confused with runtime-api, a different
component)

- [.github/workflows/ghcr-build.yml](https://github.com/All-Hands-AI/OpenHands/blob/main/.github/workflows/ghcr-build.yml#L89)

Runtime image builds in two flavors, Nikolaik (based on debian) and Ubuntu.
Probably we should publish both for your purposes. The Ubuntu flavor was created
for a partner (maybe Modern health, not sure), in order to get CVEs down.

It's thought that in the future more flavors will be desired but there are
various difficulties with doing that (can of worms omitted for now).

### 5.2 Chuck's Manual S3 Upload Script

The following script demonstrates the current manual process for uploading Docker images to S3. This script will be replaced by the automated workflows described in sections 4.2 and 4.3.

```bash
# Pull images from GHCR
docker pull ghcr.io/all-hands-ai/deploy:sha-5fc93b8
docker pull ghcr.io/all-hands-ai/runtime:174c69174410a871e2064fc41c7351add7543dd5-nikolaik
docker pull ghcr.io/all-hands-ai/runtime-api:sha-9c9a3ce

# Export images to compressed tarballs (run in parallel)
docker save ghcr.io/all-hands-ai/deploy:sha-5fc93b8 | gzip > deploy_5fc93b8.tar.gz &
docker save ghcr.io/all-hands-ai/runtime:174c69174410a871e2064fc41c7351add7543dd5-nikolaik | gzip > runtime_174c69174410a871e2064fc41c7351add7543dd5.tar.gz &
docker save ghcr.io/all-hands-ai/runtime-api:sha-9c9a3ce | gzip > runtime_api_9c9a3ce.tar.gz &

# Wait for all background jobs to complete
wait

# Set AWS credentials (export these environment variables)
# export AWS_ACCESS_KEY_ID=your_access_key
# export AWS_SECRET_ACCESS_KEY=your_secret_key
# export AWS_DEFAULT_REGION=us-east-1

# Upload to S3
aws s3 cp deploy_5fc93b8.tar.gz s3://jpmc-images
aws s3 cp runtime_174c69174410a871e2064fc41c7351add7543dd5.tar.gz s3://jpmc-images
aws s3 cp runtime_api_9c9a3ce.tar.gz s3://jpmc-images
```

**Notes:**

- This script demonstrates the current manual process for creating prerelease images
- Images are pulled from GHCR, exported to compressed tarballs, and uploaded to S3
- The `&` operator runs the `docker save` commands in parallel for efficiency
- The `wait` command ensures all background jobs complete before proceeding
- AWS credentials must be set as environment variables before running the upload commands
- This process will be automated by the workflows described in sections 4.2 and 4.3
