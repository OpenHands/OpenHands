# OpenHands Deployment on Railway

This guide will help you deploy OpenHands on [Railway](https://railway.app).

## Prerequisites

- A Railway account ([Sign up here](https://railway.app))
- GitHub account (for connecting your repository)
- An OpenAI API key or other LLM provider API key

## Quick Deploy

### Option 1: Deploy from GitHub (Recommended)

1. **Fork or Clone this Repository**
   - Fork this repository to your GitHub account
   - Or push your local changes to a GitHub repository

2. **Create a New Project on Railway**
   - Go to [Railway Dashboard](https://railway.app/dashboard)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your OpenHands repository

3. **Configure Environment Variables**

   Railway will automatically detect the `railway.toml` configuration. Set these required environment variables in the Railway dashboard:

   **Required Variables:**
   ```bash
   # LLM Configuration (choose one provider)
   LLM_API_KEY=your-openai-api-key-here
   LLM_MODEL=gpt-4o
   LLM_BASE_URL=https://api.openai.com/v1  # Optional, for custom endpoints

   # Port (Railway provides this automatically)
   PORT=3000

   # Workspace Configuration
   WORKSPACE_BASE=/opt/workspace_base
   WORKSPACE_MOUNT_PATH=/opt/workspace_base

   # File Store Configuration
   FILE_STORE=local
   FILE_STORE_PATH=/.openhands

   # Runtime Configuration
   SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.openhands.dev/openhands/runtime:1.2-nikolaik

   # Security (generate a random secret)
   JWT_SECRET=your-random-secret-here-change-this
   ```

   **Optional Variables:**
   ```bash
   # Enable debugging
   DEBUG=false

   # Maximum iterations
   MAX_ITERATIONS=500

   # Maximum budget per task (0 = no limit)
   MAX_BUDGET_PER_TASK=0.0

   # OpenTelemetry (if using observability)
   OTEL_EXPORTER_OTLP_ENDPOINT=your-otlp-endpoint
   ```

4. **Deploy**
   - Railway will automatically build and deploy your application
   - The build process uses the Dockerfile at `containers/app/Dockerfile`
   - Deployment typically takes 3-5 minutes

5. **Access Your Application**
   - Once deployed, Railway will provide a public URL
   - Click on the deployment to get your URL (e.g., `https://your-app.railway.app`)

### Option 2: Deploy from Railway CLI

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   # or
   brew install railway
   ```

2. **Login to Railway**
   ```bash
   railway login
   ```

3. **Initialize Project**
   ```bash
   railway init
   ```

4. **Set Environment Variables**
   ```bash
   railway variables set LLM_API_KEY=your-api-key-here
   railway variables set LLM_MODEL=gpt-4o
   railway variables set JWT_SECRET=your-random-secret
   railway variables set WORKSPACE_BASE=/opt/workspace_base
   railway variables set FILE_STORE=local
   railway variables set FILE_STORE_PATH=/.openhands
   ```

5. **Deploy**
   ```bash
   railway up
   ```

## Important Notes

### Limitations on Railway

⚠️ **Docker Socket Access**: Railway does not provide Docker socket access (`/var/run/docker.sock`). This means:

- The default Docker runtime for sandboxing **will not work** on Railway
- You need to use an alternative runtime

### Recommended Runtime Solutions

Choose one of these alternatives for running sandboxed code:

#### 1. **Use E2B Runtime** (Recommended for Railway)

E2B provides cloud-based sandboxes that work perfectly with Railway.

```bash
# Set these environment variables
railway variables set RUNTIME=e2b
railway variables set E2B_API_KEY=your-e2b-api-key
```

Get an E2B API key at: https://e2b.dev

#### 2. **Use Modal Runtime**

Modal provides serverless containers for code execution.

```bash
railway variables set RUNTIME=modal
railway variables set MODAL_API_KEY=your-modal-api-key
railway variables set MODAL_SECRET=your-modal-secret
```

Get Modal credentials at: https://modal.com

#### 3. **Use Remote Docker Host**

If you have a remote Docker host with exposed Docker API:

```bash
railway variables set RUNTIME=docker
railway variables set DOCKER_HOST=tcp://your-docker-host:2375
```

⚠️ **Security Warning**: Only use this with a properly secured Docker daemon.

### Storage Considerations

Railway provides ephemeral storage, which means:

- Files are lost when the container restarts
- For persistent storage, consider:
  - Adding Railway Volume (persistent storage)
  - Using external storage like S3
  - Configuring PostgreSQL for conversation history

To use PostgreSQL for persistence:

```bash
# Add PostgreSQL service in Railway, then set:
railway variables set DATABASE_URL=postgresql://user:password@host:port/dbname
```

### Performance Tips

1. **Use Smaller Models**: Start with `gpt-4o-mini` to reduce costs and latency
2. **Set Iteration Limits**: Configure `MAX_ITERATIONS` to prevent runaway costs
3. **Enable Budget Limits**: Set `MAX_BUDGET_PER_TASK` to control API spending
4. **Monitor Logs**: Use Railway's built-in logging to track errors

## Configuration Files

This deployment uses:

- **`railway.toml`**: Railway-specific configuration
- **`.railwayignore`**: Files excluded from deployment
- **`containers/app/Dockerfile`**: Application container definition
- **`config.template.toml`**: Configuration template (reference only)

## Troubleshooting

### Build Failures

If the build fails:

1. Check Railway build logs for specific errors
2. Ensure all required files are not in `.railwayignore`
3. Verify Dockerfile path in `railway.toml`

### Runtime Errors

**"Cannot connect to Docker daemon"**
- You're using Docker runtime on Railway (not supported)
- Switch to E2B or Modal runtime (see above)

**"Module not found" errors**
- Ensure all Python dependencies are in `pyproject.toml`
- Check that Poetry is installing correctly in the Dockerfile

**Port binding errors**
- Railway automatically provides `$PORT` environment variable
- The `railway.toml` uses `$PORT` in the start command

### Application Not Responding

1. Check the Health check endpoint (if configured)
2. Review application logs in Railway dashboard
3. Verify environment variables are set correctly
4. Ensure `PORT` is being used correctly (Railway assigns it automatically)

## Cost Considerations

Railway pricing includes:

- **Free Tier**: $5 credit per month
- **Pro Plan**: Pay for what you use

OpenHands on Railway costs:

- Railway infrastructure: ~$5-20/month (depending on usage)
- LLM API costs: Variable (depends on usage)
- Runtime costs (E2B/Modal): ~$0.10-1.00 per hour of active usage

💡 **Tip**: Use the `MAX_BUDGET_PER_TASK` setting to control LLM costs.

## Support

- **OpenHands Documentation**: https://docs.openhands.dev
- **Railway Documentation**: https://docs.railway.app
- **OpenHands GitHub**: https://github.com/OpenHands/OpenHands
- **Railway Discord**: https://discord.gg/railway

## Advanced Configuration

### Custom Dockerfile

To use a custom Dockerfile, update `railway.toml`:

```toml
[build]
dockerfilePath = "./path/to/your/Dockerfile"
```

### Health Checks

Uncomment in `railway.toml`:

```toml
[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 100
```

Then implement a `/health` endpoint in your application.

### Multiple Services

For a production setup with PostgreSQL:

1. Add PostgreSQL service in Railway
2. Link it to your OpenHands service
3. Update environment variables to use the database

```bash
railway variables set DATABASE_URL=${{Postgres.DATABASE_URL}}
```

### Custom Domain

1. Go to your service settings in Railway
2. Click "Settings" → "Domains"
3. Add your custom domain
4. Update DNS records as instructed

## Next Steps

After successful deployment:

1. **Test the Application**: Visit your Railway URL
2. **Configure LLM Settings**: Adjust model and parameters as needed
3. **Set Up Monitoring**: Consider adding error tracking (e.g., Sentry)
4. **Enable Authentication**: Configure JWT and user management
5. **Optimize Costs**: Monitor usage and adjust settings

## License

OpenHands is licensed under the MIT License for core functionality. See LICENSE file for details.

---

**Happy Deploying! 🚂**

If you encounter issues, please open an issue on the [OpenHands GitHub repository](https://github.com/OpenHands/OpenHands/issues).
