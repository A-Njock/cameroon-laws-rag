# Cloudflare Deployment - Cameroon Laws RAG API

## Prerequisites

1. **Cloudflare Account** (free tier works)
2. **Index Files** (built from Colab notebook):
   - `index_file.index`
   - `index_file.meta.json`
   - `index_file.chunks.json`

## Step 1: Create R2 Bucket

1. Login to Cloudflare Dashboard
2. Go to **R2** → **Create bucket**
3. Name: `cameroon-laws-index`
4. Region: Automatic
5. Click **Create bucket**

## Step 2: Upload Index Files to R2

```bash
# Install Wrangler CLI
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Upload files to R2
wrangler r2 object put cameroon-laws-index/index_file.index --file=index_file.index
wrangler r2 object put cameroon-laws-index/index_file.meta.json --file=index_file.meta.json
wrangler r2 object put cameroon-laws-index/index_file.chunks.json --file=index_file.chunks.json
```

## Step 3: Deploy to Cloudflare Workers

```bash
cd cloudflare

# Install dependencies
pip install -r requirements.txt

# Deploy
wrangler deploy
```

## Step 4: Test the API

```bash
# Get your Worker URL from deployment output
# Example: https://cameroon-laws-rag.your-subdomain.workers.dev

# Health check
curl https://cameroon-laws-rag.your-subdomain.workers.dev/health

# Test query
curl -X POST https://cameroon-laws-rag.your-subdomain.workers.dev/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Code pénal article 15"}'
```

## Alternative: Cloudflare Pages

If Workers doesn't work, we can use Cloudflare Pages:

1. Push code to GitHub
2. Connect Cloudflare Pages to repo
3. Deploy as Pages Functions
4. Same API endpoints

## Troubleshooting

**Worker size too large?**
- Cloudflare Workers has 10MB limit
- Solution: Use external dependencies or Cloudflare Pages

**R2 access issues?**
- Check R2 bucket permissions
- Verify wrangler is logged in

**API not responding?**
- Check Worker logs in Cloudflare dashboard
- Verify index files uploaded correctly

## Cost

- **R2 Storage**: ~$0.01/month for 65MB
- **Workers**: Free tier (100k requests/day)
- **Total**: Essentially free!

## Next Steps

1. Run Colab notebook to build index
2. Download index files
3. Follow steps above to deploy
4. Integrate with Kemet AI chatbot
