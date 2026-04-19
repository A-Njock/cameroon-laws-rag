# Quick Deploy to GitHub + Coolify

## 1. Create GitHub Repo
Go to: https://github.com/new
- Name: `cameroon-laws-rag`
- Public
- Don't initialize

## 2. Push Code
```bash
cd "e:\YORK.A\ME\KEMET AI\code-github\RAG_BUILDING"
git remote add origin https://github.com/YOUR_USERNAME/cameroon-laws-rag.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## 3. Deploy on Coolify
1. Login: http://167.235.153.112:8000
2. Add New Application
3. Connect GitHub repo
4. Coolify auto-detects Dockerfile
5. Deploy!

## 4. Test
```bash
curl http://your-coolify-url/health
```

Done! 🎉
