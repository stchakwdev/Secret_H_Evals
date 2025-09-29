# Security Guidelines

## 🔒 API Key Security

### ✅ DO:
- Store API keys in `.env` file
- Use `load_dotenv()` to load environment variables
- Validate API keys exist before using them
- Keep `.env` file in `.gitignore`
- Use unique API keys for different environments
- Rotate API keys regularly
- Monitor API key usage

### ❌ DON'T:
- Hardcode API keys in source code
- Commit `.env` files to version control
- Share API keys in messages or emails
- Use production keys for testing
- Store keys in configuration files that get committed

## 🚨 If API Key is Exposed:

1. **Immediately rotate the key** at the provider (OpenRouter)
2. **Remove the key** from all files where it appears
3. **Check git history** for the exposed key
4. **Update all applications** using the old key
5. **Review access logs** for unauthorized usage

## 🛡️ Code Review Checklist:

Before committing code, verify:
- [ ] No hardcoded API keys
- [ ] `.env` file not committed  
- [ ] Environment variables properly loaded
- [ ] Error handling for missing keys
- [ ] No keys in logs or debug output

## 📱 OpenRouter Specific:

- Get keys at: https://openrouter.ai/keys
- Keys start with: `sk-or-v1-`
- Monitor usage at: https://openrouter.ai/activity
- Set spending limits to prevent unexpected charges

## 🔧 Environment Setup:

```bash
# 1. Copy example file
cp .env.example .env

# 2. Edit with your real keys (NEVER commit this)
nano .env

# 3. Verify .env is ignored
git status  # Should not show .env file

# 4. Test loading in your code
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API key loaded:', bool(os.environ.get('OPENROUTER_API_KEY')))"
```

## 🔍 Scan for Exposed Keys:

```bash
# Search for potential API keys in codebase
grep -r "sk-or-v1-" . --exclude-dir=node_modules --exclude-dir=.git
grep -r "OPENROUTER_API_KEY.*=" . --exclude-dir=node_modules --exclude-dir=.git
```

Remember: Security is everyone's responsibility! 🛡️