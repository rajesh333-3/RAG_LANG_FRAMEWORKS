# Mac Dev Setup & Git Reference Guide

## 1. Python on Mac

macOS removed the `python` command in macOS 12.3+. Only `python3` ships by default.

**Fix — add alias to `~/.zshrc`:**
```bash
alias python=python3
```
Then reload:
```bash
source ~/.zshrc
```

---

## 2. Starting a New Project — Routine

Every time you start a new project:

| Step | Command |
|------|---------|
| Create folder | `mkdir my-project && cd my-project` |
| Init git | `git init` |
| Add .gitignore | see below |
| Create GitHub repo | go to github.com → New Repository |
| Link remote | `gremote <repo-name>` |
| First commit & push | `git add . && git commit -m "Initial commit" && git push -u origin main` |

**Python `.gitignore` essentials:**
```
__pycache__/
*.pyc
.env
.venv/
*.egg-info/
.DS_Store
```

---

## 3. Daily Git Workflow

```bash
git add <files>           # stage changes
git commit -m "message"   # commit
git push                  # push to GitHub
```

Check status anytime:
```bash
git status
git remote -v             # see all remotes
git log --oneline -5      # recent commits
```

---

## 4. Managing Git Remotes

### Switch origin to a new repo
```bash
git remote remove origin
git remote add origin git@github.com:rajesh333-3/<repo-name>.git
```

### Add a second remote (keep origin)
```bash
git remote add <alias> git@github.com:rajesh333-3/<repo-name>.git
# example:
git remote add claude301 git@github.com:rajesh333-3/calude_301_by_rajesh.git
```

### Push to a specific remote
```bash
git push origin main       # push to origin
git push claude301 main    # push to named remote
```

---

## 5. Shell Helper Functions (in `~/.zshrc`)

```bash
GITHUB_BASE="git@github.com:rajesh333-3"

# Replace origin with a new repo
function gremote() {
    git remote remove origin 2>/dev/null
    git remote add origin "$GITHUB_BASE/$1.git"
    echo "Remote set to $GITHUB_BASE/$1.git"
}

# Add a named remote (keeps existing origin)
function gadd() {
    git remote add "$1" "$GITHUB_BASE/$2.git"
    echo "Remote '$1' set to $GITHUB_BASE/$2.git"
}
```

**Usage:**
```bash
gremote RAG_LANG_FRAMEWORKS          # set origin to this repo
gadd claude301 calude_301_by_rajesh  # add a second remote named claude301
```

Reload after editing `~/.zshrc`:
```bash
source ~/.zshrc
```

---

## 6. Installing GitHub CLI (`gh`) Without Homebrew

Download the `.pkg` installer from:
```
https://github.com/cli/cli/releases/latest
```
Pick `gh_*_macOS_arm64.pkg` (M1/M2/M3) or `amd64` (Intel).

Check your chip:
```bash
uname -m   # arm64 = Apple Silicon, x86_64 = Intel
```

Authenticate after install:
```bash
gh auth login
```

---

## 7. Quick Reference — Key Files

| File | Purpose |
|------|---------|
| `~/.zshrc` | Shell config — aliases, functions, PATH |
| `~/.gitignore` | Global git ignore (optional) |
| `.gitignore` | Per-project ignore rules |

---

## 8. Common Fixes

| Problem | Fix |
|---------|-----|
| `python: command not found` | Add `alias python=python3` to `~/.zshrc` |
| `gh: command not found` | Install from github.com/cli/cli/releases |
| Wrong remote repo | `git remote remove origin` then `gremote <new-repo>` |
| Changes not pushed | `git add . && git commit -m "..." && git push` |
