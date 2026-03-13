---
name: github-pr
description: Create or update a GitHub pull request after committing and pushing changes. Use when the user asks to create a PR, submit changes for review, or open a pull request.
---

# GitHub Pull Request Workflow

## Step 1: Check Current State

```bash
BRANCH_NAME=$(git branch --show-current)
git status --porcelain
git fetch origin
git rev-list HEAD --not origin/main --count
```

## Step 2: Route

| On main? | Uncommitted changes? | Action |
| -------- | -------------------- | ------ |
| Yes | Yes | Create new branch, commit via `/git-commit`, then create PR |
| Yes | No | Error — nothing to PR |
| No | Yes | Commit on current branch via `/git-commit`, then create PR |
| No | No | Already committed — proceed to push and create PR |

### Create Branch (if on main)

Auto-generate a branch name with a meaningful prefix. Do NOT ask the user.

| Prefix | Usage |
| ------ | ----- |
| `feat/` | new example or tensor function |
| `fix/` | bug fix |
| `docs/` | documentation changes |
| `ci/` | CI/CD changes |
| `refactor/` | restructuring |

```bash
git checkout -b <branch-name>
```

### Commit (if uncommitted changes)

Delegate to `/git-commit` skill.

## Step 3: Check for Existing PR

```bash
gh pr list --head "$BRANCH_NAME" --state open
```

If PR already exists, display with `gh pr view` and exit.

## Step 4: Rebase and Push

```bash
git fetch origin
git rebase origin/main
git push --set-upstream origin "$BRANCH_NAME"
```

After rebase (if already pushed):

```bash
git push --force-with-lease origin "$BRANCH_NAME"
```

## Step 5: Create PR

```bash
gh pr create \
  --title "Brief description of changes" \
  --body "$(cat <<'EOF'
## Summary
- Key change 1
- Key change 2

## Testing
- [ ] Example runs successfully
- [ ] Code follows pypto frontend coding style
EOF
)"
```

**Rules**:
- Auto-generate title and body from commit messages
- Keep title under 72 characters
- Do NOT add AI co-author footers or branding

## Common Issues

| Issue | Solution |
| ----- | -------- |
| PR already exists | `gh pr view` then exit |
| Merge conflicts | Resolve, `git add`, `git rebase --continue` |
| Push rejected | `git push --force-with-lease` |
| gh not authenticated | Tell user to run `gh auth login` |

## Checklist

- [ ] Branch created (if was on main)
- [ ] Changes committed via `/git-commit`
- [ ] Rebased onto `origin/main`
- [ ] Pushed to origin
- [ ] PR created with clear title and summary
- [ ] No AI co-author footers
