---
name: reference-github-repo
description: Team GitHub repo for the project and how the user commits to it
metadata:
  node_type: memory
  type: reference
  originSessionId: 863030b2-bf46-4af3-93bc-9361b0c03370
  modified: 2026-11-20T15:10:42.176Z
---

Repo: `git@github.com:ai-academy-bish/ds8-lalafo-cars-proj.git` (remote `origin`, branch `main`),
working copy `/workspace/app`. The user commits to `main` themselves with short messages
(e.g. "pilot | LM for mileage"). `dataset/` and every `data/` folder are gitignored, so raw data
and built tables are never in the repo - scripts must rebuild them.

A copy of this memory folder is kept in the repo at `claude-memory/` at the user's request
(2026-11-20); it is a snapshot - refresh it when memories change.
Related: [[project-mileage-task]].
