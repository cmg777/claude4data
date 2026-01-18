# CLAUDE.md – AI Assistant Instructions

**READ THIS FILE FIRST** upon entering this project.

This file contains critical rules and context for working on **[INSERT PROJECT NAME]**. These rules are non-negotiable.

---

## Critical Rules

### 1. NEVER DELETE DATA
Under no circumstances are you ever to DELETE any data files. This includes `.dta`, `.xlsx`, `.csv`, `.shp`, or any other data format.

### 2. NEVER DELETE PROGRAMS
Under no circumstances are you ever to DELETE any program files. This includes `.do`, `.R`, `.py`, or any other script format.

### 3. USE THE LEGACY FOLDER
The `./legacy/` folder contains a complete snapshot of the original project structure (created **[INSERT DATE]**). This is sacred and should never be modified.

**One-Time Exception (COMPLETED):** On **[INSERT DATE]**, we performed a one-time move of all original files into `./legacy/` to preserve the original project state. This was the only permitted "move" operation.

**Going forward:**
- NEVER move files directly between working directories
- ALWAYS copy from `./legacy/` when you need original files
- If reorganizing, copy files to new locations (never move)

### 4. STAY WITHIN THIS DIRECTORY
Under no circumstances are you ever to GO UP OUT OF THIS ONE FOLDER called `[./INSERT_ROOT_FOLDER_NAME]`. All work must remain within this project directory.

### 5. COPY, DON'T MOVE
When working with files:
- COPY from `./legacy/` to working directories
- COPY between working directories if needed
- NEVER move files (except the one-time legacy migration, now complete)

### 6. MAINTAIN PROGRESS LOGS
The `./log/` directory contains progress logs that preserve conversation context across sessions.

**Why:** Chat sessions can die unexpectedly. When a new Claude starts, it has no memory of previous work. Logs bridge this gap.

**When to log:**
- After completing significant work
- Before ending a session
- After major decisions
- When context is building up

**How:** Create timestamped entries (`YYYY-MM-DD_HHMM.md`) documenting what was done, current state, and next steps.

**On startup:** Always check `./log/` for recent entries to understand what was happening before.

---

## Project Context

- **Project Title:** [INSERT TITLE]
- **Authors:** [INSERT NAMES]
- **Goal:** [INSERT BRIEF GOAL]