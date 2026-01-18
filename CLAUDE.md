# CLAUDE.md – AI Assistant Instructions

**READ THIS FILE FIRST** upon entering this project.

This file contains critical rules and context for working on claude4data. These rules are non-negotiable.

---

## Critical Rules

### 1. NEVER DELETE DATA
Under no circumstances are you ever to DELETE any data files. This includes `.dta`, `.xlsx`, `.csv`, `.shp` `.geojson`, or any other data format.

### 2. NEVER DELETE PROGRAMS
Under no circumstances are you ever to DELETE any program files. This includes `.do`, `.R`, `.py` `.ipynb`, or any other script format.

### 3. USE THE LEGACY FOLDER
The `./legacy/` folder contains a complete snapshot of the original project structure (created 20260118 ). This is sacred and should never be modified.

**One-Time Exception (COMPLETED):** On 20260118 , we performed a one-time move of all original files into `./legacy/` to preserve the original project state. This was the only permitted "move" operation.

**Going forward:**
- NEVER move files directly between working directories
- ALWAYS copy from `./legacy/` when you need original files
- If reorganizing, copy files to new locations (never move)

### 4. STAY WITHIN THIS DIRECTORY
Under no circumstances are you ever to GO UP OUT OF THIS ONE FOLDER called claude4data. All work must remain within this project directory.

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

**What to include:**
- Current state of the project
- Summary of work done (include key results, tables or figures)
- Key decisions made
- Any issues or blockers
- Next steps planned

**How:** Create timestamped entries (`YYYY-MM-DD_HHMM.md`) documenting what was done, current state, and next steps.

**On startup:** Always check `./log/` for recent entries to understand what was happening before.

---

## Project Context

- **Project Title:** Using Claude for Data Science Workflows
- **Project Directory:** claude4data
- **Legacy Directory:** ./legacy/
- **Log Directory:** ./log/
- **Data Formats:** .dta, .xlsx, .csv, .shp .geojson
- **Program Formats:** .do, .R, .py, .ipynb
- **Primary Tools:** Claude Code, Python, R, Stata
- **Authors:** Carlos Mendez
- **Goal:** Data science tasks using Claude Code
  
## About the slides folder
The `slides/` folder contains Quarto presentations created to showcase the results of the current session or specific analyses. The design and structure of these slides are tailored to effectively communicate the findings from the data analyses performed in this project. The slides should creatly interpret the results. Also, the design should be clean, beautiful, professional, suitable for academic or professional presentations. Titles should be in blue (#2874A6) and bold letters should be in green (#229954). Custom CSS is embedded directly in the .qmd file for easy portability