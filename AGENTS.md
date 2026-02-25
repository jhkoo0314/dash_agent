# AGENTS.md

## Purpose
This project integrates Sales, Target, and CRM datasets to generate:
- an interactive Streamlit SFE dashboard
- a final HTML strategic report

## Project Summary
- App entrypoint: `scripts/sfe_sandbox.py`
- Report engine: `scripts/report_builder_v12.py`
- Map assets: `scripts/map_component/index.html`, `hospital_map.html`
- Report template: `templates/report_template.html`
- Column mapping config: `config/mapping.json`

## Tech Stack
- Python
- Streamlit
- pandas, numpy
- plotly
- scikit-learn
- jinja2
- openpyxl

Install dependencies:
```bash
pip install -r requirements.txt
```

## Directory Guide
- `data/`: source datasets (`sales`, `targets`, `crm`)
- `scripts/`: dashboard and report logic
- `templates/`: HTML template(s)
- `config/`: mapping and configuration
- `docs/`: planning notes
- `output/`: generated outputs
  - `output/processed_data/standardized_sales_*.csv`
  - `output/Strategic_Full_Dashboard_*.html`

## Runbook
1. Launch dashboard:
```bash
streamlit run scripts/sfe_sandbox.py
```

2. Build final report:
```bash
python scripts/report_builder_v12.py
```

## Data and Processing Notes
- Canonical columns are auto-mapped through `config/mapping.json`.
- `report_builder_v12.py` first tries standardized sales files in `output/processed_data/`.
- If standardized files are missing, it falls back to raw files under `data/sales/`.
- Output filenames are date-based with collision-safe suffixes.

## Agent Working Rules
- Prioritize edits in this order:
  1. `scripts/sfe_sandbox.py` for UI/input/mapping behavior
  2. `scripts/report_builder_v12.py` for analytics/report generation
  3. `templates/report_template.html` for presentation layer
- Do not delete or reset user data in `data/` or `output/`.
- Keep `config/mapping.json` backward-compatible when updating mapping logic.
- Preserve existing output naming conventions.

## Encoding Caution
- Some existing Korean text may appear garbled in terminals with mismatched encoding.
- Prefer UTF-8 when editing files.
- Do not rename Korean key fields blindly; verify all usages first with `rg`.
- Encoding rule applies to **all files** (`.py`, `.md`, `.json`, `.html`, `.ipynb`, etc.), not only notebooks.
- Never commit/write code or comments containing garbled placeholders such as `??` caused by encoding mismatch.
- Before running any Python-based validation that prints Korean text, set UTF-8 I/O: `$env:PYTHONIOENCODING='utf-8'`.
- If console output is garbled, do not patch from that output; inspect/edit UTF-8 source directly and re-check.
- For `.ipynb` / `.ipynb_checkpoints` edits, never write Korean comments/strings via escaped replacement flows that can produce `??`.
- When executing notebook code in terminal, set UTF-8 I/O first (`$env:PYTHONIOENCODING='utf-8'`) before validation.
- After any notebook edit, re-open the edited cell source and verify Korean literals are intact (no `??`), then verify output columns by exact Korean names.
- If Korean text appears garbled in console output, do not trust that output for patching; inspect and patch raw file content in UTF-8 and re-validate.

## Quick Validation Checklist
- `streamlit run scripts/sfe_sandbox.py` starts successfully
- `python scripts/report_builder_v12.py` generates report output
- latest files exist in `output/`
- mapping behavior still works with `config/mapping.json`
