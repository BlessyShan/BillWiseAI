# BillWise AI — Teal Theme

A local-first invoice intelligence prototype with a deep teal `#023A38` theme.

## Windows + VS Code setup (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If activation is blocked:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501.

## OCR note
Image OCR requires the separate Tesseract OCR application in addition to `pytesseract`. Searchable PDFs use PyMuPDF. Scanned PDFs need OCR support.

## Included
- Deep teal UI theme and dashboard
- Invoice upload and preview
- OCR for images (when Tesseract is installed) and text-based PDFs
- Heuristic extraction of invoice fields; review before saving
- SQLite invoice ledger, editing and deletion
- Starter TF-IDF + Logistic Regression expense categorization
- Duplicate screening and Isolation Forest anomaly screening
- Analytics and CSV/PDF reports

## Limitations
The expense classifier uses a small illustrative training dataset and is a demonstration only. OCR and field extraction can be wrong. Anomaly alerts are not proof of fraud. There is no authentication or encryption-at-rest. Use synthetic/redacted invoices and do not expose the app publicly.
