from __future__ import annotations
import hashlib, io, re, sqlite3, uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

try:
    import pytesseract
except ImportError:
    pytesseract = None
try:
    import fitz
except ImportError:
    fitz = None

APP = "BillWise AI"
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
UPLOADS = DATA / "uploads"
DB = DATA / "billwise.db"
MAX_SIZE = 10 * 1024 * 1024
EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}
DATA.mkdir(exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title=APP, page_icon="🧾", layout="wide")

# ---- Visual theme: deep teal #023A38, pale teal, white and muted gold ----
st.markdown("""
<style>
:root {--primary:#023A38;--primary2:#0B5B55;--pale:#E8F1EF;--gold:#C6A76A;
--ink:#173331;--muted:#667B78;--line:#DCE7E4;}
.stApp {background:#F6F8F7;color:var(--ink);}
[data-testid="stHeader"] {background:rgba(246,248,247,.94);}
[data-testid="stSidebar"] {background:#023A38;border-right:1px solid #164D49;}
[data-testid="stSidebar"] * {color:#F4F8F7!important;}
h1,h2,h3 {color:#023A38!important;letter-spacing:-.025em;}
h1 {font-weight:800!important;}
[data-testid="stMetric"] {background:white;border:1px solid #DCE7E4;padding:1rem;
border-radius:16px;box-shadow:0 4px 18px rgba(2,58,56,.04);}
[data-testid="stMetricLabel"] {color:#667B78!important;}
[data-testid="stMetricValue"] {color:#023A38!important;font-weight:800;}
[data-testid="stForm"],[data-testid="stExpander"],[data-testid="stFileUploader"] {
border-radius:15px!important;border-color:#DCE7E4!important;}
[data-testid="stFileUploader"] {background:white;border:1px dashed #8FB7B0;padding:.6rem;}
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button {
background:#023A38;color:white;border:1px solid #023A38;border-radius:10px;font-weight:650;}
.stButton>button:hover,.stDownloadButton>button:hover,
[data-testid="stFormSubmitButton"]>button:hover {background:#0B5B55;color:white;border-color:#0B5B55;}
hr {border-color:#DCE7E4!important;}
.bw-hero {background:linear-gradient(120deg,#E8F1EF 0%,#fff 78%);
border:1px solid #DCE7E4;border-radius:22px;padding:1.5rem 1.6rem;margin:.2rem 0 1.2rem;}
.bw-eyebrow {color:#0B5B55;text-transform:uppercase;font-size:.75rem;font-weight:800;letter-spacing:.12em;}
.bw-hero h1 {font-size:2.5rem;margin:.25rem 0 .45rem;line-height:1.12;}
.bw-hero p {color:#526966;font-size:1rem;max-width:800px;margin:0;}
.bw-brand {background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.14);
padding:.9rem;border-radius:14px;margin:.2rem 0 1rem;}
.bw-brand strong {display:block;font-size:1.15rem;}
.bw-brand small {color:#C4DAD6!important;}
.bw-card {background:white;border:1px solid #DCE7E4;border-radius:15px;padding:1rem;min-height:105px;}
.bw-card b {color:#023A38;font-size:1rem;}
.bw-card p {color:#667B78;font-size:.84rem;margin:.35rem 0 0;}
</style>
""", unsafe_allow_html=True)

# ---- SQLite storage ----
def connect():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with connect() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS invoices(
        id TEXT PRIMARY KEY, created_at TEXT NOT NULL, original_filename TEXT NOT NULL,
        stored_filename TEXT NOT NULL, file_hash TEXT NOT NULL, vendor TEXT NOT NULL DEFAULT '',
        invoice_number TEXT NOT NULL DEFAULT '', invoice_date TEXT NOT NULL DEFAULT '',
        subtotal REAL NOT NULL DEFAULT 0, tax REAL NOT NULL DEFAULT 0,
        total REAL NOT NULL DEFAULT 0, category TEXT NOT NULL DEFAULT 'Other',
        ocr_text TEXT NOT NULL DEFAULT '', notes TEXT NOT NULL DEFAULT '')""")
init_db()

def load_df():
    with connect() as c:
        return pd.read_sql_query("SELECT * FROM invoices ORDER BY created_at DESC", c)

def extract_text(filename, content):
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        if fitz is None:
            return "", "PDF support missing. Install PyMuPDF."
        try:
            with fitz.open(stream=content, filetype="pdf") as doc:
                text = "\n".join(page.get_text("text") for page in doc).strip()
            return text, "Text extracted from PDF." if text else "No selectable text found. Scanned PDF OCR is not enabled."
        except Exception as e:
            return "", f"Could not read PDF: {e}"
    if pytesseract is None:
        return "", "Install pytesseract and the Tesseract OCR application to read images."
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        text = pytesseract.image_to_string(image).strip()
        return text, "OCR completed." if text else "OCR found no readable text."
    except Exception as e:
        return "", f"OCR failed. Check that Tesseract is installed and on PATH. {e}"

def match(patterns, text):
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m: return m.group(1).strip(" :#\t")
    return ""

def amount(value):
    try: return round(float(value.replace(",", "").replace("₹", "").replace("Rs.", "").strip()), 2)
    except (ValueError, AttributeError): return 0.0

def parse_invoice(text):
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    vendor = lines[0][:120] if lines else ""
    number = match([r"(?:invoice|bill|receipt)\s*(?:no\.?|number|#|id)\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-\/]+)"], text)
    inv_date = match([
        r"(?:invoice\s*date|bill\s*date|date)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(?:invoice\s*date|bill\s*date|date)\s*[:\-]?\s*(\d{4}[/-]\d{1,2}[/-]\d{1,2})"], text)
    total = amount(match([r"(?:grand\s*total|total\s*amount|amount\s*due|net\s*payable|total)\s*[:\-]?\s*(?:INR|Rs\.?|₹|\$)?\s*([\d,]+(?:\.\d{1,2})?)"], text))
    tax = amount(match([r"(?:total\s*tax|gst|tax)\s*[:\-]?\s*(?:INR|Rs\.?|₹|\$)?\s*([\d,]+(?:\.\d{1,2})?)"], text))
    subtotal = amount(match([r"(?:subtotal|sub\s*total|taxable\s*value)\s*[:\-]?\s*(?:INR|Rs\.?|₹|\$)?\s*([\d,]+(?:\.\d{1,2})?)"], text))
    if subtotal == 0 and total > 0: subtotal = max(0, total-tax)
    return dict(vendor=vendor, invoice_number=number, invoice_date=inv_date,
                subtotal=subtotal, tax=tax, total=total)

TRAIN_TEXTS = [
"diesel petrol fuel vehicle service motor oil", "bus taxi train transport freight delivery",
"office paper pens stationery printer ink laptop keyboard", "internet broadband phone telecom electricity water bill",
"rice vegetables groceries food restaurant catering", "raw materials inventory stock wholesale purchase merchandise",
"advertising marketing promotion printing flyers", "rent lease office building warehouse maintenance",
"salary wages payroll employee contractor fees", "software subscription cloud hosting domain license",
"courier shipping postage parcel logistics", "repair tools equipment spare parts maintenance"]
TRAIN_LABELS = ["Transport","Transport","Office Supplies","Utilities","Food","Inventory",
"Marketing","Rent & Maintenance","Payroll","Software","Transport","Rent & Maintenance"]

@st.cache_resource
def category_model():
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2))),
                      ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    model.fit(TRAIN_TEXTS, TRAIN_LABELS)
    return model

def predict_category(vendor, text):
    if not (vendor.strip() or text.strip()): return "Other", 0.0
    model = category_model()
    probs = model.predict_proba([f"{vendor} {text}"])[0]
    idx = probs.argmax()
    return str(model.classes_[idx]), float(probs[idx])

def duplicate_check(df, file_hash, vendor, number, total):
    if df.empty: return None
    hit = df[df.file_hash == file_hash]
    if len(hit): return f"Exact same file as record {hit.iloc[0]['id']}."
    if number.strip():
        hit = df[(df.invoice_number.fillna("").str.casefold() == number.strip().casefold()) &
                 (df.vendor.fillna("").str.casefold() == vendor.strip().casefold())]
        if len(hit): return f"Vendor and invoice number match record {hit.iloc[0]['id']}."
    if vendor.strip() and total > 0:
        hit = df[(df.vendor.fillna("").str.casefold() == vendor.strip().casefold()) &
                 (pd.to_numeric(df.total, errors="coerce").fillna(0).round(2) == round(total,2))]
        if len(hit): return f"Vendor and total match record {hit.iloc[0]['id']}; review manually."
    return None

def save_record(rec):
    with connect() as c:
        c.execute("""INSERT INTO invoices VALUES(
        :id,:created_at,:original_filename,:stored_filename,:file_hash,:vendor,
        :invoice_number,:invoice_date,:subtotal,:tax,:total,:category,:ocr_text,:notes)""", rec)

def pdf_report(df):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=landscape(A4), rightMargin=12*mm,leftMargin=12*mm,topMargin=12*mm,bottomMargin=12*mm)
    styles = getSampleStyleSheet()
    story = [Paragraph("BillWise AI — Invoice Summary", styles["Title"]),
             Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M}", styles["Normal"]), Spacer(1,5*mm)]
    if df.empty:
        story.append(Paragraph("No invoices found.", styles["Normal"]))
    else:
        story.append(Paragraph(f"Invoice count: {len(df)} | Recorded total: ₹{pd.to_numeric(df.total, errors='coerce').fillna(0).sum():,.2f}", styles["Heading2"]))
        cols = ["created_at","vendor","invoice_number","invoice_date","category","total"]
        data = [["Saved at","Vendor","Invoice No.","Invoice Date","Category","Total"]]
        for _, r in df[cols].head(100).iterrows():
            data.append([str(r.created_at)[:16],str(r.vendor)[:25],str(r.invoice_number)[:18],
                         str(r.invoice_date)[:16],str(r.category)[:18],f"₹{float(r.total or 0):,.2f}"])
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#023A38")),
          ("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.3,colors.grey),
          ("FONTSIZE",(0,0),(-1,-1),8),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.whitesmoke])]))
        story.append(t)
    doc.build(story)
    return out.getvalue()

# ---- Sidebar ----
with st.sidebar:
    st.markdown('<div class="bw-brand"><strong>🧾 BillWise AI</strong><small>Smarter invoice management</small></div>', unsafe_allow_html=True)
    page = st.radio("WORKSPACE", ["Dashboard", "Upload invoice", "Invoice ledger", "Analytics", "Reports", "About"])
    st.divider()
    st.caption("Local-first student prototype")
    st.caption("Use synthetic or redacted invoices.")

st.markdown("""<div class="bw-hero">
<div class="bw-eyebrow">INVOICE INTELLIGENCE · LOCAL-FIRST</div>
<h1>BillWise AI</h1>
<p>Turn everyday bills into clearer business insights. Extract invoice details, organize expenses,
and review potential duplicates or unusual transactions in one focused workspace.</p></div>""", unsafe_allow_html=True)

df = load_df()

if page == "Dashboard":
    st.subheader("Your business at a glance")
    total = pd.to_numeric(df["total"], errors="coerce").fillna(0).sum() if not df.empty else 0
    a,b,c,d = st.columns(4)
    a.metric("Invoices recorded", f"{len(df):,}")
    b.metric("Recorded spend", f"₹{total:,.2f}")
    b.caption("Based on saved invoice totals")
    c.metric("Expense categories", str(df["category"].nunique()) if not df.empty else "0")
    d.metric("Needs review", "—", help="Potential alerts will be shown in Analytics as records accumulate.")
    st.markdown("### Get started")
    cards = st.columns(3)
    for col, (n,title,desc) in zip(cards, [
        ("01","Upload","Add an invoice image or PDF."),
        ("02","Review","Confirm extracted fields before saving."),
        ("03","Understand","Explore spending and reports.")]):
        col.markdown(f'<div class="bw-card"><span style="color:#C6A76A;font-weight:800">{n}</span><br><b>{title}</b><p>{desc}</p></div>', unsafe_allow_html=True)
    st.write("")
    if not df.empty:
        st.markdown("### Recent invoices")
        st.dataframe(df[["created_at","vendor","invoice_number","category","total"]].head(8),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Your invoice workspace is ready. Upload a sample invoice to begin.")

elif page == "Upload invoice":
    st.subheader("Upload and review an invoice")
    st.write("Upload an image or PDF. BillWise attempts text extraction, suggests a category, and lets you verify every field before saving.")
    cc = st.columns(3)
    for col,(n,title,desc) in zip(cc,[("01","Extract","Read invoice text and key fields"),("02","Review","Correct values before saving"),("03","Understand","Track spending and review alerts")]):
        col.markdown(f'<div class="bw-card"><span style="color:#C6A76A;font-weight:800">{n}</span><br><b>{title}</b><p>{desc}</p></div>',unsafe_allow_html=True)
    st.write("")
    uploaded = st.file_uploader("Choose invoice file", type=sorted(EXTENSIONS), help="PNG, JPG, JPEG, PDF; max 10 MB")
    if uploaded:
        content = uploaded.getvalue()
        ext = Path(uploaded.name).suffix.lower().lstrip(".")
        if not content: st.error("The uploaded file is empty.")
        elif len(content) > MAX_SIZE: st.error("File exceeds the 10 MB size limit.")
        elif ext not in EXTENSIONS: st.error("Unsupported file type.")
        else:
            left,right = st.columns([1,1])
            with left:
                st.markdown("#### Document preview")
                st.write(f"**File:** {uploaded.name}")
                st.write(f"**Size:** {len(content)/1024:.1f} KB")
                if ext in {"png","jpg","jpeg"}:
                    try:
                        im = Image.open(io.BytesIO(content)); im.verify()
                        st.image(content, use_container_width=True)
                    except (UnidentifiedImageError,OSError): st.error("This is not a readable image.")
                else:
                    st.info("PDF uploaded. Text extraction will be attempted.")
                    st.download_button("Download original PDF",content, file_name=Path(uploaded.name).name, mime="application/pdf")
            with right:
                st.markdown("#### Extracted information")
                text, status = extract_text(uploaded.name, content)
                st.caption(status)
                fields = parse_invoice(text)
                with st.expander("View extracted text", expanded=bool(text)):
                    st.text_area("Document text", text, height=150, disabled=True)
                cat_guess, confidence = predict_category(fields["vendor"], text)
                cats = ["Other","Inventory","Transport","Office Supplies","Utilities","Food","Marketing","Rent & Maintenance","Payroll","Software"]
                default_cat = cat_guess if cat_guess in cats else "Other"
                with st.form("review_invoice"):
                    vendor = st.text_input("Vendor / merchant", fields["vendor"])
                    invno = st.text_input("Invoice number", fields["invoice_number"])
                    invdate = st.text_input("Invoice date (as printed)", fields["invoice_date"])
                    subtotal = st.number_input("Subtotal (₹)", min_value=0.0, value=float(fields["subtotal"]), step=1.0)
                    tax = st.number_input("Tax (₹)", min_value=0.0, value=float(fields["tax"]), step=1.0)
                    total_amt = st.number_input("Total (₹)", min_value=0.0, value=float(fields["total"]), step=1.0)
                    category = st.selectbox("Expense category", cats, index=cats.index(default_cat))
                    notes = st.text_area("Review notes", placeholder="Optional")
                    st.caption(f"Suggested category: {cat_guess} · starter-model confidence {confidence:.0%}")
                    submit = st.form_submit_button("Save reviewed invoice", type="primary", use_container_width=True)
                if submit:
                    digest = hashlib.sha256(content).hexdigest()
                    current = load_df()
                    warning = duplicate_check(current,digest,vendor,invno,float(total_amt))
                    if warning:
                        st.warning("Possible duplicate: "+warning)
                        allow = st.checkbox("I reviewed this warning and still want to save it.", key="allow_"+digest)
                    else: allow = True
                    if allow:
                        ident = uuid.uuid4().hex[:12]
                        stored = f"{ident}.{ext}"
                        try:
                            (UPLOADS/stored).write_bytes(content)
                            save_record({"id":ident,"created_at":datetime.now().isoformat(timespec="seconds"),
                              "original_filename":Path(uploaded.name).name,"stored_filename":stored,"file_hash":digest,
                              "vendor":vendor.strip(),"invoice_number":invno.strip(),"invoice_date":invdate.strip(),
                              "subtotal":round(float(subtotal),2),"tax":round(float(tax),2),"total":round(float(total_amt),2),
                              "category":category,"ocr_text":text[:50000],"notes":notes.strip()})
                            st.success(f"Invoice saved. Record ID: {ident}")
                            st.rerun()
                        except (OSError,sqlite3.Error) as e:
                            st.error(f"Could not save invoice: {e}")

elif page == "Invoice ledger":
    st.subheader("All invoices")
    df = load_df()
    if df.empty: st.info("No invoices yet. Upload your first invoice.")
    else:
        q = st.text_input("Search vendor, invoice number, category or notes")
        filtered = df
        if q.strip():
            mask = (df.vendor.fillna("").str.contains(q,case=False,regex=False) |
                    df.invoice_number.fillna("").str.contains(q,case=False,regex=False) |
                    df.category.fillna("").str.contains(q,case=False,regex=False) |
                    df.notes.fillna("").str.contains(q,case=False,regex=False))
            filtered = df[mask]
        cols = ["id","created_at","vendor","invoice_number","invoice_date","category","subtotal","tax","total","notes"]
        st.dataframe(filtered[cols], use_container_width=True, hide_index=True)
        st.download_button("Download ledger CSV",filtered[cols].to_csv(index=False).encode("utf-8-sig"),
                           file_name="billwise_invoices.csv",mime="text/csv")
        st.markdown("#### Edit or delete record")
        labels = {f"{r.vendor or 'Unknown vendor'} | {r.invoice_number or r.id} | ₹{float(r.total or 0):.2f}":r.id for _,r in df.iterrows()}
        label = st.selectbox("Select invoice",list(labels.keys()))
        sid = labels[label]; row = df[df.id == sid].iloc[0]
        with st.form("edit"):
            ev = st.text_input("Vendor",str(row.vendor or ""))
            en = st.text_input("Invoice number",str(row.invoice_number or ""))
            ed = st.text_input("Invoice date",str(row.invoice_date or ""))
            es = st.number_input("Subtotal (₹)",min_value=0.0,value=float(row.subtotal or 0),step=1.0)
            et = st.number_input("Tax (₹)",min_value=0.0,value=float(row.tax or 0),step=1.0)
            ea = st.number_input("Total (₹)",min_value=0.0,value=float(row.total or 0),step=1.0)
            cats = ["Other","Inventory","Transport","Office Supplies","Utilities","Food","Marketing","Rent & Maintenance","Payroll","Software"]
            ec = st.selectbox("Category",cats,index=cats.index(row.category) if row.category in cats else 0)
            enotes = st.text_area("Notes",str(row.notes or ""))
            save = st.form_submit_button("Save changes",type="primary")
        if save:
            with connect() as con:
                con.execute("UPDATE invoices SET vendor=?,invoice_number=?,invoice_date=?,subtotal=?,tax=?,total=?,category=?,notes=? WHERE id=?",
                            (ev.strip(),en.strip(),ed.strip(),es,et,ea,ec,enotes.strip(),sid))
            st.success("Invoice updated."); st.rerun()
        if st.button("Delete selected invoice"):
            st.session_state["delete_candidate"] = sid
        if st.session_state.get("delete_candidate") == sid:
            ok = st.checkbox("Confirm permanent deletion",key="confirm_delete")
            if st.button("Confirm delete",disabled=not ok):
                with connect() as con:
                    saved = con.execute("SELECT stored_filename FROM invoices WHERE id=?",(sid,)).fetchone()
                    con.execute("DELETE FROM invoices WHERE id=?",(sid,))
                if saved: (UPLOADS/saved["stored_filename"]).unlink(missing_ok=True)
                st.session_state.pop("delete_candidate",None); st.success("Deleted."); st.rerun()

elif page == "Analytics":
    st.subheader("Expense overview")
    df = load_df()
    if df.empty: st.info("Save invoices to see your analytics.")
    else:
        df["total"] = pd.to_numeric(df.total,errors="coerce").fillna(0)
        a,b,c = st.columns(3)
        a.metric("Recorded spend",f"₹{df.total.sum():,.2f}")
        b.metric("Invoices",f"{len(df):,}")
        c.metric("Average invoice",f"₹{df.total.mean():,.2f}")
        st.markdown("#### Spending by category")
        st.bar_chart(df.groupby("category")["total"].sum().sort_values(ascending=False))
        st.markdown("#### Monthly spending")
        dates = pd.to_datetime(df.invoice_date,errors="coerce",dayfirst=True)
        monthly = df.assign(parsed_date=dates).dropna(subset=["parsed_date"]).copy()
        if not monthly.empty:
            monthly["month"] = monthly.parsed_date.dt.to_period("M").astype(str)
            st.line_chart(monthly.groupby("month")["total"].sum())
        else: st.caption("Monthly chart appears when invoice dates can be parsed.")
        st.markdown("#### Potentially unusual amounts")
        amounts = df.loc[df.total > 0,"total"]
        if len(amounts) < 10:
            st.info("At least 10 positive invoice totals are needed for an initial anomaly model.")
        else:
            model = IsolationForest(n_estimators=100,contamination=.1,random_state=42)
            model.fit(amounts.to_numpy().reshape(-1,1))
            df["anomaly"] = ""
            valid = df.total > 0
            df.loc[valid,"anomaly"] = model.predict(df.loc[valid,["total"]]).astype(str)
            flagged = df[(df.anomaly == "-1") & valid]
            if flagged.empty: st.success("No amounts flagged by the current model.")
            else:
                st.warning("These are review signals, not proof of fraud.")
                st.dataframe(flagged[["vendor","invoice_number","total","category"]],use_container_width=True,hide_index=True)

elif page == "Reports":
    st.subheader("Generate reports")
    df = load_df()
    if df.empty: st.info("Save invoices before generating a report.")
    else:
        st.write("Download a CSV ledger or a printable summary PDF.")
        st.download_button("Download CSV report",df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="billwise_report.csv",mime="text/csv")
        try:
            st.download_button("Download PDF summary",pdf_report(df),file_name="billwise_summary.pdf",mime="application/pdf")
        except Exception as e: st.error(f"PDF report could not be created: {e}")

else:
    st.subheader("About BillWise AI")
    st.markdown("""
**BillWise AI** is a local-first final-year AIML prototype for invoice processing and expense review.

**Included:** invoice uploads, OCR/text extraction, editable records, SQLite storage, a starter TF-IDF + Logistic Regression expense classifier, duplicate screening, anomaly screening, analytics, CSV exports, and PDF reports.

**Important limitations:** the classifier uses a tiny illustrative dataset and is not production-ready. OCR field extraction is heuristic and must be checked by a person. Anomaly flags are not proof of fraud. This prototype has no login, encryption-at-rest, or multi-user controls.

Use synthetic or redacted bills. Do not expose this app publicly or rely on it as the only accounting record.
""")
    st.warning("Before real business use, add authentication, secure storage, backups, audit logging, stronger file validation, and a privacy policy.")

st.divider()
st.caption("BillWise AI · Prototype · Theme: #023A38 · Verify extracted invoice data before relying on it.")
