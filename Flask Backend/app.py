# app.py
import os
import zipfile
import logging
import traceback
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO
from easyocr import Reader
import cv2
import pandas as pd
from fuzzywuzzy import fuzz
import re
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Database config (MySQL)
database_url = os.getenv("DATABASE_URL")
if not database_url:
    logging.error("DATABASE_URL is not set. Example: mysql+pymysql://root:root@localhost:3306/frauddb")
    raise Exception("Please set DATABASE_URL environment variable (mysql+pymysql://user:pass@host:port/db)")

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# ----------------------------
# Database models (same as you used)
# ----------------------------
class FileDetails(db.Model):
    __tablename__ = "file_details"
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed_at = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(50), nullable=False)


class ExtractedDetails(db.Model):
    __tablename__ = "extracted_details"
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255))
    uid = db.Column(db.String(255))
    address = db.Column(db.String(1024))
    name_match_score = db.Column(db.Float)
    address_match_score = db.Column(db.Float)
    uid_match_score = db.Column(db.Float)
    overall_match_score = db.Column(db.Float)
    status = db.Column(db.String(50))
    reason = db.Column(db.String(255))
    processed_at = db.Column(db.DateTime, default=datetime.utcnow)


class Verification(db.Model):
    __tablename__ = "verification"
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    processed_at = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()


# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


# ----------------------------
# AadhaarVerificationSystem - detection, OCR, compare
# ----------------------------
class AadhaarVerificationSystem:
    def __init__(self, upload_folder, classifier_path=None, detector_path=None, ocr_langs=None):
        self.upload_folder = upload_folder
        self.extract_folder = os.path.join(upload_folder, "extracted_files")
        ensure_dir(self.extract_folder)

        # model paths
        self.classifier_path = classifier_path
        self.detector_path = detector_path

        # load YOLO models (defensive)
        self.classifier = None
        self.detector = None
        try:
            if self.classifier_path and os.path.exists(self.classifier_path):
                logging.info("Loading classifier from %s", self.classifier_path)
                self.classifier = YOLO(self.classifier_path)
            else:
                logging.warning("Classifier .pt not found - classification will be skipped.")
        except Exception as e:
            logging.exception("Failed loading classifier: %s", e)
            self.classifier = None

        try:
            if self.detector_path and os.path.exists(self.detector_path):
                logging.info("Loading detector from %s", self.detector_path)
                self.detector = YOLO(self.detector_path)
            else:
                logging.warning("Detector .pt not found - detection will be skipped.")
        except Exception as e:
            logging.exception("Failed loading detector: %s", e)
            self.detector = None

        # EasyOCR Reader initialization
        self.ocr_langs = (ocr_langs or "en").split(",")
        try:
            self.ocr_reader = Reader(self.ocr_langs, gpu=False)
        except Exception as e:
            logging.exception("Failed to init EasyOCR reader: %s", e)
            self.ocr_reader = None

    # simple cleaning
    def clean_text(self, text):
        if text is None:
            return ""
        text = str(text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ascii
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def clean_uid(self, uid):
        if not uid:
            return ""
        return re.sub(r"\D", "", str(uid))

    def clean_address(self, address):
        if not address:
            return ""
        s = str(address).lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\b(marg|lane|township|block|street|plot)\b", " ", s)
        s = re.sub(r"[^a-z0-9\s,.-]", " ", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        return s

    def extract_uid_from_text(self, text):
        if not text:
            return None
        # common grouped or ungrouped patterns
        m = re.search(r"(\d{4}\s?\d{4}\s?\d{4})", text)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        digits = re.sub(r"\D", "", text)
        if len(digits) >= 12:
            return digits[:12]
        return None

    def _resize_for_ocr(self, img):
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) < 300:
            scale = 2.0
        elif max(h, w) < 500:
            scale = 1.5
        if scale != 1.0:
            return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return img

    def _preprocess_crop_for_ocr(self, crop):
        # grayscale
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = crop if len(crop.shape) == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # denoise
        gray = cv2.bilateralFilter(gray, 5, 75, 75)
        # CLAHE for contrast
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass
        # threshold
        if max(gray.shape) < 300:
            try:
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            except Exception:
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # resize small crops
        thresh = self._resize_for_ocr(thresh)
        return thresh

    def _ocr_on_image(self, img):
        """Run EasyOCR readtext returning a single cleaned string (line breaks preserved)."""
        if self.ocr_reader is None:
            return ""
        try:
            # prefer paragraph=True on modern versions
            try:
                texts = self.ocr_reader.readtext(img, detail=0, paragraph=True)
            except TypeError:
                texts = self.ocr_reader.readtext(img, detail=0)
            if isinstance(texts, (list, tuple)):
                joined = "\n".join([str(t) for t in texts if t])
            else:
                joined = str(texts)
            return self.clean_text(joined)
        except Exception as e:
            logging.warning("EasyOCR error: %s", e)
            return ""

    def _fallback_full_image_name_search(self, full_image):
        """
        If detector didn't find name label, attempt scanning areas
        likely to contain the name: top-left card area and lower-left photo area.
        Returns best name candidate or None.
        """
        h, w = full_image.shape[:2]
        candidates = []

        # region 1: upper-left quarter (where front left panel often has name)
        r1 = full_image[int(0.05 * h):int(0.4 * h), int(0.03 * w):int(0.55 * w)]
        # region 2: below photo (lower-left blocks)
        r2 = full_image[int(0.55 * h):int(0.9 * h), int(0.03 * w):int(0.55 * w)]
        # region 3: left-center vertical strip
        r3 = full_image[int(0.2 * h):int(0.6 * h), int(0.03 * w):int(0.45 * w)]

        for region in (r1, r2, r3):
            try:
                thresh = self._preprocess_crop_for_ocr(region)
                txt = self._ocr_on_image(thresh)
                if txt:
                    # split to lines and choose name-like lines using the same heuristics below
                    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
                    for ln in lines:
                        low = ln.lower()
                        # reject lines with digits or obvious keywords
                        if re.search(r'\d', ln):  # contains digit -> not a name
                            continue
                        if re.search(r'\b(dob|birth|male|female|year|yob|son|daughter|father|mother)\b', low):
                            continue
                        # prefer lines with 2-4 words and moderate length
                        if 1 <= len(ln.split()) <= 6 and 2 < len(ln) < 80:
                            candidates.append(ln)
            except Exception:
                continue

        # choose the best candidate if any
        if candidates:
            # pick candidate with fewer punctuation and moderate length
            candidates.sort(key=lambda c: (len(re.sub(r'[^A-Za-z]', '', c)), -len(c)), reverse=True)
            best = candidates[0]
            best = re.sub(r'[^A-Za-z\s\.\-]', ' ', best).strip()
            best = re.sub(r'\s{2,}', ' ', best)
            return best
        return None

    def detect_fields(self, image_path):
        """
        Detect regions (name/uid/address), OCR them and return fields dict.
        Robust: uses best-box per label and fallback full-image name search.
        """
        try:
            if self.detector is None:
                logging.warning("Detector model not loaded - performing full-image OCR fallback.")
                image = cv2.imread(image_path)
                if image is None:
                    return {}
                thresh = self._preprocess_crop_for_ocr(image)
                txt = self._ocr_on_image(thresh)
                # try to extract uid and name from the full text
                uid = self.extract_uid_from_text(txt)
                name = None
                # try fallback name heuristics on full text
                lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
                for ln in lines:
                    if re.search(r'\d', ln):
                        continue
                    low = ln.lower()
                    if re.search(r'\b(dob|birth|male|female|yob)\b', low):
                        continue
                    if 1 <= len(ln.split()) <= 6:
                        name = re.sub(r'[^A-Za-z\s\.\-]', ' ', ln).strip()
                        break
                address = None
                pin = re.search(r'\b(\d{6})\b', txt)
                if pin:
                    address = pin.group(1)
                fields = {}
                if name:
                    fields["name"] = name
                if uid:
                    fields["uid"] = uid
                if address:
                    fields["address"] = address
                return fields

            # run detector with thresholds - tune conf/iou if needed
            results = self.detector(image_path, conf=0.45, iou=0.4)[0]

            # select best (highest-conf) box per label
            best_boxes = {}
            for box in results.boxes:
                try:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label_raw = str(self.detector.names[class_id])
                    label = label_raw.strip().lower()
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    if (x2 - x1) < 18 or (y2 - y1) < 8:
                        continue
                    if label not in best_boxes or conf > best_boxes[label]["conf"]:
                        best_boxes[label] = {"conf": conf, "coords": coords}
                except Exception:
                    continue

            image = cv2.imread(image_path)
            if image is None:
                logging.error("Failed to read image: %s", image_path)
                return {}

            fields = {}
            # OCR each best box
            for label, info in best_boxes.items():
                x1, y1, x2, y2 = info["coords"]
                pad_x = max(2, int(0.03 * (x2 - x1)))
                pad_y = max(2, int(0.03 * (y2 - y1)))
                x1p = max(0, x1 - pad_x)
                y1p = max(0, y1 - pad_y)
                x2p = min(image.shape[1], x2 + pad_x)
                y2p = min(image.shape[0], y2 + pad_y)
                crop = image[y1p:y2p, x1p:x2p]
                if crop.size == 0:
                    continue
                thresh = self._preprocess_crop_for_ocr(crop)
                ocr_text = self._ocr_on_image(thresh)

                if "uid" in label or "aadhar" in label or "number" in label:
                    uid = self.extract_uid_from_text(ocr_text)
                    if uid:
                        fields["uid"] = uid
                    else:
                        # maybe OCR produced digits with spaces: try to extract digits-only
                        digits = re.sub(r"\D", "", ocr_text)
                        if len(digits) >= 12:
                            fields["uid"] = digits[:12]
                elif "name" in label:
                    lines = [ln.strip() for ln in ocr_text.split("\n") if ln.strip()]
                    filtered = []
                    for ln in lines:
                        low = ln.lower()
                        if re.search(r'\d', ln):  # digits likely not name
                            continue
                        if re.search(r'\b(dob|birth|male|female|yob|son|daughter)\b', low):
                            continue
                        filtered.append(ln)
                    name_candidate = None
                    if filtered:
                        for ln in filtered:
                            if 1 <= len(ln.split()) <= 6 and len(ln) < 90:
                                name_candidate = ln
                                break
                        name_candidate = name_candidate or filtered[0]
                    else:
                        name_candidate = lines[0] if lines else None
                    if name_candidate:
                        name_candidate = re.sub(r'[^A-Za-z\s\.\-]', ' ', name_candidate).strip()
                        name_candidate = re.sub(r'\s{2,}', ' ', name_candidate)
                        fields["name"] = name_candidate
                elif "address" in label:
                    lines = [ln.strip() for ln in ocr_text.split("\n") if ln.strip()]
                    lines = [re.sub(r'[\|\>\<\#\@\$\^\*]', ' ', ln) for ln in lines]
                    good_lines = [ln for ln in lines if len(ln) > 4 and not re.match(r'^[\W_]+$', ln)]
                    address_candidate = " ".join(good_lines[:4]) if good_lines else (lines[0] if lines else None)
                    pin = re.search(r'\b(\d{6})\b', ocr_text)
                    if pin:
                        p = pin.group(1)
                        if address_candidate and p not in address_candidate:
                            address_candidate = f"{address_candidate} {p}"
                    fields["address"] = address_candidate
                else:
                    # store generic label
                    fields[label] = ocr_text if ocr_text else None

            # fallback: if 'name' missing, try full-image name search
            if "name" not in fields or not fields.get("name"):
                fallback_name = self._fallback_full_image_name_search(image)
                if fallback_name:
                    fields["name"] = fallback_name

            # final tidy for uid
            if "uid" in fields and fields["uid"]:
                u = re.sub(r"\s+", "", str(fields["uid"]))
                if len(u) == 12:
                    fields["uid"] = f"{u[:4]} {u[4:8]} {u[8:12]}"
                else:
                    fields["uid"] = re.sub(r"\s+", " ", str(fields["uid"])).strip()

            return fields

        except Exception as e:
            logging.exception("Field detection error: %s", e)
            return {}


    def is_aadhaar_card(self, image_path):
        """
        Classification check if classifier loaded. If classifier missing, return True so
        detection+OCR runs anyway (we want to process).
        """
        try:
            if self.classifier is None:
                logging.info("Classifier not loaded - defaulting to Aadhaar=True")
                return True, 1.0
            pred = self.classifier.predict(source=image_path)
            # try to extract class name and confidence robustly
            try:
                class_index = pred[0].probs.top1
                class_name = pred[0].names[class_index]
                confidence = float(pred[0].probs.top1conf.item())
            except Exception:
                # fallback: if model returned names/probs differently
                try:
                    probs = pred[0].probs
                    arr = probs.numpy() if hasattr(probs, "numpy") else None
                    if arr is not None:
                        idx = int(arr.argmax())
                        class_name = pred[0].names[idx]
                        confidence = float(arr.max())
                    else:
                        class_name = pred[0].names[0]
                        confidence = 1.0
                except Exception:
                    class_name = "aadhar"
                    confidence = 1.0
            logging.info("Classification: %s, Confidence: %s", class_name, confidence)
            return str(class_name).lower() in ("aadhar", "aadhaar"), confidence
        except Exception as e:
            logging.exception("Classification error: %s", e)
            return False, 0.0

    # matching & comparison logic remains same as yours (but defensive)
    def match_names(self, extracted_name, excel_name):
        extracted_name = extracted_name or ""
        excel_name = excel_name or ""
        return fuzz.ratio(self.clean_text(extracted_name), self.clean_text(excel_name))

    def match_addresses(self, extracted_address, row):
        address_components = ["Street Road Name", "City", "State", "PINCODE"]
        full_address = " ".join([str(row.get(c, "")) for c in address_components if row.get(c, "")])
        cleaned_extracted = self.clean_address(extracted_address or "")
        cleaned_full = self.clean_address(full_address or "")
        address_score = fuzz.partial_ratio(cleaned_extracted, cleaned_full)
        extracted_pin = re.sub(r"\D", "", str(extracted_address or ""))
        row_pin = str(row.get("PINCODE", "") or "")
        if extracted_pin and row_pin and extracted_pin == row_pin:
            address_score = 100.0
        return address_score, full_address

    def compare_with_excel(self, fields, excel_path):
        try:
            excel_data = pd.read_excel(excel_path, dtype=str)
        except Exception as e:
            logging.exception("Excel read error: %s", e)
            return [{"status": "Error", "reason": f"Excel read error: {e}"}]

        uid = fields.get("uid")
        extracted_name = fields.get("name", "N/A")
        extracted_address = fields.get("address")

        if not uid:
            return [{"status": "Rejected", "reason": "UID not found in image."}]

        uid_cleaned = self.clean_uid(uid)
        best_match = None
        highest_score = -1.0

        for _, row in excel_data.iterrows():
            row = row.to_dict()
            excel_uid_cleaned = self.clean_uid(row.get("UID", ""))
            name_score = self.match_names(extracted_name, row.get("Name", "")) if extracted_name != "N/A" else 0.0
            address_score, full_address = self.match_addresses(extracted_address, row) if extracted_address else (0.0, None)
            uid_score = fuzz.ratio(uid_cleaned, excel_uid_cleaned)

            name_score = safe_float(name_score, 0.0)
            address_score = safe_float(address_score, 0.0)
            uid_score = safe_float(uid_score, 0.0)

            overall_score = (name_score + uid_score + address_score) / 3.0
            status = "Accepted" if overall_score >= 70.0 else "Rejected"

            if overall_score > highest_score:
                highest_score = overall_score
                best_match = {
                    "SrNo": row.get("SrNo"),
                    "Name": row.get("Name"),
                    "Extracted Name": extracted_name,
                    "UID": row.get("UID"),
                    "Address Match Score": address_score,
                    "Address Reference": full_address,
                    "Name Match Score": name_score,
                    "UID Match Score": uid_score,
                    "Overall Match Score": overall_score,
                    "status": status,
                    "reason": "Matching scores calculated.",
                }

        if not best_match:
            return [{"status": "Rejected", "reason": "No matching record found."}]
        return [best_match]

    def process_zip_file(self, zip_path, excel_path):
        results_out = []
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(self.extract_folder)
        except Exception as e:
            logging.exception("Zip extraction error: %s", e)
            raise

        for root, _, files in os.walk(self.extract_folder):
            for file in files:
                if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                image_path = os.path.join(root, file)
                try:
                    is_aadhar, confidence = self.is_aadhaar_card(image_path)
                    if is_aadhar:
                        fields = self.detect_fields(image_path)
                        fields["filename"] = file
                        match_results = self.compare_with_excel(fields, excel_path)
                        results_out.append({
                            "filename": file,
                            "is_aadhar": is_aadhar,
                            "confidence": safe_float(confidence, 0.0),
                            "fields": fields,
                            "match_results": match_results
                        })
                    else:
                        results_out.append({
                            "filename": file,
                            "is_aadhar": False,
                            "confidence": safe_float(confidence, 0.0),
                            "reason": "Not an Aadhaar card."
                        })
                except Exception as e:
                    logging.exception("Error processing image %s: %s", image_path, e)
                    results_out.append({"filename": file, "error": str(e)})
        return results_out


# ----------------------------
# Initialize verifier with paths from env or defaults
# ----------------------------
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
ensure_dir(UPLOAD_FOLDER)

DEFAULT_CLASSIFIER = os.path.join(os.getcwd(), "models", "classifier.pt")
DEFAULT_DETECTOR = os.path.join(os.getcwd(), "models", "detector.pt")

verifier = AadhaarVerificationSystem(
    upload_folder=UPLOAD_FOLDER,
    classifier_path=os.getenv("CLASSIFIER_PT", DEFAULT_CLASSIFIER),
    detector_path=os.getenv("DETECTOR_PT", DEFAULT_DETECTOR),
    ocr_langs=os.getenv("OCR_LANGS", "en")
)


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analytics")
def analytics_dashboard():
    try:
        total_files = FileDetails.query.count()
        processed_files = FileDetails.query.filter_by(status="Processed").count()

        verification_stats = {}
        q = db.session.query(Verification.status, db.func.count(Verification.id).label("count")).group_by(Verification.status).all()
        for row in q:
            verification_stats[row.status] = {"count": int(row.count), "percentage": round(100.0 * row.count / total_files if total_files else 0.0, 2)}

        match_stats = db.session.query(
            db.func.min(ExtractedDetails.overall_match_score).label("min_score"),
            db.func.max(ExtractedDetails.overall_match_score).label("max_score"),
            db.func.avg(ExtractedDetails.overall_match_score).label("avg_score"),
            db.func.stddev(ExtractedDetails.overall_match_score).label("score_deviation")
        ).first()

        score_ranges = {}
        if total_files:
            try:
                q2 = db.session.query(
                    db.case(
                        (ExtractedDetails.overall_match_score < 50, "Low Match (0-50)"),
                        (ExtractedDetails.overall_match_score.between(50, 70), "Moderate Match (50-70)"),
                        (ExtractedDetails.overall_match_score.between(70, 85), "Good Match (70-85)"),
                        (ExtractedDetails.overall_match_score >= 85, "Excellent Match (85-100)"),
                        else_ = "Unknown"
                    ).label("range"),
                    db.func.count(ExtractedDetails.id).label("count")
                ).group_by("range").all()
                for row in q2:
                    score_ranges[row.range] = {"count": int(row.count), "percentage": round(100.0 * row.count / total_files, 2)}
            except Exception:
                logging.debug("Score range aggregation failed", exc_info=True)

        monthly_trend = []
        try:
            mt = db.session.query(
                db.func.date_format(Verification.processed_at, "%Y-%m").label("month"),
                Verification.status,
                db.func.count(Verification.id).label("count")
            ).group_by("month", Verification.status).order_by("month").all()
            for row in mt:
                monthly_trend.append({"month": row.month, "status": row.status, "count": int(row.count)})
        except Exception:
            logging.debug("Monthly trend query failed", exc_info=True)

        fm = db.session.query(
            db.func.avg(ExtractedDetails.name_match_score).label("avg_name"),
            db.func.avg(ExtractedDetails.address_match_score).label("avg_address"),
            db.func.avg(ExtractedDetails.uid_match_score).label("avg_uid")
        ).first()
        field_matching = {"avg_name_match": 0, "avg_address_match": 0, "avg_uid_match": 0}
        if fm:
            field_matching = {
                "avg_name_match": round(float(fm.avg_name or 0.0), 2),
                "avg_address_match": round(float(fm.avg_address or 0.0), 2),
                "avg_uid_match": round(float(fm.avg_uid or 0.0), 2),
            }

        analytics_data = {
            "total_files": int(total_files),
            "processed_files": int(processed_files),
            "match_score_analysis": {
                "avg_score": round(float(match_stats.avg_score or 0.0), 2) if match_stats else 0.0,
                "score_deviation": round(float(match_stats.score_deviation or 0.0), 2) if match_stats else 0.0
            },
            "verification_stats": verification_stats,
            "score_ranges": score_ranges,
            "monthly_trend": monthly_trend,
            "field_matching": field_matching
        }
        return render_template("analytics.html", analytics_data=analytics_data)
    except Exception as e:
        logging.exception("Analytics generation error: %s", e)
        return jsonify({"error": f"Analytics generation error: {e}"}), 500


@app.route("/upload", methods=["GET", "POST"])
def upload_files():
    if request.method == "POST":
        if "zipfile" not in request.files or "excelfile" not in request.files:
            return jsonify({"error": "Both files are required.", "results": None}), 400

        zip_file = request.files["zipfile"]
        excel_file = request.files["excelfile"]

        # safe names
        zip_filename = os.path.basename(zip_file.filename)
        excel_filename = os.path.basename(excel_file.filename)
        zip_path = os.path.join(verifier.upload_folder, zip_filename)
        excel_path = os.path.join(verifier.upload_folder, excel_filename)

        try:
            zip_file.save(zip_path)
            excel_file.save(excel_path)
        except Exception as e:
            logging.exception("File save error: %s", e)
            return jsonify({"error": f"File save error: {e}", "results": None}), 500

        try:
            results = verifier.process_zip_file(zip_path, excel_path)

            # record file
            new_file = FileDetails(filename=zip_filename, status="Processed", processed_at=datetime.utcnow())
            db.session.add(new_file)
            db.session.commit()

            # store extracted details + verification entries
            for res in results:
                matches = res.get("match_results") or []
                if matches:
                    for match in matches:
                        overall = safe_float(match.get("Overall Match Score"), 0.0)
                        status = match.get("status") or ("Accepted" if overall >= 70.0 else "Rejected")
                        if overall >= 70.0:
                            try:
                                ed = ExtractedDetails(
                                    filename=res.get("filename"),
                                    name=match.get("Extracted Name"),
                                    uid=match.get("UID"),
                                    address=match.get("Address Reference"),
                                    name_match_score=safe_float(match.get("Name Match Score")),
                                    address_match_score=safe_float(match.get("Address Match Score")),
                                    uid_match_score=safe_float(match.get("UID Match Score")),
                                    overall_match_score=overall,
                                    status=status,
                                    reason=match.get("reason")
                                )
                                db.session.add(ed)
                                db.session.add(Verification(filename=res.get("filename"), status="Accepted"))
                            except Exception:
                                logging.exception("DB insert error for accepted match")
                        else:
                            db.session.add(Verification(filename=res.get("filename"), status="Rejected"))
                else:
                    # if no match_results, create a rejection verification entry
                    db.session.add(Verification(filename=res.get("filename"), status="Rejected"))

            db.session.commit()
            return jsonify({"results": results, "success": True})
        except Exception as e:
            logging.exception("Processing error: %s", e)
            return jsonify({"error": str(e), "results": None}), 500
        finally:
            try:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                if os.path.exists(excel_path):
                    os.remove(excel_path)
            except Exception:
                logging.debug("Cleanup failed", exc_info=True)

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
