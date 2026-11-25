
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from pathlib import Path
import subprocess
import sys
import cv2
import json
import shutil
import uuid


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

UPLOAD_DIR = PROJECT_ROOT / "uploads"
RESULT_DIR = PROJECT_ROOT / "results" / "web"

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

PREPROCESS_SCRIPT = PROJECT_ROOT / "preprocessing" / "preprocess.py"
CFAR_SCRIPT = PROJECT_ROOT / "cfar" / "os_cfar_detector.py"
YOLO_SCRIPT = PROJECT_ROOT / "yolov5" / "detect.py"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def run_command(command):
    """
    Run an external Python script and return its output.
    """

    print("\n========================================")
    print("RUNNING:")
    print(" ".join(str(x) for x in command))
    print("========================================")

    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"{result.stderr}"
        )

    return result.stdout


def find_file(folder, stem, extensions):
    """
    Find a file generated for the uploaded image.
    """

    for ext in extensions:
        path = folder / f"{stem}{ext}"

        if path.exists():
            return path

    return None


def copy_result(source, destination):
    """
    Copy a generated result into results/web.
    """

    if source is None or not source.exists():
        return None

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        source,
        destination
    )

    return destination


def create_browser_image(input_path, output_path):
    """
    Convert uploaded SAR image to PNG
    so the browser can display it.
    """

    image = cv2.imread(
        str(input_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            "Unable to read uploaded image."
        )

    cv2.imwrite(
        str(output_path),
        image
    )

    return output_path


def create_comparison(cfar_path, yolo_path, output_path):
    """
    Create side-by-side OS-CFAR vs YOLOv5 comparison.
    """

    cfar = cv2.imread(
        str(cfar_path)
    )

    yolo = cv2.imread(
        str(yolo_path)
    )

    if cfar is None:
        raise ValueError(
            "Could not read CFAR result."
        )

    if yolo is None:
        raise ValueError(
            "Could not read YOLO result."
        )

    # --------------------------------------------------------
    # Resize both images to same height
    # --------------------------------------------------------

    target_height = max(
        cfar.shape[0],
        yolo.shape[0]
    )

    def resize_height(img, height):

        if img.shape[0] == height:
            return img

        ratio = height / img.shape[0]

        width = int(
            img.shape[1] * ratio
        )

        return cv2.resize(
            img,
            (width, height),
            interpolation=cv2.INTER_AREA
        )

    cfar = resize_height(
        cfar,
        target_height
    )

    yolo = resize_height(
        yolo,
        target_height
    )

    # --------------------------------------------------------
    # Side-by-side
    # --------------------------------------------------------

    comparison = cv2.hconcat(
        [cfar, yolo]
    )

    divider_x = cfar.shape[1]

    # --------------------------------------------------------
    # Divider
    # --------------------------------------------------------

    cv2.line(
        comparison,
        (divider_x, 0),
        (divider_x, target_height),
        (255, 255, 255),
        3
    )

    # --------------------------------------------------------
    # OS-CFAR title
    # --------------------------------------------------------

    cv2.rectangle(
        comparison,
        (10, 10),
        (190, 48),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        comparison,
        "OS-CFAR",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # --------------------------------------------------------
    # YOLOv5 title
    # --------------------------------------------------------

    cv2.rectangle(
        comparison,
        (
            divider_x + 10,
            10
        ),
        (
            divider_x + 200,
            48
        ),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        comparison,
        "YOLOv5",
        (
            divider_x + 20,
            38
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    cv2.imwrite(
        str(output_path),
        comparison
    )

    return output_path


def parse_yolo_results(
    txt_path,
    image_width,
    image_height
):
    """
    Convert YOLO normalized coordinates
    into pixel coordinates.
    """

    detections = []

    if not txt_path.exists():
        return detections

    lines = txt_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for index, line in enumerate(
        lines,
        start=1
    ):

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        try:

            class_id = int(
                float(parts[0])
            )

            x_center = float(parts[1])
            y_center = float(parts[2])

            width = float(parts[3])
            height = float(parts[4])

            confidence = (
                float(parts[5])
                if len(parts) >= 6
                else 0.0
            )

        except ValueError:
            continue

        # ----------------------------------------------------
        # Convert normalized YOLO coordinates
        # to pixel coordinates
        # ----------------------------------------------------

        x1 = int(
            (x_center - width / 2)
            * image_width
        )

        y1 = int(
            (y_center - height / 2)
            * image_height
        )

        x2 = int(
            (x_center + width / 2)
            * image_width
        )

        y2 = int(
            (y_center + height / 2)
            * image_height
        )

        detections.append({

            "id": index,

            "label": (
                "ship"
                if class_id == 0
                else str(class_id)
            ),

            "confidence": round(
                confidence,
                3
            ),

            "bbox": [
                x1,
                y1,
                x2,
                y2
            ]

        })

    return detections


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# DETECTION API
# ============================================================

@app.route(
    "/detect",
    methods=["POST"]
)
def detect():

    try:

        # ====================================================
        # CHECK FILE
        # ====================================================

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No file uploaded."
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "No file selected."
            }), 400

        if not allowed_file(
            file.filename
        ):

            return jsonify({
                "success": False,
                "error": "Unsupported image format."
            }), 400


        # ====================================================
        # SAVE UPLOAD
        # ====================================================

        original_name = secure_filename(
            file.filename
        )

        unique_id = uuid.uuid4().hex[:8]

        stem = Path(
            original_name
        ).stem

        extension = Path(
            original_name
        ).suffix.lower()

        filename = (
            f"{stem}_{unique_id}{extension}"
        )

        upload_path = (
            UPLOAD_DIR / filename
        )

        file.save(
            str(upload_path)
        )

        print(
            "\nUploaded:",
            upload_path
        )


        # ====================================================
        # ORIGINAL IMAGE
        # ====================================================

        original_browser_path = (
            RESULT_DIR /
            f"{stem}_{unique_id}_original.png"
        )

        create_browser_image(
            upload_path,
            original_browser_path
        )


        # ====================================================
        # STEP 1 — PREPROCESSING
        # ====================================================

        run_command([
            sys.executable,
            str(PREPROCESS_SCRIPT),
            "--image",
            str(upload_path)
        ])


        normalized_path = (
            PROJECT_ROOT /
            "results" /
            "preprocessing" /
            "normalized" /
            f"{upload_path.stem}_normalized.png"
        )

        enhanced_path = (
            PROJECT_ROOT /
            "results" /
            "preprocessing" /
            "enhanced" /
            f"{upload_path.stem}_enhanced.png"
        )


        if not normalized_path.exists():

            raise RuntimeError(
                "Preprocessing completed but "
                "normalized image was not created."
            )


        if not enhanced_path.exists():

            raise RuntimeError(
                "Preprocessing completed but "
                "enhanced image was not created."
            )


        # ----------------------------------------------------
        # Copy preprocessed image to web results
        # ----------------------------------------------------

        processed_web_path = (
            RESULT_DIR /
            f"{stem}_{unique_id}_processed.png"
        )

        copy_result(
            enhanced_path,
            processed_web_path
        )


        # ====================================================
        # STEP 2 — OS-CFAR
        # ====================================================

        run_command([
            sys.executable,
            str(CFAR_SCRIPT),
            "--image",
            str(upload_path)
        ])


        cfar_path = (
            PROJECT_ROOT /
            "results" /
            "cfar" /
            "visualizations" /
            f"{upload_path.stem}_os_cfar.png"
        )


        if not cfar_path.exists():

            raise RuntimeError(
                "OS-CFAR completed but "
                "visualization was not created."
            )


        cfar_web_path = (
            RESULT_DIR /
            f"{stem}_{unique_id}_cfar.png"
        )


        copy_result(
            cfar_path,
            cfar_web_path
        )


        # ====================================================
        # STEP 3 — YOLOv5
        # ====================================================

        yolo_project = (
            RESULT_DIR /
            "yolo"
        )

        yolo_name = (
            f"{stem}_{unique_id}"
        )


        run_command([
            sys.executable,
            str(YOLO_SCRIPT),

            "--weights",
            str(MODEL_PATH),

            "--source",
            str(upload_path),

            "--img",
            "640",

            "--conf-thres",
            "0.25",

            "--device",
            "cpu",

            "--save-txt",

            "--save-conf",

            "--project",
            str(yolo_project),

            "--name",
            yolo_name,

            "--exist-ok"
        ])


        yolo_output_dir = (
            yolo_project /
            yolo_name
        )


        # ----------------------------------------------------
        # Find YOLO prediction image
        # ----------------------------------------------------

        yolo_image = find_file(
            yolo_output_dir,
            upload_path.stem,
            [
                ".jpg",
                ".jpeg",
                ".png"
            ]
        )


        # ----------------------------------------------------
        # YOLO labels
        # ----------------------------------------------------

        yolo_txt = (
            yolo_output_dir /
            "labels" /
            f"{upload_path.stem}.txt"
        )


        if yolo_image is None:

            raise RuntimeError(
                "YOLO completed but "
                "prediction image was not found."
            )


        # ====================================================
        # YOLO DETECTION DETAILS
        # ====================================================

        original_gray = cv2.imread(
            str(upload_path),
            cv2.IMREAD_GRAYSCALE
        )


        if original_gray is None:

            raise RuntimeError(
                "Could not read uploaded image "
                "for detection details."
            )


        image_height, image_width = (
            original_gray.shape
        )


        detections = parse_yolo_results(
            yolo_txt,
            image_width,
            image_height
        )


        # ====================================================
        # STEP 4 — COMPARISON
        # ====================================================

        comparison_path = (
            RESULT_DIR /
            f"{stem}_{unique_id}_comparison.png"
        )


        create_comparison(
            cfar_web_path,
            yolo_image,
            comparison_path
        )


        # ====================================================
        # CFAR JSON
        # ====================================================

        cfar_json = (
            PROJECT_ROOT /
            "results" /
            "cfar" /
            "detections" /
            f"{upload_path.stem}_os_cfar.json"
        )


        cfar_data = {}


        if cfar_json.exists():

            try:

                cfar_data = json.loads(
                    cfar_json.read_text(
                        encoding="utf-8"
                    )
                )

            except Exception:

                cfar_data = {}


        cfar_metrics = (
            cfar_data.get(
                "metrics",
                {}
            )
        )


        cfar_count = (
            cfar_data.get(
                "candidate_count",
                0
            )
        )


        # ====================================================
        # CURRENT IMAGE YOLO METRICS
        # ====================================================

        total_detections = len(
            detections
        )


        average_confidence = (

            round(
                sum(
                    d["confidence"]
                    for d in detections
                ) / total_detections,
                3
            )

            if total_detections > 0

            else 0

        )


        # ====================================================
        # RETURN RESULTS
        # ====================================================

        return jsonify({

            "success": True,

            "message":
                "Detection completed successfully.",


            # ------------------------------------------------
            # ORIGINAL IMAGE
            # ------------------------------------------------

            "original":
                "/results/web/"
                + original_browser_path.name,


            # ------------------------------------------------
            # PREPROCESSED IMAGE
            # ------------------------------------------------

            "processed":
                "/results/web/"
                + processed_web_path.name,


            # ------------------------------------------------
            # CFAR
            #
            # IMPORTANT:
            # image is now inside the CFAR object.
            # This fixes the duplicate "cfar" JSON key.
            # ------------------------------------------------

            "cfar": {

                "image":
                    "/results/web/"
                    + cfar_web_path.name,

                "detections":
                    cfar_count,

                "tp":
                    cfar_metrics.get(
                        "TP",
                        0
                    ),

                "fp":
                    cfar_metrics.get(
                        "FP",
                        0
                    ),

                "fn":
                    cfar_metrics.get(
                        "FN",
                        0
                    ),

                "precision":
                    cfar_metrics.get(
                        "precision",
                        0
                    ),

                "recall":
                    cfar_metrics.get(
                        "recall",
                        0
                    ),

                "f1":
                    cfar_metrics.get(
                        "f1",
                        0
                    )
            },


            # ------------------------------------------------
            # YOLO IMAGE
            # ------------------------------------------------

            "yolo":
                "/results/web/yolo/"
                + yolo_name
                + "/"
                + yolo_image.name,


            # ------------------------------------------------
            # SIDE-BY-SIDE COMPARISON
            # ------------------------------------------------

            "comparison":
                "/results/web/"
                + comparison_path.name,


            # ------------------------------------------------
            # YOLO DETECTION DETAILS
            # ------------------------------------------------

            "detections":
                detections,


            # ------------------------------------------------
            # CURRENT IMAGE TOTAL SHIPS
            # ------------------------------------------------

            "total_detections":
                total_detections,


            # ------------------------------------------------
            # CURRENT IMAGE AVERAGE CONFIDENCE
            # ------------------------------------------------

            "average_confidence":
                average_confidence,


            # ------------------------------------------------
            # YOLO TEST-SET MODEL METRICS
            #
            # These remain fixed because they describe
            # the trained model's evaluation on the test set,
            # not the currently uploaded image.
            # ------------------------------------------------

        "model_metrics": {
            "precision": 0.957,
            "recall": 0.937,
            "f1": 0.947,
            "map50": 0.979,
            "map5095": 0.648
        }

        })


    except Exception as e:

        print(
            "\nERROR:"
        )

        print(
            str(e)
        )

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# SERVE WEB RESULTS
# ============================================================

@app.route(
    "/results/web/<path:filename>"
)
def serve_result(filename):

    return send_from_directory(
        RESULT_DIR,
        filename
    )


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "SAR SHIP DETECTOR"
    )

    print(
        "========================================"
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Preprocessing:",
        PREPROCESS_SCRIPT
    )

    print(
        "CFAR:",
        CFAR_SCRIPT
    )

    print(
        "YOLO:",
        YOLO_SCRIPT
    )

    print(
        "========================================"
    )

    print(
        "Open: http://127.0.0.1:5000"
    )

    print(
        "========================================\n"
    )


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )