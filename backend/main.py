from pathlib import Path
import json
import os
import tempfile
from datetime import datetime

import cv2
import mediapipe as mp

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.measurement import SIZCALMeasurementEngine
from backend.recommender import SIZCALRecommender


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
HISTORY_DIR = PROJECT_ROOT / "data" / "measurements"
HISTORY_FILE = HISTORY_DIR / "history.json"

HISTORY_DIR.mkdir(parents=True, exist_ok=True)

if not HISTORY_FILE.exists():
    HISTORY_FILE.write_text("[]", encoding="utf-8")


app = FastAPI(
    title="SIZCAL",
    description="AI-Powered Clothing Size Recommendation System",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

measurement_engine = SIZCALMeasurementEngine()
recommender = SIZCALRecommender(
    project_root=PROJECT_ROOT
)


def load_history():
    try:
        return json.loads(
            HISTORY_FILE.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return []


def save_history(history):
    HISTORY_FILE.write_text(
        json.dumps(
            history,
            indent=2
        ),
        encoding="utf-8"
    )


@app.get("/")
def frontend():
    return FileResponse(
        FRONTEND_DIR / "index.html"
    )


app.mount(
    "/static",
    StaticFiles(
        directory=FRONTEND_DIR
    ),
    name="static"
)


@app.get("/api/health")
def health():
    return {
        "success": True,
        "status": "healthy",
        "service": "SIZCAL",
        "components": {
            "mediapipe": True,
            "xgboost": True,
            "recommender": True,
            "camera": True,
            "brand_dataset": len(
                recommender.brands
            ),
        },
    }


@app.get("/api/brands")
def get_brands():
    return {
        "success": True,
        "brands": recommender.get_brands(),
    }


@app.get("/api/sizes")
def get_sizes():
    return {
        "success": True,
        "sizes": recommender.get_sizes(),
    }


def detect_body_height_pixels(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    height, width = image.shape[:2]

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        result = pose.process(rgb)

    if not result.pose_landmarks:
        return None

    visible_y = []

    for landmark in result.pose_landmarks.landmark:

        if landmark.visibility < 0.5:
            continue

        y = landmark.y * height

        if 0 <= y <= height:
            visible_y.append(y)

    if len(visible_y) < 5:
        return None

    detected_height = max(visible_y) - min(visible_y)

    if detected_height <= 0:
        return None

    return float(detected_height * 1.06)


def process_image(
    image_bytes,
    filename,
    reference_pixel_length,
    reference_length_cm,
):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(filename).suffix or ".jpg",
        ) as temp:
            temp.write(image_bytes)
            temp_path = temp.name

        result = measurement_engine.measure(
            temp_path,
            reference_pixel_length=float(
                reference_pixel_length
            ),
            reference_length_cm=float(
                reference_length_cm
            ),
        )

        return result

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    reference_pixel_length: float = Form(200.0),
    reference_length_cm: float = Form(10.0),
    brand: str = Form(""),
):
    image_bytes = await file.read()

    measurement = process_image(
        image_bytes,
        file.filename or "image.jpg",
        reference_pixel_length,
        reference_length_cm,
    )

    if not measurement.get("success"):
        return measurement

    measurements = measurement["measurements"]

    recommendation = recommender.recommend(
        measurements,
        brand=brand or None,
    )

    return {
        "success": True,
        "measurement": measurement,
        "recommendation": recommendation,
    }

def detect_body_height_pixels(
    image_path
):
    """
    Estimate full body height in pixels using
    MediaPipe Pose landmarks.

    The detected landmark span is expanded slightly
    because the top of the head and bottom of the feet
    are not represented exactly by the standard pose
    landmarks.
    """

    import cv2

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return None

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = (
        measurement_engine.pose.process(
            image_rgb
        )
    )

    if not results.pose_landmarks:
        return None

    landmarks = (
        results.pose_landmarks.landmark
    )

    visible_y = []

    for landmark in landmarks:

        if landmark.visibility >= 0.50:

            visible_y.append(
                landmark.y
            )

    if len(visible_y) < 5:
        return None

    height_normalized = (
        max(visible_y)
        -
        min(visible_y)
    )

    image_height = image.shape[0]

    detected_height_pixels = (
        height_normalized *
        image_height
    )

    # Small correction for head/foot extremities
    estimated_body_height = (
        detected_height_pixels * 1.06
    )

    return float(
        estimated_body_height
    )

@app.post("/api/camera/measure-multiview")
async def camera_measure_multiview(
    front_image: UploadFile = File(...),
    left_image: UploadFile = File(...),
    back_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
    user_height_cm: float = Form(...),
    category: str = Form("top_wear"),
    brand: str = Form("")
):
    """
    Multi-view body measurement and clothing-size recommendation.

    category:
        top_wear
        bottom_wear

    The user's actual height is used to calculate the
    pixel-to-centimeter scale instead of using a fixed
    artificial calibration.
    """

    import tempfile
    import os

    temp_files = []

    try:

        # ----------------------------------------------------
        # VALIDATE CATEGORY
        # ----------------------------------------------------

        category = category.strip().lower()

        if category not in {
            "top_wear",
            "bottom_wear"
        }:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid category. "
                    "Use top_wear or bottom_wear."
                )
            )

        # ----------------------------------------------------
        # VALIDATE HEIGHT
        # ----------------------------------------------------

        user_height_cm = float(user_height_cm)

        if not 100 <= user_height_cm <= 250:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Height must be between "
                    "100 and 250 cm."
                )
            )

        # ----------------------------------------------------
        # UPLOADS
        # ----------------------------------------------------

        uploaded_images = {
            "front": front_image,
            "left": left_image,
            "back": back_image,
            "right": right_image
        }

        measurements_by_view = {}

        # ----------------------------------------------------
        # PROCESS EACH VIEW
        # ----------------------------------------------------

        for view_name, upload in uploaded_images.items():

            image_bytes = await upload.read()

            if not image_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"{view_name} image is empty."
                )

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as temp:

                temp.write(image_bytes)

                temp_path = temp.name

            temp_files.append(temp_path)

            # ------------------------------------------------
            # DETECT BODY HEIGHT IN PIXELS
            # ------------------------------------------------

            body_height_pixels = (
                detect_body_height_pixels(
                    temp_path
                )
            )

            if (
                body_height_pixels is None
                or body_height_pixels <= 0
            ):
                measurements_by_view[
                    view_name
                ] = {
                    "success": False,
                    "error": (
                        "Could not detect "
                        "body height."
                    )
                }

                continue

            # ------------------------------------------------
            # REAL SCALE
            # ------------------------------------------------

            scale_cm_per_pixel = (
                user_height_cm /
                body_height_pixels
            )

            # ------------------------------------------------
            # MEASUREMENT ENGINE
            # ------------------------------------------------

            measurement_result = (
                measurement_engine.measure(
                    temp_path,
                    reference_pixel_length=1.0,
                    reference_length_cm=(
                        scale_cm_per_pixel
                    )
                )
            )

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            if measurement_result:

                measurement_result[
                    "body_height_pixels"
                ] = body_height_pixels

                measurement_result[
                    "scale_cm_per_pixel"
                ] = scale_cm_per_pixel

            measurements_by_view[
                view_name
            ] = measurement_result

        # ----------------------------------------------------
        # SUCCESSFUL VIEWS
        # ----------------------------------------------------

        successful_views = []

        for view_name, result in (
            measurements_by_view.items()
        ):

            if (
                result
                and result.get("success")
            ):
                successful_views.append(
                    view_name
                )

        if not successful_views:

            raise HTTPException(
                status_code=422,
                detail=(
                    "MediaPipe could not detect "
                    "a usable body pose."
                )
            )

        # ----------------------------------------------------
        # HELPER
        # ----------------------------------------------------

        def get_measurement(
            view_name,
            field_name
        ):

            result = measurements_by_view.get(
                view_name
            )

            if not result:
                return None

            values = result.get(
                "measurements",
                {}
            )

            return values.get(
                field_name
            )

        def first_available(
            field_name,
            preferred_views
        ):

            for view_name in preferred_views:

                value = get_measurement(
                    view_name,
                    field_name
                )

                if value is not None:
                    return float(value)

            return None

        # ----------------------------------------------------
        # MEASUREMENT PRIORITIES
        # ----------------------------------------------------

        if category == "top_wear":

            preferred_width_views = [
                "front",
                "back",
                "left",
                "right"
            ]

            preferred_inseam_views = [
                "front",
                "left",
                "right",
                "back"
            ]

        else:

            preferred_width_views = [
                "front",
                "back",
                "left",
                "right"
            ]

            preferred_inseam_views = [
                "left",
                "right",
                "front",
                "back"
            ]

        # ----------------------------------------------------
        # COMBINE MEASUREMENTS
        # ----------------------------------------------------

        combined_measurements = {

            "shoulder_width_cm":
                first_available(
                    "shoulder_width_cm",
                    preferred_width_views
                ),

            "chest_circumference_cm":
                first_available(
                    "chest_circumference_cm",
                    preferred_width_views
                ),

            "waist_circumference_cm":
                first_available(
                    "waist_circumference_cm",
                    preferred_width_views
                ),

            "hip_circumference_cm":
                first_available(
                    "hip_circumference_cm",
                    preferred_width_views
                ),

            "inseam_cm":
                first_available(
                    "inseam_cm",
                    preferred_inseam_views
                )
        }

        # ----------------------------------------------------
        # CATEGORY-SPECIFIC REQUIRED FIELDS
        # ----------------------------------------------------

        if category == "top_wear":

            required_fields = [
                "shoulder_width_cm",
                "chest_circumference_cm",
                "waist_circumference_cm"
            ]

        else:

            required_fields = [
                "waist_circumference_cm",
                "hip_circumference_cm",
                "inseam_cm"
            ]

        missing_fields = [
            field
            for field in required_fields
            if combined_measurements.get(field)
            is None
        ]

        if missing_fields:

            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "Required measurements "
                        "could not be extracted."
                    ),
                    "category": category,
                    "missing_fields":
                        missing_fields,
                    "successful_views":
                        successful_views
                }
            )

        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        # Current trained brand models are top-wear models.
        # Therefore bottom-wear recommendations are not
        # falsely generated from top-wear charts.

        if category == "top_wear":

            recommendation = (
                recommender.recommend(
                    combined_measurements,
                    brand=brand or None
                )
            )

        else:

            recommendation = {
                "category": "bottom_wear",
                "available": False,
                "message": (
                    "Bottom-wear brand models "
                    "are not trained yet. "
                    "Measurements were extracted "
                    "successfully."
                ),
                "measurement_basis": [
                    "waist_circumference_cm",
                    "hip_circumference_cm",
                    "inseam_cm"
                ]
            }

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return {

            "success": True,

            "category": category,

            "category_name": (
                "Top Wear"
                if category == "top_wear"
                else "Bottom Wear"
            ),

            "user_height_cm":
                user_height_cm,

            "pipeline": {

                "front_view":
                    "front" in successful_views,

                "left_view":
                    "left" in successful_views,

                "back_view":
                    "back" in successful_views,

                "right_view":
                    "right" in successful_views,

                "pose_detection": True,

                "measurement_extraction": True,

                "xgboost_recommendation":
                    category == "top_wear"
            },

            "views_captured": [
                "front",
                "left",
                "back",
                "right"
            ],

            "successful_views":
                successful_views,

            "measurements_by_view":
                measurements_by_view,

            "measurement": {

                "success": True,

                "measurements":
                    combined_measurements
            },

            "recommendation":
                recommendation
        }

    except HTTPException:
        raise

    except Exception as exc:

        print(
            "Multi-view measurement error:",
            exc
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    finally:

        for path in temp_files:

            try:

                if os.path.exists(path):
                    os.remove(path)

            except Exception:
                pass

@app.post("/api/history/save")
async def save_measurement(
    data: str = Form(...)
):
    try:
        record = json.loads(data)

        record["saved_at"] = (
            datetime.now().isoformat()
        )

        history = load_history()
        history.insert(0, record)

        save_history(history[:100])

        return {
            "success": True,
            "record": record,
        }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }


@app.get("/api/history")
def get_history():
    return {
        "success": True,
        "history": load_history(),
    }


@app.delete("/api/history")
def clear_history():
    save_history([])

    return {
        "success": True,
    }


@app.on_event("startup")
async def startup():
    print()
    print("=" * 70)
    print("SIZCAL SERVER")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Brand dataset: {len(recommender.brands)} brands")
    print()
    print("Endpoints:")
    print("  GET    /")
    print("  GET    /api/health")
    print("  GET    /api/brands")
    print("  GET    /api/sizes")
    print("  POST   /api/analyze")
    print("  POST   /api/camera/measure-multiview")
    print("  POST   /api/history/save")
    print("  GET    /api/history")
    print("  DELETE /api/history")
    print()
    print("Brand-aware recommendation: ENABLED")
    print("SIZCAL READY")
    print("=" * 70)
