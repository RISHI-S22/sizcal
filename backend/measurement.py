from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class SIZCALMeasurementEngine:
    """
    SIZCAL body-measurement engine.

    Pipeline:

        Image
          ↓
        MediaPipe Pose
          ↓
        body landmarks
          ↓
        pixel geometry
          ↓
        scale calibration
          ↓
        centimeter estimates

    Important:
        MediaPipe provides body landmarks.
        It does not directly provide garment/body circumference
        measurements in centimeters.

        A known physical reference length is therefore required
        for metric scale calibration.
    """

    # --------------------------------------------------------
    # MediaPipe landmark indices
    # --------------------------------------------------------

    NOSE = 0

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14

    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    LEFT_HIP = 23
    RIGHT_HIP = 24

    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    LEFT_HEEL = 29
    RIGHT_HEEL = 30

    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 2,
    ):
        self.min_detection_confidence = (
            min_detection_confidence
        )

        self.min_tracking_confidence = (
            min_tracking_confidence
        )

        self.model_complexity = model_complexity

        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=(
                min_detection_confidence
            ),
            min_tracking_confidence=(
                min_tracking_confidence
            ),
        )

    # ========================================================
    # IMAGE LOADING
    # ========================================================

    @staticmethod
    def load_image(
        image_path: str,
    ) -> np.ndarray:

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{path}"
            )

        image = cv2.imread(
            str(path)
        )

        if image is None:
            raise ValueError(
                f"Unable to read image:\n{path}"
            )

        return image

    # ========================================================
    # POSE DETECTION
    # ========================================================

    def detect_pose(
        self,
        image: np.ndarray,
    ):
        """
        Detect pose landmarks.

        Returns:
            MediaPipe pose result.
        """

        if image is None:
            raise ValueError(
                "Input image is None."
            )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        result = self.pose.process(
            rgb
        )

        if not result.pose_landmarks:
            raise RuntimeError(
                "No human pose detected in image."
            )

        return result

    # ========================================================
    # LANDMARK EXTRACTION
    # ========================================================

    @staticmethod
    def landmark_to_pixel(
        landmark,
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float]:

        x = (
            landmark.x
            * image_width
        )

        y = (
            landmark.y
            * image_height
        )

        return x, y

    def get_landmarks(
        self,
        result,
        image_width: int,
        image_height: int,
    ) -> Dict[int, Tuple[float, float]]:

        landmarks = {}

        for index, landmark in enumerate(
            result.pose_landmarks.landmark
        ):

            landmarks[index] = (
                self.landmark_to_pixel(
                    landmark,
                    image_width,
                    image_height,
                )
            )

        return landmarks

    # ========================================================
    # VISIBILITY
    # ========================================================

    @staticmethod
    def get_landmark_visibility(
        result,
        index: int,
    ) -> float:

        return float(
            result.pose_landmarks
            .landmark[index]
            .visibility
        )

    @staticmethod
    def check_landmark_visibility(
        result,
        indices,
        threshold: float = 0.5,
    ) -> bool:

        for index in indices:

            visibility = (
                SIZCALMeasurementEngine
                .get_landmark_visibility(
                    result,
                    index,
                )
            )

            if visibility < threshold:
                return False

        return True

    # ========================================================
    # DISTANCE
    # ========================================================

    @staticmethod
    def euclidean_distance(
        point_a,
        point_b,
    ) -> float:

        return float(
            np.sqrt(
                (
                    point_a[0]
                    - point_b[0]
                ) ** 2
                +
                (
                    point_a[1]
                    - point_b[1]
                ) ** 2
            )
        )

    # ========================================================
    # SCALE CALIBRATION
    # ========================================================

    @staticmethod
    def calculate_scale(
        reference_pixel_length: float,
        reference_length_cm: float,
    ) -> float:
        """
        Calculate centimeters per pixel.

            alpha = real_length_cm / pixel_length

        """

        if reference_pixel_length <= 0:
            raise ValueError(
                "Reference pixel length "
                "must be greater than zero."
            )

        if reference_length_cm <= 0:
            raise ValueError(
                "Reference physical length "
                "must be greater than zero."
            )

        alpha = (
            reference_length_cm
            / reference_pixel_length
        )

        return float(alpha)

    # ========================================================
    # PIXELS → CM
    # ========================================================

    @staticmethod
    def pixels_to_cm(
        pixel_length: float,
        scale_cm_per_pixel: float,
    ) -> float:

        return float(
            pixel_length
            * scale_cm_per_pixel
        )

    # ========================================================
    # SHOULDER WIDTH
    # ========================================================

    def shoulder_width(
        self,
        landmarks,
        scale,
    ) -> Optional[float]:

        left = landmarks[
            self.LEFT_SHOULDER
        ]

        right = landmarks[
            self.RIGHT_SHOULDER
        ]

        pixel_distance = (
            self.euclidean_distance(
                left,
                right,
            )
        )

        return self.pixels_to_cm(
            pixel_distance,
            scale,
        )

    # ========================================================
    # HIP WIDTH
    # ========================================================

    def hip_width(
        self,
        landmarks,
        scale,
    ) -> Optional[float]:

        left = landmarks[
            self.LEFT_HIP
        ]

        right = landmarks[
            self.RIGHT_HIP
        ]

        pixel_distance = (
            self.euclidean_distance(
                left,
                right,
            )
        )

        return self.pixels_to_cm(
            pixel_distance,
            scale,
        )

    # ========================================================
    # TORSO HEIGHT
    # ========================================================

    def torso_height(
        self,
        landmarks,
        scale,
    ) -> Optional[float]:

        left_shoulder = landmarks[
            self.LEFT_SHOULDER
        ]

        right_shoulder = landmarks[
            self.RIGHT_SHOULDER
        ]

        left_hip = landmarks[
            self.LEFT_HIP
        ]

        right_hip = landmarks[
            self.RIGHT_HIP
        ]

        shoulder_center = (
            (
                left_shoulder[0]
                + right_shoulder[0]
            ) / 2.0,
            (
                left_shoulder[1]
                + right_shoulder[1]
            ) / 2.0,
        )

        hip_center = (
            (
                left_hip[0]
                + right_hip[0]
            ) / 2.0,
            (
                left_hip[1]
                + right_hip[1]
            ) / 2.0,
        )

        pixel_distance = (
            self.euclidean_distance(
                shoulder_center,
                hip_center,
            )
        )

        return self.pixels_to_cm(
            pixel_distance,
            scale,
        )

    # ========================================================
    # INSEAM
    # ========================================================

    def inseam(
        self,
        landmarks,
        scale,
    ) -> Optional[float]:

        left_hip = landmarks[
            self.LEFT_HIP
        ]

        left_knee = landmarks[
            self.LEFT_KNEE
        ]

        left_ankle = landmarks[
            self.LEFT_ANKLE
        ]

        right_hip = landmarks[
            self.RIGHT_HIP
        ]

        right_knee = landmarks[
            self.RIGHT_KNEE
        ]

        right_ankle = landmarks[
            self.RIGHT_ANKLE
        ]

        left_leg = (
            self.euclidean_distance(
                left_hip,
                left_knee,
            )
            +
            self.euclidean_distance(
                left_knee,
                left_ankle,
            )
        )

        right_leg = (
            self.euclidean_distance(
                right_hip,
                right_knee,
            )
            +
            self.euclidean_distance(
                right_knee,
                right_ankle,
            )
        )

        # Average both sides.
        pixel_length = (
            left_leg + right_leg
        ) / 2.0

        return self.pixels_to_cm(
            pixel_length,
            scale,
        )

    # ========================================================
    # ELLIPSE CIRCUMFERENCE
    # ========================================================

    @staticmethod
    def ellipse_circumference(
        semi_major_axis: float,
        semi_minor_axis: float,
    ) -> float:
        """
        Ramanujan approximation for ellipse circumference.

        C ≈ π [3(a+b) - sqrt((3a+b)(a+3b))]
        """

        a = max(
            semi_major_axis,
            semi_minor_axis,
        )

        b = min(
            semi_major_axis,
            semi_minor_axis,
        )

        if a <= 0 or b <= 0:
            return 0.0

        circumference = (
            np.pi
            * (
                3.0 * (a + b)
                - np.sqrt(
                    (3.0 * a + b)
                    * (a + 3.0 * b)
                )
            )
        )

        return float(
            circumference
        )

    # ========================================================
    # CHEST CIRCUMFERENCE
    # ========================================================

    def chest_circumference(
        self,
        landmarks,
        scale,
    ) -> Optional[float]:

        left_shoulder = landmarks[
            self.LEFT_SHOULDER
        ]

        right_shoulder = landmarks[
            self.RIGHT_SHOULDER
        ]

        shoulder_width_px = (
            self.euclidean_distance(
                left_shoulder,
                right_shoulder,
            )
        )

        # Estimate torso depth from shoulder width.
        #
        # This is a geometric approximation, not a direct
        # circumference measurement.
        semi_major = (
            shoulder_width_px
            * 0.5
        )

        semi_minor = (
            shoulder_width_px
            * 0.35
        )

        circumference_px = (
            self.ellipse_circumference(
                semi_major,
                semi_minor,
            )
        )

        return self.pixels_to_cm(
            circumference_px,
            scale,
        )

    # ========================================================
    # WAIST CIRCUMFERENCE
    # ========================================================

    def waist_circumference(
        self,
        landmarks,
        scale,
    ) -> Optional[float]:

        left_hip = landmarks[
            self.LEFT_HIP
        ]

        right_hip = landmarks[
            self.RIGHT_HIP
        ]

        hip_width_px = (
            self.euclidean_distance(
                left_hip,
                right_hip,
            )
        )

        # Approximate waist width as a fraction
        # of the hip width.
        waist_width_px = (
            hip_width_px
            * 0.85
        )

        semi_major = (
            waist_width_px
            * 0.5
        )

        semi_minor = (
            waist_width_px
            * 0.35
        )

        circumference_px = (
            self.ellipse_circumference(
                semi_major,
                semi_minor,
            )
        )

        return self.pixels_to_cm(
            circumference_px,
            scale,
        )

    # ========================================================
    # HIP CIRCUMFERENCE
    # ========================================================

    def hip_circumference(
        self,
        landmarks,
        scale,
    ) -> Optional[float]:

        left_hip = landmarks[
            self.LEFT_HIP
        ]

        right_hip = landmarks[
            self.RIGHT_HIP
        ]

        hip_width_px = (
            self.euclidean_distance(
                left_hip,
                right_hip,
            )
        )

        semi_major = (
            hip_width_px
            * 0.5
        )

        semi_minor = (
            hip_width_px
            * 0.40
        )

        circumference_px = (
            self.ellipse_circumference(
                semi_major,
                semi_minor,
            )
        )

        return self.pixels_to_cm(
            circumference_px,
            scale,
        )

    # ========================================================
    # COMPLETE MEASUREMENT EXTRACTION
    # ========================================================

    def measure(
    self,
    image,
    reference_pixel_length,
    reference_length_cm,
    ):
        if isinstance(image, str):
            image = cv2.imread(image)

            if image is None:
                raise ValueError(
                    f"Could not read image: {image}"
                )

        if image is None:
            raise ValueError(
                "Input image is None."
            )

        if not hasattr(image, "shape"):
            raise ValueError(
                "Input must be an image array "
                "or a valid image file path."
            )

        height, width = image.shape[:2]

        # Convert OpenCV BGR → RGB for MediaPipe.
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        result = self.pose.process(
            rgb_image
        )

        if not result.pose_landmarks:
            raise RuntimeError(
                "No human pose detected in image."
            )

        # ----------------------------------------------------
        # Landmarks
        # ----------------------------------------------------

        landmarks = self.get_landmarks(
            result,
            width,
            height,
        )

        # ----------------------------------------------------
        # Scale
        # ----------------------------------------------------

        scale = self.calculate_scale(
            reference_pixel_length,
            reference_length_cm,
        )

        visibility_threshold = 0.5

        # ----------------------------------------------------
        # Required landmarks
        # ----------------------------------------------------

        required = [
            self.LEFT_SHOULDER,
            self.RIGHT_SHOULDER,
            self.LEFT_HIP,
            self.RIGHT_HIP,
            self.LEFT_KNEE,
            self.RIGHT_KNEE,
            self.LEFT_ANKLE,
            self.RIGHT_ANKLE,
        ]

        visible = (
            self.check_landmark_visibility(
                result,
                required,
                visibility_threshold,
            )
        )

        # ----------------------------------------------------
        # Measurements
        # ----------------------------------------------------

        measurements = {
            "shoulder_width_cm": (
                self.shoulder_width(
                    landmarks,
                    scale,
                )
            ),

            "chest_circumference_cm": (
                self.chest_circumference(
                    landmarks,
                    scale,
                )
            ),

            "waist_circumference_cm": (
                self.waist_circumference(
                    landmarks,
                    scale,
                )
            ),

            "hip_circumference_cm": (
                self.hip_circumference(
                    landmarks,
                    scale,
                )
            ),

            "inseam_cm": (
                self.inseam(
                    landmarks,
                    scale,
                )
            ),

            "torso_height_cm": (
                self.torso_height(
                    landmarks,
                    scale,
                )
            ),
        }

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        return {
            "success": True,
            "image_width": width,
            "image_height": height,
            "reference_length_cm": reference_length_cm,
            "reference_pixel_length": reference_pixel_length,
            "scale_cm_per_pixel": scale,
            "landmarks_visible": visible,
            "measurements": measurements,
        }

    # ========================================================

    def draw_landmarks(
        self,
        image: np.ndarray,
        result,
    ) -> np.ndarray:

        output = image.copy()

        self.mp_pose.POSE_CONNECTIONS

        mp.solutions.drawing_utils.draw_landmarks(
            output,
            result.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
        )

        return output

    # ========================================================
    # CLEANUP
    # ========================================================

    def close(self):
        self.pose.close()