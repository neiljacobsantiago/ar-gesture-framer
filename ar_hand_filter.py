"""AR Hand Filter: a real-time "director's frame" gesture filter.

Captures a webcam feed, tracks hands with Google MediaPipe, and detects the
"director's frame" gesture: two hands each making an L-shape with the thumb and
index finger. While the gesture is held, a thin white rectangle is drawn
between the hands and the active image filter is applied inside it (or to the
whole frame, see the ``F`` key).

Controls:
    N        Normal (no filter)
    S        Sepia
    G        Grayscale
    I        Invert
    TAB      Cycle to the next filter
    F        Toggle filter scope: inside the virtual frame / full image
    D        Toggle landmark debug overlay
    Q / ESC  Quit

Usage:
    python ar_hand_filter.py [--camera 0] [--max-hands 2] [--no-mirror]
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_NAME = "AR Hand Filter"

MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
DEFAULT_MAX_HANDS = 2

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20

# Landmarks whose extremes define the virtual frame.
FRAME_LANDMARKS = (THUMB_TIP, INDEX_TIP)

# Gesture thresholds, expressed relative to palm size so they hold at any
# distance from the camera.
PINCH_CLOSED_RATIO = 0.25   # thumb/index tip gap at which a pinch is certain
PINCH_OPEN_RATIO = 0.80     # gap at which the fingers are clearly apart
FRAME_CONFIDENCE_THRESHOLD = 0.5
MIN_BOX_SIZE_PX = 40

# Box smoothing: exponential moving average weight for the newest box and the
# number of frames the box survives a momentary tracking dropout.
BOX_SMOOTHING_ALPHA = 0.45
BOX_HOLD_FRAMES = 6

# Tasks API fallback (mediapipe releases that no longer ship mp.solutions).
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

# Sepia kernel for BGR pixels. Rows are output channels (B, G, R), columns
# are input channels (B, G, R): the standard RGB sepia matrix reordered.
SEPIA_KERNEL_BGR = np.array(
    [
        [0.131, 0.534, 0.272],
        [0.168, 0.686, 0.349],
        [0.189, 0.769, 0.393],
    ],
    dtype=np.float32,
)

BOX_COLOR = (255, 255, 255)
BOX_THICKNESS = 2
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 (inclusive top-left, exclusive bottom-right)


# ---------------------------------------------------------------------------
# Geometry and gesture recognition
# ---------------------------------------------------------------------------

def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle ABC in degrees (the angle at vertex ``b``)."""
    ba = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    bc = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return vector_angle(ba, bc)


def vector_angle(u: np.ndarray, v: np.ndarray) -> float:
    """Return the angle between two vectors in degrees (0-180)."""
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom < 1e-6:
        return 0.0
    cos = float(np.dot(u, v)) / denom
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def ramp(value: float, low: float, high: float) -> float:
    """Map ``value`` linearly from [low, high] to [0, 1], clamped.

    ``low`` may be greater than ``high`` for a decreasing ramp.
    """
    if high == low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def palm_size(points: np.ndarray) -> float:
    """Return a scale reference for a hand: wrist to middle-finger knuckle."""
    return distance(points[WRIST], points[MIDDLE_MCP])


def pinch_confidence(points: np.ndarray, palm_size: float) -> float:
    """Return how strongly the thumb and index tips are pinched together.

    Args:
        points: (21, 2) array of landmark pixel coordinates for one hand.
        palm_size: Scale reference for the hand (see :func:`palm_size`).

    Returns:
        1.0 when the thumb tip (4) touches the index tip (8), falling to 0.0
        once the gap reaches ``PINCH_OPEN_RATIO`` palm sizes. An L-shape or
        pointing pose therefore scores close to 0.
    """
    thumb_index = distance(points[THUMB_TIP], points[INDEX_TIP]) / max(palm_size, 1e-6)
    return ramp(thumb_index, PINCH_OPEN_RATIO, PINCH_CLOSED_RATIO)


def curled_score(points: np.ndarray, tip: int, pip: int) -> float:
    """Return 1.0 when a finger is folded (tip nearer the wrist than its PIP joint)."""
    wrist = points[WRIST]
    ratio = distance(points[tip], wrist) / max(distance(points[pip], wrist), 1e-6)
    return ramp(ratio, 1.25, 0.95)


def frame_confidence(points: np.ndarray, palm_size: float) -> float:
    """Score how closely one hand matches the director's-frame L-shape.

    The L-shape needs the thumb and index finger both extended, roughly
    perpendicular and well apart, with the remaining fingers folded.

    Args:
        points: (21, 2) array of landmark pixel coordinates for one hand.
        palm_size: Scale reference for the hand (see :func:`palm_size`).

    Returns:
        A confidence in [0, 1].
    """
    palm = max(palm_size, 1e-6)

    # Thumb and index tips apart (the opposite of a pinch).
    open_score = 1.0 - pinch_confidence(points, palm)

    # Index finger straight: angle at the PIP joint close to 180 degrees.
    index_angle = angle(points[INDEX_MCP], points[INDEX_PIP], points[INDEX_TIP])
    index_straight = ramp(index_angle, 120.0, 160.0)

    # Thumb pushed out, away from the base of the index finger.
    thumb_reach = distance(points[THUMB_TIP], points[INDEX_MCP]) / palm
    thumb_extended = ramp(thumb_reach, 0.35, 0.65)

    # Thumb and index directions form an "L" (around 90 degrees).
    l_angle = vector_angle(
        points[THUMB_TIP] - points[THUMB_MCP],
        points[INDEX_TIP] - points[INDEX_MCP],
    )
    l_shape = ramp(abs(l_angle - 90.0), 55.0, 25.0)

    # Middle, ring and pinky folded. This separates an L from an open palm.
    folded = float(np.mean([
        curled_score(points, MIDDLE_TIP, MIDDLE_PIP),
        curled_score(points, RING_TIP, RING_PIP),
        curled_score(points, PINKY_TIP, PINKY_PIP),
    ]))

    core = min(open_score, index_straight, thumb_extended, l_shape)
    return core * (0.4 + 0.6 * folded)


def is_frame_hand(points: np.ndarray) -> bool:
    """Return True if a hand is making the L-shaped half of the frame."""
    return frame_confidence(points, palm_size(points)) >= FRAME_CONFIDENCE_THRESHOLD


def compute_frame_box(hands: Sequence[np.ndarray], frame_shape: Tuple[int, ...]) -> Optional[Box]:
    """Find the director's frame formed by two L-shaped hands.

    The box spans the minimum and maximum X/Y of the thumb and index tips of
    both hands. When more than two hands qualify, the pair producing the
    largest box is used.

    Args:
        hands: Landmark arrays, one (21, 2) array per detected hand.
        frame_shape: Shape of the image, used to clip the box.

    Returns:
        ``(x1, y1, x2, y2)`` in pixels, or ``None`` if no frame is formed.
    """
    candidates = [h for h in hands if is_frame_hand(h)]
    if len(candidates) < 2:
        return None

    height, width = frame_shape[:2]
    best: Optional[Box] = None
    best_area = 0
    for hand_a, hand_b in itertools.combinations(candidates, 2):
        tips = np.vstack([hand_a[list(FRAME_LANDMARKS)], hand_b[list(FRAME_LANDMARKS)]])
        x1, y1 = np.min(tips, axis=0)
        x2, y2 = np.max(tips, axis=0)
        x1, x2 = int(np.clip(x1, 0, width - 1)), int(np.clip(x2, 0, width - 1))
        y1, y2 = int(np.clip(y1, 0, height - 1)), int(np.clip(y2, 0, height - 1))
        if x2 - x1 < MIN_BOX_SIZE_PX or y2 - y1 < MIN_BOX_SIZE_PX:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best, best_area = (x1, y1, x2, y2), area
    return best


class BoxSmoother:
    """Smooths the frame box over time and rides out short tracking dropouts."""

    def __init__(self, alpha: float = BOX_SMOOTHING_ALPHA, hold_frames: int = BOX_HOLD_FRAMES) -> None:
        self.alpha = alpha
        self.hold_frames = hold_frames
        self._box: Optional[np.ndarray] = None
        self._missed = 0

    def update(self, box: Optional[Box]) -> Optional[Box]:
        """Feed the latest raw box (or ``None``) and return the smoothed box."""
        if box is None:
            self._missed += 1
            if self._missed > self.hold_frames:
                self._box = None
        else:
            self._missed = 0
            new = np.asarray(box, dtype=np.float32)
            self._box = new if self._box is None else self.alpha * new + (1.0 - self.alpha) * self._box

        if self._box is None:
            return None
        x1, y1, x2, y2 = (int(round(v)) for v in self._box)
        return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def filter_normal(image: np.ndarray) -> np.ndarray:
    """Return the image unchanged."""
    return image


def filter_sepia(image: np.ndarray) -> np.ndarray:
    """Apply the standard sepia colour matrix with ``cv2.transform``."""
    return cv2.transform(image, SEPIA_KERNEL_BGR)  # saturates to 0-255 for uint8


def filter_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale, keeping three channels."""
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def filter_invert(image: np.ndarray) -> np.ndarray:
    """Invert all colours."""
    return cv2.bitwise_not(image)


@dataclass(frozen=True)
class ImageFilter:
    """A named filter bound to a keyboard shortcut."""

    name: str
    key: str
    apply: Callable[[np.ndarray], np.ndarray]


FILTERS: Tuple[ImageFilter, ...] = (
    ImageFilter("Normal", "n", filter_normal),
    ImageFilter("Sepia", "s", filter_sepia),
    ImageFilter("Grayscale", "g", filter_grayscale),
    ImageFilter("Invert", "i", filter_invert),
)


class FilterState:
    """Minimal state machine holding the active filter and where it applies."""

    def __init__(self, filters: Sequence[ImageFilter] = FILTERS, initial: str = "Sepia") -> None:
        self.filters = list(filters)
        self._by_key: Dict[int, int] = {ord(f.key): i for i, f in enumerate(self.filters)}
        self._by_key.update({ord(f.key.upper()): i for i, f in enumerate(self.filters)})
        names = [f.name.lower() for f in self.filters]
        self.index = names.index(initial.lower()) if initial.lower() in names else 0
        self.inside_frame_only = True

    @property
    def current(self) -> ImageFilter:
        """The active filter."""
        return self.filters[self.index]

    def next(self) -> None:
        """Advance to the next filter, wrapping around."""
        self.index = (self.index + 1) % len(self.filters)

    def handle_key(self, key: int) -> bool:
        """Switch filter for a shortcut key. Returns True if the key was used."""
        if key in self._by_key:
            self.index = self._by_key[key]
            return True
        if key == 9:  # TAB
            self.next()
            return True
        if key in (ord("f"), ord("F")):
            self.inside_frame_only = not self.inside_frame_only
            return True
        return False

    def apply(self, frame: np.ndarray, box: Optional[Box]) -> None:
        """Apply the active filter to ``frame`` in place.

        In frame-only mode the filter is applied inside ``box`` and nothing
        changes when no box is shown. Otherwise the whole image is filtered.
        """
        if self.current.apply is filter_normal:
            return
        if not self.inside_frame_only:
            frame[:] = self.current.apply(frame)
        elif box is not None:
            x1, y1, x2, y2 = box
            roi = frame[y1:y2, x1:x2]
            if roi.size:
                roi[:] = self.current.apply(roi)


# ---------------------------------------------------------------------------
# Hand tracking
# ---------------------------------------------------------------------------

class HandTrackerError(RuntimeError):
    """Raised when the hand tracker cannot be initialised."""


class HandTracker:
    """MediaPipe hand tracker returning landmarks in pixel coordinates.

    Uses the classic ``mp.solutions.hands`` API when the installed mediapipe
    provides it. Newer releases removed that module, in which case the
    equivalent Tasks ``HandLandmarker`` is used (its model file is downloaded
    on first run).
    """

    def __init__(
        self,
        max_hands: int = DEFAULT_MAX_HANDS,
        detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ) -> None:
        self._last_timestamp_ms = -1
        if hasattr(mp, "solutions"):
            self.backend = "solutions"
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
        else:
            self.backend = "tasks"
            self._hands = self._create_landmarker(max_hands, detection_confidence, tracking_confidence)

    @staticmethod
    def _create_landmarker(max_hands: int, detection_confidence: float, tracking_confidence: float):
        """Build a Tasks API HandLandmarker in VIDEO mode."""
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        model_path = ensure_hand_model()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        try:
            return vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # mediapipe raises plain RuntimeError/ValueError
            raise HandTrackerError(f"Could not start the MediaPipe hand landmarker: {exc}") from exc

    def process(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        """Detect hands in a BGR frame.

        Returns:
            One (21, 2) float32 array of pixel coordinates per detected hand.
        """
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        scale = np.array([width, height], dtype=np.float32)

        if self.backend == "solutions":
            rgb.flags.writeable = False
            results = self._hands.process(rgb)
            hand_list = [h.landmark for h in (results.multi_hand_landmarks or [])]
        else:
            # VIDEO mode requires strictly increasing timestamps.
            timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp_ms
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            hand_list = self._hands.detect_for_video(image, timestamp_ms).hand_landmarks

        return [
            np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32) * scale
            for landmarks in hand_list
        ]

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()


def ensure_hand_model(path: Path = HAND_MODEL_PATH, url: str = HAND_MODEL_URL) -> Path:
    """Download the HandLandmarker model if it is not already on disk."""
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".part")
    print(f"Downloading hand landmark model to {path} ...")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HandTrackerError(
            f"Could not download the hand model ({exc}).\n"
            f"Download it manually from {url}\nand save it as {path}"
        ) from exc
    return path


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read."""


def camera_backends() -> List[int]:
    """Return the OpenCV capture backends to try, in order.

    On Windows the default Media Foundation backend often fails to deliver
    frames from some webcams, so DirectShow is tried first.
    """
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    return [cv2.CAP_ANY]


def wait_for_frame(cap: cv2.VideoCapture, timeout_s: float) -> bool:
    """Read until a frame arrives or ``timeout_s`` passes (cameras need warm-up)."""
    deadline = time.monotonic() + timeout_s
    while True:
        ok, frame = cap.read()
        if ok and frame is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def open_camera(
    index: int = 0, width: int = 1280, height: int = 720, warmup_s: float = 3.0
) -> cv2.VideoCapture:
    """Open a webcam and confirm it delivers frames.

    A camera held by another application often opens "successfully" but
    never returns a frame, so a test read is part of the check. Each capture
    backend gets ``warmup_s`` seconds to produce its first frame.

    Raises:
        CameraError: If the device cannot be opened or produces no frames.
    """
    opened = False
    for backend in camera_backends():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        opened = True
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if wait_for_frame(cap, warmup_s):
            return cap
        cap.release()

    if not opened:
        raise CameraError(
            f"Could not open camera {index}. Check that a webcam is connected "
            "and that the app has camera permission."
        )
    raise CameraError(
        f"Camera {index} opened but returned no frames. Check that:\n"
        "  - no other app is using it (Zoom, Teams, OBS, a browser tab...)\n"
        "  - Windows Settings > Privacy & security > Camera allows desktop apps\n"
        "  - the laptop's camera privacy shutter/kill switch is off\n"
        "  - it is the right device: try --camera 1 or --camera 2"
    )


# ---------------------------------------------------------------------------
# UI overlays
# ---------------------------------------------------------------------------

def draw_label(
    image: np.ndarray,
    text: str,
    anchor: str,
    scale: float = 0.6,
    margin: int = 10,
    padding: int = 6,
) -> None:
    """Draw white text on a solid black box.

    Args:
        image: Frame to draw on (modified in place).
        text: Label text.
        anchor: ``"top-left"``, ``"top-right"`` or ``"bottom-center"``.
        scale: Font scale.
        margin: Distance from the image edge, in pixels.
        padding: Space between the text and the box edge, in pixels.
    """
    thickness = 1 if scale < 0.6 else 2
    (text_w, text_h), baseline = cv2.getTextSize(text, LABEL_FONT, scale, thickness)
    box_w, box_h = text_w + 2 * padding, text_h + baseline + 2 * padding
    height, width = image.shape[:2]

    if anchor == "top-left":
        x, y = margin, margin
    elif anchor == "top-right":
        x, y = width - margin - box_w, margin
    elif anchor == "bottom-center":
        x, y = (width - box_w) // 2, height - margin - box_h
    else:
        raise ValueError(f"Unknown anchor: {anchor}")

    cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        image, text, (x + padding, y + padding + text_h),
        LABEL_FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )


def draw_debug_landmarks(image: np.ndarray, hands: Sequence[np.ndarray]) -> None:
    """Mark each hand's landmarks, colouring hands that form an L-shape green."""
    for points in hands:
        confidence = frame_confidence(points, palm_size(points))
        color = (0, 220, 0) if confidence >= FRAME_CONFIDENCE_THRESHOLD else (0, 0, 220)
        for x, y in points.astype(int):
            cv2.circle(image, (int(x), int(y)), 3, color, cv2.FILLED)
        wx, wy = points[WRIST].astype(int)
        cv2.putText(image, f"{confidence:.2f}", (int(wx), int(wy) + 20),
                    LABEL_FONT, 0.5, color, 1, cv2.LINE_AA)


class FpsCounter:
    """Frames-per-second estimate smoothed with an exponential moving average."""

    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self.fps = 0.0
        self._last = time.perf_counter()

    def tick(self) -> float:
        """Register a new frame and return the smoothed FPS."""
        now = time.perf_counter()
        elapsed, self._last = now - self._last, now
        if elapsed > 0:
            instant = 1.0 / elapsed
            self.fps = instant if self.fps == 0 else self.smoothing * self.fps + (1 - self.smoothing) * instant
        return self.fps


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class ARHandFilterApp:
    """Main loop: capture, track, detect the frame gesture, filter, render."""

    HINT = "N/S/G/I or TAB: switch filter   F: frame/full   D: debug   Q: quit"

    def __init__(self, camera_index: int = 0, max_hands: int = DEFAULT_MAX_HANDS, mirror: bool = True) -> None:
        self.camera_index = camera_index
        self.max_hands = max_hands
        self.mirror = mirror
        self.filters = FilterState()
        self.smoother = BoxSmoother()
        self.fps = FpsCounter()
        self.debug = False

    def run(self) -> int:
        """Run until the user quits. Returns a process exit code."""
        try:
            cap = open_camera(self.camera_index)
        except CameraError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        try:
            tracker = HandTracker(max_hands=self.max_hands)
        except HandTrackerError as exc:
            cap.release()
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Error: lost connection to the camera.", file=sys.stderr)
                    return 1

                self.render(frame, tracker)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("d"), ord("D")):
                    self.debug = not self.debug
                elif key != 0xFF:
                    self.filters.handle_key(key)

                # Closing the window with its X button ends the app.
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            tracker.close()
            cap.release()
            cv2.destroyAllWindows()
        return 0

    def render(self, frame: np.ndarray, tracker: HandTracker) -> None:
        """Process one frame in place: track hands, filter and draw overlays."""
        if self.mirror:
            cv2.flip(frame, 1, dst=frame)

        hands = tracker.process(frame)
        box = self.smoother.update(compute_frame_box(hands, frame.shape))

        self.filters.apply(frame, box)
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS, cv2.LINE_AA)
        if self.debug:
            draw_debug_landmarks(frame, hands)

        scope = "" if self.filters.inside_frame_only else " (full)"
        draw_label(frame, f"Filter: {self.filters.current.name}{scope}", "top-left")
        draw_label(frame, f"{self.fps.tick():.0f} FPS", "top-right")
        draw_label(frame, self.HINT, "bottom-center", scale=0.45)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Real-time AR director's-frame hand filter.")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--max-hands", type=int, default=DEFAULT_MAX_HANDS,
                        help="maximum hands to track; use 4 for two people (default: 2)")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the preview")
    args = parser.parse_args(argv)
    if args.max_hands < 2:
        parser.error("--max-hands must be at least 2 to form a frame")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    app = ARHandFilterApp(camera_index=args.camera, max_hands=args.max_hands, mirror=not args.no_mirror)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
