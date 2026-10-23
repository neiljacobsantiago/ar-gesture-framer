"""AR Hand Filter: a real-time "director's frame" gesture filter.

Captures a webcam feed, tracks hands with Google MediaPipe, and detects the
"director's frame" gesture: two hands with the thumb and index finger spread
apart. A thin white frame joins the four fingertips and the active image filter
is applied inside it (or to the whole image, see the ``F`` key).

Camera capture and hand tracking run on their own threads, so the preview runs
at the camera's frame rate even when tracking is slower.

Controls:
    N / S / G / I / H   Normal / Sepia / Grayscale / Invert / Hologram
    TAB                 Cycle to the next filter
    Pinch               Touch thumb and middle fingertips: next filter
    F                   Toggle filter scope: inside the frame / full image
    D                   Toggle landmark debug overlay
    Q / ESC             Quit

Usage:
    python ar_hand_filter.py [--camera 0] [--max-hands 2] [--no-mirror] [--rect]
    python ar_hand_filter.py --list-cameras
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import threading
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_NAME = "AR Hand Filter"

MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
MIN_PRESENCE_CONFIDENCE = 0.5  # lower than detection so a found hand is kept
DEFAULT_MAX_HANDS = 2

# Hand tracking runs on a downscaled copy of the frame. MediaPipe's models
# work at ~224 px internally, so this costs no accuracy and saves time.
TRACKING_MAX_WIDTH = 640

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12

# Gesture thresholds, expressed relative to palm size so they hold at any
# distance from the camera.
PINCH_CLOSED_RATIO = 0.20   # thumb/index tip gap at which a pinch is certain
PINCH_OPEN_RATIO = 0.60     # gap at which the fingers are clearly apart
FRAME_ENTER_THRESHOLD = 0.5  # hand confidence needed to start a frame
FRAME_STAY_THRESHOLD = 0.3   # lower bar to keep an existing frame (no flicker)
MIN_FRAME_SIZE_PX = 40

# Thumb + middle fingertip pinch that switches filter.
SWITCH_PINCH_ON = 0.6       # confidence that fires a switch
SWITCH_PINCH_OFF = 0.2      # confidence below which the pinch counts as released
SWITCH_COOLDOWN_S = 0.6

# Frame smoothing (One Euro filter) and how long the frame survives a
# momentary tracking dropout.
SMOOTH_MIN_CUTOFF = 1.5     # Hz; lower = steadier when still
SMOOTH_BETA = 0.01          # higher = less lag when moving fast
FRAME_HOLD_S = 0.25

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

# Hologram tint: a lookup table mapping each gray level to cyan-blue (B, G, R).
_GRAY_LEVELS = np.arange(256, dtype=np.float32)[:, None]
HOLOGRAM_LUT = np.clip(
    _GRAY_LEVELS * np.array([1.0, 0.85, 0.35]) + np.array([70, 40, 0]), 0, 255
).astype(np.uint8).reshape(256, 1, 3)
HOLOGRAM_LINE_PERIOD = 4    # rows per scanline cycle

FRAME_COLOR = (255, 255, 255)
FRAME_THICKNESS = 2
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX

# A frame as 4 corner points, shape (4, 2): left index tip, right index tip,
# right thumb tip, left thumb tip.
Quad = np.ndarray


# ---------------------------------------------------------------------------
# Geometry and gesture recognition
# ---------------------------------------------------------------------------

def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Euclidean distance between two 2D points."""
    return float(np.hypot(*(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


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


def index_extended(points: np.ndarray, palm_size: float) -> float:
    """Return 1.0 when the index finger is stretched out from the palm."""
    reach = distance(points[INDEX_TIP], points[WRIST]) / max(palm_size, 1e-6)
    return ramp(reach, 1.1, 1.45)


def frame_confidence(points: np.ndarray, palm_size: float) -> float:
    """Score how well one hand can hold a corner of the director's frame.

    Deliberately forgiving: the thumb and index finger only need to be
    extended and spread apart. An L-shape, an open hand and a loose "C" all
    count; a pinch or a fist does not.

    Args:
        points: (21, 2) array of landmark pixel coordinates for one hand.
        palm_size: Scale reference for the hand (see :func:`palm_size`).

    Returns:
        A confidence in [0, 1].
    """
    palm = max(palm_size, 1e-6)
    spread = 1.0 - pinch_confidence(points, palm)
    thumb_out = ramp(distance(points[THUMB_TIP], points[INDEX_MCP]) / palm, 0.3, 0.55)
    return min(spread, thumb_out, index_extended(points, palm))


def switch_pinch_confidence(points: np.ndarray, palm_size: float) -> float:
    """Return how clearly a hand pinches thumb and middle fingertips together.

    The index finger must stay extended and the middle finger must reach out
    to the thumb, so a fist or a pointing hand (thumb resting on the curled
    middle finger) does not count.
    """
    palm = max(palm_size, 1e-6)
    thumb_middle = distance(points[THUMB_TIP], points[MIDDLE_TIP]) / palm
    middle_reach = distance(points[MIDDLE_TIP], points[MIDDLE_MCP]) / palm
    return min(
        ramp(thumb_middle, 0.45, 0.2),
        ramp(middle_reach, 0.4, 0.6),
        index_extended(points, palm),
    )


def frame_quad(hand_a: np.ndarray, hand_b: np.ndarray) -> Quad:
    """Join two hands' thumb and index tips into a four-corner frame."""
    left, right = sorted(
        (hand_a, hand_b), key=lambda h: h[THUMB_TIP, 0] + h[INDEX_TIP, 0]
    )
    return np.array(
        [left[INDEX_TIP], right[INDEX_TIP], right[THUMB_TIP], left[THUMB_TIP]],
        dtype=np.float32,
    )


def quad_to_rect(quad: Quad) -> Quad:
    """Return the axis-aligned bounding box of a quad, as a quad."""
    (x1, y1), (x2, y2) = quad.min(axis=0), quad.max(axis=0)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def compute_frame_quad(
    hands: Sequence[np.ndarray], threshold: float = FRAME_ENTER_THRESHOLD
) -> Optional[Quad]:
    """Find the director's frame formed by two hands.

    When more than two hands qualify, the pair framing the largest area is
    used, so a bystander's hand does not hijack the frame.

    Args:
        hands: Landmark arrays, one (21, 2) array per detected hand.
        threshold: Minimum :func:`frame_confidence` for a hand to take part.

    Returns:
        The (4, 2) corner points in pixels, or ``None`` if no frame is formed.
    """
    candidates = [h for h in hands if frame_confidence(h, palm_size(h)) >= threshold]
    best: Optional[Quad] = None
    best_area = 0.0
    for hand_a, hand_b in itertools.combinations(candidates, 2):
        quad = frame_quad(hand_a, hand_b)
        width, height = quad.max(axis=0) - quad.min(axis=0)
        if width < MIN_FRAME_SIZE_PX or height < MIN_FRAME_SIZE_PX:
            continue
        area = abs(cv2.contourArea(quad))
        if area >= best_area:
            best, best_area = quad, area
    return best


class OneEuroFilter:
    """One Euro filter for arrays of points (Casiez et al., CHI 2012).

    Adapts its smoothing to speed: heavy when the hands are still (no
    jitter), light when they move (little lag).
    """

    def __init__(
        self,
        min_cutoff: float = SMOOTH_MIN_CUTOFF,
        beta: float = SMOOTH_BETA,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        """Forget all history; the next sample passes through unchanged."""
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._t = 0.0

    @staticmethod
    def _alpha(cutoff: np.ndarray, dt: float) -> np.ndarray:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Filter sample ``x`` taken at time ``t`` (seconds)."""
        x = np.asarray(x, dtype=np.float32)
        if self._x is None or t <= self._t:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return x
        dt = t - self._t
        dx = (x - self._x) / dt
        self._dx = self._dx + self._alpha(np.float32(self.d_cutoff), dt) * (dx - self._dx)
        speed = np.linalg.norm(self._dx, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        self._x = self._x + self._alpha(cutoff, dt) * (x - self._x)
        self._t = t
        return self._x


class FrameTracker:
    """Turns per-detection hands into a stable, smoothed frame."""

    def __init__(self, rect: bool = False, hold_s: float = FRAME_HOLD_S) -> None:
        self.rect = rect
        self.hold_s = hold_s
        self.smoother = OneEuroFilter()
        self.quad: Optional[Quad] = None
        self._last_seen = -math.inf

    def update(self, hands: Sequence[np.ndarray], t: float) -> Optional[Quad]:
        """Feed one tracking result taken at time ``t`` and return the frame."""
        # Hysteresis: once a frame is showing, hands may relax a little.
        threshold = FRAME_STAY_THRESHOLD if self.quad is not None else FRAME_ENTER_THRESHOLD
        raw = compute_frame_quad(hands, threshold)
        if raw is not None:
            if self.quad is None:
                self.smoother.reset()
            self.quad = self.smoother(quad_to_rect(raw) if self.rect else raw, t)
            self._last_seen = t
        elif t - self._last_seen > self.hold_s:
            self.quad = None
        return self.quad


class PinchSwitch:
    """Fires once per thumb + middle pinch, with a cooldown against repeats."""

    def __init__(self, cooldown_s: float = SWITCH_COOLDOWN_S) -> None:
        self.cooldown_s = cooldown_s
        self._armed = True
        self._last_fire = -math.inf

    def update(self, hands: Sequence[np.ndarray], t: float) -> bool:
        """Feed one tracking result; return True when a switch should happen."""
        strength = max((switch_pinch_confidence(h, palm_size(h)) for h in hands), default=0.0)
        if strength <= SWITCH_PINCH_OFF:
            self._armed = True
        elif strength >= SWITCH_PINCH_ON and self._armed and t - self._last_fire >= self.cooldown_s:
            self._armed = False
            self._last_fire = t
            return True
        return False


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


def filter_hologram(image: np.ndarray) -> np.ndarray:
    """Cyan-blue tint with drifting scanlines, like a sci-fi hologram."""
    holo = cv2.applyColorMap(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), HOLOGRAM_LUT)
    phase = int(time.monotonic() * 20) % HOLOGRAM_LINE_PERIOD
    holo[phase::HOLOGRAM_LINE_PERIOD] >>= 1  # darken every 4th row
    return cv2.addWeighted(holo, 0.75, image, 0.25, 0)  # keep some detail showing


@dataclass(frozen=True)
class ImageFilter:
    """A named filter bound to a keyboard shortcut."""

    name: str
    key: str
    apply: Callable[[np.ndarray], np.ndarray]


FILTERS: tuple[ImageFilter, ...] = (
    ImageFilter("Normal", "n", filter_normal),
    ImageFilter("Sepia", "s", filter_sepia),
    ImageFilter("Grayscale", "g", filter_grayscale),
    ImageFilter("Invert", "i", filter_invert),
    ImageFilter("Hologram", "h", filter_hologram),
)


class FilterState:
    """Minimal state machine holding the active filter and where it applies."""

    def __init__(self, filters: Sequence[ImageFilter] = FILTERS, initial: str = "Sepia") -> None:
        self.filters = list(filters)
        self._by_key: dict[int, int] = {ord(f.key): i for i, f in enumerate(self.filters)}
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

    def apply(self, frame: np.ndarray, quad: Optional[Quad]) -> None:
        """Apply the active filter to ``frame`` in place.

        In frame-only mode the filter is applied inside ``quad`` and nothing
        changes when no frame is shown. Otherwise the whole image is filtered.
        Only the frame's bounding box is filtered, then masked to its shape.
        """
        if self.current.apply is filter_normal:
            return
        if not self.inside_frame_only:
            frame[:] = self.current.apply(frame)
            return
        if quad is None:
            return
        height, width = frame.shape[:2]
        pts = np.round(quad).astype(np.int32)
        x1, y1 = np.clip(pts.min(axis=0), 0, [width, height])
        x2, y2 = np.clip(pts.max(axis=0) + 1, 0, [width, height])
        if x2 - x1 < 2 or y2 - y1 < 2:
            return
        roi = frame[y1:y2, x1:x2]
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts - (x1, y1)], 255)
        np.copyto(roi, self.current.apply(roi), where=mask[..., None].astype(bool))


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
            min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=tracking_confidence,
        )
        try:
            return vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # mediapipe raises plain RuntimeError/ValueError
            raise HandTrackerError(f"Could not start the MediaPipe hand landmarker: {exc}") from exc

    def process(
        self, frame_bgr: np.ndarray, output_size: Optional[tuple[int, int]] = None
    ) -> list[np.ndarray]:
        """Detect hands in a BGR frame.

        Args:
            frame_bgr: Image to search.
            output_size: ``(width, height)`` to express landmarks in, e.g. the
                full-resolution frame when ``frame_bgr`` is a downscaled copy.
                Defaults to the size of ``frame_bgr``.

        Returns:
            One (21, 2) float32 array of pixel coordinates per detected hand.
        """
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        scale = np.array(output_size or (width, height), dtype=np.float32)

        if self.backend == "solutions":
            rgb.flags.writeable = False
            results = self._hands.process(rgb)
            hand_list = [h.landmark for h in (results.multi_hand_landmarks or [])]
        else:
            # VIDEO mode requires strictly increasing timestamps.
            timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp_ms
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_list = self._hands.detect_for_video(image, timestamp_ms).hand_landmarks

        return [
            np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32) * scale
            for landmarks in hand_list
        ]

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()


@dataclass(frozen=True)
class TrackingResult:
    """Hands found in one frame, numbered so callers can spot new results."""

    seq: int
    timestamp: float
    hands: list[np.ndarray]


class HandWorker:
    """Runs a :class:`HandTracker` on a background thread.

    The main loop submits every frame without waiting. The worker always
    processes the most recent one and skips any it had no time for, so slow
    tracking never stalls the preview.
    """

    def __init__(self, tracker: HandTracker, max_width: int = TRACKING_MAX_WIDTH) -> None:
        self.tracker = tracker
        self.max_width = max_width
        self.result = TrackingResult(0, time.monotonic(), [])
        self.error: Optional[BaseException] = None
        self._pending: Optional[tuple[np.ndarray, tuple[int, int], float]] = None
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="hand-tracker", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Queue a frame for tracking, replacing any frame still waiting."""
        height, width = frame.shape[:2]
        if width > self.max_width:
            small = cv2.resize(
                frame, (self.max_width, round(height * self.max_width / width)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame.copy()  # the caller keeps drawing on ``frame``
        with self._cond:
            self._pending = (small, (width, height), time.monotonic())
            self._cond.notify()

    def _run(self) -> None:
        seq = 0
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._pending is not None or not self._running)
                if not self._running:
                    return
                small, size, timestamp = self._pending
                self._pending = None
            try:
                hands = self.tracker.process(small, size)
            except Exception as exc:  # surface tracker crashes to the main thread
                self.error = exc
                return
            seq += 1
            self.result = TrackingResult(seq, timestamp, hands)

    def close(self) -> None:
        """Stop the thread and release the tracker."""
        with self._cond:
            self._running = False
            self._cond.notify()
        self._thread.join(timeout=2.0)
        self.tracker.close()


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


CAMERA_BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
BACKEND_NAMES = {value: name for name, value in CAMERA_BACKENDS.items()}

# A frame whose brightest pixel is below this is treated as black: what a
# closed privacy shutter, a disabled camera or an idle virtual camera sends.
BLACK_FRAME_MAX = 8


def is_black(frame: np.ndarray) -> bool:
    """Return True if every pixel in the frame is (almost) black."""
    return int(frame[::8, ::8].max()) < BLACK_FRAME_MAX


def camera_backends(preference: str = "auto") -> list[int]:
    """Return the OpenCV capture backends to try, in order.

    On Windows the default Media Foundation backend often fails to deliver
    frames from some webcams, so DirectShow is tried first.
    """
    if preference != "auto":
        return [CAMERA_BACKENDS[preference]]
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    return [cv2.CAP_ANY]


def read_first_frame(cap: cv2.VideoCapture, timeout_s: float) -> Optional[np.ndarray]:
    """Read until a non-black frame arrives or ``timeout_s`` passes.

    Cameras often need a moment to warm up, and some send black frames
    before the first real one.

    Returns:
        The first non-black frame, else the last black frame read, else
        ``None`` if the camera produced nothing at all.
    """
    deadline = time.monotonic() + timeout_s
    last = None
    while True:
        ok, frame = cap.read()
        if ok and frame is not None:
            if not is_black(frame):
                return frame
            last = frame
        if time.monotonic() >= deadline:
            return last
        time.sleep(0.05)


def _open_capture(index: int, backend: int, width: int, height: int) -> Optional[cv2.VideoCapture]:
    """Open one camera with one backend, or return None if it will not open."""
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    # MJPG lets USB webcams send 720p at full frame rate; uncompressed
    # formats are often capped at a few FPS. Ignored if unsupported.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def open_camera(
    index: int = 0,
    backend: str = "auto",
    width: int = 1280,
    height: int = 720,
    warmup_s: float = 3.0,
) -> cv2.VideoCapture:
    """Open a webcam and confirm it delivers frames.

    A camera held by another application often opens "successfully" but
    never returns a frame, so a test read is part of the check. Each capture
    backend gets ``warmup_s`` seconds to produce its first frame, and one
    that sends real images is preferred over one that only sends black.

    Raises:
        CameraError: If the device cannot be opened or produces no frames.
    """
    opened = False
    black_cap: Optional[cv2.VideoCapture] = None
    for api in camera_backends(backend):
        cap = _open_capture(index, api, width, height)
        if cap is None:
            continue
        opened = True
        frame = read_first_frame(cap, warmup_s)
        if frame is None:
            cap.release()
        elif not is_black(frame):
            if black_cap is not None:
                black_cap.release()
            return cap
        elif black_cap is None:
            black_cap = cap
        else:
            cap.release()

    if black_cap is not None:
        print(
            f"Warning: camera {index} is only sending black frames. Check the privacy "
            "shutter / camera kill switch, or run with --list-cameras to find the "
            "right --camera index.",
            file=sys.stderr,
        )
        return black_cap
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
        "  - it is the right device: run with --list-cameras"
    )


def list_cameras(max_index: int = 5, backend: str = "auto") -> int:
    """Probe camera indices and print what each one delivers. Returns 0."""
    print("Probing cameras (this takes a few seconds)...")
    for index in range(max_index):
        for api in camera_backends(backend):
            cap = _open_capture(index, api, 1280, 720)
            if cap is None:
                continue
            frame = read_first_frame(cap, 1.5)
            cap.release()
            if frame is None:
                status = "opens, but no frames (in use?)"
            else:
                status = f"{frame.shape[1]}x{frame.shape[0]}" + (", BLACK frames" if is_black(frame) else ", OK")
            print(f"  --camera {index} --backend {BACKEND_NAMES[api]}: {status}")
    print("Use an index marked OK, e.g. python ar_hand_filter.py --camera 1")
    return 0


class CameraStream:
    """Reads a camera on a background thread and keeps only the newest frame.

    ``cap.read()`` blocks until the camera's next frame, so reading on the
    main thread caps everything else at the camera's pace. Here the main
    loop just picks up whatever frame is newest.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        self.failed = False
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            with self._cond:
                if not ok or frame is None:
                    self.failed = True
                    self._cond.notify_all()
                    return
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, last_seq: int, timeout_s: float = 2.0) -> tuple[int, Optional[np.ndarray]]:
        """Wait for a frame newer than ``last_seq``.

        Returns:
            ``(seq, frame)``, or ``(last_seq, None)`` if the camera failed or
            no new frame arrived within ``timeout_s``.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or self.failed, timeout_s)
            if self._seq == last_seq:
                return last_seq, None
            return self._seq, self._frame

    def close(self) -> None:
        """Stop the reader thread and release the camera."""
        self._running = False
        self._thread.join(timeout=2.0)
        self.cap.release()


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
        anchor: ``"top-left"``, ``"top-right"``, ``"bottom-center"`` or ``"center"``.
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
    elif anchor == "center":
        x, y = (width - box_w) // 2, (height - box_h) // 2
    else:
        raise ValueError(f"Unknown anchor: {anchor}")

    cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        image, text, (x + padding, y + padding + text_h),
        LABEL_FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )


def draw_frame(image: np.ndarray, quad: Quad) -> None:
    """Draw the director's frame: thin white edges with small corner dots."""
    pts = np.round(quad).astype(np.int32)
    cv2.polylines(image, [pts], True, FRAME_COLOR, FRAME_THICKNESS, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(image, (int(x), int(y)), 4, FRAME_COLOR, cv2.FILLED, cv2.LINE_AA)


def draw_debug_landmarks(image: np.ndarray, hands: Sequence[np.ndarray]) -> None:
    """Mark each hand's landmarks, colouring hands that can hold a frame green."""
    for points in hands:
        confidence = frame_confidence(points, palm_size(points))
        color = (0, 220, 0) if confidence >= FRAME_ENTER_THRESHOLD else (0, 0, 220)
        for x, y in points.astype(int):
            cv2.circle(image, (int(x), int(y)), 3, color, cv2.FILLED)
        wx, wy = points[WRIST].astype(int)
        cv2.putText(image, f"{confidence:.2f}", (int(wx), int(wy) + 20),
                    LABEL_FONT, 0.5, color, 1, cv2.LINE_AA)


class FpsCounter:
    """Events-per-second estimate smoothed with an exponential moving average."""

    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self.fps = 0.0
        self._last = time.perf_counter()

    def tick(self) -> float:
        """Register a new event and return the smoothed rate."""
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

    HINT = "Pinch thumb+middle or TAB: next filter   N/S/G/I/H   F: frame/full   D: debug   Q: quit"

    def __init__(
        self,
        camera_index: int = 0,
        max_hands: int = DEFAULT_MAX_HANDS,
        mirror: bool = True,
        backend: str = "auto",
        rect: bool = False,
    ) -> None:
        self.camera_index = camera_index
        self.backend = backend
        self.max_hands = max_hands
        self.mirror = mirror
        self.filters = FilterState()
        self.frame_tracker = FrameTracker(rect=rect)
        self.pinch_switch = PinchSwitch()
        self.fps = FpsCounter()
        self.tracking_fps = FpsCounter()
        self.debug = False
        self._result_seq = 0
        self._hands: list[np.ndarray] = []

    def run(self) -> int:
        """Run until the user quits. Returns a process exit code."""
        try:
            cap = open_camera(self.camera_index, self.backend)
        except CameraError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        try:
            worker = HandWorker(HandTracker(max_hands=self.max_hands))
        except HandTrackerError as exc:
            cap.release()
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        camera = CameraStream(cap)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        seq = 0
        try:
            while True:
                seq, frame = camera.read(seq)
                if frame is None:
                    print("Error: lost connection to the camera.", file=sys.stderr)
                    return 1
                if worker.error is not None:
                    print(f"Error: hand tracking failed: {worker.error}", file=sys.stderr)
                    return 1

                self.render(frame, worker)
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
            camera.close()
            worker.close()
            cv2.destroyAllWindows()
        return 0

    def render(self, frame: np.ndarray, worker: HandWorker) -> None:
        """Process one frame in place: track hands, filter and draw overlays."""
        if self.mirror:
            cv2.flip(frame, 1, dst=frame)

        black = is_black(frame)
        if not black:
            worker.submit(frame)

        # Gesture state advances only when the tracker has something new.
        result = worker.result
        if result.seq != self._result_seq:
            self._result_seq = result.seq
            self._hands = result.hands
            self.tracking_fps.tick()
            self.frame_tracker.update(result.hands, result.timestamp)
            if self.pinch_switch.update(result.hands, result.timestamp):
                self.filters.next()
        quad = self.frame_tracker.quad

        self.filters.apply(frame, quad)
        if quad is not None:
            draw_frame(frame, quad)
        if self.debug:
            draw_debug_landmarks(frame, self._hands)

        scope = "" if self.filters.inside_frame_only else " (full)"
        draw_label(frame, f"Filter: {self.filters.current.name}{scope}", "top-left")
        fps = f"{self.fps.tick():.0f} FPS"
        if self.debug:
            fps += f" | track {self.tracking_fps.fps:.0f}/s"
        draw_label(frame, fps, "top-right")
        draw_label(frame, self.HINT, "bottom-center", scale=0.45)
        if black:
            draw_label(frame, "Camera sends black frames: open the shutter or try --list-cameras", "center", scale=0.55)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Real-time AR director's-frame hand filter.")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--max-hands", type=int, default=DEFAULT_MAX_HANDS,
                        help="maximum hands to track; use 4 for two people (default: 2)")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the preview")
    parser.add_argument("--rect", action="store_true",
                        help="draw an axis-aligned rectangle instead of following the fingertips")
    parser.add_argument("--backend", choices=["auto", *CAMERA_BACKENDS], default="auto",
                        help="OpenCV capture backend (default: auto)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="probe camera indices and exit")
    args = parser.parse_args(argv)
    if args.max_hands < 2:
        parser.error("--max-hands must be at least 2 to form a frame")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    if args.list_cameras:
        return list_cameras(backend=args.backend)
    app = ARHandFilterApp(
        camera_index=args.camera,
        max_hands=args.max_hands,
        mirror=not args.no_mirror,
        backend=args.backend,
        rect=args.rect,
    )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
