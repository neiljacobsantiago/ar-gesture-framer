# AR Gesture Framer

A real-time AR webcam filter. Make a "director's frame" with two hands (each thumb and index finger
in an L-shape) and a thin white frame appears between them, with the active filter applied inside it.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python ar_hand_filter.py
```

Options: `--camera 1` picks another webcam, `--max-hands 4` lets two people each use both hands,
and `--no-mirror` turns off the selfie-style flip.

## Controls

| Key       | Action                                          |
|-----------|-------------------------------------------------|
| `N`       | Normal                                          |
| `S`       | Sepia                                           |
| `G`       | Grayscale                                       |
| `I`       | Invert                                          |
| `TAB`     | Next filter                                     |
| `F`       | Apply filter inside the frame / to whole image  |
| `D`       | Show landmarks and each hand's L-shape score    |
| `Q`/`ESC` | Quit (closing the window also works)            |

## How it works

- **Tracking:** MediaPipe Hands, at 0.7 detection and tracking confidence. Recent mediapipe
  releases (0.10.30+) removed `mp.solutions`. In that case the script uses the equivalent Tasks
  `HandLandmarker` and downloads its model (~7.5 MB) to `models/` on first run.
- **Gesture:** `pinch_confidence(points, palm_size)` measures the thumb-tip (4) to index-tip (8)
  distance relative to palm size. `frame_confidence` combines it with index straightness, thumb
  extension, the thumb/index angle (about 90°) and folded middle/ring/pinky fingers.
- **Frame:** when two hands score at least 0.5, the box spans the min/max X/Y of their thumb and
  index tips. It is smoothed over time and held through brief tracking dropouts.
- **Sepia:** `cv2.transform` with the standard sepia matrix, reordered for BGR.

If the webcam is missing or held by another app (Zoom, Teams, OBS...), the script prints a clear
error and exits with code 1 instead of crashing.
