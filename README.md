# AR Gesture Framer

A real-time AR webcam filter. Make a "director's frame" with two hands (thumb and index finger
spread apart) and a thin white frame joins your fingertips, with the active filter applied inside it.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python ar_hand_filter.py
```

Options: `--camera 1` picks another webcam (`--list-cameras` shows which ones work),
`--max-hands 4` lets two people each use both hands, `--rect` draws an upright rectangle, and
`--no-mirror` turns off the selfie-style flip.

## Controls

| Key / gesture            | Action                                          |
|--------------------------|-------------------------------------------------|
| `N` `S` `G` `I` `H`      | Normal / Sepia / Grayscale / Invert / Hologram  |
| `TAB`                    | Next filter                                     |
| Pinch thumb + middle tip | Next filter (keep the index finger up)          |
| `F`                      | Apply filter inside the frame / to whole image  |
| `D`                      | Show landmarks, hand scores and tracking rate   |
| `Q`/`ESC`                | Quit (closing the window also works)            |

## How it works

- **Gesture:** hold up two hands with thumb and index finger spread apart (an L, a "C" or an
  open hand all work). `pinch_confidence(points, palm_size)` scores the thumb-tip (4) to
  index-tip (8) gap relative to palm size, so a pinch or a fist never starts a frame.
- **Frame:** a thin white frame joins both hands' index and thumb tips and follows them as
  they tilt. Pass `--rect` for an axis-aligned rectangle (min/max X/Y of the same points).
  Corners are smoothed with a One Euro filter (steady when still, quick when moving) and the
  frame survives brief tracking dropouts.
- **Speed:** the camera and MediaPipe each run on their own thread, and tracking runs on a
  640 px-wide copy of the frame. The preview runs at the camera's frame rate even when
  tracking is slower. Press `D` to see both rates.
- **Tracking:** MediaPipe Hands, at 0.7 detection and tracking confidence. Recent mediapipe
  releases (0.10.30+) removed `mp.solutions`. In that case the script uses the equivalent Tasks
  `HandLandmarker` and downloads its model (~7.5 MB) to `models/` on first run.
- **Sepia:** `cv2.transform` with the standard sepia matrix, reordered for BGR.

## Tips for a high frame rate

- Light the room well. Webcams lower their frame rate in dim light to expose longer.
- Close other apps that use the camera or the GPU heavily (video calls, OBS).
- With two people in view, use `--max-hands 4`. Each tracked hand costs time, so keep 2 otherwise.

## Troubleshooting

If the webcam is missing or held by another app (Zoom, Teams, OBS...), the script prints a clear
error and exits with code 1 instead of crashing. If the preview is black, the camera's privacy
shutter or kill switch is probably on, or index 0 is a virtual/IR camera: run
`python ar_hand_filter.py --list-cameras` and pick an index marked OK.
