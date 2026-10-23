# AR Gesture Framer

A real-time AR webcam filter. Make a "director's frame" with your hands (thumb and index finger
spread apart) and a thin white frame joins your fingertips, with the active filter applied inside it.
Up to four hands are tracked, so a couple can share one frame (one hand each) or each make their own.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell (Git Bash: source .venv/Scripts/activate)
pip install -r requirements.txt
python ar_hand_filter.py
```

| Option                   | Effect                                                    |
|--------------------------|-----------------------------------------------------------|
| `--fullscreen`           | Start fullscreen (or press `F`)                           |
| `--camera 1`             | Use another webcam; `--list-cameras` shows which ones work |
| `--resolution 1920x1080` | Ask the camera for a sharper picture (default 1280x720)   |
| `--max-hands 2`          | Track fewer hands for speed when you are alone (default 4) |
| `--rect`                 | Draw upright rectangles instead of following fingertips   |
| `--no-mirror`            | Turn off the selfie-style flip                            |

## Controls

| Key / gesture            | Action                                                 |
|--------------------------|--------------------------------------------------------|
| `N` `S` `G` `I` `H`      | Normal / Sepia / Grayscale / Invert / Hologram         |
| `TAB`                    | Next filter                                            |
| Pinch thumb + middle tip | Next filter (keep the index finger up)                 |
| `F`                      | Fullscreen on/off                                      |
| `Z`                      | Fit (black bars) / fill (crop) when shapes differ      |
| `A`                      | Filter inside the frames / all of the image            |
| `U`                      | Hide / show the key hints (clean recording)            |
| `D`                      | Show landmarks, hand sides and scores, tracking rate   |
| `ESC`                    | Leave fullscreen (quits when windowed)                 |
| `Q`                      | Quit (closing the window also works)                   |

## How it works

- **Gesture:** each hand needs its thumb and index finger spread apart (an L, a "C" or an open
  hand all work). `pinch_confidence(points, palm_size)` scores the thumb-tip (4) to index-tip (8)
  gap relative to palm size, so a pinch or a fist never starts a frame.
- **Several frames:** hands are paired so the most frames form, preferring a left + right hand
  close together. Two people with all four hands up get one frame each; with one hand each
  up, those two hands share a frame.
- **Frame:** the white frame joins each pair's index and thumb tips and tilts with them
  (`--rect` gives the min/max X/Y rectangle instead). Corners are smoothed with a One Euro
  filter (steady when still, quick when moving), and frames survive brief tracking dropouts.
- **No stretching:** the picture is scaled to the window or screen keeping its aspect ratio;
  `Z` chooses black bars or a crop when the window shape differs from the camera's. Frames and
  labels are drawn at screen resolution, so they stay sharp in fullscreen.
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
- Alone in view? `--max-hands 2` saves some tracking time (each tracked hand costs a little).

## Troubleshooting

If the webcam is missing or held by another app (Zoom, Teams, OBS...), the script prints a clear
error and exits with code 1 instead of crashing. If the preview is black, the camera's privacy
shutter or kill switch is probably on, or index 0 is a virtual/IR camera: run
`python ar_hand_filter.py --list-cameras` and pick an index marked OK.
