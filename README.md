# Driver Drowsiness Detection System

A real-time drowsiness detector that watches a driver through a webcam and sounds an alarm when their eyes stay closed or they start yawning. It was a university project for SOF106 Principles of Artificial Intelligence.

In Malaysia, roughly one in five fatal crashes involves a drowsy driver. A passenger can wake a tired driver, but a driver alone has nobody. This project tries to fill that gap with a camera and a few classic AI techniques.

<!-- Add a screenshot or GIF here: ![Demo](demo.gif) -->

## How it works

1. MediaPipe Face Mesh finds facial landmarks in each frame.
2. The Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) are calculated and smoothed with an exponential moving average to filter out jitter and normal blinks.
3. For the first 80 frames the system calibrates to the driver's own eyes, so it works for different face shapes instead of relying on fixed thresholds.
4. Several AI components then decide how worried to be, and an alarm plays if the driver reaches the critical state.

The EAR is the vertical eyelid distance divided by the eye width, averaged across both eyes. A low value means the eye is closing. The MAR works the same way for the mouth, and a high value means a wide yawn.

## AI components

| Component | What it does |
|---|---|
| Genetic algorithm | Tunes the EAR drop ratio, MAR threshold and consecutive-frame limit. Runs once at startup and again after calibration. |
| Alert state machine | Maps eye state (open, half-closed, closed) and mouth state (closed, open) onto a graph. BFS finds the shortest path from Alert to Critical. |
| KNN classifier | Predicts a LOW, MEDIUM or HIGH risk level from EAR, MAR and drowsiness score (k = 5, majority vote). |
| K-Means clustering | Groups session frames into alert, warning and drowsy clusters (k = 3). |
| Autoencoder | A 3-2-3 network written from scratch in NumPy, trained on calibration data to flag unusual facial patterns. |

The models retrain every 150 frames so the system keeps up with changes in posture.

## Results

Tested on 18 simulated cases (15 drowsy, 3 alert) while adding one component at a time:

| Configuration | Precision | Recall | F1 |
|---|---|---|---|
| State machine only | 0.857 | 0.800 | 0.827 |
| + KNN | 0.867 | 0.867 | 0.867 |
| + K-Means | 0.923 | 0.800 | 0.858 |
| + Autoencoder | 0.933 | 0.933 | 0.933 |

Other measurements:

- The alarm triggers after 18 consecutive closed-eye frames and stops within one frame of the eyes reopening.
- It handled bright light, moonlight, distance from the camera, slight head turns and tilts, and clear glasses. It struggled in complete darkness and with sunglasses.
- Ten volunteers with driving licences rated alarm timing 5/5 and overall quality 4.25/5. Willingness to use it in a real car was only 2.83/5.

The sample is small and the drowsiness was acted, not real, so treat these numbers as a prototype result. Testing in a moving car would not have been safe.

## Getting started

Python 3.10 is recommended. MediaPipe does not work well on 3.12 or newer, and NumPy must stay on 1.x.

```bash
git clone https://github.com/mohammedadalachi/driver-drowsiness-detection.git
cd driver-drowsiness-detection
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python driver_drowsiness_detector_system.py
```

Keep your eyes open and look at the camera during calibration. Press `Q` or `Esc` to quit.

When you quit, the session is saved to `session_log.csv` (time, EAR, MAR, drowsiness, blinks, yawns, alert level) and a short summary is printed with the average EAR and MAR, the longest drowsy run, and the correlation between EAR and MAR.

## Troubleshooting

- **PermissionError saving the CSV:** the file is open in Excel or being synced by OneDrive. Close it, or run the project outside a synced folder.
- **Camera does not open:** close other apps using the webcam, or change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.
- **Install errors with mediapipe or numpy:** check you are on Python 3.10 and numpy 1.26.4.

## Known limitations

- It has not been tested in a real moving car.
- It stays active when the car is stopped, so it can alarm during a rest at a red light.
- It needs a clear view of the driver's face and does not work in complete darkness.
- The genetic algorithm does not keep its best chromosome between generations, and K-Means uses random initialisation. Elitism and K-Means++ would be the first fixes.

## Author

Dalachi Mohammed Abderrahmane ([@mohammedadalachi](https://github.com/mohammedadalachi))

## License

MIT. See [LICENSE](LICENSE).
