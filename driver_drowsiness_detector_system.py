import csv
import math
import random
import time
import os
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pygame


def euclidean(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


# settings, tuned mostly by trial and error during testing
EAR_DROP = 0.72
MAR_THRESH = 0.35  # tried 0.3 first but it kept triggering on normal talking
CONSEC_FRAMES = 18
CALIB_FRAMES = 80
SMOOTH_ALPHA = 0.35
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_log.csv")
SIDEBAR_W = 210
EPSILON = 1e-6  # avoid divide by zero when horizontal distance is tiny
EAR_SEVERE_RATIO = 0.85
RISK_EAR_LOW = 0.85
RISK_EAR_MID = 0.95
RISK_MAR_HIGH = 1.30
RETRAIN_EVERY = 150

# mediapipe face mesh landmark indices for the eyes and mouth
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_TOP2 = 312
MOUTH_BOTTOM2 = 317
MOUTH_TOP3 = 82
MOUTH_BOTTOM3 = 87
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# bgr colours for opencv
GREEN = (57, 255, 20)
YELLOW = (0, 215, 255)
RED = (50, 50, 255)
WHITE = (255, 255, 255)
GREY = (120, 120, 120)
BLUE = (255, 160, 50)
PURPLE = (200, 80, 180)
CYAN = (255, 220, 50)
ORANGE = (0, 165, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def get_mouth_point(landmarks, index, frame_w, frame_h):
    return [
        landmarks[index].x * frame_w,
        landmarks[index].y * frame_h,
        landmarks[index].z * frame_w,
    ]


def compute_ear(landmarks, eye_points, frame_w, frame_h):
    pts = []
    for i in eye_points:
        x = landmarks[i].x * frame_w
        y = landmarks[i].y * frame_h
        z = landmarks[i].z * frame_w
        pts.append([x, y, z])

    vertical_1 = euclidean(pts[1], pts[5])
    vertical_2 = euclidean(pts[2], pts[4])
    horizontal = euclidean(pts[0], pts[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal + EPSILON)


# mouth aspect ratio using 3 vertical pairs instead of 1 since a single
# pair was too noisy when tested on different people's faces
def compute_mar(landmarks, frame_w, frame_h):
    top1 = get_mouth_point(landmarks, MOUTH_TOP, frame_w, frame_h)
    bot1 = get_mouth_point(landmarks, MOUTH_BOTTOM, frame_w, frame_h)
    top2 = get_mouth_point(landmarks, MOUTH_TOP2, frame_w, frame_h)
    bot2 = get_mouth_point(landmarks, MOUTH_BOTTOM2, frame_w, frame_h)
    top3 = get_mouth_point(landmarks, MOUTH_TOP3, frame_w, frame_h)
    bot3 = get_mouth_point(landmarks, MOUTH_BOTTOM3, frame_w, frame_h)
    left = get_mouth_point(landmarks, MOUTH_LEFT, frame_w, frame_h)
    right = get_mouth_point(landmarks, MOUTH_RIGHT, frame_w, frame_h)

    v1 = euclidean(top1, bot1)
    v2 = euclidean(top2, bot2)
    v3 = euclidean(top3, bot3)
    horizontal = euclidean(left, right)

    return (v1 + v2 + v3) / (3.0 * horizontal + EPSILON)


class SessionLogger:

    def __init__(self):
        self.records = []
        self.start_time = time.time()

    def log(self, ear, mar, drowsy_score, blinks, yawns, alert_level):
        elapsed = round(time.time() - self.start_time, 2)
        self.records.append({
            "time_s": elapsed,
            "ear": round(ear, 4),
            "mar": round(mar, 4),
            "drowsy_pct": drowsy_score,
            "blinks": blinks,
            "yawns": yawns,
            "alert_level": alert_level,
        })

    def save_to_csv(self):
        if not self.records:
            return
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)
        print(f"Saved {len(self.records)} frames to {LOG_FILE}")

    def mine_patterns(self):
        if len(self.records) < 20:
            print("not enough data to mine patterns yet")
            return

        all_ear = [r["ear"] for r in self.records]
        all_mar = [r["mar"] for r in self.records]
        all_alerts = [r["alert_level"] for r in self.records]

        avg_ear = sum(all_ear) / len(all_ear)
        avg_mar = sum(all_mar) / len(all_mar)

        # longest stretch stuck at the critical alert level
        longest_drowsy = 0
        current_run = 0
        for level in all_alerts:
            if level == 2:
                current_run += 1
                longest_drowsy = max(longest_drowsy, current_run)
            else:
                current_run = 0

        # pearson correlation, ear vs mar — negative means eyes closing
        # while mouth opens simultaneously, which is the drowsy pattern
        numerator = sum(
            (all_ear[i] - avg_ear) * (all_mar[i] - avg_mar)
            for i in range(len(all_ear))
        )
        std_ear = math.sqrt(sum((e - avg_ear) ** 2 for e in all_ear))
        std_mar = math.sqrt(sum((m - avg_mar) ** 2 for m in all_mar))
        correlation = round(numerator / (std_ear * std_mar + 1e-9), 3)

        print("\n--- mined patterns ---")
        print(f"avg ear: {avg_ear:.3f}")
        print(f"avg mar: {avg_mar:.3f}")
        print(f"longest drowsy run: {longest_drowsy} frames")
        print(f"ear-mar correlation: {correlation}")
        print(f"total frames logged: {len(self.records)}")
        print("----------------------\n")


# genetic algorithm that evolves the ear/mar/consec thresholds.
# chromosome = [ear_drop, mar_threshold, consec_frames]
class GeneticAlgorithm:

    POPULATION_SIZE = 16
    GENERATIONS = 25
    MUTATION_RATE = 0.15
    EAR_RANGE = (0.55, 0.85)
    MAR_RANGE = (0.30, 0.50)
    CONSEC_RANGE = (10, 30)

    def __init__(self):
        self.population = [self._make_chromosome() for _ in range(self.POPULATION_SIZE)]
        self.calib_baseline = None

    def set_baseline(self, calib_values):
        sorted_vals = sorted(calib_values)
        self.calib_baseline = sorted_vals[len(sorted_vals) // 2]
        print(f"calibration baseline received: {self.calib_baseline:.3f}")

    def _make_chromosome(self):
        ear_drop = random.uniform(self.EAR_RANGE[0], self.EAR_RANGE[1])
        mar_thresh = random.uniform(self.MAR_RANGE[0], self.MAR_RANGE[1])
        consec = float(random.randint(self.CONSEC_RANGE[0], self.CONSEC_RANGE[1]))
        return [ear_drop, mar_thresh, consec]

    def _fitness(self, chromosome):
        ear, mar, consec = chromosome

        if self.calib_baseline is not None:
            # score against the user's actual calibrated baseline
            ideal_thresh = self.calib_baseline * 0.70
            actual_thresh = self.calib_baseline * ear
            ear_score = 1.0 - abs(actual_thresh - ideal_thresh) / (self.calib_baseline + EPSILON)
        else:
            # fall back to global constants so fitness stays consistent
            ear_score = 1.0 - abs(ear - EAR_DROP) * 4

        # use global constants here too for the same reason
        mar_score = 1.0 - abs(mar - MAR_THRESH) * 3
        consec_score = 1.0 - abs(consec - CONSEC_FRAMES) / 20.0

        total = 0.5 * ear_score + 0.3 * mar_score + 0.2 * consec_score
        return max(0.0, total)

    def _tournament_select(self):
        contestants = random.sample(self.population, 3)
        return max(contestants, key=self._fitness)

    def _crossover(self, parent_a, parent_b):
        cut = random.randint(1, len(parent_a) - 1)
        child = parent_a[:cut] + parent_b[cut:]
        return child

    def _mutate(self, chromosome):
        chromosome = chromosome[:]
        for i in range(len(chromosome)):
            if random.random() < self.MUTATION_RATE:
                if i == 0:
                    chromosome[i] = random.uniform(self.EAR_RANGE[0], self.EAR_RANGE[1])
                elif i == 1:
                    chromosome[i] = random.uniform(self.MAR_RANGE[0], self.MAR_RANGE[1])
                else:
                    chromosome[i] = float(random.randint(*self.CONSEC_RANGE))
        return chromosome

    def run(self):
        print("running genetic algorithm to evolve detection thresholds...")
        for gen in range(self.GENERATIONS):
            next_generation = []
            for _ in range(self.POPULATION_SIZE):
                parent_a = self._tournament_select()
                parent_b = self._tournament_select()
                child = self._crossover(parent_a, parent_b)
                child = self._mutate(child)
                next_generation.append(child)
            self.population = next_generation

        best = max(self.population, key=self._fitness)
        ear, mar, c = best
        print(f"done. best thresholds -> ear_drop={ear:.3f}  mar={mar:.3f}  consec={int(c)}")
        return ear, mar, int(c)

    def rerun_with_calibration(self):
        print("re-running ga with personal calibration data...")
        return self.run()


class AlertStateMachine:

    STATE_NAMES = {
        (0, 0): "ALERT",
        (1, 0): "MILD_EYE",
        (0, 1): "MILD_MOUTH",
        (1, 1): "WARNING",
        (2, 0): "SEVERE_EYE",
        (2, 1): "CRITICAL",
    }

    ACTIONS = {
        "ALERT": "no action",
        "MILD_EYE": "log only",
        "MILD_MOUTH": "log only",
        "WARNING": "flash yellow warning",
        "SEVERE_EYE": "beep once",
        "CRITICAL": "full alarm",
    }

    ACTION_SEVERITY = {
        "no action": 0,
        "log only": 1,
        "flash yellow warning": 2,
        "beep once": 3,
        "full alarm": 4,
    }

    def __init__(self, eye_threshold, mouth_threshold):
        self.eye_thresh = eye_threshold
        self.mouth_thresh = mouth_threshold
        self.graph = self._build_graph()
        self.bfs_path = self._bfs_alert_to_critical()
        self.distance_to_critical = self._bfs_distances_from((2, 1))

    def _build_graph(self):
        all_states = [(e, m) for e in range(3) for m in range(2)]
        graph = {state: [] for state in all_states}

        for state in all_states:
            eye_z, mouth_z = state
            for d_eye in [-1, 0, 1]:
                for d_mouth in [-1, 0, 1]:
                    neighbour = (
                        max(0, min(2, eye_z + d_eye)),
                        max(0, min(1, mouth_z + d_mouth)),
                    )
                    if neighbour != state:
                        graph[state].append(neighbour)

        total_edges = sum(len(v) for v in graph.values())
        print(f"graph built: {len(graph)} nodes, {total_edges} edges")
        return graph

    def _bfs_alert_to_critical(self):
        start = (0, 0)
        goal = (2, 1)
        queue = deque([[start]])
        visited = {start}

        while queue:
            path = queue.popleft()
            current = path[-1]

            if current == goal:
                path_str = " -> ".join(str(s) for s in path)
                print(f"bfs: alert to critical in {len(path)-1} steps: {path_str}")
                return path

            for neighbour in self.graph[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(path + [neighbour])

        return []

    def _bfs_distances_from(self, start):
        distances = {start: 0}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for neighbour in self.graph[current]:
                if neighbour not in distances:
                    distances[neighbour] = distances[current] + 1
                    queue.append(neighbour)

        return distances

    def evaluate(self, smoothed_ear, mar):
        if smoothed_ear is None:
            return "ALERT", "no action"

        if smoothed_ear >= self.eye_thresh:
            eye_zone = 0
        elif smoothed_ear >= self.eye_thresh * EAR_SEVERE_RATIO:
            eye_zone = 1
        else:
            eye_zone = 2

        mouth_zone = 1 if mar >= self.mouth_thresh else 0
        state = (eye_zone, mouth_zone)
        state_name = self.STATE_NAMES.get(state, "ALERT")
        action = self.ACTIONS.get(state_name, "no action")

        steps_away = self.distance_to_critical.get(state)
        if steps_away == 0:
            min_action = "full alarm"
        elif steps_away == 1:
            min_action = "flash yellow warning"
        else:
            min_action = "no action"

        if self.ACTION_SEVERITY[min_action] > self.ACTION_SEVERITY[action]:
            action = min_action

        return state_name, action


class RiskClassifier:

    def __init__(self):
        self.avg_ear = 0.28
        self.avg_mar = 0.40
        self.is_trained = False
        self.training_data = []

    def train(self, records):
        if len(records) < 30:
            return

        all_ear = [r['ear'] for r in records]
        all_mar = [r['mar'] for r in records]
        self.avg_ear = sum(all_ear) / len(all_ear)
        self.avg_mar = sum(all_mar) / len(all_mar)

        self.training_data = []
        for r in records:
            ear = r['ear']
            mar = r['mar']
            drowsy_pct = r['drowsy_pct']

            if drowsy_pct > 60 or ear < self.avg_ear * RISK_EAR_LOW:
                label = 'HIGH'
            elif drowsy_pct > 30 or ear < self.avg_ear * RISK_EAR_MID or mar > self.avg_mar * RISK_MAR_HIGH:
                label = 'MEDIUM'
            else:
                label = 'LOW'

            self.training_data.append((ear, mar, drowsy_pct, label))

        self.is_trained = True
        print(f"knn trained on {len(self.training_data)} samples (avg_ear={self.avg_ear:.3f}, avg_mar={self.avg_mar:.3f})")

    def predict(self, ear, mar, drowsy_score):
        if not self.is_trained or len(self.training_data) < 30:
            if drowsy_score > 60 or ear < self.avg_ear * RISK_EAR_LOW:
                return 'HIGH'
            elif drowsy_score > 30 or ear < self.avg_ear * RISK_EAR_MID:
                return 'MEDIUM'
            return 'LOW'

        k = 5
        distances = []
        for (t_ear, t_mar, t_score, label) in self.training_data:
            dist = math.sqrt(
                (ear - t_ear) ** 2 +
                (mar - t_mar) ** 2 +
                ((drowsy_score - t_score) / 100.0) ** 2
            )
            distances.append((dist, label))

        distances.sort(key=lambda x: x[0])
        top_k = [label for _, label in distances[:k]]

        return max(set(top_k), key=top_k.count)


class KMeansClustering:

    def __init__(self, k=3, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.cluster_labels = {}
        self.is_fitted = False

    def _euclidean_3d(self, a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

    def _init_centroids(self, data):
        # k-means++ spreads initial centroids out probabilistically to reduce
        # the chance of unlucky starts that cause label shuffling between retrains
        centroids = [random.choice(data)]
        for _ in range(self.k - 1):
            dists = [
                min(self._euclidean_3d(p, c) ** 2 for c in centroids)
                for p in data
            ]
            total = sum(dists)
            probs = [d / total for d in dists]
            cumulative, running = [], 0
            for p in probs:
                running += p
                cumulative.append(running)
            r = random.random()
            for i, threshold in enumerate(cumulative):
                if r <= threshold:
                    centroids.append(data[i])
                    break
        return centroids

    def fit(self, records):
        if len(records) < self.k * 10:
            return

        data = [
            [r['ear'], r['mar'], r['drowsy_pct'] / 100.0]
            for r in records
        ]

        self.centroids = self._init_centroids(data)

        for iteration in range(self.max_iterations):
            assignments = []
            for point in data:
                dists = [self._euclidean_3d(point, c) for c in self.centroids]
                assignments.append(dists.index(min(dists)))

            new_centroids = []
            for cluster_idx in range(self.k):
                cluster_points = [
                    data[i] for i in range(len(data))
                    if assignments[i] == cluster_idx
                ]
                if not cluster_points:
                    new_centroids.append(self.centroids[cluster_idx])
                else:
                    mean = [
                        sum(p[dim] for p in cluster_points) / len(cluster_points)
                        for dim in range(3)
                    ]
                    new_centroids.append(mean)

            moved = any(
                self._euclidean_3d(new_centroids[i], self.centroids[i]) > 1e-6
                for i in range(self.k)
            )
            self.centroids = new_centroids
            if not moved:
                print(f"k-means converged at iteration {iteration + 1}")
                break

        # rank clusters by avg drowsy_pct so the assignment is always
        # consistent: lowest avg = ALERT, highest = DROWSY
        cluster_drowsy_avg = []
        for cluster_idx in range(self.k):
            cluster_scores = [
                records[i]['drowsy_pct']
                for i in range(len(records))
                if assignments[i] == cluster_idx
            ]
            avg = sum(cluster_scores) / len(cluster_scores) if cluster_scores else 0
            cluster_drowsy_avg.append((cluster_idx, avg))

        cluster_drowsy_avg.sort(key=lambda x: x[1])
        label_names = ['ALERT', 'WARNING', 'DROWSY']
        self.cluster_labels = {
            cluster_idx: label_names[rank]
            for rank, (cluster_idx, _) in enumerate(cluster_drowsy_avg)
        }

        self.is_fitted = True
        print(f"k-means fitted, cluster labels: {self.cluster_labels}")
        print(f"centroids: {[[round(v, 3) for v in c] for c in self.centroids]}")

    def predict(self, ear, mar, drowsy_score):
        if not self.is_fitted:
            return '...'

        point = [ear, mar, drowsy_score / 100.0]
        dists = [self._euclidean_3d(point, c) for c in self.centroids]
        nearest = dists.index(min(dists))
        return self.cluster_labels.get(nearest, '?')


# small autoencoder built with plain numpy. trained only on calibration
# frames where the user is alert, so drowsy frames reconstruct poorly
# and the error spikes above the threshold
class Autoencoder:

    def __init__(self, input_dim=3, hidden_dim=2, learning_rate=0.05):
        self.lr = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.is_trained = False
        self.error_threshold = 0.05

        scale_1 = math.sqrt(1.0 / input_dim)
        scale_2 = math.sqrt(1.0 / hidden_dim)

        self.W1 = np.random.randn(input_dim, hidden_dim) * scale_1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, input_dim) * scale_2
        self.b2 = np.zeros((1, input_dim))

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_grad(self, x):
        return (x > 0).astype(float)

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_grad(self, s):
        return s * (1 - s)

    def _forward(self, x):
        z = self._relu(x @ self.W1 + self.b1)
        xhat = self._sigmoid(z @ self.W2 + self.b2)
        return z, xhat

    def _backward(self, x, z, xhat):
        d_xhat = -(x - xhat)
        d_b2 = d_xhat * self._sigmoid_grad(xhat)
        d_W2 = z.T @ d_b2

        d_z = d_b2 @ self.W2.T
        d_b1 = d_z * self._relu_grad(z)
        d_W1 = x.T @ d_b1

        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1

    def _normalise(self, ear, mar, drowsy_score):
        return np.array([[
            min(ear / 0.5, 1.0),
            min(mar / 0.8, 1.0),
            min(drowsy_score / 100.0, 1.0),
        ]])

    def train(self, calib_records, epochs=200):
        if len(calib_records) < 10:
            print("not enough calibration data to train the autoencoder")
            return

        print(f"training autoencoder on {len(calib_records)} calibration frames...")

        for epoch in range(epochs):
            total_loss = 0.0
            random.shuffle(calib_records)

            for r in calib_records:
                x = self._normalise(r['ear'], r['mar'], r['drowsy_pct'])
                z, xhat = self._forward(x)
                loss = float(np.mean((x - xhat) ** 2))
                total_loss += loss
                self._backward(x, z, xhat)

            if (epoch + 1) % 50 == 0:
                avg_loss = total_loss / len(calib_records)
                print(f"  epoch {epoch+1}/{epochs}  avg_loss={avg_loss:.6f}")

        final_errors = []
        for r in calib_records:
            x = self._normalise(r['ear'], r['mar'], r['drowsy_pct'])
            _, xhat = self._forward(x)
            final_errors.append(float(np.mean((x - xhat) ** 2)))

        avg_train_error = sum(final_errors) / len(final_errors)
        # floor lowered to 0.005 so anomaly detection fires more readily
        self.error_threshold = max(avg_train_error * 2.0, 0.005)
        self.is_trained = True

        print("training complete")
        print(f"avg training error: {avg_train_error:.6f}")
        print(f"anomaly threshold: {self.error_threshold:.6f}")

    def predict(self, ear, mar, drowsy_score):
        if not self.is_trained:
            return 0.0, False

        x = self._normalise(ear, mar, drowsy_score)
        _, xhat = self._forward(x)
        error = float(np.mean((x - xhat) ** 2))
        is_anomaly = error > self.error_threshold
        return error, is_anomaly


def build_alarm():
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)

    duration = 0.6
    sample_rate = 22050
    time_array = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    half = len(time_array) // 2

    wave = np.concatenate([
        np.sin(2 * np.pi * 1100 * time_array[:half]),
        np.sin(2 * np.pi * 880 * time_array[half:]),
    ]) * 0.8

    samples = np.column_stack([(wave * 32767).astype(np.int16)] * 2)
    return pygame.sndarray.make_sound(samples)


def draw_sidebar(frame, h, ear, thresh, mar,
                 score, risk_label, state_name,
                 blinks, yawns,
                 cluster_label, ae_error, ae_anomaly):

    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (SIDEBAR_W, h), (18, 18, 18), -1)
    cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)
    cv2.line(frame, (SIDEBAR_W, 0), (SIDEBAR_W, h), GREY, 1)

    cv2.putText(frame, "DROWSY DETECT", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.50, BLUE, 1)

    ear_col = RED if (ear is not None and ear < thresh) else GREEN
    ear_text = f"EAR : {ear:.3f}" if ear is not None else "EAR : --"
    cv2.putText(frame, ear_text, (10, 88), FONT, 0.50, ear_col, 2)
    cv2.putText(frame, f"Thr : {thresh:.3f}", (10, 108), FONT, 0.37, GREY, 1)
    cv2.putText(frame, f"MAR : {mar:.2f}", (10, 128), FONT, 0.45, WHITE, 1)

    bar_x1, bar_x2 = 10, SIDEBAR_W - 10
    cv2.rectangle(frame, (bar_x1, 144), (bar_x2, 158), (50, 50, 55), -1)
    fill_px = int(score * (bar_x2 - bar_x1) / 100)
    bar_col = RED if score > 70 else YELLOW if score > 40 else GREEN
    cv2.rectangle(frame, (bar_x1, 144), (bar_x1 + fill_px, 158), bar_col, -1)
    cv2.putText(frame, f"Drowsy: {score}%", (10, 175), FONT, 0.40, WHITE, 1)

    risk_col = RED if risk_label == "HIGH" else YELLOW if risk_label == "MEDIUM" else GREEN
    cv2.putText(frame, f"Risk: {risk_label}", (10, 195), FONT, 0.42, risk_col, 1)

    cv2.putText(frame, f"State:{state_name[:9]}", (10, 215), FONT, 0.37, PURPLE, 1)

    cl_col = RED if cluster_label == "DROWSY" else YELLOW if cluster_label == "WARNING" else GREEN
    cv2.putText(frame, f"Clust:{cluster_label[:7]}", (10, 235), FONT, 0.37, cl_col, 1)

    ae_col = RED if ae_anomaly else CYAN
    ae_text = f"AE:{ae_error:.4f}{'!!' if ae_anomaly else ''}"
    cv2.putText(frame, ae_text, (10, 255), FONT, 0.37, ae_col, 1)

    # changed from GREY to WHITE so the counts are clearly readable
    # on the dark sidebar background
    cv2.putText(frame, f"Blinks: {blinks}", (10, 278), FONT, 0.37, WHITE, 1)
    cv2.putText(frame, f"Yawns : {yawns}", (10, 296), FONT, 0.37, WHITE, 1)


def get_camera_fps(cap):
    # read fps from the camera, fall back to 30 if it returns nothing useful
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        return fps
    return 30.0


def main():

    print()

    ga = GeneticAlgorithm()
    ear_drop, mar_threshold, consec_frames = ga.run()
    mar_threshold = max(mar_threshold, 0.28)

    logger = SessionLogger()
    classifier = RiskClassifier()
    kmeans = KMeansClustering(k=3)
    autoenc = Autoencoder(input_dim=3, hidden_dim=2, learning_rate=0.05)
    sm = None

    alarm_sound = build_alarm()
    alarm_playing = False

    eye_closed_frames = 0
    total_blinks = 0
    total_yawns = 0
    yawn_active = False
    smoothed_ear = None
    risk_label = "LOW"
    state_name = "ALERT"
    cluster_label = "..."
    ae_error = 0.0
    ae_anomaly = False
    calib_records = []
    calib_values = []
    is_calibrated = False
    baseline_ear = 0.30
    dynamic_thresh = 0.22
    frames_since_retrain = 0
    camera_fps = 30.0  # updated once the camera opens

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open camera. Check that a webcam is connected and not in use.")
        return

    camera_fps = get_camera_fps(cap)
    print(f"[INFO] Camera FPS: {camera_fps:.1f}")

    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    print("\n[INFO] Camera ready. Keep eyes open normally during calibration.")
    print("[INFO] Press Q or ESC to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame from camera. Ending session.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                face_lm = result.multi_face_landmarks[0]
                lm = face_lm.landmark

                mp_drawing.draw_landmarks(
                    frame, face_lm,
                    mp_face_mesh.FACEMESH_CONTOURS, None,
                    mp_styles.get_default_face_mesh_contours_style())

                for connections, colour in [
                    (mp_face_mesh.FACEMESH_LEFT_EYE, (0, 0, 255)),
                    (mp_face_mesh.FACEMESH_RIGHT_EYE, (0, 0, 255)),
                    (mp_face_mesh.FACEMESH_LIPS, (255, 180, 0)),
                ]:
                    mp_drawing.draw_landmarks(
                        frame, face_lm, connections, None,
                        mp_drawing.DrawingSpec(color=colour, thickness=2, circle_radius=1))

                left_ear = compute_ear(lm, LEFT_EYE, w, h)
                right_ear = compute_ear(lm, RIGHT_EYE, w, h)
                raw_ear = (left_ear + right_ear) / 2.0
                mar = compute_mar(lm, w, h)

                if smoothed_ear is None:
                    smoothed_ear = raw_ear
                else:
                    smoothed_ear = SMOOTH_ALPHA * raw_ear + (1 - SMOOTH_ALPHA) * smoothed_ear

                if not is_calibrated:
                    calib_values.append(smoothed_ear)
                    calib_records.append({
                        "ear": round(smoothed_ear, 4),
                        "mar": round(mar, 4),
                        "drowsy_pct": 0,
                    })

                    progress = int(len(calib_values) / CALIB_FRAMES * 100)
                    cv2.putText(frame, f"CALIBRATING {progress}%",
                                (SIDEBAR_W + 15, 50), FONT, 0.8, YELLOW, 2)
                    cv2.putText(frame, "Keep eyes open normally",
                                (SIDEBAR_W + 15, 80), FONT, 0.55, WHITE, 1)

                    if len(calib_values) >= CALIB_FRAMES:
                        sorted_vals = sorted(calib_values)
                        # 75th percentile filters out any blinks during calibration
                        baseline_ear = sorted_vals[int(len(sorted_vals) * 0.75)]

                        ga.set_baseline(calib_values)
                        ear_drop, mar_threshold, consec_frames = ga.rerun_with_calibration()
                        mar_threshold = max(mar_threshold, 0.28)
                        dynamic_thresh = max(baseline_ear * ear_drop, 0.15)
                        is_calibrated = True
                        sm = AlertStateMachine(dynamic_thresh, mar_threshold)

                        autoenc.train(calib_records, epochs=200)

                        print(f"[INFO] Calibrated. Baseline EAR={baseline_ear:.3f}  "
                              f"Threshold={dynamic_thresh:.3f}")

                    draw_sidebar(frame, h, smoothed_ear, dynamic_thresh,
                                 mar, 0, risk_label, state_name,
                                 total_blinks, total_yawns,
                                 cluster_label, ae_error, ae_anomaly)
                    cv2.imshow("Driver Drowsiness Detection", frame)
                    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                        break
                    continue

                if smoothed_ear < dynamic_thresh:
                    eye_closed_frames += 1
                    alert_level = 2 if eye_closed_frames >= consec_frames else 1
                    if alert_level == 2 and not alarm_playing:
                        alarm_sound.play(-1)
                        alarm_playing = True
                else:
                    if eye_closed_frames >= consec_frames:
                        total_blinks += 1
                    eye_closed_frames = max(0, eye_closed_frames - 2)
                    if eye_closed_frames == 0 and alarm_playing:
                        alarm_sound.stop()
                        alarm_playing = False
                    alert_level = 0

                if mar > mar_threshold and not yawn_active:
                    total_yawns += 1
                    yawn_active = True
                elif mar <= mar_threshold:
                    yawn_active = False

                drowsy_score = min(int(eye_closed_frames / consec_frames * 100), 100)

                logger.log(smoothed_ear, mar, drowsy_score,
                           total_blinks, total_yawns, alert_level)
                frames_since_retrain += 1

                if frames_since_retrain >= RETRAIN_EVERY:
                    classifier.train(logger.records)
                    kmeans.fit(logger.records)
                    frames_since_retrain = 0

                risk_label = classifier.predict(smoothed_ear, mar, drowsy_score)
                cluster_label = kmeans.predict(smoothed_ear, mar, drowsy_score)
                ae_error, ae_anomaly = autoenc.predict(smoothed_ear, mar, drowsy_score)

                if sm is not None:
                    state_name, action = sm.evaluate(smoothed_ear, mar)

                draw_sidebar(frame, h, smoothed_ear, dynamic_thresh,
                             mar, drowsy_score, risk_label,
                             state_name, total_blinks, total_yawns,
                             cluster_label, ae_error, ae_anomaly)

                if alert_level == 2:
                    cv2.rectangle(frame, (SIDEBAR_W, 0), (w, h), (0, 0, 180), 8)
                    cv2.putText(frame, "WAKE UP!",
                                (w // 2 - 100, h // 2),
                                cv2.FONT_HERSHEY_DUPLEX, 1.8, WHITE, 3)
                elif alert_level == 1:
                    cv2.putText(frame, "Eyes closing...",
                                (SIDEBAR_W + 15, 50), FONT, 0.7, YELLOW, 2)
                else:
                    if ae_anomaly:
                        cv2.putText(frame, "AE ANOMALY",
                                    (SIDEBAR_W + 15, 50), FONT, 0.7, ORANGE, 2)
                    else:
                        cv2.putText(frame, "ALERT",
                                    (SIDEBAR_W + 15, 50), FONT, 0.7, GREEN, 2)

            else:
                cv2.putText(frame, "No face detected",
                            (SIDEBAR_W + 15, 50), FONT, 0.8, RED, 2)
                if alarm_playing:
                    alarm_sound.stop()
                    alarm_playing = False
                draw_sidebar(frame, h, smoothed_ear, dynamic_thresh,
                             0, 0, risk_label, state_name,
                             total_blinks, total_yawns,
                             cluster_label, ae_error, ae_anomaly)

            cv2.imshow("Driver Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")

    finally:
        # always runs, cleans up even if an exception fires mid-session
        print("\n[INFO] Session ended.")
        if alarm_playing:
            alarm_sound.stop()
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        logger.save_to_csv()
        logger.mine_patterns()
        print("[INFO] Done. Check session_log.csv for your session data.")


if __name__ == "__main__":
    main()