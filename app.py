import cv2
import numpy as np
import random
import math
from collections import deque

# ✅ FIXED MEDIAPIPE IMPORT (WORKS WITH PYTHON 3.10+)
import mediapipe as mp

try:
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing


# =========================
# MEDIAPIPE SETUP
# =========================
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4,
)

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)

# =========================
# HELPER FUNCTIONS
# =========================
def lm_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def draw_glow_dot(img, center, color, r=2):
    cv2.circle(img, center, r + 4,
               (color[0] // 4, color[1] // 4, color[2] // 4),
               -1, cv2.LINE_AA)
    cv2.circle(img, center, r, color, -1, cv2.LINE_AA)

def draw_glow_line(img, p1, p2, color, thickness=1):
    cv2.line(img, p1, p2,
             (color[0] // 3, color[1] // 3, color[2] // 3),
             thickness + 2, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


# =========================
# PARTICLE CLASS
# =========================
class Particle:
    def __init__(self, x, y, layer=0):
        spreads = [3, 10, 22, 40]
        self.x = float(x) + random.gauss(0, spreads[layer])
        self.y = float(y) + random.gauss(0, spreads[layer])
        self.vx = 0.0
        self.vy = 0.0
        self.layer = layer
        self.life = random.randint(60, 160)
        self.max_life = self.life

        self.radius = random.choice([1, 2, 3])
        self.phase = random.uniform(0, math.pi * 2)
        self.orbit_speed = random.uniform(0.01, 0.04)
        self.jitter = random.uniform(0.5, 2.5)

    def update(self, target_x, target_y):
        self.phase += self.orbit_speed
        jx = math.cos(self.phase) * self.jitter
        jy = math.sin(self.phase) * self.jitter

        goal_x = target_x + jx
        goal_y = target_y + jy

        self.vx += (goal_x - self.x) * 0.5
        self.vy += (goal_y - self.y) * 0.5
        self.vx *= 0.45
        self.vy *= 0.45

        self.x += self.vx
        self.y += self.vy
        self.life -= 1


# =========================
# MAIN VARIABLES
# =========================
face_particles = []
max_particles = 3000

exploding = False
explode_frame = 0


# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_result = face_mesh.process(rgb)
    hand_result = hands.process(rgb)

    display = frame.copy()

    # =========================
    # FACE PARTICLES
    # =========================
    if face_result.multi_face_landmarks:
        face_landmarks = face_result.multi_face_landmarks[0]
        points = [lm_to_px(lm, w, h) for lm in face_landmarks.landmark]

        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)

        shifted = []
        for x, y in points:
            sx = cx + (x - cx) * 1.3 - int(w * 0.25)
            sy = cy + (y - cy) * 1.3
            shifted.append((sx, sy))

        # spawn particles
        while len(face_particles) < max_particles:
            px, py = random.choice(shifted)
            face_particles.append(Particle(px, py))

        # update particles
        new_particles = []
        for p in face_particles:
            tx, ty = random.choice(shifted)

            if exploding:
                dx = p.x - cx
                dy = p.y - cy
                dist = math.sqrt(dx * dx + dy * dy) + 1
                p.vx += (dx / dist) * 2
                p.vy += (dy / dist) * 2
                p.x += p.vx
                p.y += p.vy
                p.life -= 1
            else:
                p.update(tx, ty)

            if p.life > 0:
                new_particles.append(p)
            else:
                new_particles.append(Particle(tx, ty))

        face_particles = new_particles

        # draw particles
        particle_layer = np.zeros_like(display)

        for p in face_particles:
            if p.life > 0:
                cv2.circle(particle_layer,
                           (int(p.x), int(p.y)),
                           p.radius,
                           (255, 255, 255), -1)

        glow = cv2.GaussianBlur(particle_layer, (15, 15), 0)
        display = cv2.add(display, particle_layer)
        display = cv2.add(display, glow)

    # =========================
    # HAND TRACKING
    # =========================
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                display, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # detect fist → explosion
            wrist = hand_landmarks.landmark[0]
            tip = hand_landmarks.landmark[8]

            dist = math.hypot(wrist.x - tip.x, wrist.y - tip.y)

            if dist < 0.1:
                exploding = True
                explode_frame = 0

    if exploding:
        explode_frame += 1
        if explode_frame > 30:
            exploding = False

    # =========================
    # TEXT
    # =========================
    cv2.putText(display, "DIGITAL SHADOW",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (180, 180, 180), 2)

    # =========================
    # SHOW
    # =========================
    cv2.imshow("Digital Shadow", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
