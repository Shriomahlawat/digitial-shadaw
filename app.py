"""
Digital Shadow — Face Tracking + Particle Shadow + Hand Landmarks
Real-time face mesh particle effect with hand gesture controls.
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import math
from collections import deque



#  1. MEDIAPIPE SETUP

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4,
)



#  2. WEBCAM

cap = cv2.VideoCapture(0)



#  3. HELPER FUNCTIONS


def lm_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def draw_glow_dot(img, center, color, r=2):
    """Fast glow dot — no copies."""
    cv2.circle(img, center, r + 4,
               (color[0] // 4, color[1] // 4, color[2] // 4),
               -1, cv2.LINE_AA)
    cv2.circle(img, center, r, color, -1, cv2.LINE_AA)


def draw_glow_line(img, p1, p2, color, thickness=1):
    """Fast glow line — no copies."""
    cv2.line(img, p1, p2,
             (color[0] // 3, color[1] // 3, color[2] // 3),
             thickness + 2, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


# ============================================================
#  4. PARTICLE SYSTEM
# ============================================================

class Particle:
    def __init__(self, x, y, layer=0):
        # layer 0 = core tight, 1 = mid, 2 = outer haze, 3 = far outer depth
        spreads = [3, 10, 22, 40]
        self.x = float(x) + random.gauss(0, spreads[layer])
        self.y = float(y) + random.gauss(0, spreads[layer])
        self.vx = 0.0
        self.vy = 0.0
        self.layer = layer
        self.life = random.randint(60, 160)
        self.max_life = self.life
        radii = [[1, 1, 1, 2], [1, 1, 2, 2], [1, 2, 2, 3], [2, 2, 3, 3]]
        self.radius = random.choice(radii[layer])
        self.phase = random.uniform(0, math.pi * 2)
        self.orbit_speed = random.uniform(0.01, 0.04)
        self.jitter = random.uniform(0.5, 2.5) * (1 + layer * 0.6)
        self.color_idx = random.randint(0, 5)

    def update(self, target_x, target_y):
        self.phase += self.orbit_speed
        jx = math.cos(self.phase) * self.jitter
        jy = math.sin(self.phase) * self.jitter
        goal_x = target_x + jx
        goal_y = target_y + jy
        self.vx += (goal_x - self.x) * 0.50
        self.vy += (goal_y - self.y) * 0.50
        self.vx *= 0.45
        self.vy *= 0.45
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def get_draw_params(self):
        life_ratio = max(self.life / self.max_life, 0.0)
        brightness = life_ratio * [1.0, 0.85, 0.55, 0.35][self.layer]
        palette = [
            (255, 255, 0),    # cyan
            (255, 0, 255),    # magenta / pink
            (255, 180, 0),    # electric blue
            (180, 0, 255),    # purple
            (0, 220, 255),    # gold / yellow
            (200, 255, 100),  # mint green
        ]
        color = palette[self.color_idx % len(palette)]
        return int(self.x), int(self.y), self.radius, color, brightness


# ============================================================
#  5. CONSTANTS & LANDMARK GROUPS
# ============================================================

FACE_TRIS = list(mp_face_mesh.FACEMESH_TESSELATION)

FACE_OUTLINE = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)


# ============================================================
#  6. STATE VARIABLES
# ============================================================

face_particles   = []
max_particles    = 4000
frame_count      = 0
face_trails      = deque(maxlen=20)

# Gesture control
smooth_spread    = 1.0    # particle spread factor
smooth_scale     = 1.3    # face size multiplier
exploding        = False  # explosion active?
explode_frame    = 0      # frames since explosion start
explode_cx       = 0.0    # explosion center x
explode_cy       = 0.0    # explosion center y
prev_hand_size   = 0.0    # previous avg finger dist (fist detection)
smooth_hand_ratio = 0.0   # smoothed hand ratio for stable zoom
hand_lost_frames = 0      # frames since hand was last seen


# ============================================================
#  7. MAIN LOOP
# ============================================================

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

    # --------------------------------------------------
    #  7a. GESTURE DETECTION (hand size → zoom / spread)
    # --------------------------------------------------
    target_spread = 1.0
    target_scale  = 1.3
    raw_hand_ratio = 0.0
    debug_state = "no hand"
    is_fist = False

    if hand_result.multi_hand_landmarks:
        hand_lost_frames = 0
        hl = hand_result.multi_hand_landmarks[0]

        wrist      = hl.landmark[0]
        thumb_tip  = hl.landmark[4]
        index_tip  = hl.landmark[8]
        middle_tip = hl.landmark[12]
        ring_tip   = hl.landmark[16]
        pinky_tip  = hl.landmark[20]

        # Normalized wrist-to-middle-fingertip distance (resolution-independent)
        raw_hand_ratio = math.hypot(wrist.x - middle_tip.x,
                                    wrist.y - middle_tip.y)

        # Near-instant hand ratio follow
        smooth_hand_ratio += (raw_hand_ratio - smooth_hand_ratio) * 0.85

        # Fist detection: average fingertip distance from wrist
        avg_finger_dist = (
            math.hypot(index_tip.x  - wrist.x, index_tip.y  - wrist.y) +
            math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y) +
            math.hypot(ring_tip.x   - wrist.x, ring_tip.y   - wrist.y) +
            math.hypot(pinky_tip.x  - wrist.x, pinky_tip.y  - wrist.y)
        ) / 4.0

        is_fist = avg_finger_dist < 0.10

        if is_fist and prev_hand_size > 0.10 and not exploding:
            exploding = True
            explode_frame = 0

        prev_hand_size = avg_finger_dist

        # Continuous linear mapping  (hand_ratio ≈ 0.05–0.55)
        if is_fist:
            debug_state = "FIST"
            target_scale  = 0.8
            target_spread = 0.5
        else:
            r = max(0.05, min(smooth_hand_ratio, 0.55))
            t = (r - 0.05) / 0.50                  # normalize → 0..1
            target_scale  = 0.8 + t * 1.7           # far 0.8 → close 2.5
            target_spread = 3.5 - t * 3.0            # far 3.5 → close 0.5
            debug_state = f"t={t:.2f} r={r:.3f}"
    else:
        hand_lost_frames += 1
        prev_hand_size = 0.0
        if hand_lost_frames > 10:
            smooth_hand_ratio *= 0.95

    # Near-instant scale / spread response
    smooth_scale  += (target_scale  - smooth_scale)  * 0.8
    smooth_spread += (target_spread - smooth_spread) * 0.8

    # Explosion timer
    if exploding:
        explode_frame += 1
        if explode_frame > 60:
            exploding = False

    # --------------------------------------------------
    #  7b. FACE PARTICLE SHADOW
    # --------------------------------------------------
    if face_result.multi_face_landmarks:
        face_landmarks = face_result.multi_face_landmarks[0]
        lm_points = [lm_to_px(lm, w, h) for lm in face_landmarks.landmark]

        # Compute face centre & shift left + scale
        face_cx = sum(p[0] for p in lm_points) / len(lm_points)
        face_cy = sum(p[1] for p in lm_points) / len(lm_points)
        shift_amount = int(w * 0.25)

        shifted_points = []
        for (x, y) in lm_points:
            sx = face_cx + (x - face_cx) * smooth_scale - shift_amount
            sy = face_cy + (y - face_cy) * smooth_scale
            shifted_points.append((sx, sy))

        # Explosion reference centre
        explode_cx = sum(p[0] for p in shifted_points) / len(shifted_points)
        explode_cy = sum(p[1] for p in shifted_points) / len(shifted_points)

        # --- Dense fill points via tessellation ---
        dense_points = list(shifted_points)

        # Edge midpoints
        for conn in FACE_TRIS:
            i1, i2 = conn
            if i1 < len(shifted_points) and i2 < len(shifted_points):
                x1, y1 = shifted_points[i1]
                x2, y2 = shifted_points[i2]
                dense_points.append(((x1 + x2) / 2, (y1 + y2) / 2))

        # Barycentric interior points from consecutive edge pairs
        tris_list = list(FACE_TRIS)
        for i in range(0, len(tris_list) - 2, 1):
            c1 = tris_list[i]
            c2 = tris_list[i + 1]
            verts = list(set([c1[0], c1[1], c2[0], c2[1]]))
            if len(verts) == 3:
                a, b, c = verts
                if (a < len(shifted_points) and b < len(shifted_points)
                        and c < len(shifted_points)):
                    ax, ay = shifted_points[a]
                    bx, by = shifted_points[b]
                    ccx, ccy = shifted_points[c]
                    r1 = random.random()
                    r2 = random.random()
                    if r1 + r2 > 1:
                        r1, r2 = 1 - r1, 1 - r2
                    px = ax + r1 * (bx - ax) + r2 * (ccx - ax)
                    py = ay + r1 * (by - ay) + r2 * (ccy - ay)
                    dense_points.append((px, py))

        # Apply spread from face centre
        cx = sum(p[0] for p in shifted_points) / len(shifted_points)
        cy = sum(p[1] for p in shifted_points) / len(shifted_points)

        spread_points = []
        for (x, y) in dense_points:
            dx = x - cx
            dy = y - cy
            spread_points.append((cx + dx * smooth_spread,
                                  cy + dy * smooth_spread))

        selected_points = spread_points
        num_targets = len(selected_points)

        # Spawn particles
        while len(face_particles) < max_particles:
            idx = random.randint(0, num_targets - 1)
            px, py = selected_points[idx]
            layer = random.choices([0, 1, 2, 3], weights=[6, 3, 1, 1])[0]
            p = Particle(px, py, layer)
            p.anchor_idx = idx
            face_particles.append(p)

        # Update particles (normal follow or explosion)
        for p in face_particles:
            idx = p.anchor_idx % num_targets
            tx, ty = selected_points[idx]

            if exploding:
                dx = p.x - explode_cx
                dy = p.y - explode_cy
                dist = math.sqrt(dx * dx + dy * dy) + 0.1
                force = 800.0 / (dist + 50)
                p.vx += (dx / dist) * force * 0.3
                p.vy += (dy / dist) * force * 0.3
                p.vx *= 0.96
                p.vy *= 0.96
                p.x += p.vx
                p.y += p.vy
                p.life -= 1
            else:
                p.update(tx, ty)

        # Respawn dead particles
        new_particles = []
        for p in face_particles:
            if p.life > 0:
                new_particles.append(p)
            else:
                idx = random.randint(0, num_targets - 1)
                tx, ty = selected_points[idx]
                layer = random.choices([0, 1, 2, 3], weights=[6, 3, 1, 1])[0]
                np_ = Particle(tx, ty, layer)
                np_.anchor_idx = idx
                new_particles.append(np_)
        face_particles = new_particles

        # --- Batch particle rendering ---
        particle_layer = np.zeros_like(display)

        for p in face_particles:
            px, py, r, color, bright = p.get_draw_params()
            if bright > 0.1:
                sc = (int(color[0] * bright),
                      int(color[1] * bright),
                      int(color[2] * bright))
                cv2.circle(particle_layer, (px, py), r, sc, -1, cv2.LINE_AA)

        glow = cv2.GaussianBlur(particle_layer, (15, 15), 0)
        display = cv2.add(display, particle_layer)
        display = cv2.add(display, glow)

    # --------------------------------------------------
    #  7c. HAND SKELETON
    # --------------------------------------------------
    if hand_result.multi_hand_landmarks:
        hand_landmarks = hand_result.multi_hand_landmarks[0]
        hand_points = [lm_to_px(lm, w, h) for lm in hand_landmarks.landmark]

        for a, b in HAND_CONNECTIONS:
            draw_glow_line(display, hand_points[a], hand_points[b],
                           (220, 220, 220), 2)

        for pt in hand_points:
            draw_glow_dot(display, pt, (240, 240, 240), r=3)

    # --------------------------------------------------
    #  7d. OVERLAY TEXT
    # --------------------------------------------------
    cv2.putText(display, "DIGITAL SHADOW",
                (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (180, 180, 180), 2, cv2.LINE_AA)

    cv2.putText(display,
                "FACE TRACKING + PARTICLE SHADOW + HAND LANDMARKS",
                (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120, 120, 120), 1, cv2.LINE_AA)

    cv2.putText(display,
                f"hand: {smooth_hand_ratio:.3f}  scale: {smooth_scale:.1f}"
                f"  spread: {smooth_spread:.1f}  [{debug_state}]",
                (25, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 200, 200), 1, cv2.LINE_AA)

    # --------------------------------------------------
    #  7e. DISPLAY & EXIT
    # --------------------------------------------------
    cv2.imshow("Digital Shadow", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
