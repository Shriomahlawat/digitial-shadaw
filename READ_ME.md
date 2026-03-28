Digital Shadow — Face Tracking + Particle Shadow + Hand Landmarks
A real-time computer vision application that creates a colorful particle shadow of your face using your webcam. It also tracks hand gestures to let you dynamically control the particle effect — zoom in, spread out, or trigger an explosion.

Python OpenCV MediaPipe

How It Works
Face Mesh Detection — MediaPipe detects 468+ facial landmarks in real time.
Particle System — Thousands of colored particles are spawned on the face mesh and continuously track the landmark positions, creating a glowing digital shadow shifted to the left of your actual face.
Hand Gesture Control — A single hand is tracked; opening/closing your hand controls particle zoom and spread. Making a fist triggers an explosion effect.
Features
Real-time face mesh particle rendering with up to 4,000 particles across 4 depth layers (core, mid, outer haze, far outer).
Multi-color neon palette — cyan, magenta, electric blue, purple, gold, mint green — with per-particle glow effects.
Hand gesture interaction:
Open hand (close to camera): particles zoom in tight around the face.
Open hand (far from camera): particles spread outward.
Fist: triggers an explosion that scatters all particles outward, then they reform.
Hand skeleton overlay with glowing joints and connections.
Gaussian blur glow pass for a soft neon aesthetic.
Requirements
Python 3.8+
A webcam
Python Packages
Package	Purpose
opencv-python	Webcam capture & rendering
mediapipe	Face mesh & hand tracking
numpy	Image array operations
Setup
# Clone or download the project
cd Facemesh

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install opencv-python mediapipe numpy
Usage
python app.py
A window titled "Digital Shadow" will open showing your webcam feed with the particle shadow effect.
Move your hand in front of the camera to control the particles.
Press q or Esc to quit.
Controls
Gesture	Effect
No hand visible	Default scale (1.3×) and spread
Open hand (far)	Particles spread wide, small scale
Open hand (close)	Particles tight, large scale (up to 2.5×)
Close fist	Explosion — particles scatter outward for ~60 frames then reform
Project Structure
Facemesh/
├── app.py        # Main application (all logic in one file)
├── venv/         # Python virtual environment
└── README.md     # This file
License
This project is for personal/educational use.
