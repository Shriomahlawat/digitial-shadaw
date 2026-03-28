I built "Digital Shadow," a real-time face tracking particle system that uses hand gesture control. This computer vision project transforms your face into a glowing particle shadow using just a webcam.

What it does:
• Detects 468+ facial landmarks with MediaPipe and renders 4,000 neon particles that follow your face.
• Tracks hand gestures: open your hand to zoom or spread the particles, and make a fist to trigger an explosion effect.
• Creates a cyberpunk-style neon glow with a multi-color palette including cyan, magenta, electric blue, purple, gold, and mint green.

Tech Stack: Python | OpenCV | MediaPipe | NumPy

How it works:
The app captures your webcam feed, runs face mesh detection to obtain landmark positions, and spawns particles across four depth layers that continuously track those points. A hand tracking module maps finger spread to particle scale and spread in real time. A Gaussian blur pass adds the neon glow aesthetic.

This project was a fun exploration of real-time computer vision, particle systems, and gesture-based interaction, all running live from a single Python script.
