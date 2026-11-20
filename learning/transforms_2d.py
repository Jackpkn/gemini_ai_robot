"""Day 5 (Part 1): Coordinate Frames & Transformation Matrices (2D, plain Mac Python).

Run directly on your Mac (no Docker needed):
  .venv/bin/python learning/transforms_2d.py

Why learn this?
- A Transformation Matrix packs BOTH rotation (angle) and translation (x, y) into one grid.
- Chaining robot joints becomes simple multiplication: `T_world_to_hand = T1 @ T2 @ T3 ...`
- Going backwards ("where is the cube from my hand's view?") is just matrix inverse!
- In Day 5 (Part 2), ROS 2's TF2 library does this exact same math in 3D automatically.
"""

import numpy as np


def make_transform(dx: float, dy: float, angle_deg: float) -> np.ndarray:
    """Build a 3x3 2D Homogeneous Transformation Matrix.

    Layout:
      [ cos(a)  -sin(a)   dx ]   <-- top-left 2x2 = rotation; right column = (dx, dy) shift
      [ sin(a)   cos(a)   dy ]
      [   0        0       1 ]   <-- bottom row [0, 0, 1] lets us multiply matrices cleanly!
    """
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, dx],
        [s,  c, dy],
        [0,  0,  1],
    ], dtype=float)


def arm_fk_matrix(L1: float, L2: float, theta1_deg: float, theta2_deg: float) -> np.ndarray:
    """Compute world -> hand transform for a 2-link arm by multiplying simple steps."""
    # Step 1: Rotate shoulder by theta1, then walk along link 1 (length L1) to the elbow.
    T_base_to_shoulder = make_transform(0.0, 0.0, theta1_deg)
    T_shoulder_to_elbow = make_transform(L1, 0.0, 0.0)

    # Step 2: Rotate elbow by theta2, then walk along link 2 (length L2) to the hand.
    T_elbow_rot = make_transform(0.0, 0.0, theta2_deg)
    T_elbow_to_hand = make_transform(L2, 0.0, 0.0)

    # Chain them with `@` (matrix multiplication)!
    return T_base_to_shoulder @ T_shoulder_to_elbow @ T_elbow_rot @ T_elbow_to_hand


def main() -> None:
    L1, L2 = 5.0, 5.0

    # 1. Verify our matrix gives the exact same hand (x, y) as manual cos/sin trig:
    for t1, t2 in [(0, 0), (90, 0), (0, 90), (45, 45)]:
        T_world_hand = arm_fk_matrix(L1, L2, t1, t2)
        hand_x, hand_y = T_world_hand[0, 2], T_world_hand[1, 2]
        print(f"Angles ({t1:>2}°, {t2:>2}°) -> Hand position in world: ({hand_x:6.2f}, {hand_y:6.2f})")

    # 2. The superpower: switch point of view!
    #    Suppose the arm is at (0°, 90°), so the hand is at (5, 5) pointing UP (+Y).
    #    A target cube sits at (5, 8) in the world.
    #    Where is the cube FROM THE HAND'S OWN EYES?
    T_world_hand = arm_fk_matrix(L1, L2, theta1_deg=0, theta2_deg=90)
    cube_in_world = np.array([5.0, 8.0, 1.0])  # trailing 1 = a 2D point in homogeneous coords

    # Invert T_world_hand to get T_hand_world ("world seen from the hand")!
    T_hand_world = np.linalg.inv(T_world_hand)
    cube_in_hand = T_hand_world @ cube_in_world

    print("\n--- Switching Point of View (What ROS 2 TF2 does in 3D!) ---")
    print("Arm at (0°, 90°): hand is at (5.0, 5.0) pointing straight UP (+Y).")
    print("Cube in world coordinates: (5.0, 8.0)")
    print(f"Cube seen FROM THE HAND:   forward={cube_in_hand[0]:.2f}, left={cube_in_hand[1]:.2f}")


if __name__ == "__main__":
    main()
