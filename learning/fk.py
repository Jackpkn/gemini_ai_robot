import numpy as np

def forward_kinematics(L1, L2, theta1, theta2):
    theta1 = np.radians(theta1)
    theta2 = np.radians(theta2)

    # calculate the end effector position
    X = ( L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2))

    y = ( L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2))

    return X, y

print(forward_kinematics(5, 5, 0, 0))

# we need to learn transformation matrics
