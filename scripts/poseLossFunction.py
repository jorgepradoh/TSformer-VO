# PoseLoss: Custom loss function for 6-DoF pose estimation.
# It combines:
# - Translation loss: MSE between predicted and ground truth positions (x, y, z)
# - Rotation loss: Geodesic distance (angular difference) between predicted and ground truth orientations, converted from Euler angles to quaternions
# Both losses are weighted using learnable log-variance terms (log_sigma_t and log_sigma_r) to balance their contribution during training.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Convert euler angles (in radians) to quaternion
def euler_to_quaternion(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    # Quaternion components (w, x, y, z)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return torch.stack([qw, qx, qy, qz], dim=-1)

# Geodesic loss between quaternions
def quaternion_geodesic_loss(q_pred, q_gt, epsilon=1e-6):
    # Ensure quaternions are normalized
    q_pred = F.normalize(q_pred, dim=-1, eps=epsilon)
    q_gt = F.normalize(q_gt, dim=-1, eps=epsilon)

    # Compute dot product
    dot_product = torch.abs(torch.sum(q_pred * q_gt, dim=-1))  # absolute to handle double cover
    # Clamp for numerical stability - acos domain is [0, 1] for abs(dot)
    dot_product = torch.clamp(dot_product, epsilon, 1.0 - epsilon)

    # Angular distance in radians
    loss = 2.0 * torch.acos(dot_product)
    return loss.mean()

# Full Pose Loss: Translation + Rotation
class PoseLoss(nn.Module):
    def __init__(self, init_log_sigma_t=0.0, init_log_sigma_r=0.0):
        super(PoseLoss, self).__init__()
        self.log_sigma_t = nn.Parameter(torch.tensor(init_log_sigma_t))
        self.log_sigma_r = nn.Parameter(torch.tensor(init_log_sigma_r))

    def forward(self, pred, target):
        """
        pred, target: shape (batch_size, 6)
        [x, y, z, roll, pitch, yaw]  — in meters & radians
        """
        # Split translation and rotation
        t_pred, r_pred = pred[:, :3], pred[:, 3:]
        t_gt, r_gt = target[:, :3], target[:, 3:]

        # Translation Loss (MSE)
        trans_loss = F.mse_loss(t_pred, t_gt)

        # Convert Euler to Quaternion
        q_pred = euler_to_quaternion(r_pred[:, 0], r_pred[:, 1], r_pred[:, 2])
        q_gt = euler_to_quaternion(r_gt[:, 0], r_gt[:, 1], r_gt[:, 2])

        # Rotation Geodesic Loss
        rot_loss = quaternion_geodesic_loss(q_pred, q_gt)

        # Combine
        total_loss = (torch.exp(-self.log_sigma_t) * trans_loss +
                      torch.exp(-self.log_sigma_r) * rot_loss +
                      (self.log_sigma_t + self.log_sigma_r))
        return total_loss
