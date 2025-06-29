import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (ZYX order) to quaternions.
    Args:
        roll, pitch, yaw: torch.Tensor, angles in radians
    Returns:
        torch.Tensor: quaternion [w, x, y, z]
    """
    # Handle batch dimensions properly
    if roll.dim() == 0:
        roll = roll.unsqueeze(0)
    if pitch.dim() == 0:
        pitch = pitch.unsqueeze(0)
    if yaw.dim() == 0:
        yaw = yaw.unsqueeze(0)
    
    # Half angles for quaternion conversion
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

def quaternion_geodesic_loss(q_pred, q_gt, epsilon=1e-6):
    """
    Compute geodesic loss between quaternions with numerical stability.
    
    Args:
        q_pred: predicted quaternions [batch_size, 4]
        q_gt: ground truth quaternions [batch_size, 4]
        epsilon: small value for numerical stability
    
    Returns:
        torch.Tensor: mean geodesic loss in radians
    """
    # Ensure quaternions are normalized
    q_pred = F.normalize(q_pred, dim=-1, eps=epsilon)
    q_gt = F.normalize(q_gt, dim=-1, eps=epsilon)
    
    # Compute dot product
    dot_product = torch.sum(q_pred * q_gt, dim=-1)
    
    # Handle quaternion double cover: use the closer representation
    # This is crucial for Mars exploration where rotations can be large
    dot_product = torch.abs(dot_product)
    
    # Clamp for numerical stability - acos domain is [0, 1] for abs(dot)
    dot_product = torch.clamp(dot_product, 0.0, 1.0 - epsilon)
    
    # Angular distance in radians
    angular_distance = 2.0 * torch.acos(dot_product)
    
    return angular_distance.mean()

def quaternion_chordal_loss(q_pred, q_gt, epsilon=1e-6):
    """
    Alternative chordal distance loss - more stable for small angles.
    Useful when rover movements are small and precise.
    """
    q_pred = F.normalize(q_pred, dim=-1, eps=epsilon)
    q_gt = F.normalize(q_gt, dim=-1, eps=epsilon)
    
    # Chordal distance: ||q1 - q2||² or ||q1 + q2||² (choose minimum)
    diff1 = torch.norm(q_pred - q_gt, dim=-1)
    diff2 = torch.norm(q_pred + q_gt, dim=-1)  # Handle double cover
    chordal_dist = torch.minimum(diff1, diff2)
    
    return chordal_dist.mean()

class MarsVisualOdometryLoss(nn.Module):
    """
    Specialized pose loss for Mars visual odometry with adaptive weighting
    and robust handling of challenging Martian conditions.
    """
    
    def __init__(self, 
                 trans_weight=1.0, 
                 rot_weight=1.0,
                 use_adaptive_weighting=True,
                 rotation_loss_type='geodesic',
                 scale_invariant_trans=False,
                 max_trans_error=10.0,  # meters - reasonable for Mars rover
                 max_rot_error=math.pi/2):  # radians - 90 degrees max
        super(MarsVisualOdometryLoss, self).__init__()
        
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
        self.use_adaptive_weighting = use_adaptive_weighting
        self.rotation_loss_type = rotation_loss_type
        self.scale_invariant_trans = scale_invariant_trans
        self.max_trans_error = max_trans_error
        self.max_rot_error = max_rot_error
        
        # For adaptive weighting - track running statistics
        self.register_buffer('trans_loss_ema', torch.tensor(1.0))
        self.register_buffer('rot_loss_ema', torch.tensor(1.0))
        self.ema_decay = 0.99
        
    def forward(self, pred, target):
        """
        Forward pass for pose loss computation.
        
        Args:
            pred: predicted poses [batch_size, 6] -> [x, y, z, roll, pitch, yaw]
            target: ground truth poses [batch_size, 6] -> [x, y, z, roll, pitch, yaw]
            
        Returns:
            dict: containing total loss and individual components
        """
        batch_size = pred.shape[0]
        
        # Split translation and rotation
        t_pred, r_pred = pred[:, :3], pred[:, 3:]
        t_gt, r_gt = target[:, :3], target[:, 3:]
        
        # === Translation Loss ===
        if self.scale_invariant_trans:
            # Scale-invariant translation loss - useful for varying distances
            trans_diff = t_pred - t_gt
            trans_loss = torch.log(1 + torch.norm(trans_diff, dim=-1)).mean()
        else:
            # Standard MSE translation loss
            trans_loss = F.mse_loss(t_pred, t_gt)
        
        # Clamp translation loss for extreme outliers
        trans_loss = torch.clamp(trans_loss, 0, self.max_trans_error**2)
        
        # === Rotation Loss ===
        # Convert Euler angles to quaternions
        q_pred = euler_to_quaternion(r_pred[:, 0], r_pred[:, 1], r_pred[:, 2])
        q_gt = euler_to_quaternion(r_gt[:, 0], r_gt[:, 1], r_gt[:, 2])
        
        if self.rotation_loss_type == 'geodesic':
            rot_loss = quaternion_geodesic_loss(q_pred, q_gt)
        elif self.rotation_loss_type == 'chordal':
            rot_loss = quaternion_chordal_loss(q_pred, q_gt)
        else:
            raise ValueError(f"Unknown rotation loss type: {self.rotation_loss_type}")
        
        # Clamp rotation loss for extreme outliers
        rot_loss = torch.clamp(rot_loss, 0, self.max_rot_error)
        
        # === Adaptive Weighting ===
        if self.use_adaptive_weighting and self.training:
            # Update exponential moving averages
            self.trans_loss_ema = self.ema_decay * self.trans_loss_ema + (1 - self.ema_decay) * trans_loss.detach()
            self.rot_loss_ema = self.ema_decay * self.rot_loss_ema + (1 - self.ema_decay) * rot_loss.detach()
            
            # Adaptive weights to balance translation and rotation losses
            trans_weight = self.trans_weight / (self.trans_loss_ema + 1e-8)
            rot_weight = self.rot_weight / (self.rot_loss_ema + 1e-8)
        else:
            trans_weight = self.trans_weight
            rot_weight = self.rot_weight
        
        # === Combine Losses ===
        total_loss = trans_weight * trans_loss + rot_weight * rot_loss
        
        # Return detailed loss information for monitoring
        return {
            'total_loss': total_loss,
            'translation_loss': trans_loss,
            'rotation_loss': rot_loss,
            'trans_weight': trans_weight,
            'rot_weight': rot_weight
        }

# Alternative: Simple robust version if you prefer minimal changes
class RobustPoseLoss(nn.Module):
    """Simplified robust version of the original code with key fixes."""
    
    def __init__(self, trans_weight=1.0, rot_weight=1.0):
        super(RobustPoseLoss, self).__init__()
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
    
    def forward(self, pred, target):
        """
        pred, target: shape (batch_size, 6)
        [x, y, z, roll, pitch, yaw] — in meters & radians
        """
        # Split translation and rotation
        t_pred, r_pred = pred[:, :3], pred[:, 3:]
        t_gt, r_gt = target[:, :3], target[:, 3:]
        
        # Translation Loss (MSE)
        trans_loss = F.mse_loss(t_pred, t_gt)
        
        # Convert Euler to Quaternion
        q_pred = euler_to_quaternion(r_pred[:, 0], r_pred[:, 1], r_pred[:, 2])
        q_gt = euler_to_quaternion(r_gt[:, 0], r_gt[:, 1], r_gt[:, 2])
        
        # Rotation Geodesic Loss (FIXED VERSION)
        rot_loss = quaternion_geodesic_loss(q_pred, q_gt)
        
        # Combine
        total_loss = self.trans_weight * trans_loss + self.rot_weight * rot_loss
        return total_loss

# Example usage for Mars rover visual odometry
def create_mars_vo_loss():
    """Factory function to create loss optimized for Mars conditions."""
    return MarsVisualOdometryLoss(
        trans_weight=1.0,
        rot_weight=10.0,  # Higher weight for rotation - critical for navigation
        use_adaptive_weighting=True,
        rotation_loss_type='geodesic',
        scale_invariant_trans=True,  # Handle varying scales in Martian terrain
        max_trans_error=5.0,  # Conservative for precise rover navigation
        max_rot_error=math.pi/4  # 45 degrees max error tolerance
    )










class UncertaintyWeightedPoseLoss(nn.Module):
    def __init__(self, init_log_sigma_t=0.0, init_log_sigma_r=0.0):
        super(UncertaintyWeightedPoseLoss, self).__init__()
        self.log_sigma_t = nn.Parameter(torch.tensor(init_log_sigma_t))
        self.log_sigma_r = nn.Parameter(torch.tensor(init_log_sigma_r))

    def forward(self, pred, target):
        t_pred, r_pred = pred[:, :3], pred[:, 3:]
        t_gt, r_gt = target[:, :3], target[:, 3:]

        # Translation MSE
        trans_loss = F.mse_loss(t_pred, t_gt)

        # Convert Euler to Quaternion
        q_pred = euler_to_quaternion(r_pred[:, 0], r_pred[:, 1], r_pred[:, 2])
        q_gt = euler_to_quaternion(r_gt[:, 0], r_gt[:, 1], r_gt[:, 2])

        # Quaternion Geodesic loss
        rot_loss = quaternion_geodesic_loss(q_pred, q_gt)

        # Compute total loss with learned uncertainty
        total_loss = (torch.exp(-self.log_sigma_t) * trans_loss +
                      torch.exp(-self.log_sigma_r) * rot_loss +
                      (self.log_sigma_t + self.log_sigma_r))

        return total_loss
