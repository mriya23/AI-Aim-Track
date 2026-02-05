from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class Candidate:
    abs_x: int
    abs_y: int
    rel_x: int
    rel_y: int
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    crosshair_dist: float
    area: float


@dataclass
class LockState:
    x: Optional[int] = None
    y: Optional[int] = None
    last_seen_time: float = 0.0
    last_seen_x: Optional[int] = None
    last_seen_y: Optional[int] = None
    last_area: float = 0.0
    last_update_time: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    lost_since: float = 0.0
    lost_frames: int = 0
    # Advanced movement tracking
    accel_x: float = 0.0
    accel_y: float = 0.0
    prev_vel_x: float = 0.0
    prev_vel_y: float = 0.0
    movement_history: list = None
    predicted_x: Optional[int] = None
    predicted_y: Optional[int] = None
    movement_pattern: str = "unknown"  # "stationary", "linear", "strafing", "zigzag"
    confidence: float = 1.0


def compute_dynamic_radius(state: LockState, base_radius: float, ping_ms: float, ping_scale: float) -> float:
    speed = math.hypot(state.vel_x, state.vel_y)
    # Increase radius for fast-moving targets and during prediction
    speed_bonus = speed * (ping_ms / 1000.0) * ping_scale
    # Additional bonus for high acceleration targets (strafing)
    accel = math.hypot(state.accel_x, state.accel_y)
    accel_bonus = accel * 2.0
    return base_radius + speed_bonus + accel_bonus

def analyze_movement_pattern(state: LockState, history_size: int = 5) -> str:
    """Analyze target movement pattern to improve prediction accuracy"""
    if state.movement_history is None or len(state.movement_history) < 3:
        return "unknown"
    
    # Calculate velocity variance to detect strafing patterns
    recent_velocities = state.movement_history[-history_size:]
    vx_values = [v[0] for v in recent_velocities]
    vy_values = [v[1] for v in recent_velocities]
    
    # Detect strafing (rapid direction changes)
    vx_changes = sum(1 for i in range(1, len(vx_values)) if vx_values[i] * vx_values[i-1] < 0)
    vy_changes = sum(1 for i in range(1, len(vy_values)) if vy_values[i] * vy_values[i-1] < 0)
    
    if vx_changes >= 2 or vy_changes >= 2:
        return "strafing"
    
    # Detect zigzag (alternating X and Y direction changes)
    if len(recent_velocities) >= 4:
        recent_accelerations = []
        for i in range(1, len(recent_velocities)):
            ax = recent_velocities[i][0] - recent_velocities[i-1][0]
            ay = recent_velocities[i][1] - recent_velocities[i-1][1]
            recent_accelerations.append((ax, ay))
        
        # Check for zigzag pattern
        accel_sign_changes = 0
        for i in range(1, len(recent_accelerations)):
            if (recent_accelerations[i][0] * recent_accelerations[i-1][0] < 0 or 
                recent_accelerations[i][1] * recent_accelerations[i-1][1] < 0):
                accel_sign_changes += 1
        
        if accel_sign_changes >= 2:
            return "zigzag"
    
    # Check if relatively stationary
    avg_speed = sum(math.hypot(v[0], v[1]) for v in recent_velocities) / len(recent_velocities)
    if avg_speed < 50:  # Less than 50 pixels/second
        return "stationary"
    
    # Linear movement (consistent direction)
    direction_variance = sum(abs(vx_values[i] - vx_values[i-1]) for i in range(1, len(vx_values)))
    if direction_variance < 100:  # Low variance in X velocity
        return "linear"
    
    return "unknown"

def predict_future_position(state: LockState, prediction_time: float = 0.1) -> Tuple[float, float]:
    """Predict target position based on movement pattern and physics"""
    if state.movement_pattern == "unknown" or state.vel_x == 0 and state.vel_y == 0:
        return float(state.x or 0), float(state.y or 0)
    
    base_x = float(state.x or 0)
    base_y = float(state.y or 0)
    
    if state.movement_pattern == "stationary":
        return base_x, base_y
    
    elif state.movement_pattern == "linear":
        # Simple linear prediction with velocity
        pred_x = base_x + state.vel_x * prediction_time
        pred_y = base_y + state.vel_y * prediction_time
        return pred_x, pred_y
    
    elif state.movement_pattern == "strafing":
        # Enhanced prediction for strafing - account for acceleration
        pred_x = base_x + state.vel_x * prediction_time + 0.5 * state.accel_x * prediction_time ** 2
        pred_y = base_y + state.vel_y * prediction_time + 0.5 * state.accel_y * prediction_time ** 2
        return pred_x, pred_y
    
    elif state.movement_pattern == "zigzag":
        # Conservative prediction for zigzag - reduce prediction time
        short_prediction = prediction_time * 0.5
        pred_x = base_x + state.vel_x * short_prediction
        pred_y = base_y + state.vel_y * short_prediction
        return pred_x, pred_y
    
    else:
        # Default prediction
        return base_x + state.vel_x * prediction_time, base_y + state.vel_y * prediction_time


def update_lock_state(state: LockState, candidate: Optional[Candidate], now: float) -> LockState:
    if candidate is None:
        if state.lost_since == 0.0:
            state.lost_since = now
        state.lost_frames += 1
        
        # Predict position when target is lost
        if state.movement_pattern != "unknown" and state.vel_x != 0 and state.vel_y != 0:
            time_lost = now - state.lost_since
            if time_lost < 0.3:  # Only predict for short periods
                pred_x, pred_y = predict_future_position(state, time_lost)
                state.predicted_x = int(pred_x)
                state.predicted_y = int(pred_y)
        
        return state
    
    # Initialize movement history if needed
    if state.movement_history is None:
        state.movement_history = []
    
    # Calculate advanced movement metrics
    if state.last_seen_x is not None and state.last_seen_y is not None:
        dt = now - state.last_update_time
        if dt > 0:
            # Calculate velocity
            vx = (candidate.abs_x - state.last_seen_x) / dt
            vy = (candidate.abs_y - state.last_seen_y) / dt
            
            # Store previous velocity for acceleration calculation
            state.prev_vel_x = state.vel_x
            state.prev_vel_y = state.vel_y
            
            # Smooth velocity with adaptive factor based on movement consistency
            speed = math.hypot(vx, vy)
            prev_speed = math.hypot(state.vel_x, state.vel_y)
            
            # Use higher smoothing for consistent movement, lower for erratic
            if abs(speed - prev_speed) < 100:  # Consistent speed
                smoothing_factor = 0.8
            else:  # Erratic movement
                smoothing_factor = 0.4
            
            state.vel_x = smoothing_factor * state.vel_x + (1 - smoothing_factor) * vx
            state.vel_y = smoothing_factor * state.vel_y + (1 - smoothing_factor) * vy
            
            # Calculate acceleration
            state.accel_x = (state.vel_x - state.prev_vel_x) / dt if state.prev_vel_x != 0 else 0
            state.accel_y = (state.vel_y - state.prev_vel_y) / dt if state.prev_vel_y != 0 else 0
            
            # Add to movement history
            state.movement_history.append((state.vel_x, state.vel_y))
            if len(state.movement_history) > 10:  # Keep last 10 movements
                state.movement_history.pop(0)
            
            # Analyze movement pattern
            state.movement_pattern = analyze_movement_pattern(state)
            
            # Predict future position
            pred_x, pred_y = predict_future_position(state, 0.1)  # 100ms prediction
            state.predicted_x = int(pred_x)
            state.predicted_y = int(pred_y)
    
    state.x = candidate.abs_x
    state.y = candidate.abs_y
    state.last_seen_x = candidate.abs_x
    state.last_seen_y = candidate.abs_y
    state.last_seen_time = now
    state.last_update_time = now
    state.last_area = candidate.area
    state.lost_since = 0.0
    state.lost_frames = 0
    return state


def select_locked_candidate(
    candidates: Tuple[Candidate, ...],
    state: LockState,
    now: float,
    base_radius: float,
    reacquire_radius: float,
    grace_time: float,
    min_conf: float,
    min_area_ratio: float,
    ping_ms: float,
    ping_scale: float,
    sticky_multiplier: float = 1.5,
    teleport_threshold: float = 300.0,
    use_prediction: bool = True,
) -> Tuple[Optional[Candidate], bool]:
    if state.x is None or state.y is None:
        return None, False

    dynamic_radius = compute_dynamic_radius(state, base_radius, ping_ms, ping_scale)
    
    # Use predicted position if available and target is temporarily lost
    search_x = state.predicted_x if (state.predicted_x is not None and 
                                   state.lost_since > 0 and 
                                   now - state.lost_since < 0.15) else state.x
    search_y = state.predicted_y if (state.predicted_y is not None and 
                                   state.lost_since > 0 and 
                                   now - state.lost_since < 0.15) else state.y
    
    # Sticky lock: When already locked, use larger radius to maintain lock
    effective_radius = dynamic_radius * sticky_multiplier
    
    # Increase radius for strafing targets
    if state.movement_pattern in ["strafing", "zigzag"]:
        effective_radius *= 1.3

    def area_ratio_ok(c: Candidate) -> bool:
        if state.last_area <= 0:
            return True
        return (c.area / state.last_area) >= min_area_ratio
    
    def is_not_teleport(c: Candidate) -> bool:
        """Reject candidates that 'teleported' (moved too far in one frame)."""
        if state.last_seen_x is None or state.last_seen_y is None:
            return True
        dist = math.dist((c.abs_x, c.abs_y), (state.last_seen_x, state.last_seen_y))
        return dist < teleport_threshold

    best = None
    best_dist = float("inf")
    for c in candidates:
        if c.conf < min_conf:
            continue
        if not area_ratio_ok(c):
            continue
        if not is_not_teleport(c):
            continue
        dist = math.dist((c.abs_x, c.abs_y), (search_x, search_y))
        if dist <= effective_radius and dist < best_dist:
            best = c
            best_dist = dist

    if best is not None:
        return best, True

    if now - state.last_seen_time <= grace_time:
        fallback_radius = reacquire_radius + dynamic_radius
        best = None
        best_dist = float("inf")
        relaxed_conf = min_conf * 0.75
        relaxed_ratio = min_area_ratio * 0.5
        for c in candidates:
            if c.conf < relaxed_conf:
                continue
            if state.last_area > 0 and (c.area / state.last_area) < relaxed_ratio:
                continue
            dist = math.dist((c.abs_x, c.abs_y), (search_x, search_y))
            if dist <= fallback_radius and dist < best_dist:
                best = c
                best_dist = dist
        if best is not None:
            return best, False

    return None, False
