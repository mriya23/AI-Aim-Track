import ctypes
import cv2
import json
import math
import mss
import numpy as np
import os
import sys
import time
import torch
import uuid
import win32api
import winsound
import random
import gc
import psutil
from typing import Optional, Tuple, Any
from termcolor import colored
from lib.presets import get_preset_manager, get_active_preset, PresetConfig
from lib.lock_logic import Candidate, LockState, select_locked_candidate, update_lock_state

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt
    TRS_AVAILABLE = True
except ImportError:
    TRS_AVAILABLE = False

# HIGH PRIORITY PROCESS BOOST
try:
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)
    print(colored("[+] Process Priority set to HIGH", "green")) 
except:
    pass

# Try to import interception driver for anti-cheat bypass
try:
    import interception
    # Test if driver is actually installed by checking for the requires_driver decorator
    # If driver is not installed, any function call will raise DriverNotFoundError
    interception.auto_capture_devices(keyboard=False, mouse=True)
    INTERCEPTION_AVAILABLE = True
    print(colored("[+] Interception Driver LOADED - Anti-cheat bypass enabled", "green"))
except Exception:
    INTERCEPTION_AVAILABLE = False
    print(colored("[!] Interception Driver not available - using SendInput", "yellow"))


PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

# ========== SENDINPUT MOUSE MOVEMENT HELPER ==========
def move_mouse_relative(dx: int, dy: int):
    """
    Move mouse relative to current position using either Interception or SendInput.
    This is the unified function for all mouse movements.
    """
    if dx == 0 and dy == 0:
        return
        
    if INTERCEPTION_AVAILABLE:
        interception.move_relative(dx, dy)
    else:
        # Fallback to mouse_event (more compatible with games than SendInput)
        # MOUSEEVENTF_MOVE = 0x0001
        ctypes.windll.user32.mouse_event(0x0001, dx, dy, 0, 0)
        
        # Debug print (remove after testing)
        # print(f"[DEBUG] Moving mouse: dx={dx}, dy={dy}")
# ======================================================

class TRTModel:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem, 'name': name})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'name': name})

    def __call__(self, img):
        # Image preprocessing
        # RESIZE to 320x320
        img_resized = cv2.resize(img, (320, 320))
        img_resized = img_resized.astype(np.float32) / 255.0
        img_resized = np.transpose(img_resized, (2, 0, 1))
        img_resized = np.expand_dims(img_resized, axis=0)
        img_resized = np.ascontiguousarray(img_resized)
        
        # Copy to input buffer
        self.inputs[0]['host'][:] = img_resized.flatten()
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Bindings for V3
        for i in range(len(self.bindings)):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
            
        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Result back to host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        
        return [out['host'] for out in self.outputs]

    def half(self): pass # Compatibility


class PIDController:
    """
    Professional PID Controller for smooth aim movement.
    - P (Proportional): Reacts to current error (farther = faster)
    - I (Integral): Eliminates steady-state error (ensures aim reaches target)
    - D (Derivative): Prevents overshoot (smooth braking)
    """
    def __init__(self, kp=0.7, ki=0.05, kd=0.15):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_x = 0
        self.integral_y = 0
        self.prev_error_x = 0
        self.prev_error_y = 0
        self.prev_time = time.perf_counter()
        # Anti-windup limits
        self.integral_limit = 50
        
    def update(self, error_x, error_y):
        """Calculate PID output for X and Y axes"""
        current_time = time.perf_counter()
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 0.016  # Fallback to ~60fps
        
        # Proportional term
        p_x = self.kp * error_x
        p_y = self.kp * error_y
        
        # Integral term (with anti-windup)
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        self.integral_x = max(-self.integral_limit, min(self.integral_limit, self.integral_x))
        self.integral_y = max(-self.integral_limit, min(self.integral_limit, self.integral_y))
        i_x = self.ki * self.integral_x
        i_y = self.ki * self.integral_y
        
        # Derivative term
        d_x = self.kd * (error_x - self.prev_error_x) / dt
        d_y = self.kd * (error_y - self.prev_error_y) / dt
        
        # Store for next iteration
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_time = current_time
        
        # Combined output
        output_x = p_x + i_x + d_x
        output_y = p_y + i_y + d_y
        
        return output_x, output_y
    
    def reset(self):
        """Reset PID state when target is lost"""
        self.integral_x = 0
        self.integral_y = 0
        self.prev_error_x = 0
        self.prev_error_y = 0



class Aimbot:
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    screen = mss.mss()
    pixel_increment = 1 #controls how many pixels the mouse moves for each relative movement (lower = smoother/more human)
    with open("lib/config/config.json") as f:
        sens_config = json.load(f)
    config_reload_time: float = 0.0
    
    last_config_mtime: float = 0.0
    
    @staticmethod
    def reload_config():
        """Hot-reload config from file only if modified"""
        config_path = "lib/config/config.json"
        try:
            mtime = os.path.getmtime(config_path)
            if mtime > Aimbot.last_config_mtime:
                with open(config_path) as f:
                    Aimbot.sens_config = json.load(f)
                Aimbot.last_config_mtime = mtime
                # print("[INFO] Config reloaded") 
        except:
            pass
    aimbot_status = colored("ENABLED", 'green')
    status_display_time: float = 0.0  # Time when status was last changed
    status_display_duration = 2  # Show status overlay for 2 seconds
    last_move_info = ""  # Debug info for last move attempt
    
    # Target locking variables - prevent switching between multiple enemies
    locked_target_x: Optional[int] = None
    locked_target_y: Optional[int] = None
    lock_time: float = 0.0
    lock_duration = 0.5  # Stay locked to same target for 500ms
    lock_state = LockState()
    
    # Triggerbot timing
    last_shot_time: float = 0.0
    
    # Aim prediction variables (unused but kept for compatibility)
    # Aim prediction variables
    prev_target_x = 960
    prev_target_y = 540
    prev_time = time.time()
    velocity_x = 0
    velocity_y = 0
    smooth_vx = 0
    smooth_vy = 0
    
    # Micro-movement filtering
    micro_movement_threshold = 2.0  # Minimum movement in pixels to respond to
    micro_movement_counter = 0
    last_significant_move = time.time()
    
    # Advanced interpolation for lost targets
    interpolation_enabled = True
    max_interpolation_time = 0.3  # Maximum time to interpolate (seconds)
    
    # Smart RCS tracking variables
    spray_start_time: float = 0.0
    is_spraying: bool = False
    last_rcs_strength: float = 0.0
    shots_fired: int = 0
    
    # PID Controller instance for smooth aim
    pid_controller = PIDController(
        kp=sens_config.get("pid_kp", 0.7),
        ki=sens_config.get("pid_ki", 0.05),
        kd=sens_config.get("pid_kd", 0.15)
    )

    def __init__(self, box_constant = 416, collect_data = False, mouse_delay = 0.0005, debug = False):
        # ROI Size from config (smaller = faster FPS)
        config_roi = Aimbot.sens_config.get("roi_size", 320)
        self.box_constant = config_roi  # Use config value for detection box size

        engine_path = 'lib/best.engine'
        if TRS_AVAILABLE and os.path.exists(engine_path):
            print(colored(f"[+] TensorRT Engine FOUND ({engine_path})", "green"))
            print(colored("[INFO] Initializing TensorRT Inference Engine...", "cyan"))
            self.model = TRTModel(engine_path)
            self.using_trt = True
        else:
            print(colored("[!] TensorRT Engine NOT FOUND or Library Missing", "yellow"))
            print("[INFO] Falling back to PyTorch Inference...")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='lib/best.pt', force_reload=False)
            self.using_trt = False
            
            if torch.cuda.is_available():
                print(colored("CUDA ACCELERATION [ENABLED]", "green"))
                self.model.conf = 0.40
                self.model.iou = 0.45
                self.model.half()
                print(colored("[+] FP16 Mode ENABLED - Faster inference", "green"))
            else:
                print(colored("[!] CUDA ACCELERATION IS UNAVAILABLE", "red"))
        self.collect_data = collect_data
        self.mouse_delay = mouse_delay
        self.debug = debug

        print("\n[INFO] PRESS 'F1' TO TOGGLE AIMBOT\n[INFO] PRESS 'F2' TO QUIT")

    def update_status_aimbot():
        if Aimbot.aimbot_status == colored("ENABLED", 'green'):
            Aimbot.aimbot_status = colored("DISABLED", 'red')
            # Low pitch beep for disabled
            winsound.Beep(400, 150)
        else:
            Aimbot.aimbot_status = colored("ENABLED", 'green')
            # High pitch beep for enabled
            winsound.Beep(800, 150)
        Aimbot.status_display_time = time.time()  # Update display time
        sys.stdout.write("\033[K")
        print(f"[!] AIMBOT IS [{Aimbot.aimbot_status}]", end = "\r")

    @staticmethod
    def left_click():
        ctypes.windll.user32.mouse_event(0x0002) #left mouse down
        Aimbot._sleep(0.0001)
        ctypes.windll.user32.mouse_event(0x0004) #left mouse up

    @staticmethod
    def _sleep(duration: float, get_now = time.perf_counter):
        if duration == 0: return
        now = get_now()
        end = now + duration
        while now < end:
            now = get_now()

    def is_aimbot_enabled():
        return True if Aimbot.aimbot_status == colored("ENABLED", 'green') else False

    @staticmethod
    def is_targeted():
        # Use GetAsyncKeyState instead of GetKeyState for better game compatibility
        # 0x02 = VK_RBUTTON (Right Mouse Button)
        # Check if the high-order bit is set (key is pressed)
        state = ctypes.windll.user32.GetAsyncKeyState(0x02)
        return (state & 0x8000) != 0

    @staticmethod
    def is_target_locked(x: int, y: int) -> bool:
        #plus/minus 5 pixel threshold
        threshold = 5
        return 960 - threshold <= x <= 960 + threshold and 540 - threshold <= y <= 540 + threshold

    # Momentum history removed - back to simple immediate prediction

    # Momentum history for noise filtering
    smooth_mom_x = 0
    smooth_mom_y = 0

    def triggerbot(self, x, y):
        """
        Checks if the crosshair (center screen) is inside a detected target's bounding box.
        If yes, and Triggerbot is enabled, clicks the mouse.
        """
        if not Aimbot.sens_config.get("triggerbot_enabled", False):
            return

        # Check if crosshair is reasonably close to the target center (e.g. within 30px)
        # We use the calculated target x,y which is the head position
        # If the crosshair (960, 540) is close enough to the target (x, y), fire.
        
        # Calculate distance from crosshair to target
        dist = math.dist((x, y), (960, 540))
        
        # Trigger radius (pixels) - configurable? let's hardcode a reasonable value for now or use config
        trigger_radius = Aimbot.sens_config.get("trigger_radius", 15)

        if dist <= trigger_radius:
            # Check for delay
            current_time = time.time()
            last_shot = getattr(self, "last_shot_time", 0)
            delay = Aimbot.sens_config.get("trigger_delay", 0.1)
            
            if current_time - last_shot > delay:
                # Fire!
                if INTERCEPTION_AVAILABLE:
                    interception.left_click(clicks=1, interval=0.05)
                self.last_shot_time = current_time

    def move_crosshair(self, x, y):
        # 1. Triggerbot check always runs
        self.triggerbot(x, y)
        
        # 2. SMART RCS Logic (Lock-Aware, Adaptive)
        # Runs INDEPENDENTLY of Aimbot Lock to ensure recoil is always controlled
        rcs_y_offset = 0
        current_time = time.time()
        is_left_click_held = win32api.GetKeyState(0x01) < 0
        
        if is_left_click_held and Aimbot.sens_config.get("rcs_enabled", False):
            # Track spray state
            if not Aimbot.is_spraying:
                # First shot - start spray tracking
                Aimbot.spray_start_time = current_time
                Aimbot.is_spraying = True
                Aimbot.shots_fired = 1
            else:
                Aimbot.shots_fired += 1
            
            # Get Smart RCS parameters
            base_strength = Aimbot.sens_config.get("rcs_base_strength", 8)
            max_strength = Aimbot.sens_config.get("rcs_max_strength", 25)
            ramp_rate = Aimbot.sens_config.get("rcs_ramp_rate", 1.5)
            activation_delay = Aimbot.sens_config.get("rcs_activation_delay", 0.05)
            
            spray_duration = current_time - Aimbot.spray_start_time
            
            # Check activation delay (allow single shots without RCS)
            if spray_duration > activation_delay:
                # Calculate ramp factor (increases with spray duration)
                ramp_factor = 1.0 + (spray_duration * ramp_rate)
                effective_rcs = base_strength * ramp_factor
                
                # Cap at max strength
                effective_rcs = min(effective_rcs, max_strength)
                
                # Lock-Aware Scaling:
                # If aimbot is locked on target, reduce RCS (aimbot handles vertical correction)
                is_locked = Aimbot.is_targeted()
                if is_locked:
                    effective_rcs *= 0.6  # 40% reduction when locked
                
                rcs_y_offset = effective_rcs
                Aimbot.last_rcs_strength = effective_rcs
                
                # [DEBUG] Print Smart RCS values
                print(f"SPRAY:{spray_duration:.2f}s | SHOTS:{Aimbot.shots_fired} | RCS:{effective_rcs:.1f} | LOCKED:{is_locked}")
            
            # If NO target locked, Apply RCS directly here (Standalone Mode)
            if not Aimbot.is_targeted() and rcs_y_offset > 0:
                move_mouse_relative(0, int(rcs_y_offset))
        else:
            # Reset spray tracking when not shooting
            if Aimbot.is_spraying:
                Aimbot.is_spraying = False
                Aimbot.shots_fired = 0

        # 3. AIMBOT Logic (Only runs if targeted)
        if not Aimbot.is_targeted():
            Aimbot.last_move_info = "Waiting for ADS..."
            # Reset history
            Aimbot.prev_target_x = x
            Aimbot.prev_target_y = y
            Aimbot.smooth_mom_x = 0
            Aimbot.smooth_mom_y = 0
            return

        scale = Aimbot.sens_config.get("scale", 65.0) # Ensure default is 65.0
        sensitivity = Aimbot.sens_config.get("smoothing", 0.06)
        x_speed = Aimbot.sens_config.get("x_speed", 1.0)
        y_speed = Aimbot.sens_config.get("y_speed", 1.0)
        smoothness = Aimbot.sens_config.get("smoothness", 0.5)
        
        # === ADVANCED PREDICTIVE AIMING LOGIC ===
        curr_time = time.time()
        dt = curr_time - Aimbot.prev_time
        
        target_x = x
        target_y = y
        
        if dt > 0:
            # Calculate raw velocity (pixels per second)
            vx = (x - Aimbot.prev_target_x) / dt
            vy = (y - Aimbot.prev_target_y) / dt
            
            # Get velocity smoothing from config
            smoothing_factor = Aimbot.sens_config.get("velocity_smoothing", 0.6)
            
            # Enhanced velocity smoothing with movement consistency detection
            speed = math.hypot(vx, vy)
            prev_speed = math.hypot(Aimbot.smooth_vx, Aimbot.smooth_vy)
            
            # Adaptive smoothing: higher for consistent movement, lower for erratic
            if abs(speed - prev_speed) < 100:  # Consistent speed
                adaptive_smoothing = min(smoothing_factor + 0.2, 0.9)
            else:  # Erratic movement (strafing, zigzag)
                adaptive_smoothing = max(smoothing_factor - 0.3, 0.3)
            
            # Smooth velocity (Adaptive Exponential Moving Average)
            Aimbot.smooth_vx = adaptive_smoothing * Aimbot.smooth_vx + (1 - adaptive_smoothing) * vx
            Aimbot.smooth_vy = adaptive_smoothing * Aimbot.smooth_vy + (1 - adaptive_smoothing) * vy
            
            # === ENHANCED ADAPTIVE PREDICTION ===
            # Enhanced prediction with movement pattern awareness
            speed = math.hypot(Aimbot.smooth_vx, Aimbot.smooth_vy)
            
            # Detect movement pattern for better prediction
            if Aimbot.lock_state.movement_pattern == "strafing":
                # Strafing targets need more aggressive prediction
                base_prediction = Aimbot.sens_config.get("strafe_prediction_factor", 0.25)
                pred_factor = base_prediction + (speed / 1000) * 0.1
            elif Aimbot.lock_state.movement_pattern == "zigzag":
                # Zigzag needs conservative prediction
                base_prediction = Aimbot.sens_config.get("zigzag_prediction_factor", 0.08)
                pred_factor = base_prediction
            elif speed > 800:  # Very fast (sprint + strafe)
                pred_factor = Aimbot.sens_config.get("prediction_max", 0.25)
            elif speed > 400:  # Fast (running)
                pred_min = Aimbot.sens_config.get("prediction_min", 0.05)
                pred_max = Aimbot.sens_config.get("prediction_max", 0.25)
                pred_factor = pred_min + (pred_max - pred_min) * ((speed - 400) / 400)
            elif speed > 150:  # Medium (walking)
                pred_min = Aimbot.sens_config.get("prediction_min", 0.05)
                pred_factor = pred_min + (Aimbot.sens_config.get("prediction_max", 0.25) - pred_min) * 0.4 * ((speed - 150) / 250)
            else:  # Slow/stationary
                pred_factor = Aimbot.sens_config.get("prediction_min", 0.05) * 0.3
            
            # Apply enhanced prediction with acceleration compensation
            accel_compensation = 0.5  # Compensate for acceleration
            target_x = x + (Aimbot.smooth_vx * pred_factor) + (Aimbot.lock_state.accel_x * pred_factor * accel_compensation)
            target_y = y + (Aimbot.smooth_vy * pred_factor) + (Aimbot.lock_state.accel_y * pred_factor * accel_compensation)
            
            # Micro-movement filtering
            movement_distance = math.hypot(target_x - x, target_y - y)
            if movement_distance < Aimbot.micro_movement_threshold:
                Aimbot.micro_movement_counter += 1
                if Aimbot.micro_movement_counter < 5:  # Allow a few micro-movements
                    target_x = x
                    target_y = y
                else:
                    # After threshold, allow micro-movements but with reduced sensitivity
                    reduction_factor = 0.7
                    target_x = x + (target_x - x) * reduction_factor
                    target_y = y + (target_y - y) * reduction_factor
            else:
                Aimbot.micro_movement_counter = 0
                Aimbot.last_significant_move = curr_time
        
        Aimbot.prev_time = curr_time
        Aimbot.prev_target_x = x
        Aimbot.prev_target_y = y
        
        # Use predicted coordinates for aiming
        x = target_x
        y = target_y

        # === RAGE / BRUTAL MODE Logic (If smoothness < 0.2) ===
        if smoothness < 0.2:
            # DIRECT RAW OFFSET (No Filtering, No Prediction, No Smoothing)
            raw_offset_x = (x - 960)
            raw_offset_y = (y - 540)
            
            # Apply only sensitivity/speed scaling
            diff_x = int(raw_offset_x * scale / 10 * sensitivity * x_speed)
            diff_y = int(raw_offset_y * scale / 10 * sensitivity * y_speed)
            
            move_mouse_relative(diff_x, diff_y)
            
            Aimbot.last_move_info = f"RAGE PULL: {diff_x}, {diff_y}"
            return
        # ===============================================

        # === ENHANCED LEGIT MODE WITH ADAPTIVE PID CONTROLLER ===
        # Professional-grade aim smoothing using adaptive P.I.D. algorithm
        
        pid_enabled = Aimbot.sens_config.get("pid_enabled", True)
        
        if pid_enabled:
            # Calculate error (distance from crosshair to target)
            error_x = x - 960  # Target X - Screen Center X
            error_y = y - 540  # Target Y - Screen Center Y
            
            # Adaptive PID tuning based on movement pattern and speed
            target_speed = math.hypot(Aimbot.smooth_vx, Aimbot.smooth_vy)
            
            if Aimbot.lock_state.movement_pattern == "strafing":
                # Strafing targets need aggressive response
                kp = Aimbot.sens_config.get("pid_kp_strafe", 1.2)
                ki = Aimbot.sens_config.get("pid_ki_strafe", 0.12)
                kd = Aimbot.sens_config.get("pid_kd_strafe", 0.25)
            elif Aimbot.lock_state.movement_pattern == "zigzag":
                # Zigzag needs balanced response
                kp = Aimbot.sens_config.get("pid_kp_zigzag", 0.8)
                ki = Aimbot.sens_config.get("pid_ki_zigzag", 0.06)
                kd = Aimbot.sens_config.get("pid_kd_zigzag", 0.15)
            elif target_speed > 600:  # Very fast targets
                kp = Aimbot.sens_config.get("pid_kp_fast", 1.0)
                ki = Aimbot.sens_config.get("pid_ki_fast", 0.10)
                kd = Aimbot.sens_config.get("pid_kd_fast", 0.20)
            else:  # Normal targets
                kp = Aimbot.sens_config.get("pid_kp", 0.95)
                ki = Aimbot.sens_config.get("pid_ki", 0.08)
                kd = Aimbot.sens_config.get("pid_kd", 0.20)
            
            # Update PID parameters
            Aimbot.pid_controller.kp = kp
            Aimbot.pid_controller.ki = ki
            Aimbot.pid_controller.kd = kd
            
            # Dynamic deadzone based on target speed
            if target_speed > 400:
                deadzone = 1.5  # Tighter deadzone for fast targets
            elif target_speed > 200:
                deadzone = 2.0  # Medium deadzone
            else:
                deadzone = 2.5  # Larger deadzone for slow targets
            
            # Enhanced deadzone with micro-movement tolerance
            if abs(error_x) < deadzone and abs(error_y) < deadzone:
                # Only reset if we're consistently in deadzone
                if time.time() - Aimbot.last_significant_move > 0.1:
                    Aimbot.pid_controller.reset() # Reset I-term so it doesn't wind up
                return
            
            # Get PID output (how much to move)
            pid_out_x, pid_out_y = Aimbot.pid_controller.update(error_x, error_y)
            
            # Enhanced movement scaling based on error magnitude
            error_magnitude = math.hypot(error_x, error_y)
            if error_magnitude > 100:  # Large error - fast movement
                speed_multiplier = 1.2
            elif error_magnitude > 50:  # Medium error
                speed_multiplier = 1.0
            else:  # Small error - fine adjustment
                speed_multiplier = 0.8
            
            # Apply speed scaling with enhanced multiplier
            diff_x = pid_out_x * scale / 100 * x_speed * speed_multiplier
            diff_y = (pid_out_y * scale / 100 * y_speed * speed_multiplier) + rcs_y_offset
            
            # Dynamic maximum movement based on target speed
            if target_speed > 500:
                max_move = 80  # Higher cap for very fast targets
            elif target_speed > 300:
                max_move = 70
            else:
                max_move = 60
            
            diff_x = max(-max_move, min(max_move, diff_x))
            diff_y = max(-max_move, min(max_move, diff_y))
            
            # Enhanced anti-jitter filtering with micro-movement tolerance
            if abs(diff_x) < 0.5 and abs(diff_y) < 0.5:
                return
            elif abs(diff_x) < 1.0 and abs(diff_y) < 1.0 and target_speed < 100:
                # Allow tiny movements only if target is moving slowly
                if Aimbot.micro_movement_counter > 3:
                    return
                Aimbot.micro_movement_counter += 1
            else:
                Aimbot.micro_movement_counter = 0

            diff_x = int(diff_x)
            diff_y = int(diff_y)
            
            if diff_x == 0 and diff_y == 0:
                return
            
            # Execute movement
            move_mouse_relative(diff_x, diff_y)
        else:
            # FALLBACK: CLASSIC SIMPLE SMOOTHING (STABLE)
            # 1. Calculate offset
            offset_x = x - 960
            offset_y = y - 540
            
            # 3. ABSOLUTE LOCK LOGIC (Jitter Fix)
            # Extremely dampened sensitivity
            # Divisor 200 = Heavy dampening to stop left-right overshoot
            diff_x = (offset_x * scale * sensitivity * x_speed) / 200
            diff_y = (offset_y * scale * sensitivity * y_speed) / 200
            
            # HARD CAP: Increased to 25px for faster flicks
            limit = 25
            diff_x = max(-limit, min(limit, diff_x))
            diff_y = max(-limit, min(limit, diff_y))

            # 4. DIRECT EXECUTION
            move_x = int(diff_x)
            move_y = int(diff_y)
            
            if move_x != 0 or move_y != 0:
                move_mouse_relative(move_x, move_y)


    #generator yields pixel tuples for relative movement
    def interpolate_coordinates_from_center(absolute_coordinates, scale):
        diff_x = (absolute_coordinates[0] - 960) * scale/Aimbot.pixel_increment
        diff_y = (absolute_coordinates[1] - 540) * scale/Aimbot.pixel_increment
        length = int(math.dist((0,0), (diff_x, diff_y)))
        if length == 0: return
        unit_x = (diff_x/length) * Aimbot.pixel_increment
        unit_y = (diff_y/length) * Aimbot.pixel_increment
        x = y = sum_x = sum_y = 0
        for k in range(0, length):
            sum_x += x
            sum_y += y
            x, y = round(unit_x * k - sum_x), round(unit_y * k - sum_y)
            yield x, y
            

    def start(self):
        print("[INFO] Beginning screen capture")
        Aimbot.update_status_aimbot()
        half_screen_width = ctypes.windll.user32.GetSystemMetrics(0)/2 #this should always be 960
        half_screen_height = ctypes.windll.user32.GetSystemMetrics(1)/2 #this should always be 540
        detection_box = {'left': int(half_screen_width - self.box_constant//2), #x1 coord (for top-left corner of the box)
                          'top': int(half_screen_height - self.box_constant//2), #y1 coord (for top-left corner of the box)
                          'width': int(self.box_constant),  #width of the box
                          'height': int(self.box_constant)} #height of the box
        if self.collect_data:
            collect_pause = 0

        while True:
            start_time = time.perf_counter()
            
            # Hot-reload config every 2 seconds
            if time.perf_counter() - Aimbot.config_reload_time > 2:
                Aimbot.reload_config()
                Aimbot.config_reload_time = time.perf_counter()
            
            frame = np.array(Aimbot.screen.grab(detection_box))
            # Convert BGRA to BGR and make contiguous for OpenCV
            frame = np.ascontiguousarray(frame[:, :, :3])
            
            if self.collect_data: orig_frame = np.copy((frame))
            
            # --- INFERENCE STEP ---
            if self.using_trt:
                # TensorRT Path
                raw_results = self.model(frame)
                # Custom YOLOv5 model: 6 columns = [x, y, w, h, conf, class]
                # (5 + 1 class, not 85 like COCO)
                num_cols = 6
                output = raw_results[0].reshape(-1, num_cols)
                # Use config value for confidence to avoid false positives (aim pulling down)
                conf_thres = Aimbot.sens_config.get("conf_thres", 0.45)
                valid = output[output[:, 4] > conf_thres]
                
                # SCALING: TensorRT uses 320x320, Screen ROI is self.box_constant (usually 400)
                scale_factor = self.box_constant / 320.0
                
                det_list = []
                for row in valid:
                    # [x_center, y_center, w, h, conf, class]
                    box = row[:4]
                    conf = row[4]
                    # Convert to xyxy and SCALE to ROI size
                    x1 = int((box[0] - box[2]/2) * scale_factor)
                    y1 = int((box[1] - box[3]/2) * scale_factor)
                    x2 = int((box[0] + box[2]/2) * scale_factor)
                    y2 = int((box[1] + box[3]/2) * scale_factor)
                    det_list.append([x1, y1, x2, y2, conf])
                results_xyxy = det_list
            else:
                # PyTorch Path
                results = self.model(frame)
                results_xyxy = []
                if len(results.xyxy[0]) > 0:
                    for *box, conf, cls in results.xyxy[0]:
                        results_xyxy.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]), conf.item()])

            # --- PROCESSING STEP ---
            if len(results_xyxy) != 0: #player detected
                least_crosshair_dist: float = float('inf')  # Use infinity as initial value
                closest_detection = None
                player_in_frame = False
                candidates = []
                for x1, y1, x2, y2, conf in results_xyxy: #iterate over each player detected
                    w = x2 - x1
                    h = y2 - y1
                    if w == 0 or h == 0: continue
                    
                    # --- SMART FILTER (Dynamic Logic) ---
                    # 1. Aspect Ratio Filter: Ignore flat boxes (shadows/ground)
                    aspect_ratio = h / w
                    if aspect_ratio < 0.6: 
                        continue # Target is too flat/wide -> Likely not a standing player
                    
                    # 2. Bottom Border Filter: Ignore detections touching bottom of ROI (hands/weapon)
                    if y2 > self.box_constant - 5:
                        continue
                    
                    # Calculate head position - use preset system
                    active_preset = get_active_preset()
                    aim_ratio = active_preset.aim_point_ratio
                    h_offset = active_preset.horizontal_offset
                    
                    # Apply horizontal offset
                    center_x = (x1 + x2) / 2
                    offset_x = w * h_offset
                    relative_head_X = int(center_x + offset_x)
                    relative_head_Y = int(y1 + h * aim_ratio)
                    own_player = x1 < 15 or (x1 < self.box_constant/5 and y2 > self.box_constant/1.2) 
                    
                    x1y1 = (x1, y1)
                    x2y2 = (x2, y2)

                    #calculate the distance between each detection and the crosshair
                    crosshair_dist = math.dist((relative_head_X, relative_head_Y), (self.box_constant/2, self.box_constant/2))
                    
                    if crosshair_dist < least_crosshair_dist: 
                        least_crosshair_dist = crosshair_dist

                    # FOV Filter
                    fov_radius_val = float(Aimbot.sens_config.get("fov_radius", 150))
                    if crosshair_dist > fov_radius_val:
                        continue 

                    if not own_player:
                        if crosshair_dist <= least_crosshair_dist:
                            least_crosshair_dist = crosshair_dist
                            closest_detection = {"x1y1": x1y1, "x2y2": x2y2, "relative_head_X": relative_head_X, "relative_head_Y": relative_head_Y, "conf": conf, "crosshair_dist": crosshair_dist}
                        
                        abs_head_x = relative_head_X + detection_box['left']
                        abs_head_y = relative_head_Y + detection_box['top']
                        candidates.append(Candidate(
                            abs_x=abs_head_x,
                            abs_y=abs_head_y,
                            rel_x=relative_head_X,
                            rel_y=relative_head_Y,
                            x1=int(x1),
                            y1=int(y1),
                            x2=int(x2),
                            y2=int(y2),
                            conf=float(conf),
                            crosshair_dist=crosshair_dist,
                            area=float(w * h),
                        ))
                        # Draw box - Color based on confidence (Visual Debug)
                        conf_val = float(conf) if not isinstance(conf, (list, tuple)) else 0.5
                        color = (0, 255, 0) if conf_val > 0.6 else (0, 165, 255) # Green = High Conf, Orange = Low
                        cv2.rectangle(frame, x1y1, x2y2, color, 2) 
                        cv2.putText(frame, f"{int(conf_val * 100)}%", x1y1, cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2) 
                    else:
                        own_player = False
                        if not player_in_frame:
                            player_in_frame = True

                if candidates:
                    current_time = time.time()

                    lock_grace_time = float(Aimbot.sens_config.get("lock_grace_time", Aimbot.lock_duration))
                    base_radius = float(Aimbot.sens_config.get("lock_base_radius", 100))
                    reacquire_radius = float(Aimbot.sens_config.get("lock_reacquire_radius", 180))
                    min_conf = float(Aimbot.sens_config.get("lock_min_conf", Aimbot.sens_config.get("conf_thres", 0.45)))
                    min_area_ratio = float(Aimbot.sens_config.get("lock_min_area_ratio", 0.4))
                    ping_ms = float(Aimbot.sens_config.get("ping_comp_ms", 60))
                    ping_scale = float(Aimbot.sens_config.get("ping_comp_scale", 0.6))
                    sticky_multiplier = float(Aimbot.sens_config.get("lock_sticky_multiplier", 1.5))
                    teleport_threshold = float(Aimbot.sens_config.get("target_teleport_threshold", 300))
                    use_prediction = bool(Aimbot.sens_config.get("use_prediction", True))

                    locked_candidate, _ = select_locked_candidate(
                        tuple(candidates),
                        Aimbot.lock_state,
                        current_time,
                        base_radius,
                        reacquire_radius,
                        lock_grace_time,
                        min_conf,
                        min_area_ratio,
                        ping_ms,
                        ping_scale,
                        sticky_multiplier,
                        teleport_threshold,
                        use_prediction,
                    )

                    selected = None
                    if locked_candidate is not None:
                        selected = locked_candidate
                        Aimbot.lock_time = current_time
                        Aimbot.locked_target_x = selected.abs_x
                        Aimbot.locked_target_y = selected.abs_y
                        update_lock_state(Aimbot.lock_state, selected, current_time)
                    else:
                        if Aimbot.lock_state.x is not None and (current_time - Aimbot.lock_state.last_seen_time) <= lock_grace_time:
                            update_lock_state(Aimbot.lock_state, None, current_time)
                        else:
                            selected = min(candidates, key=lambda c: c.crosshair_dist)
                            Aimbot.lock_time = current_time
                            Aimbot.locked_target_x = selected.abs_x
                            Aimbot.locked_target_y = selected.abs_y
                            update_lock_state(Aimbot.lock_state, selected, current_time)

                        if selected is None and (Aimbot.lock_state.x is None or (current_time - Aimbot.lock_state.last_seen_time) > lock_grace_time):
                            Aimbot.lock_state = LockState()
                            Aimbot.locked_target_x = None
                            Aimbot.locked_target_y = None

                    if selected is not None:
                        head_x = int(selected.rel_x)
                        head_y = int(selected.rel_y)
                        cv2.circle(frame, (head_x, head_y), 5, (115, 244, 113), -1) #draw circle on the head

                        cv2.line(frame, (head_x, head_y), (self.box_constant//2, self.box_constant//2), (244, 242, 113), 2)

                        absolute_head_X = selected.abs_x
                        absolute_head_Y = selected.abs_y

                        x1 = int(selected.x1)
                        y1 = int(selected.y1)
                        if Aimbot.is_target_locked(absolute_head_X, absolute_head_Y):
                            cv2.putText(frame, "LOCKED", (x1 + 40, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (115, 244, 113), 2) #draw the confidence labels on the bounding boxes
                        else:
                            cv2.putText(frame, "TARGETING", (x1 + 40, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (115, 113, 244), 2) #draw the confidence labels on the bounding boxes

                        if Aimbot.is_aimbot_enabled():
                            Aimbot.move_crosshair(self, absolute_head_X, absolute_head_Y)
                else:
                    current_time = time.time()
                    lock_grace_time = float(Aimbot.sens_config.get("lock_grace_time", Aimbot.lock_duration))
                    if Aimbot.lock_state.x is not None:
                        update_lock_state(Aimbot.lock_state, None, current_time)
                        if (current_time - Aimbot.lock_state.last_seen_time) > lock_grace_time:
                            Aimbot.lock_state = LockState()
                            Aimbot.locked_target_x = None
                            Aimbot.locked_target_y = None
            else:
                current_time = time.time()
                lock_grace_time = float(Aimbot.sens_config.get("lock_grace_time", Aimbot.lock_duration))
                if Aimbot.lock_state.x is not None:
                    update_lock_state(Aimbot.lock_state, None, current_time)
                    if (current_time - Aimbot.lock_state.last_seen_time) > lock_grace_time:
                        Aimbot.lock_state = LockState()
                        Aimbot.locked_target_x = None
                        Aimbot.locked_target_y = None

            if self.collect_data and time.perf_counter() - collect_pause > 1 and Aimbot.is_targeted() and Aimbot.is_aimbot_enabled() and not player_in_frame: #screenshots can only be taken every 1 second
                cv2.imwrite(f"lib/data/{str(uuid.uuid4())}.jpg", orig_frame)
                collect_pause = time.perf_counter()
            
            # ------------------ VISUALIZATION START ------------------
            # Only draw and show window if NOT in headless mode
            if not Aimbot.sens_config.get("headless", False):
                cv2.putText(frame, f"FPS: {int(1/(time.perf_counter() - start_time))}", (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (113, 116, 244), 2)
                
                # Show debug movement info
                if Aimbot.last_move_info:
                    cv2.putText(frame, Aimbot.last_move_info, (5, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)
                
                # Show aimbot status overlay for a few seconds after toggle
                if time.time() - Aimbot.status_display_time < Aimbot.status_display_duration:
                    if Aimbot.is_aimbot_enabled():
                        status_text = "AIMBOT: ON"
                        status_color = (0, 255, 0)  # Green
                    else:
                        status_text = "AIMBOT: OFF"
                        status_color = (0, 0, 255)  # Red
                    # Draw background rectangle for better visibility
                    cv2.rectangle(frame, (self.box_constant//2 - 100, 50), (self.box_constant//2 + 100, 90), (0, 0, 0), -1)
                    cv2.putText(frame, status_text, (self.box_constant//2 - 90, 80), cv2.FONT_HERSHEY_DUPLEX, 1, status_color, 2)
                
                # Draw FOV Circle (always visible)
                fov_radius = Aimbot.sens_config.get("fov_radius", 150)
                center = (self.box_constant // 2, self.box_constant // 2)
                cv2.circle(frame, center, fov_radius, (0, 255, 255), 2)  # Yellow circle
                
                # Resize preview window to be larger for better visibility
                # detection still happens on small ROI, this is just for display
                display_frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("AI AIMBOT", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('0'):
                    break
            else:
                # Headless Mode:
                # Just sleep briefly to prevent 100% CPU usage if needed, though pytorch inference usually takes time.
                pass
            # ------------------ VISUALIZATION END ------------------

    def clean_up():
        print("\n[INFO] F2 WAS PRESSED. QUITTING...")
        Aimbot.screen.close()
        os._exit(0)

if __name__ == "__main__": print("You are in the wrong directory and are running the wrong file; you must run lunar.py")
