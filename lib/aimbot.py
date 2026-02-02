import ctypes
import cv2
import json
import math
import dxcam
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

from termcolor import colored

# Try to import interception driver for anti-cheat bypass
try:
    import interception
    INTERCEPTION_AVAILABLE = True
    print(colored("[+] Interception Driver LOADED - Anti-cheat bypass enabled", "green"))
except ImportError:
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
    pixel_increment = 1 #controls how many pixels the mouse moves for each relative movement (lower = smoother/more human)
    with open("lib/config/config.json") as f:
        sens_config = json.load(f)
    config_reload_time = 0
    
    last_config_mtime = 0
    
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

    @staticmethod
    def send_input_move(x, y):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        # 0x0001 = MOUSEEVENTF_MOVE
        ii_.mi = MouseInput(int(x), int(y), 0, 0x0001, 0, ctypes.pointer(extra))
        command = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

    aimbot_status = colored("ENABLED", 'green')
    status_display_time = 0  # Time when status was last changed
    status_display_duration = 2  # Show status overlay for 2 seconds
    last_move_info = ""  # Debug info for last move attempt
    
    # Target locking variables - prevent switching between multiple enemies
    locked_target_x = None
    locked_target_y = None
    lock_time = 0
    lock_duration = 0.5  # Stay locked to same target for 500ms
    
    # Aim prediction variables (unused but kept for compatibility)
    prev_target_x = 960
    prev_target_y = 540
    prev_time = time.time()
    velocity_x = 0
    velocity_y = 0
    
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

    @staticmethod
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
        Aimbot.sleep(0.0001)
        ctypes.windll.user32.mouse_event(0x0004) #left mouse up

    @staticmethod
    def sleep(duration, get_now = time.perf_counter):
        if duration == 0: return
        now = get_now()
        end = now + duration
        while now < end:
            now = get_now()

    def is_aimbot_enabled():
        return True if Aimbot.aimbot_status == colored("ENABLED", 'green') else False

    def is_targeted():
        return True if win32api.GetKeyState(0x02) in (-127, -128) else False

    def is_target_locked(x, y):
        #plus/minus 5 pixel threshold
        threshold = 5
        return True if 960 - threshold <= x <= 960 + threshold and 540 - threshold <= y <= 540 + threshold else False

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
                else:
                    Aimbot.left_click()
                self.last_shot_time = current_time

    def rcs(self):
        """
        Recoil Control System: Pulls the mouse down when Left Click is held.
        """
        if not Aimbot.sens_config.get("rcs_enabled", False):
            return

        # Check if Left Mouse Button is held (VK_LBUTTON = 0x01)
        if win32api.GetKeyState(0x01) < 0:
            strength_y = Aimbot.sens_config.get("rcs_strength_y", 2)
            strength_x = Aimbot.sens_config.get("rcs_strength_x", 0)
            
            # Simple constant pull down for now
            # In a pro version, this would read weapon patterns, but generic pull is good for general use
            if INTERCEPTION_AVAILABLE:
                # Move pixel by pixel to be smooth
                # We need to regulate this so it doesn't pull down instantly to the floor
                # Execution happens every frame, so we keep values small
                interception.move_relative(int(strength_x), int(strength_y))
            else:
                Aimbot.send_input_move(int(strength_x), int(strength_y))

    def move_crosshair(self, x, y):
        # Triggerbot check (Always active if enabled, even if not right-clicking? No, usually when aimed)
        self.triggerbot(x, y)
        
        # RCS Logic (Runs independent of Right Click, usually on Left Click)
        self.rcs()

        # ONLY activate aim movement when right-click is held (ADS mode)
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

        # === RAGE / BRUTAL MODE Logic (If smoothness < 0.2) ===
        if smoothness < 0.2:
            # DIRECT RAW OFFSET (No Filtering, No Prediction, No Smoothing)
            raw_offset_x = (x - 960)
            raw_offset_y = (y - 540)
            
            # Apply only sensitivity/speed scaling
            diff_x = int(raw_offset_x * scale / 10 * sensitivity * x_speed)
            diff_y = int(raw_offset_y * scale / 10 * sensitivity * y_speed)
            
            if INTERCEPTION_AVAILABLE:
                interception.move_relative(diff_x, diff_y)
            else:
                Aimbot.send_input_move(diff_x, diff_y)
            
            Aimbot.last_move_info = f"RAGE PULL: {diff_x}, {diff_y}"
            return
        # ===============================================

        # === LEGIT MODE WITH PID CONTROLLER ===
        # Professional-grade aim smoothing using P.I.D. algorithm
        
        pid_enabled = Aimbot.sens_config.get("pid_enabled", True)
        
        if pid_enabled:
            # Update PID parameters from config (hot-reload)
            Aimbot.pid_controller.kp = Aimbot.sens_config.get("pid_kp", 0.7)
            Aimbot.pid_controller.ki = Aimbot.sens_config.get("pid_ki", 0.05)
            Aimbot.pid_controller.kd = Aimbot.sens_config.get("pid_kd", 0.15)
            
            # Calculate error (distance from crosshair to target)
            error_x = x - 960  # Target X - Screen Center X
            error_y = y - 540  # Target Y - Screen Center Y
            
            # DEADZONE: If very close, STOP completely (Prevents shake)
            if abs(error_x) < 4 and abs(error_y) < 4:
                Aimbot.pid_controller.reset() # Reset I-term so it doesn't wind up
                return
            
            # Get PID output (how much to move)
            pid_out_x, pid_out_y = Aimbot.pid_controller.update(error_x, error_y)
            
            # Apply speed scaling
            diff_x = pid_out_x * scale / 100 * x_speed
            diff_y = pid_out_y * scale / 100 * y_speed
            
            # CAP maximum movement
            max_move = 60
            diff_x = max(-max_move, min(max_move, diff_x))
            diff_y = max(-max_move, min(max_move, diff_y))
            
            # FILTER: Ignore very small movements (Anti-Jitter)
            if abs(diff_x) < 1.0 and abs(diff_y) < 1.0:
                return

            diff_x = int(diff_x)
            diff_y = int(diff_y)
            
            if diff_x == 0 and diff_y == 0:
                return
            
            # Execute movement
            if INTERCEPTION_AVAILABLE:
                interception.move_relative(diff_x, diff_y)
            else:
                Aimbot.send_input_move(diff_x, diff_y)
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
                if INTERCEPTION_AVAILABLE:
                    interception.move_relative(move_x, move_y)
                else:
                    Aimbot.send_input_move(move_x, move_y)


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

        left = int(half_screen_width - self.box_constant//2)
        top = int(half_screen_height - self.box_constant//2)
        right = int(left + self.box_constant)
        bottom = int(top + self.box_constant)

        detection_box = {'left': left, 'top': top, 'width': int(self.box_constant), 'height': int(self.box_constant)}
        region = (left, top, right, bottom)

        # Initialize DXCam
        self.camera = dxcam.create(output_idx=0, output_color="BGR")

        if self.collect_data:
            collect_pause = 0

        while True:
            start_time = time.perf_counter()
            
            # Hot-reload config every 2 seconds
            if time.perf_counter() - Aimbot.config_reload_time > 2:
                Aimbot.reload_config()
                Aimbot.config_reload_time = time.perf_counter()
            
            frame = self.camera.grab(region=region)
            if frame is None:
                continue
            
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
                least_crosshair_dist = closest_detection = player_in_frame = False
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
                    
                    # Calculate head position - use configurable aim_point_ratio from config
                    aim_ratio = Aimbot.sens_config.get("aim_point_ratio", 0.10)
                    relative_head_X, relative_head_Y = int((x1 + x2)/2), int(y1 + h * aim_ratio)
                    own_player = x1 < 15 or (x1 < self.box_constant/5 and y2 > self.box_constant/1.2) 
                    
                    x1y1 = (x1, y1)
                    x2y2 = (x2, y2)

                    #calculate the distance between each detection and the crosshair
                    crosshair_dist = math.dist((relative_head_X, relative_head_Y), (self.box_constant/2, self.box_constant/2))
                    
                    if not least_crosshair_dist: least_crosshair_dist = crosshair_dist

                    # FOV Filter
                    fov_radius = Aimbot.sens_config.get("fov_radius", 150)
                    if crosshair_dist > fov_radius:
                        continue 

                    if crosshair_dist <= least_crosshair_dist and not own_player:
                        least_crosshair_dist = crosshair_dist
                        closest_detection = {"x1y1": x1y1, "x2y2": x2y2, "relative_head_X": relative_head_X, "relative_head_Y": relative_head_Y, "conf": conf, "crosshair_dist": crosshair_dist}

                    if not own_player:
                        # Draw box - Color based on confidence (Visual Debug)
                        color = (0, 255, 0) if conf > 0.6 else (0, 165, 255) # Green = High Conf, Orange = Low
                        cv2.rectangle(frame, x1y1, x2y2, color, 2) 
                        cv2.putText(frame, f"{int(conf * 100)}%", x1y1, cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2) 
                    else:
                        own_player = False
                        if not player_in_frame:
                            player_in_frame = True

                if closest_detection: #if valid detection exists
                    current_time = time.time()
                    
                    # Target locking logic - prevent switching between enemies
                    locked_detection = None
                    
                    # If we have an active lock
                    if Aimbot.locked_target_x is not None and (current_time - Aimbot.lock_time) < Aimbot.lock_duration:
                        # Find detection closest to the LOCKED position (not crosshair)
                        best_dist_to_lock = 100 # Threshold distance in pixels
                        
                        for x1, y1, x2, y2, conf in results_xyxy:
                            bx1, by1 = x1, y1
                            bx2, by2 = x2, y2
                            bh = by2 - by1
                            # Correct head calculation for distance check
                            check_head_x, check_head_y = int((bx1 + bx2)/2), int(by1 + bh/10)
                            
                            # Absolute position on screen
                            abs_check_x = check_head_x + detection_box['left']
                            abs_check_y = check_head_y + detection_box['top']
                            
                            dist_to_lock = math.dist((abs_check_x, abs_check_y), (Aimbot.locked_target_x, Aimbot.locked_target_y))
                            
                            if dist_to_lock < best_dist_to_lock:
                                best_dist_to_lock = dist_to_lock
                                locked_detection = {
                                    "x1y1": [bx1, by1], 
                                    "x2y2": [bx2, by2], 
                                    "relative_head_X": check_head_x, 
                                    "relative_head_Y": check_head_y, 
                                    "conf": conf
                                }
                    
                    if locked_detection:
                        # Continue tracking the locked target
                        closest_detection = locked_detection
                        Aimbot.locked_target_x = closest_detection["relative_head_X"] + detection_box['left']
                        Aimbot.locked_target_y = closest_detection["relative_head_Y"] + detection_box['top']
                        Aimbot.lock_time = current_time
                    else:
                        # No lock or lost lock -> Acquire new target closest to crosshair
                        Aimbot.locked_target_x = closest_detection["relative_head_X"] + detection_box['left']
                        Aimbot.locked_target_y = closest_detection["relative_head_Y"] + detection_box['top']
                        Aimbot.lock_time = current_time
                    cv2.circle(frame, (closest_detection["relative_head_X"], closest_detection["relative_head_Y"]), 5, (115, 244, 113), -1) #draw circle on the head

                    #draw line from the crosshair to the head
                    cv2.line(frame, (closest_detection["relative_head_X"], closest_detection["relative_head_Y"]), (self.box_constant//2, self.box_constant//2), (244, 242, 113), 2)

                    absolute_head_X, absolute_head_Y = closest_detection["relative_head_X"] + detection_box['left'], closest_detection["relative_head_Y"] + detection_box['top']

                    x1, y1 = closest_detection["x1y1"]
                    if Aimbot.is_target_locked(absolute_head_X, absolute_head_Y):
                        cv2.putText(frame, "LOCKED", (x1 + 40, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (115, 244, 113), 2) #draw the confidence labels on the bounding boxes
                    else:
                        cv2.putText(frame, "TARGETING", (x1 + 40, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (115, 113, 244), 2) #draw the confidence labels on the bounding boxes

                    if Aimbot.is_aimbot_enabled():
                        Aimbot.move_crosshair(self, absolute_head_X, absolute_head_Y)

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
                
                cv2.imshow("AI AIMBOT", frame)
                if cv2.waitKey(1) & 0xFF == ord('0'):
                    break
            else:
                # Headless Mode:
                # Just sleep briefly to prevent 100% CPU usage if needed, though pytorch inference usually takes time.
                pass
            # ------------------ VISUALIZATION END ------------------

    def clean_up():
        print("\n[INFO] F2 WAS PRESSED. QUITTING...")
        os._exit(0)

if __name__ == "__main__": print("You are in the wrong directory and are running the wrong file; you must run lunar.py")
