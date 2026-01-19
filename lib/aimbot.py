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


class Aimbot:
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    screen = mss.mss()
    pixel_increment = 1 #controls how many pixels the mouse moves for each relative movement (lower = smoother/more human)
    with open("lib/config/config.json") as f:
        sens_config = json.load(f)
    config_reload_time = 0
    
    @staticmethod
    def reload_config():
        """Hot-reload config from file (called periodically)"""
        try:
            with open("lib/config/config.json") as f:
                Aimbot.sens_config = json.load(f)
        except:
            pass
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

    def __init__(self, box_constant = 416, collect_data = False, mouse_delay = 0.0005, debug = False):
        #controls the initial centered box width and height of the "Lunar Vision" window
        self.box_constant = box_constant #controls the size of the detection box (equaling the width and height)

        print("[INFO] Loading the neural network model")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='lib/best.pt', force_reload = True)
        if torch.cuda.is_available():
            print(colored("CUDA ACCELERATION [ENABLED]", "green"))
        else:
            print(colored("[!] CUDA ACCELERATION IS UNAVAILABLE", "red"))
            print(colored("[!] Check your PyTorch installation, else performance will be poor", "red"))

        self.model.conf = 0.45 # base confidence threshold (or base detection (0-1)
        self.model.iou = 0.45 # NMS IoU (0-1)
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

    def left_click():
        ctypes.windll.user32.mouse_event(0x0002) #left mouse down
        Aimbot.sleep(0.0001)
        ctypes.windll.user32.mouse_event(0x0004) #left mouse up

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

    def move_crosshair(self, x, y):
        # ONLY activate when right-click is held (ADS mode)
        if not Aimbot.is_targeted():
            Aimbot.last_move_info = "Waiting for ADS (right-click)..."
            # Reset history
            Aimbot.prev_target_x = x
            Aimbot.prev_target_y = y
            Aimbot.smooth_mom_x = 0
            Aimbot.smooth_mom_y = 0
            return

        # 1. Calculate Raw Momentum
        raw_dx = x - Aimbot.prev_target_x
        raw_dy = y - Aimbot.prev_target_y
        
        # 2. Noise Filter / Deadzone
        # Increased to 3px to kill bounding box "shaking" at high scale
        if abs(raw_dx) < 3: raw_dx = 0
        if abs(raw_dy) < 3: raw_dy = 0
        
        # 3. Momentum Smoothing (EMA Filter)
        # Deep filtering (20% new, 80% prev) for inertia-like stability
        Aimbot.smooth_mom_x = (raw_dx * 0.2) + (Aimbot.smooth_mom_x * 0.8)
        Aimbot.smooth_mom_y = (raw_dy * 0.2) + (Aimbot.smooth_mom_y * 0.8)
        
        # 4. Momentum Prediction
        # Factor 5.5: Omega prediction for maximum "glue" effect
        momentum_factor = 5.5
        
        if abs(raw_dx) > 60 or abs(raw_dy) > 60: # Snap protection
            pred_x, pred_y = x, y
        else:
            pred_x = x + (Aimbot.smooth_mom_x * momentum_factor)
            pred_y = y + (Aimbot.smooth_mom_y * momentum_factor)

        # Update position history
        Aimbot.prev_target_x = x
        Aimbot.prev_target_y = y

        # 5. Sensitivity & Speed Calculation
        scale = Aimbot.sens_config.get("scale", Aimbot.sens_config.get("targeting_scale", 65.0))
        
        # Sensitivity controls movement speed (how much to move per frame)
        sensitivity = Aimbot.sens_config.get("smoothing", 0.06)  # legacy name in config
        
        # X/Y Speed multipliers from config (GUI-controlled)
        x_speed = Aimbot.sens_config.get("x_speed", 1.0)
        y_speed = Aimbot.sens_config.get("y_speed", 1.0)
            
        diff_x = int((pred_x - 960) * scale / 10 * sensitivity * x_speed)
        diff_y = int((pred_y - 540) * scale / 10 * sensitivity * y_speed)
        
        # Store debug info
        Aimbot.last_move_info = f"Move: dx={diff_x}, dy={diff_y}"
        
        if diff_x == 0 and diff_y == 0:
            return
        
        # 6. Execute Movement with TRUE SMOOTHNESS
        # Smoothness value controls how many micro-steps to use
        smoothness = Aimbot.sens_config.get("smoothness", 0.5)
        
        # Calculate base steps: 3 to 15 based on GUI slider
        base_steps = int(3 + smoothness * 12)
        
        # Additional steps based on distance (to keep long flicks smooth)
        abs_dx, abs_dy = abs(diff_x), abs(diff_y)
        dist = math.sqrt(abs_dx**2 + abs_dy**2)
        dist_steps = int(dist / 5) # 1 extra step for every 5px
        
        # Total steps (capped to prevent lag)
        steps = min(30, base_steps + dist_steps)
        
        if INTERCEPTION_AVAILABLE:
            try:
                if abs_dx > 1 or abs_dy > 1:
                    sx, sy = diff_x / steps, diff_y / steps
                    remainder_x, remainder_y = 0.0, 0.0
                    for _ in range(steps):
                        remainder_x += sx
                        remainder_y += sy
                        move_x = int(remainder_x)
                        move_y = int(remainder_y)
                        remainder_x -= move_x
                        remainder_y -= move_y
                        if move_x != 0 or move_y != 0:
                            interception.move_relative(move_x, move_y)
                else:
                    interception.move_relative(diff_x, diff_y)
            except:
                pass

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
            if self.collect_data: orig_frame = np.copy((frame))
            results = self.model(frame)

            if len(results.xyxy[0]) != 0: #player detected
                least_crosshair_dist = closest_detection = player_in_frame = False
                for *box, conf, cls in results.xyxy[0]: #iterate over each player detected
                    x1y1 = [int(x.item()) for x in box[:2]]
                    x2y2 = [int(x.item()) for x in box[2:]]
                    x1, y1, x2, y2, conf = *x1y1, *x2y2, conf.item()
                    height = y2 - y1
                    # Calculate head position - use configurable aim_point_ratio from config
                    aim_ratio = Aimbot.sens_config.get("aim_point_ratio", 0.10)
                    relative_head_X, relative_head_Y = int((x1 + x2)/2), int(y1 + height * aim_ratio)
                    own_player = x1 < 15 or (x1 < self.box_constant/5 and y2 > self.box_constant/1.2) #helps ensure that your own player is not regarded as a valid detection

                    #calculate the distance between each detection and the crosshair at (self.box_constant/2, self.box_constant/2)
                    crosshair_dist = math.dist((relative_head_X, relative_head_Y), (self.box_constant/2, self.box_constant/2))

                    if not least_crosshair_dist: least_crosshair_dist = crosshair_dist #initalize least crosshair distance variable first iteration

                    # FOV Filter: Ignore targets outside the configured radius
                    fov_radius = Aimbot.sens_config.get("fov_radius", 150)
                    if crosshair_dist > fov_radius:
                        continue  # Skip this target - too far from crosshair

                    if crosshair_dist <= least_crosshair_dist and not own_player:
                        least_crosshair_dist = crosshair_dist
                        closest_detection = {"x1y1": x1y1, "x2y2": x2y2, "relative_head_X": relative_head_X, "relative_head_Y": relative_head_Y, "conf": conf, "crosshair_dist": crosshair_dist}

                    if not own_player:
                        cv2.rectangle(frame, x1y1, x2y2, (244, 113, 115), 2) #draw the bounding boxes for all of the player detections (except own)
                        cv2.putText(frame, f"{int(conf * 100)}%", x1y1, cv2.FONT_HERSHEY_DUPLEX, 0.5, (244, 113, 116), 2) #draw the confidence labels on the bounding boxes
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
                        
                        for *box, conf, cls in results.xyxy[0]:
                            bx1, by1 = int(box[0].item()), int(box[1].item())
                            bx2, by2 = int(box[2].item()), int(box[3].item())
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
                                    "conf": conf.item()
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

    def clean_up():
        print("\n[INFO] F2 WAS PRESSED. QUITTING...")
        Aimbot.screen.close()
        os._exit(0)

if __name__ == "__main__": print("You are in the wrong directory and are running the wrong file; you must run lunar.py")
