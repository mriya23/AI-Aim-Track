import customtkinter as ctk
import json
import os
import threading
import sys
import tkinter.messagebox

# --- Appearance Settings ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class GUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🎯 LUNAR AIMBOT")
        self.geometry("420x680")
        self.resizable(False, False)
        
        # Load config
        self.config_file = "lib/config/config.json"
        self.load_config()

        # --- MAIN FRAME ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # === HEADER ===
        header = ctk.CTkFrame(self.main_frame, fg_color="#1a1a2e", corner_radius=10)
        header.pack(fill="x", pady=(0,10))
        ctk.CTkLabel(header, text="🎯 LUNAR", font=("Roboto", 24, "bold"), text_color="#e94560").pack(side="left", padx=15, pady=10)
        
        self.status_label = ctk.CTkLabel(header, text="● READY", text_color="#00ff88", font=("Roboto", 12, "bold"))
        self.status_label.pack(side="right", padx=15)

        # === TOGGLE SWITCHES ===
        toggles = ctk.CTkFrame(self.main_frame, fg_color="#16213e", corner_radius=10)
        toggles.pack(fill="x", pady=5)
        
        self.switch_enabled = ctk.CTkSwitch(toggles, text="AIMBOT", command=self.on_enable_toggle, 
                                            progress_color="#e94560", font=("Roboto", 12, "bold"))
        self.switch_enabled.pack(side="left", padx=15, pady=10)
        if self.config.get("enabled", True): self.switch_enabled.select()
        
        self.switch_trigger = ctk.CTkSwitch(toggles, text="TRIGGERBOT", command=self.on_trigger_toggle,
                                           progress_color="#e94560", font=("Roboto", 12, "bold"))
        self.switch_trigger.pack(side="left", padx=15, pady=10)
        if self.config.get("triggerbot_enabled", False): self.switch_trigger.select()
        
        self.switch_rcs = ctk.CTkSwitch(toggles, text="RCS", command=self.on_rcs_toggle,
                                       progress_color="#e94560", font=("Roboto", 12, "bold"))
        self.switch_rcs.pack(side="left", padx=15, pady=10)
        if self.config.get("rcs_enabled", False): self.switch_rcs.select()

        # === AIM SETTINGS ===
        aim_frame = ctk.CTkFrame(self.main_frame, fg_color="#16213e", corner_radius=10)
        aim_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(aim_frame, text="AIM SETTINGS", font=("Roboto", 11, "bold"), text_color="#e94560").pack(anchor="w", padx=10, pady=(8,5))
        
        # FOV Radius
        self.create_slider(aim_frame, "FOV Radius", "fov_radius", 10, 800, int)

        # Bone Preset (Dropdown)
        frame_bone = ctk.CTkFrame(aim_frame, fg_color="transparent")
        frame_bone.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(frame_bone, text="Target Bone", font=("Roboto", 11), width=100, anchor="w").pack(side="left")
        
        def on_bone_change(choice):
            if choice == "Head": val = 0.12
            elif choice == "Neck": val = 0.20
            elif choice == "Chest": val = 0.30
            else: val = 0.12
            self.config["aim_point_ratio"] = val
            self.save_config()
        
        self.bone_option = ctk.CTkOptionMenu(frame_bone, values=["Head", "Neck", "Chest"], command=on_bone_change,
                                            fg_color="#e94560", button_color="#c73e54", button_hover_color="#de3b55")
        self.bone_option.pack(side="left", expand=True, fill="x", padx=10)
        self.bone_option.set("Head") 

        # Manual Offset
        self.create_slider(aim_frame, "Offset (Vertical)", "aim_point_ratio", 0.0, 0.5, float)
        
        # Speed X
        self.create_slider(aim_frame, "Aim Speed X", "x_speed", 0.1, 8.0, float)
        # Speed Y
        self.create_slider(aim_frame, "Aim Speed Y", "y_speed", 0.1, 8.0, float)
        
        # Smoothing
        self.create_slider(aim_frame, "Smoothing", "smoothing", 0.05, 1.0, float)
        
        # === TRIGGER SETTINGS ===
        trigger_frame = ctk.CTkFrame(self.main_frame, fg_color="#16213e", corner_radius=10)
        trigger_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(trigger_frame, text="TRIGGER SETTINGS", font=("Roboto", 11, "bold"), text_color="#e94560").pack(anchor="w", padx=10, pady=(8,5))
        
        self.create_slider(trigger_frame, "Trigger Radius", "trigger_radius", 1, 100, int)
        self.create_slider(trigger_frame, "Delay (ms)", "trigger_delay", 0.0, 0.3, float, scale=1000)

        # === ADVANCED ===
        adv_frame = ctk.CTkFrame(self.main_frame, fg_color="#16213e", corner_radius=10)
        adv_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(adv_frame, text="ADVANCED", font=("Roboto", 11, "bold"), text_color="#e94560").pack(anchor="w", padx=10, pady=(8,5))
        
        self.create_slider(adv_frame, "Data Scale", "scale", 10, 100, float)
        self.create_slider(adv_frame, "Confidence", "conf_thres", 0.1, 0.9, float)

        # === BUTTONS ===
        btn_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=10)
        
        self.btn_start = ctk.CTkButton(btn_frame, text="▶ START", command=self.on_start,
                                       fg_color="#e94560", hover_color="#c73e54", height=40,
                                       font=("Roboto", 14, "bold"))
        self.btn_start.pack(side="left", expand=True, fill="x", padx=5)
        
        ctk.CTkButton(btn_frame, text="💾 SAVE", command=self.save_config,
                     fg_color="#0f3460", hover_color="#1a4a7a", height=40,
                     font=("Roboto", 14, "bold")).pack(side="left", expand=True, fill="x", padx=5)

        # === FOOTER ===
        ctk.CTkLabel(self.main_frame, text="F1 = Toggle | F2 = Exit", 
                    text_color="#666", font=("Roboto", 10)).pack(pady=5)

    def create_slider(self, parent, label, config_key, min_val, max_val, val_type, scale=1):
        """Helper to create compact slider with label"""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=10, pady=2)
        
        ctk.CTkLabel(frame, text=label, font=("Roboto", 11), width=100, anchor="w").pack(side="left")
        
        current = self.config.get(config_key, min_val)
        
        # Dynamic scaler for label display
        display_val = float(current) * scale
        fmt = "{:.0f}" if (val_type == int or scale != 1) else "{:.2f}"
            
        value_label = ctk.CTkLabel(frame, text=fmt.format(display_val), 
                                  font=("Roboto", 11), width=50)
        value_label.pack(side="right")
        
        def on_change(val):
            # Special case: If user moves X Speed slider, we CAN sync Y Speed if we want.
            # But user asked for Separate X/Y. So we DO NOT sync automatically here anymore.
            # Unless it's the specific "Speed" master slider which no longer exists.
            
            self.config[config_key] = val_type(val)
            
            display_val = float(val) * scale
            value_label.configure(text=fmt.format(display_val))
            self.save_config()
        
        slider = ctk.CTkSlider(frame, from_=min_val, to=max_val, command=on_change,
                              progress_color="#e94560", button_color="#e94560")
        slider.pack(side="left", expand=True, fill="x", padx=10)
        slider.set(current)

    # --- Callbacks ---
    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        except Exception:
            self.config = {}

    def save_config(self):
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            pass

    def on_enable_toggle(self):
        self.config["enabled"] = bool(self.switch_enabled.get())
        self.save_config()

    def on_trigger_toggle(self):
        self.config["triggerbot_enabled"] = bool(self.switch_trigger.get())
        self.save_config()

    def on_rcs_toggle(self):
        self.config["rcs_enabled"] = bool(self.switch_rcs.get())
        self.save_config()

    def on_start(self):
        self.save_config()
        self.status_label.configure(text="● RUNNING", text_color="#00ff88")
        self.btn_start.configure(text="● ACTIVE", state="disabled", fg_color="#0f3460")
        
        # Start keyboard listener for F1/F2
        def setup_hotkeys():
            try:
                from pynput import keyboard as kb
                from lib.aimbot import Aimbot
                
                def on_key_release(key):
                    if key == kb.Key.f1:
                        Aimbot.update_status_aimbot()
                        if Aimbot.is_aimbot_enabled():
                            self.after(0, lambda: self.status_label.configure(text="● AIM ON", text_color="#00ff88"))
                        else:
                            self.after(0, lambda: self.status_label.configure(text="● AIM OFF", text_color="#ff6b6b"))
                    elif key == kb.Key.f2:
                        os._exit(0)
                        
                listener = kb.Listener(on_release=on_key_release)
                listener.start()
            except Exception as e:
                print(f"Hotkey Error: {e}")
        
        def run_aimbot():
            try:
                # Ensure existing instance is killed or handled? 
                # For this simplified version we assume fresh start
                from lib.aimbot import Aimbot
                bot = Aimbot()
                bot.start()
            except Exception as e:
                print(f"Aimbot Error: {e}")
        
        # Start threads
        threading.Thread(target=setup_hotkeys, daemon=True).start()
        threading.Thread(target=run_aimbot, daemon=True).start()

# --- Main ---
if __name__ == "__main__":
    app = GUI()
    app.mainloop()
