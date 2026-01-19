"""
AI AIMBOT - Desktop GUI Application
Built with CustomTkinter
"""

import customtkinter as ctk
import json
import os
import threading
import sys

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CONFIG_PATH = "lib/config/config.json"

class AIAimbotGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("AI AIMBOT")
        self.geometry("500x750")
        self.resizable(False, True)  # Allow vertical resize
        
        # Load config
        self.config = self.load_config()
        
        # Offset presets
        self.offset_presets = {
            "Head": 0.10,
            "Neck": 0.20,
            "Chest": 0.35,
            "Body": 0.50
        }
        
        # Create UI
        self.create_widgets()
        
        # Aimbot process reference
        self.aimbot_process = None
        
    def load_config(self):
        """Load configuration from JSON file"""
        default_config = {
            "enabled": True,
            "fov_radius": 150,
            "offset_preset": "Head",
            "aim_point_ratio": 0.10,
            "x_speed": 1.0,
            "y_speed": 1.0,
            "smoothing": 0.06,
            "scale": 65.0,
            "prediction": 5.5,
            "second_instance": False
        }
        
        try:
            with open(CONFIG_PATH, 'r') as f:
                loaded = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in loaded:
                        loaded[key] = default_config[key]
                return loaded
        except:
            return default_config
    
    def save_config(self):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(self.config, f, indent=4)
        print("[GUI] Config saved!")
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Create scrollable container
        self.scroll_frame = ctk.CTkScrollableFrame(self, width=480, height=700)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title = ctk.CTkLabel(
            self.scroll_frame, 
            text="🎯 AI AIMBOT", 
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.pack(pady=20)
        
        # === Enable/Disable Section ===
        enable_frame = ctk.CTkFrame(self.scroll_frame)
        enable_frame.pack(fill="x", padx=20, pady=10)
        
        self.enable_var = ctk.BooleanVar(value=self.config.get("enabled", True))
        self.enable_switch = ctk.CTkSwitch(
            enable_frame,
            text="Aimbot Enabled",
            variable=self.enable_var,
            command=self.on_enable_toggle,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.enable_switch.pack(pady=15)
        
        # === Targeting Section ===
        targeting_label = ctk.CTkLabel(
            self.scroll_frame, 
            text="─── Targeting ───", 
            font=ctk.CTkFont(size=14)
        )
        targeting_label.pack(pady=(10, 5))
        
        targeting_frame = ctk.CTkFrame(self.scroll_frame)
        targeting_frame.pack(fill="x", padx=20, pady=5)
        
        # FOV Radius
        fov_label = ctk.CTkLabel(targeting_frame, text="FOV Radius:")
        fov_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.fov_value_label = ctk.CTkLabel(
            targeting_frame, 
            text=str(self.config.get("fov_radius", 150))
        )
        self.fov_value_label.grid(row=0, column=2, padx=10)
        
        self.fov_slider = ctk.CTkSlider(
            targeting_frame,
            from_=50, to=300,
            number_of_steps=25,
            command=self.on_fov_change
        )
        self.fov_slider.set(self.config.get("fov_radius", 150))
        self.fov_slider.grid(row=0, column=1, padx=10, pady=10)
        
        # Offset Preset
        preset_label = ctk.CTkLabel(targeting_frame, text="Offset Preset:")
        preset_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        self.preset_var = ctk.StringVar(value=self.config.get("offset_preset", "Head"))
        self.preset_dropdown = ctk.CTkOptionMenu(
            targeting_frame,
            values=list(self.offset_presets.keys()),
            variable=self.preset_var,
            command=self.on_preset_change
        )
        self.preset_dropdown.grid(row=1, column=1, padx=10, pady=10, columnspan=2)
        
        # Offset Manual
        offset_label = ctk.CTkLabel(targeting_frame, text="Offset (Manual):")
        offset_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        
        self.offset_entry = ctk.CTkEntry(targeting_frame, width=80)
        self.offset_entry.insert(0, str(self.config.get("aim_point_ratio", 0.10)))
        self.offset_entry.grid(row=2, column=1, padx=10, pady=10)
        self.offset_entry.bind("<FocusOut>", self.on_offset_change)
        
        # === Speed Section ===
        speed_label = ctk.CTkLabel(
            self, 
            text="─── Aim Speed ───", 
            font=ctk.CTkFont(size=14)
        )
        speed_label.pack(pady=(15, 5))
        
        speed_frame = ctk.CTkFrame(self.scroll_frame)
        speed_frame.pack(fill="x", padx=20, pady=5)
        
        # X-Speed
        x_label = ctk.CTkLabel(speed_frame, text="X-Speed:")
        x_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.x_speed_label = ctk.CTkLabel(
            speed_frame, 
            text=f"{self.config.get('x_speed', 1.0):.2f}"
        )
        self.x_speed_label.grid(row=0, column=2, padx=10)
        
        self.x_speed_slider = ctk.CTkSlider(
            speed_frame,
            from_=0.1, to=5.0,
            number_of_steps=49,
            command=self.on_x_speed_change
        )
        self.x_speed_slider.set(self.config.get("x_speed", 1.0))
        self.x_speed_slider.grid(row=0, column=1, padx=10, pady=10)
        
        # Y-Speed
        y_label = ctk.CTkLabel(speed_frame, text="Y-Speed:")
        y_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        self.y_speed_label = ctk.CTkLabel(
            speed_frame, 
            text=f"{self.config.get('y_speed', 1.0):.2f}"
        )
        self.y_speed_label.grid(row=1, column=2, padx=10)
        
        self.y_speed_slider = ctk.CTkSlider(
            speed_frame,
            from_=0.1, to=5.0,
            number_of_steps=49,
            command=self.on_y_speed_change
        )
        self.y_speed_slider.set(self.config.get("y_speed", 1.0))
        self.y_speed_slider.grid(row=1, column=1, padx=10, pady=10)
        
        # === Smoothing Section ===
        smooth_label = ctk.CTkLabel(
            self, 
            text="─── Sensitivity ───", 
            font=ctk.CTkFont(size=14)
        )
        smooth_label.pack(pady=(15, 5))
        
        smooth_frame = ctk.CTkFrame(self.scroll_frame)
        smooth_frame.pack(fill="x", padx=20, pady=5)
        
        sens_text = ctk.CTkLabel(smooth_frame, text="Sensitivity:")
        sens_text.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.smooth_value_label = ctk.CTkLabel(
            smooth_frame, 
            text=f"{self.config.get('smoothing', 0.06):.2f}"
        )
        self.smooth_value_label.grid(row=0, column=2, padx=10)
        
        self.smooth_slider = ctk.CTkSlider(
            smooth_frame,
            from_=0.01, to=1.0,
            number_of_steps=99,
            command=self.on_smooth_change
        )
        self.smooth_slider.set(self.config.get("smoothing", 0.06))
        self.smooth_slider.grid(row=0, column=1, padx=10, pady=10)
        
        # Smoothness (TRUE smoothing - controls micro-step count)
        smoothness_text = ctk.CTkLabel(smooth_frame, text="Smoothness:")
        smoothness_text.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        self.smoothness_value_label = ctk.CTkLabel(
            smooth_frame, 
            text=f"{self.config.get('smoothness', 0.5):.2f}"
        )
        self.smoothness_value_label.grid(row=1, column=2, padx=10)
        
        self.smoothness_slider = ctk.CTkSlider(
            smooth_frame,
            from_=0.1, to=1.0,
            number_of_steps=18,
            command=self.on_smoothness_change
        )
        self.smoothness_slider.set(self.config.get("smoothness", 0.5))
        self.smoothness_slider.grid(row=1, column=1, padx=10, pady=10)
        
        # === Advanced Section ===
        adv_label = ctk.CTkLabel(
            self, 
            text="─── Advanced ───", 
            font=ctk.CTkFont(size=14)
        )
        adv_label.pack(pady=(15, 5))
        
        adv_frame = ctk.CTkFrame(self.scroll_frame)
        adv_frame.pack(fill="x", padx=20, pady=5)
        
        # Second Instance
        self.second_var = ctk.BooleanVar(value=self.config.get("second_instance", False))
        self.second_switch = ctk.CTkSwitch(
            adv_frame,
            text="Second Aimbot Instance",
            variable=self.second_var,
            command=self.on_second_toggle
        )
        self.second_switch.pack(pady=15)
        
        # === Buttons ===
        button_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=20)
        
        self.save_btn = ctk.CTkButton(
            button_frame,
            text="💾 Save Settings",
            command=self.on_save,
            width=150
        )
        self.save_btn.pack(side="left", padx=10)
        
        self.start_btn = ctk.CTkButton(
            button_frame,
            text="▶ Start Aimbot",
            command=self.on_start,
            width=150,
            fg_color="green",
            hover_color="darkgreen"
        )
        self.start_btn.pack(side="right", padx=10)
        
        # Status bar
        self.status_label = ctk.CTkLabel(
            self,
            text="Status: Ready",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(pady=10)
    
    # === Event Handlers ===
    def on_enable_toggle(self):
        self.config["enabled"] = self.enable_var.get()
        
    def on_fov_change(self, value):
        self.config["fov_radius"] = int(value)
        self.fov_value_label.configure(text=str(int(value)))
        
    def on_preset_change(self, value):
        self.config["offset_preset"] = value
        ratio = self.offset_presets.get(value, 0.10)
        self.config["aim_point_ratio"] = ratio
        self.offset_entry.delete(0, "end")
        self.offset_entry.insert(0, str(ratio))
        
    def on_offset_change(self, event):
        try:
            value = float(self.offset_entry.get())
            value = max(0.0, min(1.0, value))
            self.config["aim_point_ratio"] = value
        except:
            pass
            
    def on_x_speed_change(self, value):
        self.config["x_speed"] = round(value, 2)
        self.x_speed_label.configure(text=f"{value:.2f}")
        
    def on_y_speed_change(self, value):
        self.config["y_speed"] = round(value, 2)
        self.y_speed_label.configure(text=f"{value:.2f}")
        
    def on_smooth_change(self, value):
        self.config["smoothing"] = round(value, 2)
        self.smooth_value_label.configure(text=f"{value:.2f}")
    
    def on_smoothness_change(self, value):
        self.config["smoothness"] = round(value, 2)
        self.smoothness_value_label.configure(text=f"{value:.2f}")
        
    def on_second_toggle(self):
        self.config["second_instance"] = self.second_var.get()
        
    def on_save(self):
        self.save_config()
        self.status_label.configure(text="Status: Settings Saved! ✓")
        
    def on_start(self):
        """Start the aimbot in a separate thread"""
        self.save_config()
        self.status_label.configure(text="Status: Starting Aimbot...")
        
        # Import and run aimbot
        def run_aimbot():
            try:
                from lib.aimbot import Aimbot
                aimbot = Aimbot()
                aimbot.start()
            except Exception as e:
                print(f"[ERROR] {e}")
        
        thread = threading.Thread(target=run_aimbot, daemon=True)
        thread.start()
        
        # Start keyboard listener for F1/F2 hotkeys
        def on_key_release(key):
            try:
                from pynput import keyboard as kb
                from lib.aimbot import Aimbot
                if key == kb.Key.f1:
                    Aimbot.update_status_aimbot()
                if key == kb.Key.f2:
                    Aimbot.clean_up()
            except:
                pass
        
        from pynput import keyboard
        listener = keyboard.Listener(on_release=on_key_release)
        listener.start()
        
        self.start_btn.configure(text="⏹ Running...", fg_color="orange")
        self.status_label.configure(text="Status: Aimbot Running (F1 to toggle)")


if __name__ == "__main__":
    app = AIAimbotGUI()
    app.mainloop()
