import customtkinter as ctk
import json
import os
import threading
import time
import tkinter.messagebox
from typing import Any, Dict, Optional

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ConfigStore:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = {}
        self.last_error: Optional[str] = None
        self.load()

    def load(self) -> bool:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.last_error = None
            return True
        except Exception as e:
            self.data = {}
            self.last_error = str(e)
            return False

    def save(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4)
            self.last_error = None
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value, *, persist: bool = True) -> bool:
        self.data[key] = value
        return self.save() if persist else True


class ErrorBanner(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="#2b1d1d", corner_radius=10)
        self.grid_columnconfigure(0, weight=1)
        self._label = ctk.CTkLabel(
            self,
            text="",
            text_color="#ffb4b4",
            font=("Roboto", 12, "bold"),
            justify="left",
            anchor="w",
        )
        self._label.grid(row=0, column=0, sticky="ew", padx=12, pady=10)
        self.hide()

    def show(self, message: str):
        self._label.configure(text=message)
        self.grid()

    def hide(self):
        self.grid_remove()


class LabeledSlider(ctk.CTkFrame):
    def __init__(
        self,
        master,
        *,
        store: ConfigStore,
        key: str,
        label: str,
        min_val: float,
        max_val: float,
        val_type,
        scale: float = 1.0,
        format_override: Optional[str] = None,
    ):
        super().__init__(master, fg_color="transparent")
        self.store = store
        self.key = key
        self.val_type = val_type
        self.scale = scale
        self._page: Optional["PageBase"] = self._find_page()

        self.grid_columnconfigure(1, weight=1)
        self._label = ctk.CTkLabel(self, text=label, font=("Roboto", 12), anchor="w")
        self._label.grid(row=0, column=0, sticky="w", padx=(0, 10), pady=6)

        current = self.store.get(key, min_val)
        try:
            current = float(current)
        except Exception:
            current = float(min_val)

        fmt: str
        if format_override is not None:
            fmt = format_override
        else:
            fmt = "{:.0f}" if (val_type == int or scale != 1.0) else "{:.2f}"

        self._value_label = ctk.CTkLabel(
            self, text=fmt.format(current * scale), font=("Roboto", 12), width=70, anchor="e"
        )
        self._value_label.grid(row=0, column=2, sticky="e", padx=(10, 0), pady=6)

        def on_change(val):
            try:
                casted = self.val_type(val)
            except Exception:
                casted = self.val_type(float(val))
            ok = self.store.set(self.key, casted)
            if self._page is not None:
                if ok:
                    self._page.notify_persist_ok()
                else:
                    self._page.notify_persist_failed(self.store.last_error or "Unknown error")
            self._value_label.configure(text=fmt.format(float(val) * self.scale))

        self._slider = ctk.CTkSlider(
            self,
            from_=min_val,
            to=max_val,
            command=on_change,
            progress_color="#e94560",
            button_color="#e94560",
        )
        self._slider.grid(row=0, column=1, sticky="ew", pady=6)
        self._slider.set(current)

    def _find_page(self) -> Optional["PageBase"]:
        cur = self.master
        while cur is not None:
            if isinstance(cur, PageBase):
                return cur
            cur = getattr(cur, "master", None)
        return None


class LabeledSwitch(ctk.CTkFrame):
    def __init__(self, master, *, store: ConfigStore, key: str, label: str, default: bool):
        super().__init__(master, fg_color="transparent")
        self.store = store
        self.key = key
        self._page: Optional["PageBase"] = self._find_page()
        self._switch = ctk.CTkSwitch(
            self,
            text=label,
            progress_color="#e94560",
            font=("Roboto", 12, "bold"),
            command=self._on_toggle,
        )
        self._switch.pack(anchor="w", pady=6)
        if bool(self.store.get(self.key, default)):
            self._switch.select()

    def _on_toggle(self):
        ok = self.store.set(self.key, bool(self._switch.get()))
        if self._page is not None:
            if ok:
                self._page.notify_persist_ok()
            else:
                self._page.notify_persist_failed(self.store.last_error or "Unknown error")

    def _find_page(self) -> Optional["PageBase"]:
        cur = self.master
        while cur is not None:
            if isinstance(cur, PageBase):
                return cur
            cur = getattr(cur, "master", None)
        return None


class PageBase(ctk.CTkFrame):
    def __init__(self, master, *, controller, store: ConfigStore, title: str, subtitle: str):
        super().__init__(master, fg_color="transparent")
        self.controller = controller
        self.store = store
        self._loading = False

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self, fg_color="#1a1a2e", corner_radius=12)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text=title, font=("Roboto", 22, "bold"), text_color="#e94560").grid(
            row=0, column=0, sticky="w", padx=16, pady=(12, 2)
        )
        ctk.CTkLabel(header, text=subtitle, font=("Roboto", 12), text_color="#b7b7c8").grid(
            row=1, column=0, sticky="w", padx=16, pady=(0, 12)
        )

        self.error_banner = ErrorBanner(self)
        self.error_banner.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))

        self.content = ctk.CTkScrollableFrame(self, fg_color="#16213e", corner_radius=12)
        self.content.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.content.grid_columnconfigure(0, weight=1)

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 16))
        footer.grid_columnconfigure(0, weight=1)

        self._notice = ctk.CTkLabel(footer, text="", text_color="#9aa0b5", font=("Roboto", 11), anchor="w")
        self._notice.grid(row=0, column=0, sticky="w")

        self._busy = ctk.CTkProgressBar(footer, mode="indeterminate", width=140, progress_color="#e94560")
        self._busy.grid(row=0, column=1, sticky="e", padx=(10, 0))
        self._busy.grid_remove()

    def show_error(self, message: str):
        self.error_banner.show(message)

    def clear_error(self):
        self.error_banner.hide()

    def on_show(self):
        self.clear_error()

    def set_loading(self, is_loading: bool):
        if is_loading:
            self._busy.grid()
            self._busy.start()
        else:
            try:
                self._busy.stop()
            except Exception:
                pass
            self._busy.grid_remove()

    def notify_persist_ok(self):
        self.set_loading(True)
        self._notice.configure(text="Tersimpan", text_color="#9aa0b5")
        self.after(250, lambda: self.set_loading(False))

    def notify_persist_failed(self, message: str):
        self.set_loading(False)
        self.show_error(f"Gagal menyimpan konfigurasi: {message}")
        self._notice.configure(text="Gagal menyimpan", text_color="#ff6b6b")


class ControlPage(PageBase):
    def __init__(self, master, *, controller, store: ConfigStore):
        super().__init__(
            master,
            controller=controller,
            store=store,
            title="Kontrol",
            subtitle="Mulai aimbot, lihat status, dan kelola hotkey.",
        )

        self._status = ctk.CTkLabel(
            self.content, text="● READY", text_color="#00ff88", font=("Roboto", 12, "bold"), anchor="w"
        )
        self._status.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 6))

        self._loading_bar = ctk.CTkProgressBar(self.content, mode="indeterminate", progress_color="#e94560")
        self._loading_bar.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 16))
        self._loading_bar.grid_remove()

        switches = ctk.CTkFrame(self.content, fg_color="transparent")
        switches.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 8))
        switches.grid_columnconfigure(0, weight=1)

        self._enabled_switch = LabeledSwitch(
            switches, store=self.store, key="enabled", label="Aimbot Enabled (UI)", default=True
        )
        self._enabled_switch.grid(row=0, column=0, sticky="w")

        btns = ctk.CTkFrame(self.content, fg_color="transparent")
        btns.grid(row=3, column=0, sticky="ew", padx=16, pady=(8, 16))
        btns.grid_columnconfigure((0, 1, 2), weight=1)

        self._btn_start = ctk.CTkButton(
            btns,
            text="Start",
            fg_color="#e94560",
            hover_color="#c73e54",
            height=40,
            font=("Roboto", 14, "bold"),
            command=self._on_start,
        )
        self._btn_start.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self._btn_toggle = ctk.CTkButton(
            btns,
            text="Toggle AIM (F1)",
            fg_color="#0f3460",
            hover_color="#1a4a7a",
            height=40,
            font=("Roboto", 14, "bold"),
            command=self._on_toggle_aim,
        )
        self._btn_toggle.grid(row=0, column=1, sticky="ew", padx=8)

        self._btn_exit = ctk.CTkButton(
            btns,
            text="Exit (F2)",
            fg_color="#2b2b2b",
            hover_color="#3a3a3a",
            height=40,
            font=("Roboto", 14, "bold"),
            command=self._on_exit,
        )
        self._btn_exit.grid(row=0, column=2, sticky="ew", padx=(8, 0))

        info = ctk.CTkFrame(self.content, fg_color="transparent")
        info.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 16))
        info.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            info,
            text=f"Config: {self.store.path}",
            text_color="#9aa0b5",
            font=("Roboto", 11),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        ctk.CTkLabel(
            info,
            text="Hotkey: F1 = Toggle AIM, F2 = Exit",
            text_color="#9aa0b5",
            font=("Roboto", 11),
            anchor="w",
        ).grid(row=1, column=0, sticky="w")

        self._aimbot_started = False

    def _set_loading(self, is_loading: bool):
        if is_loading:
            self._loading_bar.grid()
            self._loading_bar.start()
            self._btn_start.configure(state="disabled")
        else:
            self._loading_bar.stop()
            self._loading_bar.grid_remove()
            self._btn_start.configure(state="normal" if not self._aimbot_started else "disabled")

    def _set_status(self, text: str, color: str):
        self._status.configure(text=text, text_color=color)

    def _on_start(self):
        self.clear_error()
        self.store.save()

        if self._aimbot_started:
            self._set_status("● RUNNING", "#00ff88")
            return

        self._set_loading(True)
        self._set_status("● LOADING", "#ffd166")

        def setup_hotkeys():
            try:
                from pynput import keyboard as kb
                from lib.aimbot import Aimbot

                def on_key_release(key):
                    if key == kb.Key.f1:
                        Aimbot.update_status_aimbot()
                        self.after(0, self._refresh_status_from_aimbot)
                    elif key == kb.Key.f2:
                        os._exit(0)

                listener = kb.Listener(on_release=on_key_release)
                listener.start()
            except Exception as e:
                self.after(0, lambda: self.show_error(f"Gagal memulai hotkey: {e}"))

        def run_aimbot():
            try:
                from lib.aimbot import Aimbot

                bot = Aimbot()
                self.after(0, self._on_aimbot_ready)
                bot.start()
            except Exception as e:
                self.after(0, lambda: self._on_aimbot_failed(e))

        threading.Thread(target=setup_hotkeys, daemon=True).start()
        threading.Thread(target=run_aimbot, daemon=True).start()

    def _on_aimbot_ready(self):
        self._aimbot_started = True
        self._set_loading(False)
        self._refresh_status_from_aimbot()

    def _on_aimbot_failed(self, err: Exception):
        self._aimbot_started = False
        self._set_loading(False)
        self._set_status("● ERROR", "#ff6b6b")
        self.show_error(f"Gagal memulai aimbot: {err}")

    def _refresh_status_from_aimbot(self):
        try:
            from lib.aimbot import Aimbot

            if Aimbot.is_aimbot_enabled():
                self._set_status("● AIM ON", "#00ff88")
            else:
                self._set_status("● AIM OFF", "#ff6b6b")
        except Exception:
            self._set_status("● RUNNING", "#00ff88")

    def _on_toggle_aim(self):
        try:
            from lib.aimbot import Aimbot

            Aimbot.update_status_aimbot()
            self._refresh_status_from_aimbot()
        except Exception as e:
            self.show_error(f"Tidak bisa toggle AIM: {e}")

    def _on_exit(self):
        os._exit(0)


class AimPage(PageBase):
    def __init__(self, master, *, controller, store: ConfigStore):
        super().__init__(
            master,
            controller=controller,
            store=store,
            title="Aim",
            subtitle="Pengaturan FOV, target point, dan smoothing pergerakan aim.",
        )

        row = 0
        enabled = LabeledSwitch(self.content, store=self.store, key="enabled", label="Aimbot Enabled (UI)", default=True)
        enabled.grid(row=row, column=0, sticky="ew", padx=16, pady=(16, 0))
        row += 1

        LabeledSlider(
            self.content, store=self.store, key="fov_radius", label="FOV Radius", min_val=10, max_val=800, val_type=int
        ).grid(row=row, column=0, sticky="ew", padx=16)
        row += 1

        bone = ctk.CTkFrame(self.content, fg_color="transparent")
        bone.grid(row=row, column=0, sticky="ew", padx=16)
        bone.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(bone, text="Target Preset", font=("Roboto", 12), anchor="w").grid(
            row=0, column=0, sticky="w", pady=6
        )

        def on_bone_change(choice: str):
            # Import preset manager
            try:
                from lib.presets import get_preset_manager
                manager = get_preset_manager()
                manager.set_active(choice.lower())
                manager.save_to_config()
                
                # Also update legacy aim_point_ratio for compatibility
                mapping = {"Head": 0.10, "Neck": 0.18, "Chest": 0.30, "Stomach": 0.45}
                self.store.set("aim_point_ratio", mapping.get(choice, 0.10))
            except Exception:
                # Fallback to legacy method
                mapping = {"Head": 0.10, "Neck": 0.18, "Chest": 0.30, "Stomach": 0.45}
                self.store.set("aim_point_ratio", mapping.get(choice, 0.10))

        option = ctk.CTkOptionMenu(
            bone,
            values=["Head", "Neck", "Chest", "Stomach"],
            command=on_bone_change,
            fg_color="#e94560",
            button_color="#c73e54",
            button_hover_color="#de3b55",
        )
        option.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=6)
        
        # Determine current preset from config
        current_ratio = float(self.store.get("aim_point_ratio", 0.10) or 0.10)
        preset = "Head"
        if current_ratio > 0.35:
            preset = "Stomach"
        elif current_ratio > 0.24:
            preset = "Chest"
        elif current_ratio > 0.14:
            preset = "Neck"
        option.set(preset)
        row += 1

        LabeledSlider(
            self.content,
            store=self.store,
            key="aim_point_ratio",
            label="Offset (Vertical)",
            min_val=0.0,
            max_val=0.5,
            val_type=float,
            format_override="{:.3f}",
        ).grid(row=row, column=0, sticky="ew", padx=16)
        row += 1

        LabeledSlider(
            self.content, store=self.store, key="x_speed", label="Aim Speed X", min_val=0.1, max_val=8.0, val_type=float
        ).grid(row=row, column=0, sticky="ew", padx=16)
        row += 1
        LabeledSlider(
            self.content, store=self.store, key="y_speed", label="Aim Speed Y", min_val=0.1, max_val=8.0, val_type=float
        ).grid(row=row, column=0, sticky="ew", padx=16)
        row += 1
        LabeledSlider(
            self.content, store=self.store, key="smoothing", label="Smoothing", min_val=0.05, max_val=1.0, val_type=float
        ).grid(row=row, column=0, sticky="ew", padx=16)
        row += 1
        LabeledSlider(
            self.content,
            store=self.store,
            key="smoothness",
            label="Mode Smoothness",
            min_val=0.05,
            max_val=1.0,
            val_type=float,
        ).grid(row=row, column=0, sticky="ew", padx=16, pady=(0, 10))


class TriggerPage(PageBase):
    def __init__(self, master, *, controller, store: ConfigStore):
        super().__init__(
            master,
            controller=controller,
            store=store,
            title="Triggerbot",
            subtitle="Tembak otomatis saat crosshair berada dekat target.",
        )

        row = 0
        LabeledSwitch(
            self.content, store=self.store, key="triggerbot_enabled", label="Triggerbot Enabled", default=False
        ).grid(row=row, column=0, sticky="ew", padx=16, pady=(16, 0))
        row += 1

        LabeledSlider(
            self.content, store=self.store, key="trigger_radius", label="Trigger Radius", min_val=1, max_val=100, val_type=int
        ).grid(row=row, column=0, sticky="ew", padx=16)
        row += 1

        LabeledSlider(
            self.content,
            store=self.store,
            key="trigger_delay",
            label="Delay (ms)",
            min_val=0.0,
            max_val=0.3,
            val_type=float,
            scale=1000.0,
        ).grid(row=row, column=0, sticky="ew", padx=16, pady=(0, 10))


class RCSPage(PageBase):
    def __init__(self, master, *, controller, store: ConfigStore):
        super().__init__(
            master,
            controller=controller,
            store=store,
            title="RCS",
            subtitle="Recoil Control System saat tombol klik kiri ditahan.",
        )

        row = 0
        LabeledSwitch(self.content, store=self.store, key="rcs_enabled", label="RCS Enabled", default=False).grid(
            row=row, column=0, sticky="ew", padx=16, pady=(16, 0)
        )
        row += 1

        LabeledSlider(
            self.content,
            store=self.store,
            key="rcs_strength_y",
            label="RCS Strength Y",
            min_val=0,
            max_val=10,
            val_type=int,
        ).grid(row=row, column=0, sticky="ew", padx=16)
        row += 1
        LabeledSlider(
            self.content,
            store=self.store,
            key="rcs_strength_x",
            label="RCS Strength X",
            min_val=-10,
            max_val=10,
            val_type=int,
        ).grid(row=row, column=0, sticky="ew", padx=16, pady=(0, 10))


class AdvancedPage(PageBase):
    def __init__(self, master, *, controller, store: ConfigStore):
        super().__init__(
            master,
            controller=controller,
            store=store,
            title="Advanced",
            subtitle="Pengaturan performa, confidence, PID, dan mode tampilan.",
        )

        row = 0
        LabeledSlider(
            self.content, store=self.store, key="roi_size", label="ROI Size", min_val=160, max_val=800, val_type=int
        ).grid(row=row, column=0, sticky="ew", padx=16, pady=(16, 0))
        row += 1

        LabeledSlider(
            self.content, store=self.store, key="scale", label="Data Scale", min_val=10, max_val=100, val_type=float
        ).grid(row=row, column=0, sticky="ew", padx=16)
        row += 1

        LabeledSlider(
            self.content, store=self.store, key="conf_thres", label="Confidence (TensorRT)", min_val=0.1, max_val=0.9, val_type=float
        ).grid(row=row, column=0, sticky="ew", padx=16)
        row += 1

        LabeledSwitch(self.content, store=self.store, key="pid_enabled", label="PID Enabled", default=True).grid(
            row=row, column=0, sticky="ew", padx=16, pady=(8, 0)
        )
        row += 1

        LabeledSlider(
            self.content, store=self.store, key="pid_kp", label="PID Kp", min_val=0.0, max_val=2.0, val_type=float
        ).grid(row=row, column=0, sticky="ew", padx=16)
        row += 1
        LabeledSlider(
            self.content, store=self.store, key="pid_ki", label="PID Ki", min_val=0.0, max_val=1.0, val_type=float
        ).grid(row=row, column=0, sticky="ew", padx=16)
        row += 1
        LabeledSlider(
            self.content, store=self.store, key="pid_kd", label="PID Kd", min_val=0.0, max_val=1.0, val_type=float
        ).grid(row=row, column=0, sticky="ew", padx=16)
        row += 1

        LabeledSwitch(self.content, store=self.store, key="headless", label="Headless Mode", default=False).grid(
            row=row, column=0, sticky="ew", padx=16, pady=(8, 16)
        )


class HelpPage(PageBase):
    def __init__(self, master, *, controller, store: ConfigStore):
        super().__init__(
            master,
            controller=controller,
            store=store,
            title="Panduan",
            subtitle="Ringkasan fitur, alur penggunaan, dan troubleshooting.",
        )

        wrapper = ctk.CTkFrame(self.content, fg_color="transparent")
        wrapper.grid(row=0, column=0, sticky="ew", padx=16, pady=16)
        wrapper.grid_columnconfigure(0, weight=1)

        text = (
            "Alur cepat:\n"
            "1) Buka halaman Kontrol → Start.\n"
            "2) Gunakan F1 untuk ON/OFF aim saat running.\n"
            "3) Atur Aim/Triggerbot/RCS/Advanced sesuai kebutuhan.\n\n"
            "Error umum:\n"
            "- Jika aimbot gagal start, pastikan dependencies terpasang dan config tersedia.\n"
            "- Jika window OpenCV mengganggu, aktifkan Headless Mode di Advanced.\n"
        )
        ctk.CTkLabel(wrapper, text=text, justify="left", anchor="w", font=("Roboto", 12)).grid(
            row=0, column=0, sticky="ew"
        )


class GUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("LUNAR AIMBOT")
        self.geometry("980x680")
        self.minsize(860, 560)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.store = ConfigStore("lib/config/config.json")

        self.sidebar = ctk.CTkFrame(self, fg_color="#0f1220", corner_radius=0, width=220)
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_rowconfigure(10, weight=1)

        brand = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="ew", padx=16, pady=(18, 12))
        ctk.CTkLabel(brand, text="LUNAR", font=("Roboto", 22, "bold"), text_color="#e94560").pack(anchor="w")
        ctk.CTkLabel(brand, text="Multi-page GUI", font=("Roboto", 11), text_color="#9aa0b5").pack(anchor="w")

        self._nav_buttons: Dict[str, ctk.CTkButton] = {}
        self._pages: Dict[str, PageBase] = {}

        self.container = ctk.CTkFrame(self, fg_color="#0b0f1a", corner_radius=0)
        self.container.grid(row=0, column=1, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self._register_page("control", "Kontrol", ControlPage)
        self._register_page("aim", "Aim", AimPage)
        self._register_page("trigger", "Triggerbot", TriggerPage)
        self._register_page("rcs", "RCS", RCSPage)
        self._register_page("advanced", "Advanced", AdvancedPage)
        self._register_page("help", "Panduan", HelpPage)

        footer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        footer.grid(row=11, column=0, sticky="sew", padx=16, pady=16)
        ctk.CTkLabel(footer, text="F1: Toggle AIM\nF2: Exit", text_color="#9aa0b5", font=("Roboto", 11), justify="left").pack(
            anchor="w"
        )

        self.show_page("control")

        if self.store.last_error:
            try:
                self._pages["control"].show_error(f"Config tidak bisa dibaca: {self.store.last_error}")
            except Exception:
                pass

    def _register_page(self, page_id: str, label: str, page_cls):
        row_idx = len(self._nav_buttons) + 1
        btn = ctk.CTkButton(
            self.sidebar,
            text=label,
            fg_color="transparent",
            hover_color="#1a1f33",
            text_color="#d6d8e0",
            font=("Roboto", 13, "bold"),
            anchor="w",
            height=44,
            command=lambda pid=page_id: self.show_page(pid),
        )
        btn.grid(row=row_idx, column=0, sticky="ew", padx=12, pady=4)
        self._nav_buttons[page_id] = btn

        page = page_cls(self.container, controller=self, store=self.store)
        page.grid(row=0, column=0, sticky="nsew")
        self._pages[page_id] = page

    def show_page(self, page_id: str):
        for pid, btn in self._nav_buttons.items():
            btn.configure(fg_color="transparent" if pid != page_id else "#1a1f33")
        page = self._pages[page_id]
        page.tkraise()
        page.on_show()


if __name__ == "__main__":
    app = GUI()
    app.mainloop()
