"""
Preset Management System for Aimbot
====================================
Modul untuk mengelola preset target (HEAD, NECK, CHEST, STOMACH)
dengan error handling, hot-reload, dan validasi parameter.

Author: AI-Aim-Track Team
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any, List
from enum import Enum
import json
import os
import time
import threading


class PresetType(Enum):
    """Enum untuk tipe preset yang tersedia."""
    HEAD = "head"
    NECK = "neck"
    CHEST = "chest"
    STOMACH = "stomach"
    CUSTOM = "custom"


@dataclass
class PresetConfig:
    """
    Konfigurasi preset untuk target aim.
    
    Attributes:
        name: Nama preset (untuk display)
        aim_point_ratio: Rasio vertikal dari atas bounding box (0.0 = kepala, 1.0 = kaki)
        horizontal_offset: Offset horizontal (-0.5 ke 0.5, 0 = tengah)
        priority_weight: Prioritas target (1.0 = normal, >1 = lebih prioritas)
        size_threshold_min: Minimum area threshold untuk validasi target
        size_threshold_max: Maximum area threshold
    """
    name: str
    aim_point_ratio: float = 0.10
    horizontal_offset: float = 0.0
    priority_weight: float = 1.0
    size_threshold_min: float = 0.01
    size_threshold_max: float = 0.5
    
    def __post_init__(self):
        """Validasi parameter setelah inisialisasi."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validasi semua parameter preset.
        
        Returns:
            True jika valid
            
        Raises:
            ValueError jika parameter tidak valid
        """
        if not 0.0 <= self.aim_point_ratio <= 1.0:
            raise ValueError(f"aim_point_ratio harus antara 0.0 dan 1.0, got {self.aim_point_ratio}")
        
        if not -0.5 <= self.horizontal_offset <= 0.5:
            raise ValueError(f"horizontal_offset harus antara -0.5 dan 0.5, got {self.horizontal_offset}")
        
        if not 0.1 <= self.priority_weight <= 2.0:
            raise ValueError(f"priority_weight harus antara 0.1 dan 2.0, got {self.priority_weight}")
        
        if not 0.0 <= self.size_threshold_min <= 1.0:
            raise ValueError(f"size_threshold_min harus antara 0.0 dan 1.0, got {self.size_threshold_min}")
        
        if not 0.0 <= self.size_threshold_max <= 1.0:
            raise ValueError(f"size_threshold_max harus antara 0.0 dan 1.0, got {self.size_threshold_max}")
        
        if self.size_threshold_min > self.size_threshold_max:
            raise ValueError("size_threshold_min tidak boleh lebih besar dari size_threshold_max")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert preset ke dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PresetConfig':
        """
        Buat PresetConfig dari dictionary.
        
        Args:
            data: Dictionary dengan parameter preset
            
        Returns:
            PresetConfig instance
        """
        return cls(
            name=data.get("name", "Unknown"),
            aim_point_ratio=float(data.get("aim_point_ratio", 0.10)),
            horizontal_offset=float(data.get("horizontal_offset", 0.0)),
            priority_weight=float(data.get("priority_weight", 1.0)),
            size_threshold_min=float(data.get("size_threshold_min", 0.01)),
            size_threshold_max=float(data.get("size_threshold_max", 0.5)),
        )


# ============ BUILT-IN PRESETS ============

PRESET_HEAD = PresetConfig(
    name="Head",
    aim_point_ratio=0.10,
    horizontal_offset=0.0,
    priority_weight=1.0,
    size_threshold_min=0.01,
    size_threshold_max=0.5,
)

PRESET_NECK = PresetConfig(
    name="Neck",
    aim_point_ratio=0.18,
    horizontal_offset=0.0,
    priority_weight=0.95,
    size_threshold_min=0.01,
    size_threshold_max=0.5,
)

PRESET_CHEST = PresetConfig(
    name="Chest",
    aim_point_ratio=0.30,
    horizontal_offset=0.0,
    priority_weight=0.90,
    size_threshold_min=0.01,
    size_threshold_max=0.5,
)

PRESET_STOMACH = PresetConfig(
    name="Stomach",
    aim_point_ratio=0.45,
    horizontal_offset=0.0,
    priority_weight=0.85,
    size_threshold_min=0.01,
    size_threshold_max=0.5,
)

# Mapping dari nama ke preset
BUILTIN_PRESETS: Dict[str, PresetConfig] = {
    "head": PRESET_HEAD,
    "neck": PRESET_NECK,
    "chest": PRESET_CHEST,
    "stomach": PRESET_STOMACH,
}


class PresetManager:
    """
    Manager untuk mengelola preset dengan fitur:
    - Hot-reload dari config file
    - Error handling dengan fallback
    - Thread-safe operations
    """
    
    _instance: Optional['PresetManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern untuk PresetManager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize PresetManager."""
        if self._initialized:
            return
            
        self._active_preset_name: str = "head"
        self._active_preset: PresetConfig = PRESET_HEAD
        self._custom_presets: Dict[str, PresetConfig] = {}
        self._config_path: str = "lib/config/config.json"
        self._last_config_mtime: float = 0.0
        self._last_error: Optional[str] = None
        self._reload_debounce: float = 0.5  # 500ms debounce
        self._last_reload_time: float = 0.0
        self._initialized = True
    
    @classmethod
    def get_instance(cls) -> 'PresetManager':
        """Get singleton instance."""
        return cls()
    
    def get_active(self) -> PresetConfig:
        """
        Get active preset dengan auto hot-reload.
        
        Returns:
            PresetConfig aktif
        """
        self._try_hot_reload()
        return self._active_preset
    
    def get_active_name(self) -> str:
        """Get nama preset aktif."""
        return self._active_preset_name
    
    def set_active(self, preset_name: str) -> bool:
        """
        Set preset aktif berdasarkan nama.
        
        Args:
            preset_name: Nama preset (head, neck, chest, stomach, atau custom)
            
        Returns:
            True jika berhasil, False jika gagal
        """
        preset_name_lower = preset_name.lower()
        
        # Check built-in presets
        if preset_name_lower in BUILTIN_PRESETS:
            self._active_preset_name = preset_name_lower
            self._active_preset = BUILTIN_PRESETS[preset_name_lower]
            self._last_error = None
            return True
        
        # Check custom presets
        if preset_name_lower in self._custom_presets:
            self._active_preset_name = preset_name_lower
            self._active_preset = self._custom_presets[preset_name_lower]
            self._last_error = None
            return True
        
        # Preset not found - fallback to HEAD
        self._last_error = f"Preset '{preset_name}' tidak ditemukan, menggunakan HEAD"
        self._active_preset_name = "head"
        self._active_preset = PRESET_HEAD
        return False
    
    def get_all_presets(self) -> Dict[str, PresetConfig]:
        """Get semua preset (built-in + custom)."""
        all_presets = dict(BUILTIN_PRESETS)
        all_presets.update(self._custom_presets)
        return all_presets
    
    def get_preset_names(self) -> List[str]:
        """Get daftar nama preset."""
        names = list(BUILTIN_PRESETS.keys())
        names.extend(self._custom_presets.keys())
        return names
    
    def add_custom_preset(self, name: str, config: PresetConfig) -> bool:
        """
        Tambah custom preset.
        
        Args:
            name: Nama preset
            config: Konfigurasi preset
            
        Returns:
            True jika berhasil
        """
        try:
            config.validate()
            self._custom_presets[name.lower()] = config
            self._last_error = None
            return True
        except ValueError as e:
            self._last_error = str(e)
            return False
    
    def remove_custom_preset(self, name: str) -> bool:
        """Hapus custom preset."""
        name_lower = name.lower()
        if name_lower in self._custom_presets:
            del self._custom_presets[name_lower]
            return True
        return False
    
    def load_from_config(self, config_path: Optional[str] = None) -> bool:
        """
        Load preset dari config file.
        
        Args:
            config_path: Path ke config file (optional)
            
        Returns:
            True jika berhasil load
        """
        if config_path:
            self._config_path = config_path
        
        try:
            if not os.path.exists(self._config_path):
                self._last_error = f"Config file tidak ditemukan: {self._config_path}"
                return False
            
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Load active preset dari config
            active_name = data.get("active_preset", "head")
            
            # Legacy support: gunakan aim_point_ratio jika ada
            if "aim_point_ratio" in data and "active_preset" not in data:
                ratio = float(data.get("aim_point_ratio", 0.10))
                active_name = self._ratio_to_preset_name(ratio)
            
            # Load custom presets jika ada
            custom_presets_data = data.get("custom_presets", {})
            for name, preset_data in custom_presets_data.items():
                try:
                    preset = PresetConfig.from_dict(preset_data)
                    self._custom_presets[name.lower()] = preset
                except Exception:
                    pass  # Skip invalid custom presets
            
            # Set active preset
            self.set_active(active_name)
            
            # Update mtime
            self._last_config_mtime = os.path.getmtime(self._config_path)
            self._last_error = None
            return True
            
        except json.JSONDecodeError as e:
            self._last_error = f"Config JSON error: {e}"
            self._fallback_to_default()
            return False
        except Exception as e:
            self._last_error = f"Config load error: {e}"
            self._fallback_to_default()
            return False
    
    def save_to_config(self) -> bool:
        """
        Save preset ke config file.
        
        Returns:
            True jika berhasil
        """
        try:
            # Load existing config
            data = {}
            if os.path.exists(self._config_path):
                try:
                    with open(self._config_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    pass
            
            # Update preset data
            data["active_preset"] = self._active_preset_name
            data["aim_point_ratio"] = self._active_preset.aim_point_ratio
            
            # Save custom presets
            if self._custom_presets:
                data["custom_presets"] = {
                    name: preset.to_dict() 
                    for name, preset in self._custom_presets.items()
                }
            
            # Write to file
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            
            self._last_config_mtime = os.path.getmtime(self._config_path)
            self._last_error = None
            return True
            
        except Exception as e:
            self._last_error = f"Config save error: {e}"
            return False
    
    def get_last_error(self) -> Optional[str]:
        """Get error terakhir."""
        return self._last_error
    
    def _try_hot_reload(self):
        """Try hot-reload config jika file berubah."""
        now = time.time()
        
        # Debounce
        if now - self._last_reload_time < self._reload_debounce:
            return
        
        try:
            if os.path.exists(self._config_path):
                mtime = os.path.getmtime(self._config_path)
                if mtime > self._last_config_mtime:
                    self._last_reload_time = now
                    self.load_from_config()
        except Exception:
            pass
    
    def _fallback_to_default(self):
        """Fallback ke preset default (HEAD)."""
        self._active_preset_name = "head"
        self._active_preset = PRESET_HEAD
    
    def _ratio_to_preset_name(self, ratio: float) -> str:
        """Convert aim_point_ratio ke nama preset terdekat."""
        # Mapping ratio ke preset
        mappings = [
            (0.15, "head"),
            (0.24, "neck"),
            (0.38, "chest"),
            (1.0, "stomach"),
        ]
        
        for threshold, name in mappings:
            if ratio <= threshold:
                return name
        
        return "head"


# ============ HELPER FUNCTIONS ============

def get_preset_manager() -> PresetManager:
    """Get singleton PresetManager instance."""
    return PresetManager.get_instance()


def get_active_preset() -> PresetConfig:
    """Get active preset (shortcut function)."""
    return get_preset_manager().get_active()


def set_active_preset(name: str) -> bool:
    """Set active preset (shortcut function)."""
    return get_preset_manager().set_active(name)


def get_preset_by_name(name: str) -> Optional[PresetConfig]:
    """Get preset by name."""
    manager = get_preset_manager()
    all_presets = manager.get_all_presets()
    return all_presets.get(name.lower())
