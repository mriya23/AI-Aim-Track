"""
Unit Tests for Preset System
==============================
Test coverage target: ≥90%

Run tests with:
    pytest tests/test_presets.py -v --cov=lib/presets --cov-report=term-missing
"""

import pytest
import json
import os
import tempfile
import shutil
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.presets import (
    PresetConfig, 
    PresetType,
    PresetManager,
    PRESET_HEAD,
    PRESET_NECK,
    PRESET_CHEST,
    PRESET_STOMACH,
    BUILTIN_PRESETS,
    get_preset_manager,
    get_active_preset,
    set_active_preset,
    get_preset_by_name,
)


class TestPresetConfig:
    """Tests for PresetConfig dataclass."""
    
    def test_preset_config_creation(self):
        """Test creating a valid PresetConfig."""
        config = PresetConfig(
            name="Test",
            aim_point_ratio=0.15,
            horizontal_offset=0.1,
            priority_weight=1.0,
        )
        assert config.name == "Test"
        assert config.aim_point_ratio == 0.15
        assert config.horizontal_offset == 0.1
        assert config.priority_weight == 1.0
    
    def test_preset_config_default_values(self):
        """Test PresetConfig with default values."""
        config = PresetConfig(name="Default")
        assert config.aim_point_ratio == 0.10
        assert config.horizontal_offset == 0.0
        assert config.priority_weight == 1.0
        assert config.size_threshold_min == 0.01
        assert config.size_threshold_max == 0.5
    
    def test_preset_config_validation_aim_ratio_low(self):
        """Test validation fails for aim_point_ratio < 0."""
        with pytest.raises(ValueError, match="aim_point_ratio"):
            PresetConfig(name="Invalid", aim_point_ratio=-0.1)
    
    def test_preset_config_validation_aim_ratio_high(self):
        """Test validation fails for aim_point_ratio > 1."""
        with pytest.raises(ValueError, match="aim_point_ratio"):
            PresetConfig(name="Invalid", aim_point_ratio=1.5)
    
    def test_preset_config_validation_horizontal_offset(self):
        """Test validation fails for invalid horizontal_offset."""
        with pytest.raises(ValueError, match="horizontal_offset"):
            PresetConfig(name="Invalid", horizontal_offset=-0.6)
    
    def test_preset_config_validation_priority_weight(self):
        """Test validation fails for invalid priority_weight."""
        with pytest.raises(ValueError, match="priority_weight"):
            PresetConfig(name="Invalid", priority_weight=0.05)
    
    def test_preset_config_validation_size_thresholds(self):
        """Test validation fails when min > max threshold."""
        with pytest.raises(ValueError, match="size_threshold"):
            PresetConfig(
                name="Invalid", 
                size_threshold_min=0.5, 
                size_threshold_max=0.1
            )
    
    def test_preset_config_to_dict(self):
        """Test converting PresetConfig to dictionary."""
        config = PresetConfig(name="Test", aim_point_ratio=0.2)
        data = config.to_dict()
        assert data["name"] == "Test"
        assert data["aim_point_ratio"] == 0.2
        assert "horizontal_offset" in data
    
    def test_preset_config_from_dict(self):
        """Test creating PresetConfig from dictionary."""
        data = {
            "name": "FromDict",
            "aim_point_ratio": 0.25,
            "horizontal_offset": 0.05,
        }
        config = PresetConfig.from_dict(data)
        assert config.name == "FromDict"
        assert config.aim_point_ratio == 0.25
        assert config.horizontal_offset == 0.05


class TestBuiltinPresets:
    """Tests for built-in presets."""
    
    def test_preset_head_exists(self):
        """Test HEAD preset exists and has correct ratio."""
        assert PRESET_HEAD is not None
        assert PRESET_HEAD.name == "Head"
        assert 0.08 <= PRESET_HEAD.aim_point_ratio <= 0.12
    
    def test_preset_neck_exists(self):
        """Test NECK preset exists."""
        assert PRESET_NECK is not None
        assert PRESET_NECK.name == "Neck"
        assert 0.15 <= PRESET_NECK.aim_point_ratio <= 0.20
    
    def test_preset_chest_exists(self):
        """Test CHEST preset exists."""
        assert PRESET_CHEST is not None
        assert PRESET_CHEST.name == "Chest"
        assert 0.25 <= PRESET_CHEST.aim_point_ratio <= 0.35
    
    def test_preset_stomach_exists(self):
        """Test STOMACH preset exists."""
        assert PRESET_STOMACH is not None
        assert PRESET_STOMACH.name == "Stomach"
        assert 0.40 <= PRESET_STOMACH.aim_point_ratio <= 0.50
    
    def test_builtin_presets_dict(self):
        """Test BUILTIN_PRESETS contains all 4 presets."""
        assert len(BUILTIN_PRESETS) == 4
        assert "head" in BUILTIN_PRESETS
        assert "neck" in BUILTIN_PRESETS
        assert "chest" in BUILTIN_PRESETS
        assert "stomach" in BUILTIN_PRESETS
    
    def test_builtin_presets_are_valid(self):
        """Test all built-in presets pass validation."""
        for name, preset in BUILTIN_PRESETS.items():
            assert preset.validate() is True


class TestPresetManager:
    """Tests for PresetManager class."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        temp_dir = tempfile.mkdtemp()
        config_dir = os.path.join(temp_dir, "lib", "config")
        os.makedirs(config_dir, exist_ok=True)
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def manager(self, temp_config_dir):
        """Create fresh PresetManager instance."""
        # Reset singleton for testing
        PresetManager._instance = None
        manager = PresetManager()
        manager._config_path = os.path.join(temp_config_dir, "lib", "config", "config.json")
        return manager
    
    def test_manager_singleton(self):
        """Test PresetManager is singleton."""
        PresetManager._instance = None
        m1 = PresetManager()
        m2 = PresetManager()
        assert m1 is m2
    
    def test_manager_default_preset(self, manager):
        """Test manager starts with HEAD as default."""
        assert manager.get_active_name() == "head"
        assert manager.get_active().name == "Head"
    
    def test_manager_set_active_builtin(self, manager):
        """Test setting active preset to built-in preset."""
        result = manager.set_active("neck")
        assert result is True
        assert manager.get_active_name() == "neck"
        assert manager.get_active().aim_point_ratio == PRESET_NECK.aim_point_ratio
    
    def test_manager_set_active_case_insensitive(self, manager):
        """Test preset names are case-insensitive."""
        assert manager.set_active("CHEST") is True
        assert manager.get_active_name() == "chest"
        
        assert manager.set_active("Stomach") is True
        assert manager.get_active_name() == "stomach"
    
    def test_manager_set_active_invalid(self, manager):
        """Test fallback when setting invalid preset."""
        result = manager.set_active("invalid_preset")
        assert result is False
        assert manager.get_active_name() == "head"  # Fallback
        assert manager.get_last_error() is not None
    
    def test_manager_get_all_presets(self, manager):
        """Test getting all presets."""
        all_presets = manager.get_all_presets()
        assert len(all_presets) >= 4
        assert "head" in all_presets
        assert "stomach" in all_presets
    
    def test_manager_get_preset_names(self, manager):
        """Test getting preset names."""
        names = manager.get_preset_names()
        assert "head" in names
        assert "neck" in names
        assert "chest" in names
        assert "stomach" in names
    
    def test_manager_add_custom_preset(self, manager):
        """Test adding custom preset."""
        custom = PresetConfig(name="Custom", aim_point_ratio=0.35)
        result = manager.add_custom_preset("my_custom", custom)
        assert result is True
        assert "my_custom" in manager.get_preset_names()
    
    def test_manager_add_invalid_custom_preset(self, manager):
        """Test adding invalid custom preset fails."""
        # Create invalid preset by bypassing __post_init__
        custom = PresetConfig.__new__(PresetConfig)
        custom.name = "Invalid"
        custom.aim_point_ratio = 2.0  # Invalid
        custom.horizontal_offset = 0.0
        custom.priority_weight = 1.0
        custom.size_threshold_min = 0.01
        custom.size_threshold_max = 0.5
        
        result = manager.add_custom_preset("invalid", custom)
        assert result is False
        assert manager.get_last_error() is not None
    
    def test_manager_remove_custom_preset(self, manager):
        """Test removing custom preset."""
        custom = PresetConfig(name="ToRemove", aim_point_ratio=0.4)
        manager.add_custom_preset("removable", custom)
        
        result = manager.remove_custom_preset("removable")
        assert result is True
        assert "removable" not in manager.get_preset_names()
    
    def test_manager_load_from_config(self, manager, temp_config_dir):
        """Test loading preset from config file."""
        config_path = os.path.join(temp_config_dir, "lib", "config", "config.json")
        config_data = {
            "active_preset": "chest",
            "aim_point_ratio": 0.30,
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f)
        
        result = manager.load_from_config()
        assert result is True
        assert manager.get_active_name() == "chest"
    
    def test_manager_load_legacy_config(self, manager, temp_config_dir):
        """Test loading from legacy config (aim_point_ratio only)."""
        config_path = os.path.join(temp_config_dir, "lib", "config", "config.json")
        config_data = {
            "aim_point_ratio": 0.45,  # Should map to stomach
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f)
        
        result = manager.load_from_config()
        assert result is True
        assert manager.get_active_name() == "stomach"
    
    def test_manager_load_corrupt_config(self, manager, temp_config_dir):
        """Test fallback when config is corrupt."""
        config_path = os.path.join(temp_config_dir, "lib", "config", "config.json")
        with open(config_path, "w") as f:
            f.write("{ invalid json }")
        
        result = manager.load_from_config()
        assert result is False
        assert manager.get_active_name() == "head"  # Fallback
        assert manager.get_last_error() is not None
    
    def test_manager_save_to_config(self, manager, temp_config_dir):
        """Test saving preset to config file."""
        manager.set_active("neck")
        result = manager.save_to_config()
        assert result is True
        
        # Verify saved
        config_path = os.path.join(temp_config_dir, "lib", "config", "config.json")
        with open(config_path, "r") as f:
            data = json.load(f)
        assert data["active_preset"] == "neck"
    
    def test_manager_hotreload(self, manager, temp_config_dir):
        """Test hot-reload when config file changes."""
        config_path = os.path.join(temp_config_dir, "lib", "config", "config.json")
        
        # Initial config
        with open(config_path, "w") as f:
            json.dump({"active_preset": "head"}, f)
        
        manager.load_from_config()
        assert manager.get_active_name() == "head"
        
        # Modify config (simulate external change)
        import time
        time.sleep(0.1)  # Ensure mtime changes
        with open(config_path, "w") as f:
            json.dump({"active_preset": "stomach"}, f)
        
        # Force reload by resetting mtime tracking
        manager._last_config_mtime = 0
        manager._last_reload_time = 0
        
        # This should trigger hot-reload
        preset = manager.get_active()
        assert manager.get_active_name() == "stomach"


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_get_preset_manager(self):
        """Test get_preset_manager returns manager."""
        PresetManager._instance = None
        manager = get_preset_manager()
        assert isinstance(manager, PresetManager)
    
    def test_get_active_preset(self):
        """Test get_active_preset shortcut."""
        PresetManager._instance = None
        preset = get_active_preset()
        assert isinstance(preset, PresetConfig)
    
    def test_set_active_preset(self):
        """Test set_active_preset shortcut."""
        # Test that set_active_preset returns True for valid preset
        result = set_active_preset("chest")
        assert result is True
        # Verify active preset was changed
        manager = get_preset_manager()
        assert manager.get_active_name() == "chest"
    
    def test_get_preset_by_name(self):
        """Test get_preset_by_name function."""
        PresetManager._instance = None
        preset = get_preset_by_name("neck")
        assert preset is not None
        assert preset.name == "Neck"
    
    def test_get_preset_by_name_invalid(self):
        """Test get_preset_by_name with invalid name."""
        PresetManager._instance = None
        preset = get_preset_by_name("nonexistent")
        assert preset is None


class TestPresetType:
    """Tests for PresetType enum."""
    
    def test_preset_type_values(self):
        """Test PresetType has correct values."""
        assert PresetType.HEAD.value == "head"
        assert PresetType.NECK.value == "neck"
        assert PresetType.CHEST.value == "chest"
        assert PresetType.STOMACH.value == "stomach"
        assert PresetType.CUSTOM.value == "custom"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=lib/presets", "--cov-report=term-missing"])
