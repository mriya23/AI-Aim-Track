import pytest
from dataclasses import dataclass

@dataclass
class MockPreset:
    aim_point_ratio: float
    horizontal_offset: float

def calculate_lock_position(box, preset):
    """
    Simulated logic from aimbot.py (post-fix)
    box: (x1, y1, x2, y2)
    """
    bx1, by1, bx2, by2 = box
    bh = by2 - by1
    bw = bx2 - bx1
    
    # Logic from aimbot.py
    offset_x = bw * preset.horizontal_offset
    check_head_x = int((bx1 + bx2)/2 + offset_x)
    check_head_y = int(by1 + bh * preset.aim_point_ratio)
    
    return check_head_x, check_head_y

class TestLockingMath:
    def test_stomach_preset_math(self):
        """Test that Stomach preset (0.45) targets correct Y coordinate."""
        # Box: Top=0, Bottom=100. Height=100.
        # Stomach (0.45) should be at Y = 45
        box = (0, 0, 100, 100)
        preset = MockPreset(aim_point_ratio=0.45, horizontal_offset=0.0)
        
        mx, my = calculate_lock_position(box, preset)
        
        assert my == 45
        # Previous bug (hardcoded 0.1) would result in Y=10
        assert my != 10 

    def test_head_preset_math(self):
        """Test Head preset."""
        box = (0, 0, 100, 100)
        preset = MockPreset(aim_point_ratio=0.10, horizontal_offset=0.0)
        mx, my = calculate_lock_position(box, preset)
        assert my == 10

    def test_chest_preset_math(self):
        """Test Chest preset."""
        box = (0, 0, 100, 100)
        preset = MockPreset(aim_point_ratio=0.30, horizontal_offset=0.0)
        mx, my = calculate_lock_position(box, preset)
        assert my == 30
