# Sistem Preset Aimbot

## Overview

Sistem preset memungkinkan pengguna memilih titik target pada anatomi musuh:
- **Head** (0.10) - Targeting kepala
- **Neck** (0.18) - Targeting leher  
- **Chest** (0.30) - Targeting dada
- **Stomach** (0.45) - Targeting perut

## Struktur File

```
lib/
├── presets.py     # Modul preset utama
├── config/
│   └── config.json  # Konfigurasi tersimpan
└── aimbot.py      # Menggunakan get_active_preset()

tests/
└── test_presets.py  # Unit tests (36 tests, 84%+ coverage)
```

## Penggunaan

### Di GUI
Pilih preset dari dropdown "Target Preset" di halaman Aim.

### Programatik
```python
from lib.presets import get_active_preset, set_active_preset

# Get preset aktif
preset = get_active_preset()
print(f"Aim ratio: {preset.aim_point_ratio}")

# Set preset
set_active_preset("stomach")
```

## Parameter Preset

| Parameter | Deskripsi | Range |
|-----------|-----------|-------|
| `aim_point_ratio` | Offset vertikal dari top bbox | 0.0 - 1.0 |
| `horizontal_offset` | Offset horizontal | -0.5 - 0.5 |
| `priority_weight` | Prioritas target | 0.1 - 2.0 |

## Custom Preset

```python
from lib.presets import PresetConfig, get_preset_manager

custom = PresetConfig(
    name="CustomTarget",
    aim_point_ratio=0.25,
    horizontal_offset=0.05,
)

manager = get_preset_manager()
manager.add_custom_preset("my_preset", custom)
manager.set_active("my_preset")
```

## Error Handling

Jika config rusak atau preset tidak ditemukan:
1. System fallback ke preset **HEAD**
2. Error log tersedia via `manager.get_last_error()`

## Hot-Reload

Config otomatis di-reload saat file berubah (debounce 500ms).
