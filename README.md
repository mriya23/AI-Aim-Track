# Lunar
Lunar is a neural network aim assist that uses real-time object detection accelerated with CUDA on Nvidia GPUs.

## About
Lunar can be modified to work with a variety of FPS games; however, it is currently configured for Fortnite. Besides being general purpose, the main advantage of using Lunar is that it does not meddle with the memory of other processes.

The basis of Lunar's player detection is the [YOLOv5](https://github.com/ultralytics/yolov5) architecture written in PyTorch.

A demo video (outdated) can be found [here](https://www.youtube.com/watch?v=XDAcQNUuT84).

## Installation
1. Install a version of [Python](https://www.python.org/downloads/) 3.8 or later.
2. Navigate to the root directory and install dependencies:
```
pip install -r requirements.txt
```

## Usage (CLI)
```
python lunar.py
```
To update sensitivity settings:
```
python lunar.py setup
```
To collect image data for annotating and training:
```
python lunar.py collect_data
```

## GUI (Multi-Page)
Jalankan:
```
python gui.py
```

### Struktur Navigasi
- Kontrol: start aimbot, status, Toggle AIM (F1), Exit (F2)
- Aim: FOV, target point, speed X/Y, smoothing
- Triggerbot: enable/disable, radius, delay
- RCS: enable/disable, strength X/Y
- Advanced: ROI size, scale, confidence (TensorRT), PID, headless mode
- Panduan: ringkasan penggunaan dan troubleshooting

### Panduan UX per Fitur
- Kontrol: gunakan Start untuk memulai; saat running, ON/OFF aim via F1 atau tombol Toggle AIM.
- Aim: kecilkan FOV untuk seleksi target lebih ketat; sesuaikan speed/smoothing agar stabil.
- Triggerbot: naikkan delay jika klik terasa terlalu agresif; radius kecil biasanya lebih presisi.
- RCS: mulai dari strength Y kecil lalu naikkan bertahap sesuai recoil.
- Advanced: kecilkan ROI size untuk FPS lebih tinggi; aktifkan Headless Mode untuk menyembunyikan jendela OpenCV.

