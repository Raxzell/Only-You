import urllib.request
from pathlib import Path
import cv2
import numpy as np

# Buat folder untuk menampung aset sample
SAMPLE_DIR = Path("samples")
SAMPLE_DIR.mkdir(exist_ok=True)

# URL foto wajah publik (dari repo sampel OpenCV / InsightFace)
# Kita ambil 1 gambar berisi beberapa wajah, lalu kita crop masing-masing wajahnya
SOURCE_IMG_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
GROUP_IMG_URL = "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg"
GROUP_IMG_URL_2 = "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg"


def download_image(url, save_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        cv2.imwrite(str(save_path), img)
        return img


def main():
    print("1. Mengunduh foto wajah sampel...")
    p1_path = SAMPLE_DIR / "person1_target.jpg"
    p2_path = SAMPLE_DIR / "person2_other.jpg"
    p3_path = SAMPLE_DIR / "person3_other.jpg"

    img_target = download_image(GROUP_IMG_URL, p1_path)
    img_other1 = download_image(GROUP_IMG_URL_2, p2_path)
    img_other2 = download_image(SOURCE_IMG_URL, p3_path)

    # Simpan foto referensi utama yang siap di-upload ke Streamlit
    ref_output = Path("sample_reference.jpg")
    cv2.imwrite(str(ref_output), img_target)
    print(f"   -> Foto referensi disimpan di: {ref_output}")

    # Resize ketiga foto agar proporsional saat bergerak di dalam video
    h_card, w_card = 240, 200
    card_target = cv2.resize(img_target, (w_card, h_card))
    card_other1 = cv2.resize(img_other1, (w_card, h_card))
    card_other2 = cv2.resize(img_other2, (w_card, h_card))

    print("2. Merender video sample_crowd_video.mp4 (6 detik @ 30 FPS)...")
    video_output = Path("sample_crowd_video.mp4")
    width, height = 960, 540
    fps = 30
    total_frames = fps * 6  # 180 frames (6 detik)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_output), fourcc, fps, (width, height))

    for f in range(total_frames):
        # Buat background abu-abu gelap dengan grid halus
        frame = np.full((height, width, 3), 28, dtype=np.uint8)
        for gx in range(0, width, 60):
            cv2.line(frame, (gx, 0), (gx, height), (38, 38, 42), 1)
        for gy in range(0, height, 60):
            cv2.line(frame, (0, gy), (width, gy), (38, 38, 42), 1)

        t = f / float(fps)

        # Pergerakan halus 3 orang di dalam frame (menggunakan fungsi sinus/cosinus)
        # Orang 1 (TARGET - bergerak di tengah)
        x1 = int(380 + 140 * np.sin(t * 1.5))
        y1 = int(150 + 40 * np.cos(t * 2.0))

        # Orang 2 (BYSTANDER 1 - bergerak di sisi kiri-tengah)
        x2 = int(90 + 70 * np.cos(t * 1.8))
        y2 = int(120 + 60 * np.sin(t * 1.4))

        # Orang 3 (BYSTANDER 2 - bergerak di sisi kanan)
        x3 = int(660 + 65 * np.sin(t * 2.2))
        y3 = int(170 + 50 * np.cos(t * 1.6))

        # Tempelkan ketiga wajah ke dalam frame
        for (x, y, card) in [(x2, y2, card_other1), (x3, y3, card_other2), (x1, y1, card_target)]:
            x = max(0, min(width - w_card, x))
            y = max(0, min(height - h_card, y))
            frame[y : y + h_card, x : x + w_card] = card

        out.write(frame)

    out.release()
    print(f"   -> Video sampel berhasil dibuat: {video_output}")
    print("\nSelesai! Sekarang lu tinggal upload 'sample_reference.jpg' dan 'sample_crowd_video.mp4' di aplikasi Streamlit.")


if __name__ == "__main__":
    main()