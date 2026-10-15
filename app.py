import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import streamlit as st


APP_NAME = "Only You"
SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "mpeg4"]
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]

MODEL_DIR = Path("models")
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


st.set_page_config(
    page_title=APP_NAME,
    page_icon="◎",
    layout="centered",
)


st.markdown(
    """
    <style>
        :root {
            --bg: #101114;
            --surface: #17191f;
            --surface-soft: #1d2027;
            --border: #2b3038;
            --border-soft: #363c46;
            --text: #f4f5f7;
            --muted: #a7adb7;
            --muted-2: #7f8792;
            --accent: #b8bec8;
            --accent-hover: #c8ced6;
            --button-text: #101114;
            --radius: 18px;
        }

        .stApp, .main {
            background: var(--bg);
            color: var(--text);
        }

        .block-container {
            max-width: 980px;
            padding-top: 2.5rem;
            padding-bottom: 2rem;
        }

        h1 {
            color: var(--text);
            letter-spacing: -0.045em;
            font-weight: 750;
            margin-bottom: 0.4rem;
        }

        .subtitle {
            color: var(--muted-2);
            font-size: 1rem;
            line-height: 1.65;
            margin-top: -0.25rem;
            margin-bottom: 1.25rem;
        }

        .info-card {
            background: var(--surface-soft);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.15rem 1.25rem;
            margin: 1.1rem 0 1.35rem 0;
        }

        .info-title {
            color: var(--text);
            font-size: 0.95rem;
            font-weight: 650;
            margin-bottom: 0.25rem;
        }

        .small-text {
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.6;
        }

        .upload-label {
            color: var(--text);
            font-size: 0.95rem;
            font-weight: 650;
            margin-bottom: 0.55rem;
        }

        div[data-testid="stFileUploader"] > label {
            display: none;
        }

        div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            min-height: 118px;
            padding: 0.75rem;
            transition: all 0.18s ease;
        }

        div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover {
            background: var(--surface-soft);
            border-color: var(--border-soft);
        }

        div[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] {
            color: var(--muted) !important;
        }

        div[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] > div {
            color: var(--text) !important;
            font-weight: 600;
        }

        div[data-testid="stFileUploaderDropzone"] small {
            color: var(--muted-2) !important;
        }

        div[data-testid="stFileUploaderDropzone"] button {
            background: #20242d !important;
            color: var(--text) !important;
            border: 1px solid var(--border-soft) !important;
            border-radius: 14px !important;
            font-weight: 650 !important;
            padding: 0.55rem 0.95rem !important;
            min-height: 44px !important;
        }

        div[data-testid="stExpander"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
            margin-top: 1.1rem;
            margin-bottom: 1.15rem;
        }

        .stButton > button {
            width: 100%;
            background: var(--accent);
            color: var(--button-text);
            border: 1px solid transparent;
            border-radius: var(--radius);
            padding: 0.85rem 1rem;
            font-weight: 750;
            transition: all 0.18s ease;
        }

        .stButton > button:hover {
            background: var(--accent-hover);
            color: var(--button-text);
        }

        div[data-testid="stAlert"] {
            background: var(--surface-soft);
            border: 1px solid var(--border);
            border-radius: 14px;
            color: var(--text);
        }

        div[data-testid="stVideo"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
        }

        .stDownloadButton > button {
            width: 100%;
            background: var(--surface-soft);
            color: var(--text);
            border: 1px solid var(--border-soft);
            border-radius: var(--radius);
            padding: 0.85rem 1rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def ensure_models_downloaded():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not YUNET_PATH.exists():
        urllib.request.urlretrieve(YUNET_URL, str(YUNET_PATH))
    if not SFACE_PATH.exists():
        urllib.request.urlretrieve(SFACE_URL, str(SFACE_PATH))


@st.cache_resource
def load_vision_models():
    ensure_models_downloaded()

    detector = cv2.FaceDetectorYN.create(
        model=str(YUNET_PATH),
        config="",
        input_size=(320, 320),
        score_threshold=0.4,
        nms_threshold=0.3,
        top_k=5000,
    )

    recognizer = cv2.FaceRecognizerSF.create(
        model=str(SFACE_PATH),
        config="",
    )

    return detector, recognizer


def decode_uploaded_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Reference image cannot be read. Use JPG, JPEG, or PNG.")
    return image


def save_uploaded_video(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        temp_video.write(uploaded_file.read())
        return temp_video.name


def convert_to_browser_compatible_mp4(input_path, output_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise ValueError("Video was processed, but conversion for browser preview failed.")
    return output_path


def detect_faces_yunet(image_bgr, detector, score_threshold=0.40):
    h, w = image_bgr.shape[:2]
    detector.setInputSize((w, h))
    detector.setScoreThreshold(score_threshold)
    _, faces = detector.detect(image_bgr)
    if faces is None:
        return []
    return faces


def extract_reference_embedding(image_bgr, detector, recognizer):
    faces = detect_faces_yunet(image_bgr, detector, score_threshold=0.60)

    if len(faces) == 0:
        raise ValueError(
            "No clear face was detected in the reference photo. Use one sharp front-facing face photo."
        )

    if len(faces) > 1:
        raise ValueError(
            f"Reference photo contains {len(faces)} faces. Please upload a photo with exactly one clear face."
        )

    aligned_face = recognizer.alignCrop(image_bgr, faces[0])
    ref_embedding = recognizer.feature(aligned_face)
    return ref_embedding


def compute_iou(boxA, boxB):
    if boxA is None or boxB is None:
        return 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return inter_area / float(areaA + areaB - inter_area)


def clamp_box(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(frame_width - 1, int(x1)))
    y1 = max(0, min(frame_height - 1, int(y1)))
    x2 = max(0, min(frame_width, int(x2)))
    y2 = max(0, min(frame_height, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_box(box, frame_width, frame_height, padding_ratio=0.25):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    expanded = (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
    return clamp_box(expanded, frame_width, frame_height)


def make_odd(value):
    value = int(value)
    return value if value % 2 == 1 else value + 1


def apply_blur_to_box(frame, box, blur_strength, blur_style="Elliptical Gaussian"):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return frame

    kernel = make_odd(blur_strength)

    if blur_style == "Pixelate / Mosaic":
        h, w = roi.shape[:2]
        blocks = max(4, int(22 - (blur_strength / 6)))
        temp = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        blurred_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = blurred_roi

    elif blur_style == "Elliptical Gaussian":
        blurred_roi = cv2.GaussianBlur(roi, (kernel, kernel), 0)
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        center = (roi.shape[1] // 2, roi.shape[0] // 2)
        axes = (max(1, roi.shape[1] // 2), max(1, roi.shape[0] // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask_3ch = cv2.merge([mask, mask, mask])
        frame[y1:y2, x1:x2] = np.where(mask_3ch == 255, blurred_roi, roi)

    else:
        blurred_roi = cv2.GaussianBlur(roi, (kernel, kernel), 0)
        frame[y1:y2, x1:x2] = blurred_roi

    return frame


def face_row_to_box(face_row):
    x, y, w, h = face_row[:4]
    return (int(x), int(y), int(x + w), int(y + h))


def init_lk_points(gray_frame, box):
    x1, y1, x2, y2 = box
    roi = gray_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    pts = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=45,
        qualityLevel=0.01,
        minDistance=4,
        blockSize=7,
    )
    if pts is None:
        return None
    pts[:, :, 0] += x1
    pts[:, :, 1] += y1
    return pts.astype(np.float32)


def track_box_lk(prev_gray, curr_gray, prev_pts, prev_box, width, height):
    if prev_gray is None or prev_pts is None or len(prev_pts) < 3:
        return prev_box, None

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
    )

    if curr_pts is None or status is None:
        return prev_box, None

    good_prev = prev_pts[status.flatten() == 1]
    good_curr = curr_pts[status.flatten() == 1]

    if len(good_curr) < 3:
        return prev_box, None

    movement = np.median(good_curr - good_prev, axis=0).reshape(2)
    dx, dy = int(movement[0]), int(movement[1])

    x1, y1, x2, y2 = prev_box
    moved_box = clamp_box((x1 + dx, y1 + dy, x2 + dx, y2 + dy), width, height)
    if moved_box is None:
        return prev_box, None

    return moved_box, good_curr.reshape(-1, 1, 2).astype(np.float32)


def process_video(
    reference_image_bgr,
    video_path,
    output_path,
    progress_callback,
    status_callback,
    det_score_threshold=0.40,
    cosine_threshold=0.36,
    blur_strength=71,
    padding_ratio=0.28,
    recognition_interval=5,
    bystander_persistence=20,
    blur_style="Elliptical Gaussian",
):
    detector, recognizer = load_vision_models()
    ref_embedding = extract_reference_embedding(reference_image_bgr, detector, recognizer)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video cannot be opened. Try MP4 with standard encoding.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise ValueError("Output video cannot be created on this system.")

    prev_gray = None
    last_target_box = None
    missing_target_frames = 0
    max_missing_grace_frames = 8

    # Menyimpan jejak wajah orang lain (bystanders) agar saat ketimpa sebagian tetap ke-blur
    active_bystanders = []

    processed_frames = 0
    target_preserved_frames = 0
    total_faces_blurred = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_yunet(frame, detector, score_threshold=det_score_threshold)
        target_idx = None

        # 1. IDENTIFIKASI TARGET
        if len(faces) > 0:
            should_run_sface = (
                processed_frames % recognition_interval == 0
                or last_target_box is None
            )

            if not should_run_sface and last_target_box is not None:
                best_iou = 0.0
                best_idx = None
                for idx, face in enumerate(faces):
                    box = face_row_to_box(face)
                    iou = compute_iou(box, last_target_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                if best_iou >= 0.40:
                    target_idx = best_idx
                else:
                    should_run_sface = True

            if should_run_sface:
                best_sim = -1.0
                best_idx = None

                for idx, face in enumerate(faces):
                    aligned = recognizer.alignCrop(frame, face)
                    feat = recognizer.feature(aligned)
                    sim = recognizer.match(
                        ref_embedding,
                        feat,
                        cv2.FaceRecognizerSF_FR_COSINE,
                    )
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = idx

                if best_sim >= cosine_threshold:
                    target_idx = best_idx
                elif last_target_box is not None and missing_target_frames <= max_missing_grace_frames:
                    for idx, face in enumerate(faces):
                        box = face_row_to_box(face)
                        if compute_iou(box, last_target_box) >= 0.45:
                            target_idx = idx
                            break

        if target_idx is not None:
            last_target_box = face_row_to_box(faces[target_idx])
            missing_target_frames = 0
            target_preserved_frames += 1
        else:
            missing_target_frames += 1
            if missing_target_frames > max_missing_grace_frames:
                last_target_box = None

        # 2. KUMPULKAN DETEKSI BYSTANDER SAAT INI
        current_bystander_boxes = []
        for idx, face in enumerate(faces):
            if idx == target_idx:
                continue
            box = face_row_to_box(face)
            # Pastikan bukan duplikat kotak target
            if last_target_box is not None and compute_iou(box, last_target_box) > 0.55:
                continue
            current_bystander_boxes.append(box)

        # 3. UPDATE TRACKER BYSTANDER (Agar yang ketutup separuh tetap ke-blur)
        updated_bystanders = []
        matched_current_indices = set()

        for b_track in active_bystanders:
            tracked_box, tracked_pts = track_box_lk(
                prev_gray,
                curr_gray,
                b_track["pts"],
                b_track["box"],
                width,
                height,
            )

            # Cek apakah tracker ini cocok dengan deteksi wajah baru di frame ini
            best_match_iou = 0.0
            best_match_idx = None
            for c_idx, c_box in enumerate(current_bystander_boxes):
                if c_idx in matched_current_indices:
                    continue
                iou = compute_iou(tracked_box, c_box)
                if iou > best_match_iou:
                    best_match_iou = iou
                    best_match_idx = c_idx

            if best_match_idx is not None and best_match_iou >= 0.25:
                # Wajah terdeteksi jelas, perbarui posisi dan titik LK
                matched_current_indices.add(best_match_idx)
                fresh_box = current_bystander_boxes[best_match_idx]
                updated_bystanders.append({
                    "box": fresh_box,
                    "pts": init_lk_points(curr_gray, fresh_box),
                    "missing": 0,
                })
            else:
                # Wajah sedang teroklusi/ketutup sebagian! Pertahankan dengan Optical Flow / posisi terakhir
                missing_count = b_track["missing"] + 1
                if missing_count <= bystander_persistence and tracked_box is not None:
                    # Jangan pertahankan jika tidak sengaja menyatu sepenuhnya dengan wajah target
                    if last_target_box is None or compute_iou(tracked_box, last_target_box) < 0.65:
                        if tracked_pts is None or len(tracked_pts) < 4:
                            tracked_pts = init_lk_points(curr_gray, tracked_box)
                        updated_bystanders.append({
                            "box": tracked_box,
                            "pts": tracked_pts,
                            "missing": missing_count,
                        })

        # Tambahkan wajah bystander yang baru muncul di frame ini
        for c_idx, c_box in enumerate(current_bystander_boxes):
            if c_idx not in matched_current_indices:
                updated_bystanders.append({
                    "box": c_box,
                    "pts": init_lk_points(curr_gray, c_box),
                    "missing": 0,
                })

        active_bystanders = updated_bystanders

        # 4. SIMPAN POTONGAN WAJAH TARGET (Target Shield agar tidak kena blur orang di belakangnya)
        target_shield_roi = None
        target_shield_box = None
        if last_target_box is not None:
            target_shield_box = clamp_box(last_target_box, width, height)
            if target_shield_box is not None:
                tx1, ty1, tx2, ty2 = target_shield_box
                target_shield_roi = frame[ty1:ty2, tx1:tx2].copy()

        # 5. BLUR SEMUA BYSTANDER (Termasuk yang sedang teroklusi/ketimpa)
        for b_track in active_bystanders:
            blur_box = expand_box(b_track["box"], width, height, padding_ratio=padding_ratio)
            if blur_box is not None:
                frame = apply_blur_to_box(
                    frame=frame,
                    box=blur_box,
                    blur_strength=blur_strength,
                    blur_style=blur_style,
                )
                total_faces_blurred += 1

        # 6. KEMBALIKAN KETAJAMAN WAJAH TARGET (Jika beririsan dengan blur bystander di belakangnya)
        if target_shield_roi is not None and target_shield_box is not None:
            tx1, ty1, tx2, ty2 = target_shield_box
            frame[ty1:ty2, tx1:tx2] = target_shield_roi

        out.write(frame)
        processed_frames += 1
        prev_gray = curr_gray.copy()

        if total_frames > 0:
            progress_callback(min(processed_frames / total_frames, 1.0))

        if processed_frames % 30 == 0:
            status_callback(
                f"Processed {processed_frames} frames | Target kept clear in {target_preserved_frames} frames | "
                f"Blurred {total_faces_blurred} bystander face instances."
            )

    cap.release()
    out.release()

    return {
        "processed_frames": processed_frames,
        "target_preserved_frames": target_preserved_frames,
        "total_faces_blurred": total_faces_blurred,
    }


st.markdown(f"# {APP_NAME}")

st.markdown(
    """
    <p class="subtitle">
        Only you I can see — upload one reference face photo and one video. Every other face in the video will be automatically blurred, leaving only your target person in focus.
    </p>
    """,
    unsafe_allow_html=True,
)


with st.container():
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">How it works</div>
            <div class="small-text">
                Upload a clear single-face photo of the person you want to keep visible. Powered by OpenCV YuNet, SFace, and Lucas-Kanade Optical Flow, the app detects and tracks all faces in every frame and blurs everyone except the matched person.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="upload-label">Reference face photo (Keep clear)</div>', unsafe_allow_html=True)
    reference_upload = st.file_uploader(
        "Reference face photo",
        type=SUPPORTED_IMAGE_TYPES,
        label_visibility="collapsed",
    )

with col2:
    st.markdown('<div class="upload-label">Input video</div>', unsafe_allow_html=True)
    video_upload = st.file_uploader(
        "Input video",
        type=SUPPORTED_VIDEO_TYPES,
        label_visibility="collapsed",
    )


with st.expander("Advanced settings"):
    blur_style = st.selectbox(
        "Blur style",
        options=["Elliptical Gaussian", "Box Gaussian", "Pixelate / Mosaic"],
        index=0,
    )

    cosine_threshold = st.slider(
        "Face recognition threshold (Cosine Similarity)",
        min_value=0.25,
        max_value=0.60,
        value=0.36,
        step=0.01,
        help="Standard SFace threshold is ~0.36. Lower values are more forgiving on angled faces; higher values are stricter.",
    )

    det_score_threshold = st.slider(
        "Face detection sensitivity (YuNet confidence)",
        min_value=0.25,
        max_value=0.80,
        value=0.40,
        step=0.05,
        help="Lower values catch partially covered/occluded faces to blur.",
    )

    bystander_persistence = st.slider(
        "Occluded face blur persistence (frames)",
        min_value=5,
        max_value=60,
        value=25,
        step=5,
        help="How many frames to keep tracking and blurring a bystander face when it gets partially hidden behind the target.",
    )

    blur_strength = st.slider(
        "Blur strength",
        min_value=21,
        max_value=101,
        value=71,
        step=2,
    )

    padding_percent = st.slider(
        "Blur padding (% of face size)",
        min_value=5,
        max_value=60,
        value=28,
        step=2,
    )

    recognition_interval = st.slider(
        "Verify identity every N frames",
        min_value=1,
        max_value=20,
        value=5,
        help="Between intervals, the target face is tracked via bounding-box IoU for faster processing.",
    )


process_button = st.button("Process video", type="primary")


if process_button:
    if reference_upload is None or video_upload is None:
        st.error("Upload both the reference photo and the video first.")
    else:
        temp_video_path = None
        output_path = None
        browser_output_path = None

        try:
            reference_image = decode_uploaded_image(reference_upload)
            temp_video_path = save_uploaded_video(video_upload)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
                output_path = temp_output.name

            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.info("Loading YuNet & SFace models and processing video...")

            result = process_video(
                reference_image_bgr=reference_image,
                video_path=temp_video_path,
                output_path=output_path,
                progress_callback=progress_bar.progress,
                status_callback=status_text.info,
                det_score_threshold=det_score_threshold,
                cosine_threshold=cosine_threshold,
                blur_strength=blur_strength,
                padding_ratio=padding_percent / 100.0,
                recognition_interval=recognition_interval,
                bystander_persistence=bystander_persistence,
                blur_style=blur_style,
            )

            progress_bar.progress(1.0)

            status_text.success(
                f"Done! Processed {result['processed_frames']} frames. Kept your target clear in "
                f"{result['target_preserved_frames']} frames and blurred {result['total_faces_blurred']} "
                f"other face instances."
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as browser_output:
                browser_output_path = browser_output.name

            convert_to_browser_compatible_mp4(output_path, browser_output_path)

            with open(browser_output_path, "rb") as output_file:
                output_bytes = output_file.read()

            st.video(output_bytes)

            st.download_button(
                label="Download video",
                data=output_bytes,
                file_name="only_you_output.mp4",
                mime="video/mp4",
            )

        except Exception as error:
            st.error(str(error))

        finally:
            for path in [temp_video_path, output_path, browser_output_path]:
                if path and os.path.exists(path):
                    os.remove(path)