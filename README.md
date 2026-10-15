# Only You

Only You is a Streamlit-based web application that automatically blurs every face in a video **except** the selected target person using one reference face photo.

The app takes two inputs:

1. A reference face photo containing the person you want to keep in focus
2. A video containing multiple people

Only You then processes the video, identifies the person matching the reference photo, and applies blur to everyone else in the scene.

## Concept & Companion Project

**Only You** (inspired by the phrase *"Only you I can see"*) is the direct inverse of [**BlurIn**](https://github.com/Raxzell/BlurIn):

* **BlurIn** blurs **only** the matched person and keeps everyone else clear.
* **Only You** keeps **only** the matched person clear and blurs everyone else.

While BlurIn relies on classical feature matching (Haar Cascade + SIFT + FLANN), **Only You** upgrades the pipeline to lightweight deep neural networks (**OpenCV YuNet & SFace**) combined with **Lucas-Kanade Optical Flow** to handle multi-face crowd anonymization and partial face occlusion reliably.

## Features

* Upload one reference face photo of the person to keep clear
* Upload one input video
* Automatically download and cache lightweight ONNX vision models (`YuNet` and `SFace`) on first run
* Detect all faces in every frame and verify the target identity via Cosine Similarity
* Blur all non-target (bystander) faces automatically
* Track partially occluded bystander faces using Lucas-Kanade Optical Flow so they stay blurred even when passing behind the target
* Protect the target face region with a restoration shield so bystander blur never bleeds onto the target
* Multiple blur styles: **Elliptical Gaussian**, **Box Gaussian**, and **Pixelate / Mosaic**
* Preview and download the browser-compatible output video
* Clean semi-monochrome Streamlit interface

## Tech Stack

* Python
* Streamlit
* OpenCV (`FaceDetectorYN` / YuNet, `FaceRecognizerSF` / SFace, Optical Flow)
* NumPy
* ImageIO-FFmpeg

## How It Works

Only You uses a hybrid deep learning and optical flow computer vision pipeline:

1. **Reference Face Detection & Alignment (YuNet + SFace)**
   The uploaded reference image is scanned using OpenCV's YuNet detector. Once a single clear face is verified, SFace aligns and crops the face using 5 facial landmarks and extracts a 128-dimensional face embedding.

2. **Multi-Face Detection per Frame (YuNet)**
   Every video frame is scanned with YuNet to locate all visible faces across different scales and angles.

3. **Target Identification (SFace Cosine Similarity & Spatial IoU)**
   Detected faces are compared against the reference embedding using Cosine Similarity. Between verification intervals, bounding-box Intersection over Union (IoU) is used to maintain smooth lock on the target face.

4. **Bystander Persistence Tracking (Lucas-Kanade Optical Flow)**
   Every non-target face is tracked across frames using Shi-Tomasi corner features (`goodFeaturesToTrack`) and Lucas-Kanade Optical Flow (`calcOpticalFlowPyrLK`). If a bystander face gets partially covered or occluded behind another person, the tracker maintains the blur box for a configurable number of frames.

5. **Target Shield & Selective Blurring**
   Before blurring all non-target faces (using Elliptical Gaussian, Box Gaussian, or Pixelation), the target's clear face region is buffered and restored afterward so overlapping blur regions never obscure the target person.

6. **Browser-Compatible Video Encoding**
   The processed video is re-encoded with `libx264` (`yuv420p`) via FFmpeg for direct playback and download inside the browser.

## Project Structure

```text
OnlyYou/
├── app.py
├── generate_sample.py
├── requirements.txt
└── README.md