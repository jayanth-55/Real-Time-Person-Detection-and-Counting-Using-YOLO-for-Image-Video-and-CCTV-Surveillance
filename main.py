"""
person_counter_allinone.py

Modes:
 - "realtime" : webcam or RTSP/CCTV stream (SOURCE can be 0, or RTSP url)
 - "image"    : single image file (SOURCE = path to image)
 - "video"    : video file (SOURCE = path to video)

Outputs:
 - Annotated images/videos saved into ./output/
 - For video mode a CSV with per-frame counts is saved: output/<name>_counts.csv
 - For realtime mode: optionally save recording to output/realtime_output.mp4

Requirements:
 pip install ultralytics opencv-python numpy
"""

from ultralytics import YOLO
import cv2
import os
import csv
import time
import numpy as np

# ---------------- CONFIG ---------------- #
MODEL_PATH = "yolov8n.pt"     # or yolov8m.pt / yolov8l.pt if you want more accuracy (but slower)
MODE = "image"             # Options: "realtime", "image", "video"
# For realtime: set to 0 for webcam, or RTSP/HTTP camera URL for CCTV
SOURCE =  "input/seq_000024.jpg"             # e.g. 0 or "rtsp://user:pass@IP:port/..." or "input/samplevideo.mp4" or "input/sample.jpg"
CONFIDENCE = 0.30            # minimum confidence threshold for detections
RESIZE = 640                  # width to resize frame for faster inference; set None to skip resizing
SAVE_REALTIME_VIDEO = False   # If True, will save realtime feed to output/realtime_output.mp4
OUTPUT_FOLDER = "output"
# ---------------------------------------- #

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"[INFO] Loading model '{MODEL_PATH}' ...")
model = YOLO(MODEL_PATH)  # load once

def _resize_keep_aspect(img, width=None):
    if width is None:
        return img
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / float(w)
    nh = int(h * scale)
    return cv2.resize(img, (width, nh))

def draw_annotations(frame, dets, confs, count, inference_time=None):
    # dets: numpy array Nx4 (x1,y1,x2,y2)
    for i, box in enumerate(dets):
        x1, y1, x2, y2 = map(int, box[:4])
        conf = float(confs[i]) if i < len(confs) else None
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"Person {conf:.2f}" if conf is not None else "Person"
        cv2.putText(frame, label, (x1, max(10, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 2)

    # Count display (large)
    cv2.putText(frame, f"Count: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)

    # small stats
    if inference_time is not None:
        cv2.putText(frame, f"{inference_time*1000:.0f} ms", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return frame

def extract_detections(results, conf_threshold=0.35):
    """
    Returns numpy arrays: boxes (N,4) and confs (N,)
    """
    if results is None or len(results.boxes) == 0:
        return np.empty((0,4)), np.array([])

    # try to fetch xyxy & conf as numpy
    try:
        boxes = results.boxes.xyxy.cpu().numpy()    # (N,4)
        confs = results.boxes.conf.cpu().numpy()    # (N,)
    except Exception:
        # fallback to list conversion
        xy = []
        cf = []
        for b in results.boxes:
            try:
                coords = np.array(b.xyxy[0].tolist(), dtype=float)
                xy.append(coords[:4])
                cf.append(float(b.conf[0]) if hasattr(b.conf, '__len__') else float(b.conf))
            except Exception:
                pass
        if len(xy) == 0:
            return np.empty((0,4)), np.array([])
        boxes = np.array(xy)
        confs = np.array(cf)

    # apply conf threshold
    keep = confs >= conf_threshold
    return boxes[keep], confs[keep]

# ---------------- IMAGE MODE ---------------- #
def process_image(source):
    img = cv2.imread(source)
    if img is None:
        print("[ERROR] Could not read image:", source)
        return
    resized = _resize_keep_aspect(img, RESIZE)
    t0 = time.time()
    results = model(resized, conf=CONFIDENCE, classes=[0], verbose=False)[0]
    t1 = time.time()
    boxes, confs = extract_detections(results, conf_threshold=CONFIDENCE)
    count = len(boxes)

    # If resized differs from original, we need to scale boxes back to original size
    if resized.shape[:2] != img.shape[:2]:
        scale_x = img.shape[1] / resized.shape[1]
        scale_y = img.shape[0] / resized.shape[0]
        boxes = boxes.copy()
        boxes[:, [0,2]] = boxes[:, [0,2]] * scale_x
        boxes[:, [1,3]] = boxes[:, [1,3]] * scale_y

    annotated = draw_annotations(img, boxes, confs, count, inference_time=(t1-t0))
    base = os.path.basename(source)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(OUTPUT_FOLDER, f"{name}_annotated{ext}")
    cv2.imwrite(out_path, annotated)
    print(f"[IMAGE] Saved annotated image to: {out_path}  | Detected persons: {count}")

# ---------------- VIDEO MODE ---------------- #
def process_video(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open video:", source)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    base = os.path.splitext(os.path.basename(source))[0]
    out_video_path = os.path.join(OUTPUT_FOLDER, f"{base}_annotated.mp4")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    csv_path = os.path.join(OUTPUT_FOLDER, f"{base}_counts.csv")

    frame_idx = 0
    counts = []

    print(f"[VIDEO] Processing. Output video -> {out_video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # maybe resize for speed
        resized = _resize_keep_aspect(frame, RESIZE)
        t0 = time.time()
        results = model(resized, conf=CONFIDENCE, classes=[0], verbose=False)[0]
        t1 = time.time()
        boxes, confs = extract_detections(results, conf_threshold=CONFIDENCE)
        # map boxes to original frame if resized
        if resized.shape[:2] != frame.shape[:2] and boxes.shape[0] > 0:
            scale_x = frame.shape[1] / resized.shape[1]
            scale_y = frame.shape[0] / resized.shape[0]
            boxes[:, [0,2]] = boxes[:, [0,2]] * scale_x
            boxes[:, [1,3]] = boxes[:, [1,3]] * scale_y

        count = len(boxes)
        counts.append((frame_idx, count))
        annotated = draw_annotations(frame, boxes, confs, count, inference_time=(t1-t0))
        out.write(annotated)

        # show (optional), comment if you process large videos headless
        cv2.imshow("Video - Person Detection (press q to stop)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # write CSV
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "person_count"])
        writer.writerows(counts)

    # stats
    counts_only = [c for _, c in counts]
    avg = np.mean(counts_only) if counts_only else 0
    mx = np.max(counts_only) if counts_only else 0
    print(f"[VIDEO] Done. Saved annotated video to: {out_video_path}")
    print(f"[VIDEO] Counts CSV saved to: {csv_path}")
    print(f"[VIDEO] Frames: {len(counts)}  | Avg count: {avg:.2f}  | Max count: {mx}")

# ---------------- REALTIME MODE ---------------- #
def process_realtime(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam / CCTV stream:", source)
        return

    # prepare writer if saving
    save_writer = None
    if SAVE_REALTIME_VIDEO:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        out_path = os.path.join(OUTPUT_FOLDER, "realtime_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        save_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[REALTIME] saving stream to: {out_path}")

    print("[REALTIME] Press 'q' to quit.")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[REALTIME] Frame grab failed, exiting.")
            break

        resized = _resize_keep_aspect(frame, RESIZE)
        t0 = time.time()
        results = model(resized, conf=CONFIDENCE, classes=[0], verbose=False)[0]
        t1 = time.time()
        boxes, confs = extract_detections(results, conf_threshold=CONFIDENCE)
        if resized.shape[:2] != frame.shape[:2] and boxes.shape[0] > 0:
            scale_x = frame.shape[1] / resized.shape[1]
            scale_y = frame.shape[0] / resized.shape[0]
            boxes[:, [0,2]] = boxes[:, [0,2]] * scale_x
            boxes[:, [1,3]] = boxes[:, [1,3]] * scale_y

        count = len(boxes)
        annotated = draw_annotations(frame, boxes, confs, count, inference_time=(t1-t0))

        # show and optionally save
        cv2.imshow("Real-Time Person Detection", annotated)
        if save_writer is not None:
            save_writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1

    cap.release()
    if save_writer:
        save_writer.release()
    cv2.destroyAllWindows()
    print("[REALTIME] Stopped.")

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    print("[INFO] MODE =", MODE)
    if MODE == "image":
        if not isinstance(SOURCE, str):
            print("[ERROR] For image mode, set SOURCE to image path.")
        else:
            process_image(SOURCE)
    elif MODE == "video":
        if not isinstance(SOURCE, str):
            print("[ERROR] For video mode, set SOURCE to video path.")
        else:
            process_video(SOURCE)
    elif MODE == "realtime":
        process_realtime(SOURCE)
    else:
        print("[ERROR] Unknown MODE. Choose 'image', 'video', or 'realtime'.")
