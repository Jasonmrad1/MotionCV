import cv2
import os
import argparse


def convert_avi_to_mp4(input_path, output_path, fps=30):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {input_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    print(f"[OK] Converted {os.path.basename(input_path)} -> {os.path.basename(output_path)} ({frame_count} frames)")
    return True


def batch_convert(directory):
    if not os.path.exists(directory):
        print("[ERROR] Directory does not exist")
        return

    output_dir = os.path.join(directory, "mp4_converted")
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(directory) if f.lower().endswith(".avi")]

    if not files:
        print("[INFO] No AVI files found")
        return

    print(f"[INFO] Found {len(files)} AVI files")

    for file in files:
        input_path = os.path.join(directory, file)
        output_name = os.path.splitext(file)[0] + ".mp4"
        output_path = os.path.join(output_dir, output_name)

        convert_avi_to_mp4(input_path, output_path)

    print("\n[DONE] All conversions completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing AVI files")

    args = parser.parse_args()

    batch_convert(args.dir)