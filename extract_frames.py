import cv2
import os
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video', help='Path to input video')
    ap.add_argument('--n', type=int, default=30, help='Max frames to extract')
    ap.add_argument('--size', type=int, nargs=2, default=[320, 240],
                    metavar=('W', 'H'))
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✓ Video: {total} total frames")

    os.makedirs('frames', exist_ok=True)
    i = 0
    while i < args.n:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, tuple(args.size))
        cv2.imwrite(f'frames/frame_{i:04d}.png', frame)
        i += 1
    cap.release()
    print(f"✓ {i} frames extracted ({args.size[0]}×{args.size[1]}) → ./frames/")

if __name__ == '__main__':
    main()