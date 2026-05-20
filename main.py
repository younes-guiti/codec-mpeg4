

import argparse, os, glob, cv2
import numpy as np

from encoder import encode_video, decode_video
from evaluation import (
    compute_metrics, visualise_pipeline,
    plot_qf_vs_compression, plot_gop_vs_compression,
    print_comparison_table,
)


def load_frames(frames_dir: str) -> list:
    paths = sorted(
        p for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        for p in glob.glob(os.path.join(frames_dir, ext))
    )
    if not paths:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")
    frames = [cv2.imread(p) for p in paths]
    print(f"Loaded {len(frames)} frames from {frames_dir}")
    return frames


def generate_synthetic_frames(n: int = 16, size: tuple = (128, 128)) -> list:
    frames = []
    h, w = size
    for i in range(n):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        t = i / n
        f[:,:,0] = (np.linspace(30, 180, w)[None,:] * (0.5 + 0.5*t)).astype(np.uint8)
        f[:,:,1] = (np.linspace(50, 200, h)[:,None] * (1 - 0.3*t)).astype(np.uint8)
        f[:,:,2] = np.full((h, w), 100 + int(60*np.sin(2*np.pi*i/n)), np.uint8)
        x = int((w - 20) * (i / n))
        cv2.rectangle(f, (x, 20), (x+20, 40), (255, 255, 0), -1)
        frames.append(f)
    return frames


def main():
    ap = argparse.ArgumentParser(description='MPEG-4 Simplified Encoder Pipeline')
    ap.add_argument('--frames_dir', type=str,   default=None)
    ap.add_argument('--output',     type=str,   default='video.bin')
    ap.add_argument('--decode',     type=str,   default=None)
    ap.add_argument('--output_dir', type=str,   default='decoded_frames')
    ap.add_argument('--qf',         type=float, default=50)
    ap.add_argument('--gop',        type=int,   default=8)
    ap.add_argument('--search',     type=int,   default=8)
    ap.add_argument('--analyse',    action='store_true')
    ap.add_argument('--demo',       action='store_true')
    args = ap.parse_args()

    if args.decode:
        print(f"\n▶ Decoding {args.decode} …")
        frames = decode_video(args.decode)
        os.makedirs(args.output_dir, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(args.output_dir, f'frame_{i:04d}.png'), f)
        print(f"✓ Decoded {len(frames)} frames → {args.output_dir}/")
        return

    if args.demo or args.frames_dir is None:
        print("\n▶ Demo mode — synthetic frames")
        frames_bgr = generate_synthetic_frames(n=16, size=(128, 128))
    else:
        frames_bgr = load_frames(args.frames_dir)

    print(f"\n▶ Encoding {len(frames_bgr)} frames "
          f"(QF={args.qf}, GOP={args.gop}, search=±{args.search}) …\n")

    compressed_bytes = encode_video(
        frames_bgr, args.output,
        qf=args.qf, gop=args.gop, search=args.search,
    )

    print("\n▶ Decoding for evaluation …")
    recon_bgr = decode_video(args.output)

    print_comparison_table(frames_bgr, args.gop, args.qf,
                           compressed_bytes, recon_bgr)

    print("▶ Generating pipeline visualisation …")
    visualise_pipeline(frames_bgr, recon_bgr, args.gop, args.qf,
                       out_path='pipeline_visualisation.png')

    if args.analyse:
        print("\n▶ QF analysis …")
        plot_qf_vs_compression(frames_bgr, gop=args.gop)
        print("\n▶ GOP analysis …")
        plot_gop_vs_compression(frames_bgr, qf=args.qf)

    print("\n✓ All done!")


if __name__ == '__main__':
    main()
