

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import cv2
import os
from scipy.fft import dctn
from skimage.metrics import structural_similarity as skimage_ssim

from encoder import (
    bgr_to_ycbcr, get_quant_matrix,
    dct_quantise_channel, idct_dequantise_channel,
    motion_estimation, motion_compensate, MB, ZIGZAG_IDX
)




def psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(255**2 / mse)


def ssim(orig: np.ndarray, recon: np.ndarray) -> float:
    if orig.ndim == 3:
        return skimage_ssim(orig, recon, channel_axis=2, data_range=255)
    return skimage_ssim(orig, recon, data_range=255)


def compute_metrics(originals: list, reconstructed: list,
                    compressed_bytes: int, gop: int = 8) -> dict:
    total_orig = sum(f.nbytes for f in originals)
    cr = total_orig / compressed_bytes
    psnr_i, psnr_p, ssim_vals, psnr_per_frame = [], [], [], []
    for idx, (o, r) in enumerate(zip(originals, reconstructed)):
        p = psnr(o, r); s = ssim(o, r)
        psnr_per_frame.append(p); ssim_vals.append(s)
        (psnr_i if idx % gop == 0 else psnr_p).append(p)
    return {
        'compression_ratio': cr,
        'avg_psnr':          np.mean(psnr_per_frame),
        'avg_ssim':          np.mean(ssim_vals),
        'avg_psnr_i':        np.mean(psnr_i) if psnr_i else float('nan'),
        'avg_psnr_p':        np.mean(psnr_p) if psnr_p else float('nan'),
        'psnr_per_frame':    psnr_per_frame,
        'ssim_per_frame':    ssim_vals,
    }




def visualise_pipeline(frames_bgr: list, recon_bgr: list, gop: int, qf: float,
                        out_path: str = 'pipeline_visualisation.png'):

    fig = plt.figure(figsize=(24, 34), facecolor='#0d0d0d')
    fig.suptitle('MPEG-4 Simplified Encoder — Pipeline Visualisation',
                 fontsize=22, color='white', weight='bold', y=0.99)

    # 7 rows now (added orig/recon comparison row)
    gs = GridSpec(7, 4, figure=fig, hspace=0.65, wspace=0.38,
                  left=0.05, right=0.97, top=0.96, bottom=0.02,
                  width_ratios=[1, 1, 1, 1.2])

    TC = '#f0c040'; TX = '#cccccc'; GM = 'gray'

    def _ax(r, c, cs=1): return fig.add_subplot(gs[r, c:c+cs])
    def _t(ax, t):        ax.set_title(t, color=TC, fontsize=9, pad=5, weight='bold')
    def _off(ax):         ax.axis('off')

    for k in range(min(4, len(frames_bgr))):
        ax = _ax(0, k)
        ax.imshow(cv2.cvtColor(frames_bgr[k], cv2.COLOR_BGR2RGB))
        _t(ax, f'Frame {k}  [{"I" if k % gop == 0 else "P"}]')
        _off(ax)


    Y, Cb, Cr = bgr_to_ycbcr(frames_bgr[0])
    for k, (img, name, cmap) in enumerate([
            (Y,  'Y — Luma', GM),
            (Cb, 'Cb (↓2)', 'Blues'),
            (Cr, 'Cr (↓2)', 'Reds'),
            (cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2RGB), 'RGB original', None)]):
        ax = _ax(1, k); ax.imshow(img, cmap=cmap); _t(ax, name); _off(ax)

    
    h0, w0 = Y.shape
    cy8 = min(((h0 // 2) // 8) * 8, h0 - 8)
    cx8 = min(((w0 // 2) // 8) * 8, w0 - 8)

    raw_px  = Y[cy8:cy8+8, cx8:cx8+8].copy().astype(np.float32)
    Q       = get_quant_matrix(qf, is_luma=True)
    dct_blk = dctn(raw_px - 128.0, norm='ortho')
    q_blk   = np.round(dct_blk / Q)
    from scipy.fft import idctn as _idctn
    recon_px = np.clip(_idctn(q_blk * Q, norm='ortho') + 128.0, 0, 255)

    for k, (img, name, cmap, show_zz) in enumerate([
            (raw_px,   'Raw 8×8 block',      GM,        False),
            (dct_blk,  'DCT coefficients',   'seismic',  False),
            (q_blk,    'Quantised + Zigzag', 'seismic',  True),
            (recon_px, 'Reconstructed block', GM,        False)]):
        ax = _ax(2, k)
        im = ax.imshow(img, cmap=cmap, interpolation='nearest')
        if show_zz:
            for fi, pos in enumerate(ZIGZAG_IDX):
                r_, c_ = divmod(pos, 8)
                ax.text(c_, r_, str(fi), ha='center', va='center',
                        color='yellow', fontsize=5, fontweight='bold')
        _t(ax, name); _off(ax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    gs3   = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3, :], wspace=0.28)
    ax_mv = fig.add_subplot(gs3[0])
    ax_rs = fig.add_subplot(gs3[1])

    p_idx = next((i for i in range(1, len(frames_bgr)) if i % gop != 0), None)
    if p_idx is not None:
        Y_cur, _, _ = bgr_to_ycbcr(frames_bgr[p_idx])
        Y_ref, _, _ = bgr_to_ycbcr(frames_bgr[p_idx - 1])
        hm = (Y_cur.shape[0] // MB) * MB
        wm = (Y_cur.shape[1] // MB) * MB
        Yc = Y_cur[:hm, :wm].astype(np.float32)
        Yr = Y_ref[:hm, :wm].astype(np.float32)

        mvs     = motion_estimation(Yc, Yr, search=8)
        pred_Y  = motion_compensate(Yr, mvs)
        residual = np.abs(Yc - pred_Y)

       
        ax_mv.imshow(cv2.cvtColor(frames_bgr[p_idx], cv2.COLOR_BGR2RGB),
                     aspect='auto', alpha=0.85)
        ax_mv.set_xlim(0, frames_bgr[p_idx].shape[1])
        ax_mv.set_ylim(frames_bgr[p_idx].shape[0], 0)

        mv_mag  = np.hypot(mvs[:,:,0].astype(float), mvs[:,:,1].astype(float))
        max_mag = max(mv_mag.max(), 0.5)
   
        asc = 0.7 * MB / max_mag

       
        for bi in range(mvs.shape[0]):
            for bj in range(mvs.shape[1]):
                dy, dx = float(mvs[bi,bj,0]), float(mvs[bi,bj,1])
                cx = bj * MB + MB // 2
                cy = bi * MB + MB // 2
                mag = np.hypot(dy, dx)
                t   = np.clip(mag / max_mag, 0, 1)
                
                color = (min(1.0, t * 2),
                         max(0.0, 1.0 - abs(t - 0.5) * 2),
                         max(0.0, 1.0 - t * 2))
                if mag < 0.5:
                    ax_mv.plot(cx, cy, 'o', color='#555555',
                               markersize=2, alpha=0.6)
                else:
                    ax_mv.annotate('',
                        xy=(cx + dx*asc, cy + dy*asc),
                        xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=2.0, mutation_scale=14))

        
        legend_patches = [
            mpatches.Patch(color=(0,0,1), label='No motion (0 px)'),
            mpatches.Patch(color=(0,1,0), label=f'Medium ({max_mag/2:.1f} px)'),
            mpatches.Patch(color=(1,0,0), label=f'Max ({max_mag:.1f} px)'),
        ]
        ax_mv.legend(handles=legend_patches, loc='lower right',
                     facecolor='#1a1a1a', labelcolor=TX,
                     fontsize=7, framealpha=0.85)
        _t(ax_mv, f'Motion Vectors — P-frame {p_idx}  '
                  f'(scale ×{asc:.2f}, max={max_mag:.1f} px, '
                  f'search=±8)')
        ax_mv.axis('off')

        
        vmax_r = max(float(np.percentile(residual, 99)), 1.0)
        im_r   = ax_rs.imshow(residual, cmap='inferno', vmin=0, vmax=vmax_r,
                               aspect='auto')
        cbar = plt.colorbar(im_r, ax=ax_rs, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=TX, labelsize=8)
        mean_r = residual.mean(); std_r = residual.std()
        _t(ax_rs, f'MC Residual |Y − Ŷ|  '
                  f'mean={mean_r:.1f}  std={std_r:.1f}  vmax={vmax_r:.0f}')
        ax_rs.axis('off')
    else:
        for ax in (ax_mv, ax_rs):
            ax.text(0.5, 0.5, 'Not enough frames for P-frame demo',
                    ha='center', va='center', color=TX, fontsize=11,
                    transform=ax.transAxes); ax.axis('off')

   
    gs4 = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[4, :3],
                                  hspace=0.05, wspace=0.08)
    compare_indices = [0,
                       min(1, len(frames_bgr)-1),
                       min(2, len(frames_bgr)-1)]
    for col, idx in enumerate(compare_indices):
        ax_o = fig.add_subplot(gs4[0, col])
        ax_r = fig.add_subplot(gs4[1, col])
        ax_o.imshow(cv2.cvtColor(frames_bgr[idx], cv2.COLOR_BGR2RGB))
        ax_r.imshow(cv2.cvtColor(recon_bgr[idx],  cv2.COLOR_BGR2RGB))
        ftype = 'I' if idx % gop == 0 else 'P'
        p = psnr(frames_bgr[idx], recon_bgr[idx])
        s = ssim(frames_bgr[idx], recon_bgr[idx])
        ax_o.set_title(f'Original  [{ftype}]  frame {idx}',
                       color='#80ff80', fontsize=8, pad=3, weight='bold')
        ax_r.set_title(f'Recon  {p:.1f} dB  SSIM {s:.3f}',
                       color='#ff9999', fontsize=8, pad=3)
        ax_o.axis('off'); ax_r.axis('off')

    
    ax_e  = _ax(4, 3)
    diff  = np.abs(frames_bgr[0].astype(float) - recon_bgr[0].astype(float)).mean(axis=2)
    vmax_e = max(float(np.percentile(diff, 99)), 2.0)
    im_e  = ax_e.imshow(diff, cmap='magma', vmin=0, vmax=vmax_e)
    cbar_e = plt.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04)
    cbar_e.ax.yaxis.set_tick_params(color=TX, labelsize=8)
    _t(ax_e, f'Pixel error map — frame 0\n(vmax={vmax_e:.1f}, 99th pct)')
    _off(ax_e)

   
    psnr_vals  = [psnr(o, r) for o, r in zip(frames_bgr, recon_bgr)]
    ssim_vals2 = [ssim(o, r) for o, r in zip(frames_bgr, recon_bgr)]
    colors     = ['#f0c040' if i % gop == 0 else '#4fc3f7'
                  for i in range(len(psnr_vals))]

    ax_b = _ax(5, 0, 3)
    bars = ax_b.bar(range(len(psnr_vals)), psnr_vals,
                    color=colors, edgecolor='none', width=0.85)
    ax_b.set_facecolor('#1a1a1a')
    ax_b.tick_params(colors=TX, labelsize=9)
    ax_b.set_xlabel('Frame index', color=TX, fontsize=10)
    ax_b.set_ylabel('PSNR (dB)', color=TX, fontsize=10)
    ax_b.set_xlim(-0.5, len(psnr_vals) - 0.5)
  
    if any(i % gop != 0 for i in range(len(psnr_vals))):
        pp_vals = [(i, psnr_vals[i]) for i in range(len(psnr_vals)) if i % gop != 0]
        worst_i, worst_v = min(pp_vals, key=lambda x: x[1])
        ax_b.annotate(f'min P\n{worst_v:.1f} dB',
                      xy=(worst_i, worst_v), xytext=(worst_i, worst_v + 2),
                      color='#ff9999', fontsize=8, ha='center',
                      arrowprops=dict(arrowstyle='->', color='#ff9999', lw=1.2))
    for sp in ax_b.spines.values(): sp.set_edgecolor('#444')
    pi_v = [psnr_vals[i] for i in range(len(psnr_vals)) if i%gop==0]
    pp_v = [psnr_vals[i] for i in range(len(psnr_vals)) if i%gop!=0]
    ax_b.legend(handles=[
        mpatches.Patch(color='#f0c040',
                       label=f'I-frame  avg {np.mean(pi_v):.1f} dB' if pi_v else 'I'),
        mpatches.Patch(color='#4fc3f7',
                       label=f'P-frame  avg {np.mean(pp_v):.1f} dB' if pp_v else 'P'),
    ], facecolor='#1a1a1a', labelcolor=TX, fontsize=9, loc='upper right')
    _t(ax_b, 'PSNR per Frame  (gold = I-frame, blue = P-frame)')

   
    ax_s = _ax(5, 3)
    ax_s.set_facecolor('#1a1a1a'); _off(ax_s)

    n_i  = sum(1 for i in range(len(frames_bgr)) if i%gop==0)
    n_p  = len(frames_bgr) - n_i
    torig = sum(f.nbytes for f in frames_bgr)

    rows = [
        ('SUMMARY', '',              True),
        ('─'*18,    '',              False),
        ('Frames',  str(len(frames_bgr)), False),
        ('I / P',   f'{n_i} / {n_p}', False),
        ('GOP',     str(gop),        False),
        ('QF',      str(qf),         False),
        ('',        '',              False),
        ('PSNR all',  f'{np.mean(psnr_vals):.2f} dB', False),
        ('PSNR I',    f'{np.mean(pi_v):.2f} dB' if pi_v else 'N/A', False),
        ('PSNR P',    f'{np.mean(pp_v):.2f} dB' if pp_v else 'N/A', False),
        ('SSIM',    f'{np.mean(ssim_vals2):.4f}', False),
        ('',        '',              False),
        ('Raw size', f'{torig/1024:.0f} KB', False),
    ]

    y0, dy = 0.97, 0.076
    for i, (lbl, val, bold) in enumerate(rows):
        y = y0 - i * dy
        if lbl == '': continue
        if bold:
            ax_s.text(0.5, y, lbl, transform=ax_s.transAxes,
                      color=TC, fontsize=13, weight='bold', va='top',
                      ha='center', family='monospace')
        elif lbl.startswith('─'):
            ax_s.plot([0.02, 0.98], [y - 0.01, y - 0.01],
                      color='#555', lw=0.8, transform=ax_s.transAxes,
                      clip_on=False)
        else:
            ax_s.text(0.04, y, lbl, transform=ax_s.transAxes,
                      color='#aaaaaa', fontsize=11, va='top', family='monospace')
            ax_s.text(0.96, y, val, transform=ax_s.transAxes,
                      color='#ffffff', fontsize=11, va='top', ha='right',
                      family='monospace', weight='bold')

    gs6 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[6, :], wspace=0.3)
    ax_pie = fig.add_subplot(gs6[0])
    ax_txt = fig.add_subplot(gs6[1])

    if n_i > 0 and n_p > 0:
        # Approximate bytes per frame type from PSNR (higher PSNR I → more bits)
        psnr_i_avg = np.mean(pi_v)
        psnr_p_avg = np.mean(pp_v)
        ax_pie.set_facecolor('#1a1a1a')
        wedges, texts, autotexts = ax_pie.pie(
            [n_i, n_p],
            labels=['I-frames', 'P-frames'],
            autopct='%1.0f%%',
            colors=['#f0c040', '#4fc3f7'],
            startangle=90,
            textprops={'color': TX, 'fontsize': 10},
            wedgeprops={'edgecolor': '#0d0d0d', 'linewidth': 2},
        )
        for at in autotexts: at.set_color('black'); at.set_fontsize(10)
        _t(ax_pie, 'Frame type distribution')

        
        ax_txt.set_facecolor('#1a1a1a'); _off(ax_txt)
        insight = (
            f"  Delta-coded MVs reduce bitstream overhead\n"
            f"  by predicting each MV from its neighbour.\n\n"
            f"  I-frames  : avg PSNR = {psnr_i_avg:.1f} dB\n"
            f"  P-frames  : avg PSNR = {psnr_p_avg:.1f} dB\n\n"
            f"  Pipeline  : YCbCr 4:2:0  →  8×8 DCT\n"
            f"              →  Zigzag + RLE  →  zlib"
        )
        ax_txt.text(0.05, 0.80, insight,
                    transform=ax_txt.transAxes,
                    color=TX, fontsize=11, va='top',
                    family='monospace', linespacing=1.6)
        _t(ax_txt, 'Encoding pipeline summary')
    else:
        for ax in (ax_pie, ax_txt): ax.axis('off')

    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Saved visualisation → {out_path}")




def plot_qf_vs_compression(frames_bgr: list, qf_range=None, gop=8,
                            out_path='qf_vs_compression.png'):
    import tempfile
    from encoder import encode_video, decode_video
    if qf_range is None:
        qf_range = [10, 20, 30, 50, 70, 90]
    ratios, pi_m, pp_m = [], [], []
    total_orig = sum(f.nbytes for f in frames_bgr)
    for qf in qf_range:
        tmp  = tempfile.mktemp(suffix='.bin')
        comp = encode_video(frames_bgr, tmp, qf=qf, gop=gop)
        rec  = decode_video(tmp); os.remove(tmp)
        ratios.append(total_orig / comp)
        pi = [psnr(frames_bgr[i], rec[i]) for i in range(len(frames_bgr)) if i%gop==0]
        pp = [psnr(frames_bgr[i], rec[i]) for i in range(len(frames_bgr)) if i%gop!=0]
        pi_m.append(np.mean(pi) if pi else float('nan'))
        pp_m.append(np.mean(pp) if pp else float('nan'))
    fig, axes = plt.subplots(1, 3, figsize=(16,4), facecolor='#0d0d0d')
    for ax, (col, ylabel, title, data) in zip(axes, [
            ('#f0c040','Compression Ratio','Compression Ratio vs QF',ratios),
            ('#4fc3f7','Avg PSNR I (dB)',  'I-frame PSNR vs QF',     pi_m),
            ('#ef9a9a','Avg PSNR P (dB)',  'P-frame PSNR vs QF',     pp_m)]):
        ax.set_facecolor('#1a1a1a')
        for sp in ax.spines.values(): sp.set_edgecolor('#444')
        ax.tick_params(colors='#ccc')
        ax.plot(qf_range, data, 'o-', color=col, lw=2)
        ax.set_xlabel('Quantisation Factor', color='#ccc')
        ax.set_ylabel(ylabel, color='#ccc')
        ax.set_title(title, color=col, weight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close()
    print(f"✓ Saved QF analysis → {out_path}")


def plot_gop_vs_compression(frames_bgr: list, gop_range=None, qf=50,
                             out_path='gop_vs_compression.png'):
    import tempfile
    from encoder import encode_video, decode_video
    if gop_range is None:
        gop_range = [1, 2, 4, 8, 16]
    ratios, pm, sm = [], [], []
    total_orig = sum(f.nbytes for f in frames_bgr)
    for gop in gop_range:
        tmp  = tempfile.mktemp(suffix='.bin')
        comp = encode_video(frames_bgr, tmp, qf=qf, gop=gop)
        rec  = decode_video(tmp); os.remove(tmp)
        ratios.append(total_orig / comp)
        pm.append(np.mean([psnr(o,r) for o,r in zip(frames_bgr,rec)]))
        sm.append(np.mean([ssim(o,r) for o,r in zip(frames_bgr,rec)]))
    fig, axes = plt.subplots(1, 3, figsize=(16,4), facecolor='#0d0d0d')
    for ax, (col, ylabel, title, data) in zip(axes, [
            ('#a5d6a7','Compression Ratio','Compression Ratio vs GOP',ratios),
            ('#ef9a9a','Avg PSNR (dB)',    'PSNR vs GOP',             pm),
            ('#ce93d8','Avg SSIM',         'SSIM vs GOP',             sm)]):
        ax.set_facecolor('#1a1a1a')
        for sp in ax.spines.values(): sp.set_edgecolor('#444')
        ax.tick_params(colors='#ccc')
        ax.plot(gop_range, data, 'o-', color=col, lw=2)
        ax.set_xlabel('GOP Size', color='#ccc')
        ax.set_ylabel(ylabel, color='#ccc')
        ax.set_title(title, color=col, weight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close()
    print(f"✓ Saved GOP analysis → {out_path}")


def print_comparison_table(frames_bgr: list, gop: int, qf: float,
                            compressed_bytes: int, recons: list):
    metrics  = compute_metrics(frames_bgr, recons, compressed_bytes, gop)
    n_i      = sum(1 for i in range(len(frames_bgr)) if i%gop==0)
    n_p      = len(frames_bgr) - n_i
    total_kb = sum(f.nbytes for f in frames_bgr) / 1024
    comp_kb  = compressed_bytes / 1024
    print(f"\n{'═'*52}")
    print(f"{'EVALUATION SUMMARY':^52}")
    print(f"{'═'*52}")
    print(f"  Frames total          : {len(frames_bgr)}")
    print(f"  I-frames / P-frames   : {n_i} / {n_p}")
    print(f"  GOP size              : {gop}")
    print(f"  Quality Factor (QF)   : {qf}")
    print(f"  Original size         : {total_kb:.1f} KB")
    print(f"  Compressed size       : {comp_kb:.1f} KB")
    print(f"  Compression ratio     : {metrics['compression_ratio']:.2f}×")
    print(f"  Avg PSNR (all)        : {metrics['avg_psnr']:.2f} dB")
    print(f"  Avg PSNR  I-frames    : {metrics['avg_psnr_i']:.2f} dB")
    print(f"  Avg PSNR  P-frames    : {metrics['avg_psnr_p']:.2f} dB")
    print(f"  Avg SSIM              : {metrics['avg_ssim']:.4f}")
    print(f"{'═'*52}\n")
