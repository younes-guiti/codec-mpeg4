

import numpy as np
import cv2
import os
import pickle
import zlib
import struct
from scipy.fft import dctn, idctn




LUMA_QUANT_BASE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99],
], dtype=np.float32)

CHROMA_QUANT_BASE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)


def _build_zigzag():
    order = []
    for s in range(15):
        if s % 2 == 0:
            r = min(s, 7); c = s - r
            while r >= 0 and c <= 7:
                order.append(r * 8 + c); r -= 1; c += 1
        else:
            c = min(s, 7); r = s - c
            while c >= 0 and r <= 7:
                order.append(r * 8 + c); r += 1; c -= 1
    return np.array(order, dtype=np.int32)

ZIGZAG_IDX = _build_zigzag()
ZIGZAG_INV = np.argsort(ZIGZAG_IDX)


def get_quant_matrix(qf: float, is_luma: bool = True) -> np.ndarray:
    """
    Scale quantisation matrix by quality factor (1–100).
    QF < 50 → aggressive quantisation (high compression, low quality).
    QF > 50 → mild quantisation (low compression, high quality).
    Formula from JPEG spec : scale = 5000/QF if QF<50 else 200-2*QF.
    """
    base = LUMA_QUANT_BASE if is_luma else CHROMA_QUANT_BASE
    scale = 5000 / qf if qf < 50 else 200 - 2 * qf
    return np.floor((base * scale + 50) / 100).clip(1, 255).astype(np.float32)



def bgr_to_ycbcr(frame_bgr: np.ndarray) -> tuple:
    """
    Convert BGR uint8 → YCbCr float, with 4:2:0 chroma subsampling.
    Chroma is downsampled by 2 in each dimension (INTER_AREA = box filter),
    halving its storage at the cost of very little perceptible quality loss
    because the human visual system is less sensitive to chroma detail.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    R, G, B = frame_rgb[:,:,0], frame_rgb[:,:,1], frame_rgb[:,:,2]

    Y  =  0.299   * R + 0.587   * G + 0.114    * B
    Cb = -0.168736* R - 0.331264* G + 0.5      * B + 128.0
    Cr =  0.5     * R - 0.418688* G - 0.081312 * B + 128.0

    Cb_sub = cv2.resize(Cb, (Cb.shape[1]//2, Cb.shape[0]//2),
                        interpolation=cv2.INTER_AREA)
    Cr_sub = cv2.resize(Cr, (Cr.shape[1]//2, Cr.shape[0]//2),
                        interpolation=cv2.INTER_AREA)
    return Y, Cb_sub, Cr_sub


def ycbcr_to_bgr(Y: np.ndarray,
                 Cb_sub: np.ndarray,
                 Cr_sub: np.ndarray) -> np.ndarray:
    """YCbCr float → BGR uint8 (bilinear upsampling for chroma)."""
    h, w = Y.shape
    Cb = cv2.resize(Cb_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr = cv2.resize(Cr_sub, (w, h), interpolation=cv2.INTER_LINEAR)

    R = (Y + 1.402   * (Cr - 128.0)).clip(0, 255)
    G = (Y - 0.344136*(Cb - 128.0) - 0.714136*(Cr - 128.0)).clip(0, 255)
    B = (Y + 1.772   * (Cb - 128.0)).clip(0, 255)

    rgb = np.stack([R, G, B], axis=2).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)




def _pad_channel(channel: np.ndarray, block: int = 8) -> tuple:
    """Pad channel so dimensions are multiples of `block`. Returns (padded, ph, pw)."""
    h, w = channel.shape
    ph = (block - h % block) % block
    pw = (block - w % block) % block
    padded = np.pad(channel, ((0, ph), (0, pw)), mode='edge')
    return padded, ph, pw


def dct_quantise_channel(channel: np.ndarray, qf: float,
                          is_luma: bool) -> np.ndarray:
    """
    Vectorised 8×8 DCT + quantisation over a full image channel.
    The input is pixel data in [0,255], so we subtract 128 to centre it
    around 0 before DCT — this ensures the DC coefficient represents the
    average deviation from mid-grey, improving coding efficiency.
    DO NOT use this for residual data (use dct_quantise_residual instead).
    """
    Q = get_quant_matrix(qf, is_luma)
    h, w = channel.shape
    padded, ph, pw = _pad_channel(channel.astype(np.float32) - 128.0)

    bh = padded.shape[0] // 8
    bw = padded.shape[1] // 8
    blocks    = padded.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
    dct_blocks = dctn(blocks, axes=(-2, -1), norm='ortho')
    q_blocks   = np.round(dct_blocks / Q)

    out = q_blocks.transpose(0, 2, 1, 3).reshape(bh*8, bw*8)
    return out[:h, :w]


def dct_quantise_residual(residual: np.ndarray, qf: float,
                           is_luma: bool) -> np.ndarray:
    """
    Vectorised 8×8 DCT + quantisation for P-frame residuals.
    WHY NO -128: residuals are pixel differences, already centred around 0
    (range ≈ [-255, 255]). Subtracting 128 would introduce a systematic
    DC bias of -128 in every block, severely degrading PSNR (~10→30+ dB fix).
    """
    Q = get_quant_matrix(qf, is_luma)
    h, w = residual.shape
    padded, ph, pw = _pad_channel(residual.astype(np.float32))  # NO -128

    bh = padded.shape[0] // 8
    bw = padded.shape[1] // 8
    blocks     = padded.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
    dct_blocks = dctn(blocks, axes=(-2, -1), norm='ortho')
    q_blocks   = np.round(dct_blocks / Q)

    out = q_blocks.transpose(0, 2, 1, 3).reshape(bh*8, bw*8)
    return out[:h, :w]


def idct_dequantise_channel(coeffs: np.ndarray, qf: float,
                             is_luma: bool) -> np.ndarray:
    """
    Vectorised dequantise + IDCT → pixel values [0,255].
    FIX: single _pad_channel call (old version had dead double-padding bug).
    """
    Q = get_quant_matrix(qf, is_luma)
    h, w = coeffs.shape
    # Single coherent padding call — no double-padding bug
    padded, _, _ = _pad_channel(coeffs.astype(np.float32), 8)

    bh = padded.shape[0] // 8
    bw = padded.shape[1] // 8
    blocks    = padded.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
    dq_blocks = blocks * Q
    idct_blks = idctn(dq_blocks, axes=(-2, -1), norm='ortho') + 128.0

    out = idct_blks.transpose(0, 2, 1, 3).reshape(bh*8, bw*8)
    return out[:h, :w].clip(0, 255)


def idct_dequantise_residual(coeffs: np.ndarray, qf: float,
                              is_luma: bool) -> np.ndarray:
    """
    Vectorised dequantise + IDCT for P-frame residuals.
    WHY NO +128: residuals were not shifted during encoding, so we must not
    shift here either. Adding 128 would corrupt every reconstructed residual.
    Returns float array (no clip), range ≈ [-255, 255].
    """
    Q = get_quant_matrix(qf, is_luma)
    h, w = coeffs.shape
    padded, _, _ = _pad_channel(coeffs.astype(np.float32), 8)

    bh = padded.shape[0] // 8
    bw = padded.shape[1] // 8
    blocks    = padded.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
    dq_blocks = blocks * Q
    idct_blks = idctn(dq_blocks, axes=(-2, -1), norm='ortho')  # NO +128

    out = idct_blks.transpose(0, 2, 1, 3).reshape(bh*8, bw*8)
    return out[:h, :w]   # float, no clip



def _zigzag_encode_block(block8x8: np.ndarray) -> np.ndarray:
    return block8x8.ravel()[ZIGZAG_IDX]


def _zigzag_decode_block(vec: np.ndarray) -> np.ndarray:
    out = np.empty(64, dtype=vec.dtype)
    out[ZIGZAG_IDX] = vec
    return out.reshape(8, 8)


def rle_encode(arr: np.ndarray) -> bytes:
  
    flat = arr.astype(np.int16).ravel()
    runs = []
    i = 0
    while i < len(flat):
        val = flat[i]
        if val == 0:
            count = 0
            while i < len(flat) and flat[i] == 0:
                count += 1
                i += 1
                # FIX: split runs longer than 255 into multiple (0, 255) pairs
                if count == 255:
                    runs.append((0, 255))
                    count = 0
            if count > 0:
                runs.append((0, count))
        else:
            runs.append((val, 1))
            i += 1
    packed = struct.pack('>' + 'hB' * len(runs),
                         *[x for pair in runs for x in pair])
    return packed


def rle_decode(data: bytes, length: int) -> np.ndarray:
    """Decode RLE back to flat int16 array."""
    out = []
    i = 0
    while i < len(data):
        val = struct.unpack_from('>h', data, i)[0]; i += 2
        cnt = struct.unpack_from('B',  data, i)[0]; i += 1
        if val == 0:
            out.extend([0] * cnt)
        else:
            out.append(val)
    return np.array(out, dtype=np.int16)[:length]


def zigzag_rle_encode_channel(coeffs: np.ndarray) -> bytes:
    """Apply zigzag scan + RLE to a quantised coefficient channel."""
    h, w = coeffs.shape
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    padded = np.pad(coeffs.astype(np.int16), ((0, ph), (0, pw)), mode='constant')
    bh, bw = padded.shape[0]//8, padded.shape[1]//8
    blocks  = padded.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    zz_all  = np.stack([_zigzag_encode_block(b) for b in blocks])
    flat    = zz_all.ravel()
    return rle_encode(flat)


def zigzag_rle_decode_channel(data: bytes, h: int, w: int) -> np.ndarray:
    """Decode zigzag+RLE back to (h,w) coefficient array."""
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    bh = (h + ph) // 8
    bw = (w + pw) // 8
    n_blocks = bh * bw
    flat = rle_decode(data, n_blocks * 64)
    zz_blocks = flat.reshape(n_blocks, 64)
    blocks = np.stack([_zigzag_decode_block(z) for z in zz_blocks]).reshape(bh, bw, 8, 8)
    full   = blocks.transpose(0, 2, 1, 3).reshape(bh*8, bw*8)
    return full[:h, :w].astype(np.float32)


def encode_iframe(frame_bgr: np.ndarray, qf: float) -> dict:
    Y, Cb, Cr = bgr_to_ycbcr(frame_bgr)
    qY  = dct_quantise_channel(Y,  qf, is_luma=True)
    qCb = dct_quantise_channel(Cb, qf, is_luma=False)
    qCr = dct_quantise_channel(Cr, qf, is_luma=False)
    return {
        'type': 'I',
        'shape': frame_bgr.shape[:2],
        'Y_h': Y.shape[0],   'Y_w': Y.shape[1],
        'C_h': Cb.shape[0],  'C_w': Cb.shape[1],
        'Y_enc':  zigzag_rle_encode_channel(qY),
        'Cb_enc': zigzag_rle_encode_channel(qCb),
        'Cr_enc': zigzag_rle_encode_channel(qCr),
        'qf': qf,
    }


def decode_iframe(data: dict) -> np.ndarray:
    qf = data['qf']
    qY  = zigzag_rle_decode_channel(data['Y_enc'],  data['Y_h'], data['Y_w'])
    qCb = zigzag_rle_decode_channel(data['Cb_enc'], data['C_h'], data['C_w'])
    qCr = zigzag_rle_decode_channel(data['Cr_enc'], data['C_h'], data['C_w'])
    Y   = idct_dequantise_channel(qY,  qf, is_luma=True)
    Cb  = idct_dequantise_channel(qCb, qf, is_luma=False)
    Cr  = idct_dequantise_channel(qCr, qf, is_luma=False)
    return ycbcr_to_bgr(Y, Cb, Cr)




MB = 16   


def motion_estimation(cur: np.ndarray, ref: np.ndarray,
                       search: int = 8) -> np.ndarray:
    
    h, w = cur.shape
    mbs_h = h // MB
    mbs_w = w // MB
    mvs   = np.zeros((mbs_h, mbs_w, 2), dtype=np.int16)  # int16 for safety

    ref_f = ref.astype(np.float32)
    cur_f = cur.astype(np.float32)

    pad = search
    ref_pad = cv2.copyMakeBorder(ref_f, pad, pad+MB, pad, pad+MB,
                                  cv2.BORDER_REPLICATE)

    for bi in range(mbs_h):
        for bj in range(mbs_w):
            y0 = bi * MB; x0 = bj * MB
            cur_block = cur_f[y0:y0+MB, x0:x0+MB]
            ry0 = y0; rx0 = x0
            region = ref_pad[ry0:ry0 + MB + 2*search,
                             rx0:rx0 + MB + 2*search]
            result = cv2.matchTemplate(region, cur_block, cv2.TM_SQDIFF)
            _, _, min_loc, _ = cv2.minMaxLoc(result)
            dx, dy = min_loc[0] - search, min_loc[1] - search
            mvs[bi, bj] = [
                np.clip(dy, -32767, 32767),
                np.clip(dx, -32767, 32767)
            ]
    return mvs


def motion_compensate(ref: np.ndarray, mvs: np.ndarray) -> np.ndarray:
    """Build predicted frame from reference + motion vectors."""
    h, w = ref.shape
    mbs_h, mbs_w = mvs.shape[:2]
    pred = np.zeros((mbs_h*MB, mbs_w*MB), dtype=np.float32)

    for bi in range(mbs_h):
        for bj in range(mbs_w):
            dy = int(mvs[bi, bj, 0])
            dx = int(mvs[bi, bj, 1])
            y0 = bi * MB; x0 = bj * MB
            ry = np.clip(y0 + dy, 0, h - MB)
            rx = np.clip(x0 + dx, 0, w - MB)
            pred[y0:y0+MB, x0:x0+MB] = ref[ry:ry+MB, rx:rx+MB]
    return pred




def _delta_encode_mvs(mvs: np.ndarray) -> np.ndarray:
   
    mbs_h, mbs_w, _ = mvs.shape
    flat = mvs.reshape(-1, 2).astype(np.int32)
    deltas = np.empty_like(flat)
    deltas[0] = flat[0]                     
    deltas[1:] = flat[1:] - flat[:-1]       
    
    return np.clip(deltas, -32767, 32767).astype(np.int16).reshape(mbs_h, mbs_w, 2)


def _delta_decode_mvs(deltas: np.ndarray) -> np.ndarray:
    """Reconstruct absolute MVs from delta-coded stream via cumulative sum."""
    flat = deltas.reshape(-1, 2).astype(np.int32)
    abs_mvs = np.cumsum(flat, axis=0)
    return np.clip(abs_mvs, -32767, 32767).astype(np.int16).reshape(deltas.shape)


def _encode_mvs(mvs: np.ndarray) -> bytes:
    """Serialize motion vectors: delta-code → flatten → RLE → bytes."""
    delta_mvs = _delta_encode_mvs(mvs)
    flat = delta_mvs.ravel().astype(np.int16)
    return rle_encode(flat)


def _decode_mvs(data: bytes, mbs_h: int, mbs_w: int) -> np.ndarray:
    """Deserialize motion vectors: RLE decode → reshape → delta decode."""
    length = mbs_h * mbs_w * 2
    flat = rle_decode(data, length).astype(np.int16)
    delta_mvs = flat.reshape(mbs_h, mbs_w, 2)
    return _delta_decode_mvs(delta_mvs)


def encode_pframe(frame_bgr: np.ndarray, ref_bgr: np.ndarray,
                  qf: float, search: int = 8) -> dict:
    
    Y_cur, Cb_cur, Cr_cur = bgr_to_ycbcr(frame_bgr)
    Y_ref, Cb_ref, Cr_ref = bgr_to_ycbcr(ref_bgr)

    h = (min(Y_cur.shape[0], Y_ref.shape[0]) // MB) * MB
    w = (min(Y_cur.shape[1], Y_ref.shape[1]) // MB) * MB
    Y_cur_c = Y_cur[:h, :w].astype(np.float32)
    Y_ref_c = Y_ref[:h, :w].astype(np.float32)

    mvs    = motion_estimation(Y_cur_c, Y_ref_c, search)
    pred_Y = motion_compensate(Y_ref_c, mvs)
    res_Y  = Y_cur_c - pred_Y   # range ~[-255, 255], centred near 0

    hC = (min(Cb_cur.shape[0], Cb_ref.shape[0]) // 8) * 8
    wC = (min(Cb_cur.shape[1], Cb_ref.shape[1]) // 8) * 8
    Cb_cur_c = Cb_cur[:hC, :wC].astype(np.float32)
    Cr_cur_c = Cr_cur[:hC, :wC].astype(np.float32)
    Cb_ref_c = Cb_ref[:hC, :wC].astype(np.float32)
    Cr_ref_c = Cr_ref[:hC, :wC].astype(np.float32)

    mvs_c    = (mvs // 2).astype(np.int16)
    pred_Cb  = _motion_compensate_chroma(Cb_ref_c, mvs_c)
    pred_Cr  = _motion_compensate_chroma(Cr_ref_c, mvs_c)
    res_Cb   = Cb_cur_c - pred_Cb
    res_Cr   = Cr_cur_c - pred_Cr

    mbs_h, mbs_w = mvs.shape[:2]

    return {
        'type':    'P',
        'shape':   frame_bgr.shape[:2],
        'mbs_h':   mbs_h,
        'mbs_w':   mbs_w,
        'mvs_enc': _encode_mvs(mvs),           
        'Y_h': h,    'Y_w': w,
        'res_Y_enc':  zigzag_rle_encode_channel(
                          dct_quantise_residual(res_Y,  qf, is_luma=True)),
        'C_h': hC,   'C_w': wC,
        'res_Cb_enc': zigzag_rle_encode_channel(
                          dct_quantise_residual(res_Cb, qf, is_luma=False)),
        'res_Cr_enc': zigzag_rle_encode_channel(
                          dct_quantise_residual(res_Cr, qf, is_luma=False)),
        'qf': qf,
    }


def _motion_compensate_chroma(ref_c: np.ndarray,
                               mvs_c: np.ndarray) -> np.ndarray:
    
    hC, wC = ref_c.shape
    mbs_h, mbs_w = mvs_c.shape[:2]
    B = 8
    pred = np.zeros((mbs_h*B, mbs_w*B), dtype=np.float32)
    for bi in range(mbs_h):
        for bj in range(mbs_w):
            dy = int(mvs_c[bi, bj, 0])
            dx = int(mvs_c[bi, bj, 1])
            y0 = bi * B; x0 = bj * B
            ry = np.clip(y0 + dy, 0, hC - B)
            rx = np.clip(x0 + dx, 0, wC - B)
            pred[y0:y0+B, x0:x0+B] = ref_c[ry:ry+B, rx:rx+B]
    return pred


def decode_pframe(data: dict, ref_bgr: np.ndarray) -> np.ndarray:
    
    qf = data['qf']
    Y_ref, Cb_ref, Cr_ref = bgr_to_ycbcr(ref_bgr)

    h = data['Y_h']; w = data['Y_w']
    hC = data['C_h']; wC = data['C_w']
    mbs_h = data['mbs_h']; mbs_w = data['mbs_w']

    
    mvs = _decode_mvs(data['mvs_enc'], mbs_h, mbs_w)

    Y_ref_c = Y_ref[:h, :w].astype(np.float32)
    pred_Y  = motion_compensate(Y_ref_c, mvs)
    qY      = zigzag_rle_decode_channel(data['res_Y_enc'], h, w)
    res_Y   = idct_dequantise_residual(qY, qf, is_luma=True)   # no +128
    Y_rec   = (pred_Y + res_Y).clip(0, 255)

    mvs_c    = (mvs // 2).astype(np.int16)
    Cb_ref_c = Cb_ref[:hC, :wC].astype(np.float32)
    Cr_ref_c = Cr_ref[:hC, :wC].astype(np.float32)
    pred_Cb  = _motion_compensate_chroma(Cb_ref_c, mvs_c)
    pred_Cr  = _motion_compensate_chroma(Cr_ref_c, mvs_c)

    qCb    = zigzag_rle_decode_channel(data['res_Cb_enc'], hC, wC)
    qCr    = zigzag_rle_decode_channel(data['res_Cr_enc'], hC, wC)
    res_Cb = idct_dequantise_residual(qCb, qf, is_luma=False)
    res_Cr = idct_dequantise_residual(qCr, qf, is_luma=False)
    Cb_rec = (pred_Cb + res_Cb).clip(0, 255)
    Cr_rec = (pred_Cr + res_Cr).clip(0, 255)

    orig_h, orig_w = data['shape']
    Y_full  = _pad_to(Y_rec,  orig_h,    orig_w,    Y_ref,  luma=True)
    Cb_full = _pad_to(Cb_rec, orig_h//2, orig_w//2, Cb_ref, luma=False)
    Cr_full = _pad_to(Cr_rec, orig_h//2, orig_w//2, Cr_ref, luma=False)

    return ycbcr_to_bgr(Y_full, Cb_full, Cr_full)


def _pad_to(rec: np.ndarray, th: int, tw: int,
            ref_plane: np.ndarray, luma: bool) -> np.ndarray:
    out = ref_plane[:th, :tw].copy().astype(np.float32)
    rh, rw = rec.shape
    out[:rh, :rw] = rec
    return out




def encode_video(frames_bgr: list, output_path: str,
                 qf: float = 50, gop: int = 8, search: int = 8) -> int:
   
    encoded_frames = []
    ref_bgr = None

    for idx, frame in enumerate(frames_bgr):
        if idx % gop == 0:
            data    = encode_iframe(frame, qf)
            ref_bgr = decode_iframe(data)
        else:
            data    = encode_pframe(frame, ref_bgr, qf, search)
            ref_bgr = decode_pframe(data, ref_bgr)
        encoded_frames.append(data)
        print(f"  Encoded frame {idx+1}/{len(frames_bgr)} [{data['type']}]")

    serialised = pickle.dumps(encoded_frames, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(serialised, level=9)

    with open(output_path, 'wb') as f:
        f.write(b'MP4S')
        f.write(struct.pack('>I', len(frames_bgr)))
        f.write(struct.pack('>f', qf))
        f.write(struct.pack('>I', gop))
        f.write(compressed)

    kb = len(compressed) / 1024
    print(f"\n✓ Written {output_path}  ({kb:.1f} KB)")
    return len(compressed)


def decode_video(input_path: str) -> list:
    """Decode .bin → list of BGR uint8 frames."""
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'MP4S':
            f.seek(0)
            n_frames = struct.unpack('>I', f.read(4))[0]
        else:
            n_frames = struct.unpack('>I', f.read(4))[0]
            _qf  = struct.unpack('>f', f.read(4))[0]
            _gop = struct.unpack('>I', f.read(4))[0]
        compressed = f.read()

    encoded_frames = pickle.loads(zlib.decompress(compressed))
    decoded = []
    ref_bgr = None

    for data in encoded_frames:
        if data['type'] == 'I':
            frame   = decode_iframe(data)
            ref_bgr = frame
        else:
            frame   = decode_pframe(data, ref_bgr)
            ref_bgr = frame
        decoded.append(frame)

    return decoded
