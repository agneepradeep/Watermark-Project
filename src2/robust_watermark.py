"""
robust_watermark.py

Implements the robust watermark embedding/extraction described:
- Convert host image to YCbCr, embed in Y channel (transform domain)
- DWT level-1 -> use LH and HL subbands
- Split LH/HL into non-overlapping 8x8 blocks, apply DCT per block
- Select a set of mid-frequency DCT coefficients (user-configurable)
- For each EB (embedding block of 8 MF coefficients): split into LHS/RHS halves
  and scale maxima of each half depending on watermark bit (bit=1 -> LHS up, RHS down)
- Fisher–Yates scramble on watermark using secret key
- Alpha (strength) selection routine: search alphas to minimize |SSIM - NCC|
  subject to PSNR >= 34 dB (as in the paper).
"""

import numpy as np
import pywt
from scipy.fftpack import dct, idct
from skimage import color, util
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Tuple, List
import math
import random

# ----------------------------
# Utilities
# ----------------------------
def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    """Expect img in [0,255] uint8. Return float64 YCbCr in same scale."""
    return color.rgb2ycbcr(img.astype(np.float64))

def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    return color.ycbcr2rgb(ycbcr).clip(0,255).astype(np.uint8)

def block_view(arr, blk_h, blk_w):
    """Return view of arr split into non-overlapping blocks of size blk_h x blk_w.
    arr shape: (H, W)
    Returns: blocks shaped (num_blocks_h, num_blocks_w, blk_h, blk_w)
    """
    H, W = arr.shape
    assert H % blk_h == 0 and W % blk_w == 0
    shape = (H//blk_h, W//blk_w, blk_h, blk_w)
    strides = (arr.strides[0]*blk_h, arr.strides[1]*blk_w, arr.strides[0], arr.strides[1])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def fisher_yates_scramble(bits: np.ndarray, key: int) -> np.ndarray:
    """Scramble binary 1D array bits using Fisher-Yates with seeded RNG."""
    bits = bits.copy()
    rng = random.Random(key)
    n = len(bits)
    for i in range(n-1, 0, -1):
        j = rng.randint(0, i)
        bits[i], bits[j] = bits[j], bits[i]
    return bits

def fisher_yates_unscramble(bits: np.ndarray, key: int) -> np.ndarray:
    """Reverse scramble by replaying swaps in reverse."""
    bits = bits.copy()
    rng = random.Random(key)
    n = len(bits)
    swaps = []
    for i in range(n-1, 0, -1):
        j = rng.randint(0, i)
        swaps.append((i,j))
    # replay swaps in reverse to invert
    for i,j in reversed(swaps):
        bits[i], bits[j] = bits[j], bits[i]
    return bits

def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """NCC for binary arrays (0/1 or -1/1). Returns in [-1,1]."""
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (a.std() * b.std()))

# ----------------------------
# Embedding core
# ----------------------------
class RobustWatermarker:
    def __init__(self,
                 mf_positions: List[Tuple[int,int]] = None,
                 block_size: int = 8,
                 dwt_wavelet: str = 'haar'):
        """
        mf_positions: list of (r,c) positions inside 8x8 DCT block that are considered
                      the mid-frequency (total 8 positions). If None, defaults to a
                      commonly used mid-frequency set (tunable).
        """
        self.block_size = block_size
        self.dwt_wavelet = dwt_wavelet
        if mf_positions is None:
            # Default mid-frequency positions (row,col) 0-indexed in 8x8,
            # these are typical mid-freq picks — adjust if paper figure suggests others.
            self.mf_positions = [(1,4),(1,5),(2,3),(2,4),(2,5),(3,2),(3,3),(3,4)]
        else:
            self.mf_positions = mf_positions

    def _prepare_watermark(self, watermark: np.ndarray) -> np.ndarray:
        """Take grayscale watermark in [0,255], threshold at 128 to produce binary bits (0/1)."""
        w = watermark.copy()
        if w.ndim == 3:
            w = color.rgb2gray(w)
            w = (w * 255).astype(np.uint8)
        bits = (w >= 128).astype(np.uint8).flatten()
        return bits

    def embed(self,
              host_img: np.ndarray,
              watermark_img: np.ndarray,
              alpha: float,
              key: int) -> Tuple[np.ndarray, dict]:
        """
        Embed watermark_img into host_img (uint8 RGB).
        Returns: watermarked_rgb (uint8), and metadata needed for extraction (dictionary).
        """
        # 1) Convert to YCbCr and take Y channel
        ycbcr = rgb_to_ycbcr(host_img)
        Y = ycbcr[...,0]  # float64

        # 2) DWT level-1
        LL, (LH, HL, HH) = pywt.dwt2(Y, self.dwt_wavelet)

        # We will use LH and HL for embedding.
        # Ensure LH and HL sizes are multiples of 8
        def pad_to_multiple(arr, m=8):
            H,W = arr.shape
            pad_h = (m - (H % m)) % m
            pad_w = (m - (W % m)) % m
            if pad_h==0 and pad_w==0:
                return arr, (0,0)
            return np.pad(arr, ((0,pad_h),(0,pad_w)), mode='symmetric'), (pad_h,pad_w)

        LHp, pad_l = pad_to_multiple(LH, self.block_size)
        HLp, pad_hl = pad_to_multiple(HL, self.block_size)

        # 3) Divide into 8x8 blocks and apply DCT on each block
        def process_subband(subband):
            H, W = subband.shape
            nbh, nbw = H//self.block_size, W//self.block_size
            blocks = np.zeros((nbh, nbw, self.block_size, self.block_size))
            for i in range(nbh):
                for j in range(nbw):
                    blk = subband[i*self.block_size:(i+1)*self.block_size,
                                  j*self.block_size:(j+1)*self.block_size]
                    blocks[i,j] = dct2(blk)
            return blocks

        LH_blocks = process_subband(LHp)
        HL_blocks = process_subband(HLp)

        # 4) Flatten watermark bits and scramble
        bits = self._prepare_watermark(watermark_img)
        scrambled = fisher_yates_scramble(bits, key)

        # If watermark longer than number of main-blocks available, truncate or repeat
        total_blocks = LH_blocks.shape[0]*LH_blocks.shape[1] + HL_blocks.shape[0]*HL_blocks.shape[1]
        if len(scrambled) > total_blocks:
            print("Warning: watermark larger than embedding capacity. Truncating watermark.")
            scrambled = scrambled[:total_blocks]
        elif len(scrambled) < total_blocks:
            # pad with zeros
            pad_len = total_blocks - len(scrambled)
            scrambled = np.concatenate([scrambled, np.zeros(pad_len, dtype=np.uint8)])

        # 5) Embedding loop: process MB-1 (HL) then MB-2 (LH) in sequence
        # We'll iterate through block indices and map each bit to one MB in order.
        def embed_in_blocks(blocks, bits_slice, mf_positions, alpha):
            nb_h, nb_w = blocks.shape[0], blocks.shape[1]
            idx = 0
            for i in range(nb_h):
                for j in range(nb_w):
                    if idx >= len(bits_slice): break
                    bit = bits_slice[idx]
                    block = blocks[i,j]  # DCT coefficients 8x8
                    # collect MF coefficients into EB (in order), keep their positions
                    eb_vals = []
                    for (r,c) in mf_positions:
                        eb_vals.append(block[r,c])
                    eb_vals = np.array(eb_vals)
                    # Split EB into LHS (first half) and RHS (second half)
                    half = len(eb_vals)//2
                    lhs_idx = np.arange(0,half)
                    rhs_idx = np.arange(half,len(eb_vals))
                    # We will scale all coefficients in LHS and RHS proportionally
                    # to maintain mean invariance similar to paper's pairwise scaling.
                    if bit == 1:
                        # scale lhs up, rhs down
                        block = block.copy()
                        for k in lhs_idx:
                            r,c = mf_positions[k]
                            block[r,c] = block[r,c] * (1.0 + alpha)
                        for k in rhs_idx:
                            r,c = mf_positions[k]
                            block[r,c] = block[r,c] * (1.0 - alpha)
                    else:
                        # bit == 0: lhs down, rhs up
                        block = block.copy()
                        for k in lhs_idx:
                            r,c = mf_positions[k]
                            block[r,c] = block[r,c] * (1.0 - alpha)
                        for k in rhs_idx:
                            r,c = mf_positions[k]
                            block[r,c] = block[r,c] * (1.0 + alpha)
                    blocks[i,j] = block
                    idx += 1
                if idx >= len(bits_slice): break
            return blocks

        # split scrambled sequence: assign first part to HL blocks, next to LH blocks
        hl_count = HL_blocks.shape[0]*HL_blocks.shape[1]
        hl_bits = scrambled[:hl_count]
        lh_bits = scrambled[hl_count:hl_count + (LH_blocks.shape[0]*LH_blocks.shape[1])]

        HL_blocks = embed_in_blocks(HL_blocks, hl_bits, self.mf_positions, alpha)
        LH_blocks = embed_in_blocks(LH_blocks, lh_bits, self.mf_positions, alpha)

        # 6) inverse DCT per block and reconstruct subbands
        def reconstruct_subband(blocks, original_shape, pad):
            nb_h, nb_w = blocks.shape[0], blocks.shape[1]
            H = nb_h * self.block_size
            W = nb_w * self.block_size
            sub = np.zeros((H,W))
            for i in range(nb_h):
                for j in range(nb_w):
                    sub[i*self.block_size:(i+1)*self.block_size,
                        j*self.block_size:(j+1)*self.block_size] = idct2(blocks[i,j])
            if pad != (0,0):
                pad_h, pad_w = pad
                sub = sub[:original_shape[0], :original_shape[1]]
            return sub

        LH_mod = reconstruct_subband(LH_blocks, LH.shape, pad_l)
        HL_mod = reconstruct_subband(HL_blocks, HL.shape, pad_hl)

        # 7) inverse DWT to reconstruct Y
        Y_mod = pywt.idwt2((LL, (LH_mod, HL_mod, HH)), self.dwt_wavelet)

        # crop Y_mod to same shape as original Y
        Y_mod = Y_mod[:Y.shape[0], :Y.shape[1]]
        # put back into ycbcr and convert to rgb
        ycbcr_mod = ycbcr.copy()
        ycbcr_mod[...,0] = Y_mod
        rgb_mod = ycbcr_to_rgb(ycbcr_mod)

        meta = {
            'alpha': alpha,
            'key': key,
            'mf_positions': self.mf_positions,
            'dwt_wavelet': self.dwt_wavelet,
            'orig_shape': Y.shape,
            'blocks_shape': (LH_blocks.shape, HL_blocks.shape),
            'pad_l': pad_l,
            'pad_hl': pad_hl
        }
        return rgb_mod, meta

    def extract(self, watermarked_img: np.ndarray, meta: dict) -> np.ndarray:
        """
        Extract the scrambled watermark bits from the watermarked image using meta.
        Returns binary array of extracted bits (0/1) in scrambled order.
        """
        key = meta['key']
        mf_positions = meta['mf_positions']
        dwt_wavelet = meta['dwt_wavelet']

        ycbcr = rgb_to_ycbcr(watermarked_img)
        Y = ycbcr[...,0]

        LL, (LH, HL, HH) = pywt.dwt2(Y, dwt_wavelet)

        # pad to multiple
        def pad_to_multiple(arr, m=8):
            H, W = arr.shape
            pad_h = (m - (H % m)) % m
            pad_w = (m - (W % m)) % m
            if pad_h==0 and pad_w==0:
                return arr, (0,0)
            return np.pad(arr, ((0,pad_h),(0,pad_w)), mode='symmetric'), (pad_h,pad_w)

        LHp, pad_l = pad_to_multiple(LH, self.block_size)
        HLp, pad_hl = pad_to_multiple(HL, self.block_size)

        def process_blocks(subband):
            H, W = subband.shape
            nb_h, nb_w = H//self.block_size, W//self.block_size
            blocks = []
            for i in range(nb_h):
                for j in range(nb_w):
                    blk = subband[i*self.block_size:(i+1)*self.block_size,
                                  j*self.block_size:(j+1)*self.block_size]
                    blocks.append(dct2(blk))
            return blocks

        LH_blocks = process_blocks(LHp)
        HL_blocks = process_blocks(HLp)

        # combine blocks in same order as embedding: HL blocks first then LH blocks
        all_blocks = HL_blocks + LH_blocks

        extracted_bits = []
        for block in all_blocks:
            # extract EB
            eb_vals = []
            for (r,c) in mf_positions:
                eb_vals.append(block[r,c])
            eb_vals = np.array(eb_vals)
            half = len(eb_vals)//2
            lhs = eb_vals[:half]
            rhs = eb_vals[half:]
            # compute means
            mean_l = lhs.mean()
            mean_r = rhs.mean()
            # If mean_l > mean_r -> bit likely 1 (since bit=1 made LHS up and RHS down)
            bit = 1 if mean_l > mean_r else 0
            extracted_bits.append(bit)

        extracted_bits = np.array(extracted_bits, dtype=np.uint8)
        # unscramble to original bit order
        unscrambled = fisher_yates_unscramble(extracted_bits[:len(extracted_bits)], key)
        return unscrambled

    # ----------------------------
    # Alpha selection routine
    # ----------------------------
    def select_alpha(self,
                     host_img: np.ndarray,
                     watermark_img: np.ndarray,
                     key: int,
                     alpha_candidates: np.ndarray = None) -> Tuple[float, dict]:
        """
        Search for alpha in alpha_candidates that minimizes |SSIM - NCC| subject to PSNR >= 34 dB.
        Returns chosen alpha and dictionary of metrics for the best alpha.
        """
        if alpha_candidates is None:
            alpha_candidates = np.linspace(0.01, 0.5, 50)

        best = None
        best_record = None

        orig_ycbcr = rgb_to_ycbcr(host_img)
        Y_orig = orig_ycbcr[...,0]

        watermark_bits = self._prepare_watermark(watermark_img)

        for alpha in alpha_candidates:
            # embed
            wm_img, meta = self.embed(host_img, watermark_img, alpha, key)
            # compute PSNR between Y channels (paper uses overall image PSNR often)
            Y_wm = rgb_to_ycbcr(wm_img)[...,0]
            p = psnr(Y_orig, Y_wm, data_range=Y_orig.max()-Y_orig.min())
            if p < 34.0:
                # constraint not satisfied
                continue
            # extract watermark bits
            ext = self.extract(wm_img, meta)
            # compute NCC between original bits and extracted (truncate/pad to length)
            L = min(len(ext), len(watermark_bits))
            n = normalized_cross_correlation(watermark_bits[:L], ext[:L])
            structural = ssim(Y_orig.astype(np.uint8), Y_wm.astype(np.uint8), data_range=255)

            score = abs(structural - n)
            if best is None or score < best:
                best = score
                best_record = {
                    'alpha': float(alpha),
                    'psnr': float(p),
                    'ssim': float(structural),
                    'ncc': float(n),
                    'score': float(score),
                    'meta': meta
                }
        if best_record is None:
            raise ValueError("No alpha candidate satisfied PSNR >= 34 dB. Try expanding alpha search or relaxing PSNR.")
        return best_record['alpha'], best_record
