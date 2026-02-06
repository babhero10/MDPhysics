# Training Stability Fixes Applied

This document summarizes the numerical stability fixes applied to address training instability (loss jumping, crashes, and recovery cycles).

## Root Cause Identified

**AMP Scaler Death Spiral** caused by:
1. Learning rate too high (0.001) → Large gradient updates
2. FP16 underflow in FFT/LayerNorm operations → Produces NaN/Inf
3. GradScaler continuously reduces scale factor → Gradients vanish
4. Model gets stuck → Then explodes when perturbed

---

## Fixes Applied

### 🔴 Fix 1: Reduced Learning Rate (CRITICAL)
**File:** `configs/train/default.yaml:1`

**Change:**
```yaml
# BEFORE
lr: 0.001

# AFTER
lr: 0.0002  # Reduced 5x for better stability with transformers + FFT
```

**Impact:** Prevents large gradient updates that can trigger instability in complex architectures with FFT and deformable operations.

---

### 🔴 Fix 2: Added NaN/Inf Detection (CRITICAL)
**File:** `src/utils/training.py:132-179`

**Added:**
1. **Input validation** - Check blur images and depth maps for NaN/Inf before forward pass
2. **Loss validation** - Check loss for NaN/Inf before backward pass, skip batch if detected
3. **Loss component validation** - Log warnings if individual loss terms contain NaN/Inf

**Impact:** Prevents NaN propagation through the network. Skips corrupted batches instead of crashing.

---

### 🔴 Fix 3: Enhanced Scaler Monitoring (CRITICAL)
**File:** `src/utils/training.py:191-200`

**Change:**
```python
# BEFORE
if scaler.get_scale() < 1:
    logging.warning(f"Scaler scale very low: {scaler.get_scale()}")

# AFTER
current_scale = scaler.get_scale()
if current_scale < 16384:  # Warning at 1/4 of normal (65536)
    logger.warning(f"⚠️  AMP scaler dropping: {current_scale:.0f} (normal: 65536)")
    logger.warning("This indicates FP16 underflow - model may be unstable")
    if current_scale < 1024:  # Critical threshold
        logger.error(f"🔴 Scaler critically low: {current_scale:.0f}")
        logger.error("Consider: (1) Reduce LR, (2) Check for NaN, (3) Disable AMP")
```

**Impact:** Provides early warning when AMP is struggling, allowing diagnosis before complete failure.

---

### 🟡 Fix 4: Stabilized FFT Loss (HIGH PRIORITY)
**File:** `src/utils/losses.py:39-57`

**Change:**
```python
# BEFORE
pred_fft = torch.fft.fft2(pred.float(), dim=(-2, -1))
target_fft = torch.fft.fft2(target.float(), dim=(-2, -1))
# ... direct L1 loss on coefficients

# AFTER
pred_fft = torch.fft.fft2(pred.float(), dim=(-2, -1))
target_fft = torch.fft.fft2(target.float(), dim=(-2, -1))

# Normalize by image size to prevent huge coefficient magnitudes
norm_factor = pred.size(-2) * pred.size(-1)
pred_fft = pred_fft / norm_factor
target_fft = target_fft / norm_factor
# ... then L1 loss
```

**Impact:** Prevents FFT DC component (can be 100x larger than high-freq) from dominating loss and causing instability.

---

### 🟡 Fix 5: Increased LayerNorm Epsilon (HIGH PRIORITY)
**Files:**
- `src/model/archs/mdt_arch.py:320, 339`
- `src/model/archs/mdt_dpt_arch.py:313, 332`

**Change:**
```python
# BEFORE
return x / torch.sqrt(sigma + 1e-5) * self.weight

# AFTER
return x / torch.sqrt(sigma + 1e-4) * self.weight  # Increased 10x for FP16 stability
```

**Impact:** Prevents numerical underflow in FP16 when variance is small. `1e-5` is near FP16 precision limit (~6e-8), while `1e-4` is much safer.

---

### 🟡 Fix 6: Ensured FP32 for FFT Operations (HIGH PRIORITY)
**Files:**
- `src/model/archs/mdt_arch.py:392-397, 448-457`
- `src/model/archs/mdt_dpt_arch.py:378-390, 427-446`

**Change:**
```python
# BEFORE (mdt_arch.py had partial protection)
with torch.amp.autocast(device_type='cuda', enabled=False):
    x_patch_fft = torch.fft.rfft2(x_patch.float())  # x_patch still FP16!

# BEFORE (mdt_dpt_arch.py had NO protection)
x_patch_fft = torch.fft.rfft2(x_patch.float())  # FFT in FP16 context!

# AFTER (both files)
x_patch = x_patch.float()  # Convert BEFORE autocast context
with torch.amp.autocast(device_type='cuda', enabled=False):
    x_patch_fft = torch.fft.rfft2(x_patch)  # Now truly FP32
```

**Impact:**
- **DFFN module:** Prevents FP16 underflow in frequency-space operations
- **FSAS module:** Prevents FP16 underflow in frequency-space attention
- **mdt_dpt_arch.py:** CRITICAL - This file had NO autocast protection at all!

---

## Expected Results

After these fixes, you should see:

### ✅ Stable Training
- Loss should decrease smoothly without sudden jumps
- No more crash → recovery cycles
- GradScaler should stay at ~65536 throughout training

### ✅ Early Warnings
- If scaler drops below 16384, you'll get clear warnings
- If NaN detected, batch is skipped with error log
- Loss components are validated separately

### ✅ Better Convergence
- Lower learning rate (0.0002) allows finer optimization
- Normalized FFT loss balances frequency components
- FP32 FFT operations prevent underflow

---

## Testing Recommendations

1. **Monitor GradScaler:**
   - Check logs for scaler scale values
   - Should stay at 65536 (or close to it)
   - If dropping, reduce LR further or disable AMP

2. **Watch for NaN warnings:**
   - "NaN/Inf detected in input images" → Check data preprocessing
   - "Non-finite loss detected" → Check model initialization
   - "Scaler critically low" → Reduce LR or disable AMP

3. **Compare with previous runs:**
   - Loss curve should be much smoother
   - Training should reach lower final loss
   - No sudden divergence after N epochs

---

## If Issues Persist

If training is still unstable after these fixes, try:

### Option A: Further Reduce Learning Rate
```yaml
lr: 0.0001  # Even more conservative
```

### Option B: Disable AMP Completely
Comment out AMP code in `src/train.py` and `src/utils/training.py`:
- Remove `torch.amp.autocast()` contexts
- Use regular `.backward()` instead of `scaler.scale(loss).backward()`
- Use `optimizer.step()` instead of `scaler.step(optimizer)`

### Option C: Reduce Model Complexity
- Decrease `num_blocks` in model config
- Reduce `dim` parameter
- Simplify FFT operations

---

## Files Modified

1. `configs/train/default.yaml` - Learning rate reduced
2. `src/utils/training.py` - NaN detection + scaler monitoring
3. `src/utils/losses.py` - FFT loss normalization
4. `src/model/archs/mdt_arch.py` - LayerNorm epsilon + FFT FP32
5. `src/model/archs/mdt_dpt_arch.py` - LayerNorm epsilon + FFT FP32 + autocast protection

---

## Summary

The primary issue was **AMP scaler collapse** due to:
- Learning rate too high for complex architecture
- FP16 underflow in FFT operations
- Insufficient numerical safeguards

These fixes address all identified stability issues. Training should now be stable and converge properly.

**Date Applied:** 2026-02-06
**Applied By:** Claude Sonnet 4.5
