# Secret Sharing Steganography Using $GF(2^8)$ Polynomial Interpolation and Least Significant Bit Adjustment

Official C++ implementation and steganalysis benchmark suite for the research paper:

> **Secret Sharing Steganography Using $GF(2^8)$ Polynomial Interpolation and Least Significant Bit Adjustment**  
> *Li-Chiun Lin, Cheng-Hsing Yang, Cheng-Ta Huang, and Shiuh-Jeng Wang*  
> Submitted to *Signal Processing: Image Communication* (Elsevier), under review, 2026.

---

## Overview

Traditional $(k, n)$ secret image sharing schemes generate noise-like random shares that immediately invite suspicion during transmission or storage. While steganographic secret image sharing embeds shares into meaningful cover containers, conventional approaches frequently suffer from pixel modular overflow, lossy recovery, and high cover distortion.

This repository provides an overflow-free steganographic secret image sharing framework operating entirely within the Galois Field $GF(2^8)$. By combining algebraic polynomial masking with combinatorial least significant bit adjustment, the scheme minimizes cover degradation without requiring auxiliary indicator metadata. Furthermore, an adaptive search variant dynamically modulates search depth according to local cover texture, pruning non-viable candidates to accelerate share generation while preserving high visual quality (with Peak Signal-to-Noise Ratio (PSNR) values exceeding 45 dB on textured covers; PSNR evaluations confirm superior fidelity across all standard test images).

---

## Key Features

- **Galois Field $GF(2^8)$ Arithmetic Engine:** Precomputed look-up tables for multiplication and inversion, Horner polynomial evaluation, and Lagrange polynomial interpolation over $GF(2^8)$.
- **Overflow-Free Steganographic Sharing:** Eliminates modulo-251 truncation artifacts and ensures lossless secret image reconstruction.
- **Adaptive Texture-Driven Search:** Implements local variance classification (smooth versus textured blocks) with beam search pruning and early termination.
- **Steganalysis and Empirical Verification:** Full experimental testbed implementing PSNR, Structural Similarity Index Measure (SSIM), Regular-Singular steganalysis, bit-plane decomposition, and difference visualization.

---

## Repository Structure

```text
polynomial_based_secret_sharing_code/
├── finite_field/                       # Core finite-field mathematical library
│   ├── binary_field.h / .cpp           # GF(2^8) arithmetic, polynomial evaluation, and Lagrange interpolation
│   ├── prime_field.h / .cpp            # Prime field GF(p) baseline implementations
│   └── finite_field.vcxproj            # Visual Studio project for finite field library
├── kn_pb_sharing_bn/                   # Core secret sharing and steganography engine
│   ├── kn_pb_sharing_bn.cpp            # Main experimental harness and benchmark suites
│   ├── adaptive_sharing.h / .cpp       # Texture-aware adaptive search and beam pruning
│   └── kn_pb_sharing_bn.vcxproj        # Visual Studio project for sharing engine
├── tiff_io/                            # Tagged Image File Format image processing
│   ├── tiff_io.h / .cpp                # Grayscale 8-bit image input/output wrappers
│   ├── psnr.cpp                        # Mean squared error and PSNR computation
│   ├── ssim.cpp                        # Structural Similarity Index Measure (SSIM)
│   ├── rs_analysis.cpp                 # Regular-Singular steganalysis implementation
│   ├── validate.cpp                    # Image format, dimension, and integrity validators
│   └── tiff_io.vcxproj                 # Visual Studio project for image processing
├── stegoanalysis/                      # Python analysis and diagnostic visualizer scripts
│   ├── rs.py                           # Regular-Singular steganalysis curve plotting
│   ├── bitplane.py / bitslicer.py      # Bit-plane decomposition and extraction
│   ├── diff-abs.py / diff-xor.py       # Stego versus cover residual error maps
│   └── diff-hist.py / hist.py          # Pixel difference and histogram analysis
├── libtiff/                            # LibTIFF C/C++ development headers and static libraries
└── polynominal_based_secret_sharing.sln # Visual Studio solution
```

---

## Build Prerequisites

- **C++ Compiler:** Supporting C++17 or later (Microsoft Visual C++ 2019/2022 on Windows, or GCC / Clang on Linux).
- **LibTIFF:** Built with 8-bit grayscale support (pre-configured headers and libraries provided in `libtiff/`).
- **Python (Optional, for `stegoanalysis/` tools):** Python 3.8+ with `numpy`, `pillow`, and `matplotlib`.

---

## Building and Running

### Building via Microsoft Visual Studio
1. Open `polynominal_based_secret_sharing.sln` in Visual Studio.
2. Select the **Release** configuration and **x64** platform.
3. Build the solution. The executable will be generated under `x64/Release/`.

### Running Experiments
The entry point in `kn_pb_sharing_bn/kn_pb_sharing_bn.cpp` executes the comprehensive paper benchmark:
```bash
# Execute the automated experimental evaluation
./kn_pb_sharing_bn
```
The program generates stego image shares, validates lossless reconstruction, logs PSNR and SSIM scores across cover sets (such as *airplane*, *baboon*, *house*, and *jetplane*), and exports diagnostic metrics to comma-separated values log files.

### Running Steganalysis Scripts
```bash
cd stegoanalysis

# Run Regular-Singular steganalysis on generated stego images
python rs.py <cover.tiff> <stego.tiff>

# Generate bit-plane decompositions
python bitplane.py <stego.tiff> <output_prefix>

# Compute absolute difference maps
python diff-abs.py <cover.tiff> <stego.tiff> <output_prefix>
```

---

## Citation

If you utilize this code or mathematical implementation in your research, please cite:

```bibtex
@article{lin2026secretsharing,
  author    = {Lin, Li-Chiun and Yang, Cheng-Hsing and Huang, Cheng-Ta and Wang, Shiuh-Jeng},
  title     = {Secret Sharing Steganography Using $GF(2^8)$ Polynomial Interpolation and Least Significant Bit Adjustment},
  journal   = {Signal Processing: Image Communication},
  year      = {2026},
  note      = {Under review}
}
```
