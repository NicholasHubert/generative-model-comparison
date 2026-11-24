# Studi Komparatif Empiris Model Generatif — Fashion-MNIST

Tugas Take-Home Deep Learning (D6421) Sesi 07. Proyek ini mengimplementasikan, melatih, dan membandingkan enam
model generatif pada dataset, split, dan seed yang sama:

| Keluarga | Model | Notebook |
|---|---|---|
| Autoencoder deterministik | Vanilla AE, Sparse AE ($\lambda\lVert h\rVert_1$) | `01_autoencoders.ipynb` |
| Autoencoder probabilistik | VAE | `01_autoencoders.ipynb` |
| Adversarial | Vanilla GAN (MLP), DCGAN | `02_gans.ipynb` |
| Difusi | DDPM (U-Net, T = 1000) | `03_ddpm.ipynb` |
| Pembuktian batasan | MLP Classifier 784→256→64→10 + 5 tahap eksperimen | `04_mlp_classifier_limits.ipynb` |
| Evaluasi | MSE/PSNR/SSIM, FID, mode coverage, latency, Figur 1–5 | `05_evaluation.ipynb` |

## Struktur repositori

```
common.py        # seed, split 80/10/10, DataLoader, metrik (MSE/PSNR/SSIM/FID), latency, plotting
models.py        # seluruh arsitektur: AE, SparseAE, VAE, MLPClassifier, GAN, DCGAN, U-Net + DDPM
01_…05_*.ipynb   # pipeline eksperimen (dijalankan berurutan)
run_all.py       # eksekusi seluruh notebook secara headless
results/         # metrik JSON/CSV, riwayat loss, sampel (uint8) — dihasilkan otomatis
figures/         # Figur 1–5 + figur suplementer (PNG 200 dpi)
checkpoints/     # bobot model (diabaikan git)
data/            # Fashion-MNIST (diunduh otomatis via torchvision, diabaikan git)
```

## Standardisasi eksperimen

* **Dataset**: Fashion-MNIST (60k train + 10k test resmi digabung = 70k), lalu di-split acak **80/10/10**
  (56.000 / 7.000 / 7.000) dengan `torch.Generator().manual_seed(42)`. Indeks split disimpan di
  `results/split_indices.npz`.
* **Seed**: `random`, `numpy`, `torch`, `torch.cuda` = 42 (`common.set_seed`).
* **DataLoader**: `batch_size=128`, `shuffle=True` (train), `shuffle=False` (val/test).
* **Epoch**: 30 untuk keenam model generatif; MLP Classifier 20 epoch; frozen latent decoder 30 epoch.
* **Evaluasi**: seluruh metrik numerik dihitung pada **Test Set**. FID memakai Inception-v3 (2048-d) dengan
  2.000 sampel sintetis per model dibandingkan 7.000 citra Test Set.

## Cara menjalankan

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

python run_all.py --quick          # uji pipeline (~10 menit, 1 epoch)
python run_all.py                  # eksperimen penuh (~1–1,5 jam pada RTX 2050 4 GB)
python run_all.py 05               # hanya menjalankan ulang evaluasi
```

Notebook juga dapat dijalankan satu per satu di Jupyter/VS Code dengan urutan 01 → 05. Notebook 04 membutuhkan
checkpoint Vanilla AE dari 01, dan notebook 05 membutuhkan seluruh checkpoint sebelumnya.

> Catatan: unduhan pertama FID akan mengambil bobot Inception-v3 milik `torch-fidelity` (~100 MB).

## Hasil (Test Set, NVIDIA RTX 2050 4 GB)

| Model | Test MSE | PSNR (dB) | SSIM | FID ↓ (2000 sampel) | Latensi GPU (ms / 64) |
|---|---|---|---|---|---|
| Vanilla AE | 0.0065 | 23.01 | 0.866 | 211.3† | 0.23 |
| Sparse AE | 0.0079 | 22.12 | 0.842 | 168.0† | 0.19 |
| VAE | 0.0129 | 19.71 | 0.770 | 86.0 | 0.10 |
| Vanilla GAN | – | – | – | 149.7 | 0.22 |
| DCGAN | – | – | – | 27.1 | 3.20 |
| DDPM | – | – | – | **17.3** | 23 367 (1000 langkah) |

† AE tidak memiliki prior; FID diukur dari Gaussian yang di-fit pada laten train (kontrol *latent voids*).
Referensi FID riil-vs-riil = 8.4. MLP Classifier: akurasi test 89.06%; frozen latent decoder MSE 0.0204 vs Vanilla AE 0.0065.

Detail lengkap: `results/metrics_table.csv`, `results/final_metrics.json`, `results/mlp_limits.json`, dan folder `figures/`.

> Catatan Windows/laptop: `cudnn.benchmark` sengaja dimatikan pada DDPM karena memicu kompilasi JIT cuDNN 9 yang
> menghabiskan RAM host (*Memory allocation failure*). Training DDPM menyimpan checkpoint tiap epoch
> (`checkpoints/ddpm_resume.pt`) dan otomatis melanjutkan bila proses terhenti.
