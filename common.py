"""
common.py — utilitas bersama untuk seluruh notebook eksperimen.

Isi:
  * konfigurasi global (seed, device, batch size, direktori)
  * pemuatan Fashion-MNIST + split terstandarisasi 80/10/10
  * metrik rekonstruksi (MSE, PSNR, SSIM) dan FID
  * pengukuran training time & inference latency
  * helper visualisasi grid citra
"""
import os        # operasi path & folder
import json      # simpan/baca hasil dalam format JSON
import time      # pengukuran waktu
import random    # generator acak bawaan Python

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset   # pembungkus dataset & pemuat batch
import torchvision                                       # untuk download Fashion-MNIST & membuat grid gambar

# ---------------------------------------------------------------------------
# Konfigurasi global
# ---------------------------------------------------------------------------
SEED = 42                 # seed tetap -> hasil eksperimen bisa diulang
BATCH_SIZE = 128          # jumlah gambar per batch
IMG_SHAPE = (1, 28, 28)   # (channel, tinggi, lebar): grayscale 28x28
IMG_DIM = 28 * 28         # 784 piksel jika gambar di-flatten
# Pakai GPU (CUDA) bila tersedia, jika tidak pakai CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folder-folder proyek (relatif terhadap lokasi file ini)
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")                # dataset hasil download
CKPT_DIR = os.path.join(ROOT, "checkpoints")         # bobot model terlatih (.pt)
RESULTS_DIR = os.path.join(ROOT, "results")          # hasil metrik (.json, .csv, sampel)
FIG_DIR = os.path.join(ROOT, "figures")              # gambar/figur untuk laporan
# Buat folder jika belum ada
for _d in (DATA_DIR, CKPT_DIR, RESULTS_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

# Nama 10 kelas Fashion-MNIST (indeks 0-9)
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def set_seed(seed: int = SEED) -> None:
    """Menetapkan seluruh sumber keacakan (Python, NumPy, PyTorch CPU/GPU)."""
    random.seed(seed)                            # acak Python
    np.random.seed(seed)                         # acak NumPy
    torch.manual_seed(seed)                      # acak PyTorch (CPU)
    torch.cuda.manual_seed_all(seed)             # acak PyTorch (semua GPU)
    torch.backends.cudnn.benchmark = False       # jangan pilih algoritma conv secara otomatis (bisa beda tiap run)
    torch.backends.cudnn.deterministic = True    # paksa cuDNN memakai algoritma deterministik


# ---------------------------------------------------------------------------
# Dataset & split 80 / 10 / 10
# ---------------------------------------------------------------------------
def load_fashion_mnist():
    """Menggabungkan train (60k) + test (10k) resmi menjadi 70k citra dalam [0, 1]."""
    # Download (jika belum ada) split train & test resmi
    tr = torchvision.datasets.FashionMNIST(DATA_DIR, train=True, download=True)
    te = torchvision.datasets.FashionMNIST(DATA_DIR, train=False, download=True)
    # Gabungkan -> tambah dimensi channel -> ubah ke float -> skala 0-255 menjadi 0-1
    x = torch.cat([tr.data, te.data]).unsqueeze(1).float().div_(255.0)  # (N,1,28,28)
    # Gabungkan label kelas (0-9)
    y = torch.cat([tr.targets, te.targets]).long()
    return x, y


def get_splits(seed: int = SEED):
    """Split acak terstandarisasi 80% train / 10% val / 10% test dengan seed tetap.

    Indeks split disimpan ke results/split_indices.npz agar dapat diaudit ulang.
    """
    x, y = load_fashion_mnist()
    n = len(x)                                    # 70.000
    g = torch.Generator().manual_seed(seed)       # generator acak khusus agar urutan acak selalu sama
    perm = torch.randperm(n, generator=g)         # urutan indeks yang diacak
    n_tr, n_va = int(0.8 * n), int(0.1 * n)       # 56.000 train, 7.000 val (sisanya 7.000 test)
    idx = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    # Simpan indeks tiap split ke file agar bisa diperiksa ulang
    np.savez(os.path.join(RESULTS_DIR, "split_indices.npz"),
             **{k: v.numpy() for k, v in idx.items()})
    # Bungkus tiap split menjadi TensorDataset (pasangan gambar, label)
    return {k: TensorDataset(x[v], y[v]) for k, v in idx.items()}


def get_loaders(batch_size: int = BATCH_SIZE, seed: int = SEED):
    """DataLoader train (shuffle=True) serta val/test (shuffle=False)."""
    ds = get_splits(seed)
    g = torch.Generator().manual_seed(seed)   # agar urutan shuffle train bisa diulang
    # Train diacak tiap epoch; val & test tidak perlu diacak
    train = DataLoader(ds["train"], batch_size=batch_size, shuffle=True, generator=g)
    val = DataLoader(ds["val"], batch_size=batch_size, shuffle=False)
    test = DataLoader(ds["test"], batch_size=batch_size, shuffle=False)
    return train, val, test


def to_tanh_range(x):
    """[0,1] -> [-1,1] (dipakai GAN, DCGAN, DDPM)."""
    return x * 2.0 - 1.0


def from_tanh_range(x):
    """[-1,1] -> [0,1]."""
    # clamp memastikan nilai tidak keluar dari rentang [0, 1]
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Metrik rekonstruksi: MSE, PSNR, SSIM (dievaluasi pada Test Set)
# ---------------------------------------------------------------------------
@torch.no_grad()   # evaluasi saja, tanpa gradien
def reconstruction_metrics(recon_fn, loader, device=DEVICE):
    """recon_fn: x(B,1,28,28 di [0,1]) -> x_hat (B,1,28,28 di [0,1]).

    MSE dirata-rata per piksel; PSNR & SSIM dihitung per citra lalu dirata-rata.
    """
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
    mse_all, psnr_all, ssim_all = [], [], []
    for x, _ in loader:
        x = x.to(device)
        xh = recon_fn(x).clamp(0, 1)                          # rekonstruksi, dijaga di [0, 1]
        mse = ((xh - x) ** 2).flatten(1).mean(1)              # MSE per gambar
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))  # PSNR (dB); clamp mencegah log(0)
        ssim = ssim_fn(xh, x, data_range=1.0, reduction="none")  # SSIM per gambar
        mse_all.append(mse.cpu()); psnr_all.append(psnr.cpu()); ssim_all.append(ssim.cpu())
    # Gabungkan hasil semua batch
    mse_all, psnr_all, ssim_all = map(torch.cat, (mse_all, psnr_all, ssim_all))
    # Kembalikan rata-rata tiap metrik (+ standar deviasi MSE)
    return {"mse": mse_all.mean().item(), "psnr": psnr_all.mean().item(),
            "ssim": ssim_all.mean().item(), "mse_std": mse_all.std().item()}


# ---------------------------------------------------------------------------
# FID (Fréchet Inception Distance) — Inception-v3 pool3 2048-d
# ---------------------------------------------------------------------------
class FIDEvaluator:
    """Statistik citra riil (Test Set) dihitung sekali, lalu dipakai ulang untuk tiap model.

    Citra grayscale 28x28 direplikasi ke 3 kanal; resize ke 299x299 dilakukan di dalam
    ekstraktor Inception milik torchmetrics/torch-fidelity.
    """

    def __init__(self, real_images, device=DEVICE, batch=100):
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.device, self.batch = device, batch
        # feature=2048: fitur dari layer pool3 Inception-v3
        # normalize=True: input berupa float di [0, 1]
        # reset_real_features=False: statistik citra riil tidak ikut terhapus saat reset()
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True,
                                            reset_real_features=False).to(device)
        # Masukkan citra riil per batch untuk menghitung statistik (mean & kovarians) fitur riil
        for i in range(0, len(real_images), batch):
            self.fid.update(self._prep(real_images[i:i + batch]), real=True)

    def _prep(self, x):
        # Inception butuh 3 channel (RGB) -> salin channel grayscale menjadi 3
        return x.to(self.device).float().clamp(0, 1).repeat(1, 3, 1, 1)

    @torch.no_grad()
    def score(self, fake_images):
        """fake_images: tensor (N,1,28,28) di [0,1]."""
        self.fid.reset()  # real features dipertahankan
        # Masukkan citra sintetis per batch
        for i in range(0, len(fake_images), self.batch):
            self.fid.update(self._prep(fake_images[i:i + self.batch]), real=False)
        # FID = jarak antara distribusi fitur riil dan sintetis (makin kecil makin baik)
        return self.fid.compute().item()


# ---------------------------------------------------------------------------
# Waktu & latensi
# ---------------------------------------------------------------------------
def sync(device=DEVICE):
    # Operasi GPU berjalan asinkron -> tunggu sampai selesai agar pengukuran waktu akurat
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_latency(fn, n_warmup=5, n_runs=30, device=DEVICE):
    """Median latensi (ms) satu pemanggilan fn() dengan sinkronisasi CUDA."""
    # Warm-up: beberapa panggilan pertama biasanya lambat (inisialisasi kernel/cache), tidak dihitung
    for _ in range(n_warmup):
        fn()
    sync(device)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)   # detik -> milidetik
    # Median lebih tahan terhadap outlier dibanding rata-rata
    return float(np.median(times)), float(np.std(times))


class EpochTimer:
    # Context manager: `with EpochTimer() as tm:` lalu durasinya ada di tm.seconds
    def __enter__(self):
        sync(); self.t0 = time.perf_counter(); return self

    def __exit__(self, *a):
        sync(); self.seconds = time.perf_counter() - self.t0


# ---------------------------------------------------------------------------
# I/O hasil
# ---------------------------------------------------------------------------
def save_json(obj, name):
    # Simpan objek (dict/list) ke results/<name>
    with open(os.path.join(RESULTS_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(name):
    # Baca results/<name>
    with open(os.path.join(RESULTS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def count_params(model):
    # Jumlah total parameter (bobot + bias) sebuah model
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Visualisasi
# ---------------------------------------------------------------------------
def show_grid(images, nrow=8, ax=None, title=None):
    """images: (N,1,28,28) di [0,1]."""
    import matplotlib.pyplot as plt
    # Susun banyak gambar menjadi satu gambar grid (nrow gambar per baris), dengan pemisah putih
    grid = torchvision.utils.make_grid(images.detach().cpu().clamp(0, 1), nrow=nrow,
                                       padding=2, pad_value=1.0)
    # Buat axes baru jika tidak diberikan
    if ax is None:
        _, ax = plt.subplots(figsize=(nrow, len(images) / nrow + 0.5))
    # (C,H,W) -> (H,W,C) karena matplotlib butuh channel di akhir
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


# Mode uji cepat (QUICK_RUN=1) untuk memeriksa pipeline end-to-end sebelum training penuh.
QUICK = os.environ.get("QUICK_RUN", "0") == "1"
