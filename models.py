"""
models.py — definisi arsitektur seluruh model.

  * VanillaAE, SparseAE, VAE        : MLP 784 -> 256 -> 64 (encoder identik dgn MLP Classifier)
  * MLPClassifier, LatentDecoder    : eksperimen batasan model diskriminatif (Bagian 2)
  * MLPGenerator / MLPDiscriminator : Vanilla GAN (fully connected)
  * DCGANGenerator / DCGANDiscriminator
  * UNet + DDPM                      : Denoising Diffusion Probabilistic Model
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_DIM = 28 * 28   # 784 piksel


# ===========================================================================
# 1. Keluarga Autoencoder
# ===========================================================================
class MLPDecoder(nn.Module):
    """Decoder MLP 64 -> 256 -> 784 dengan Sigmoid (output di [0,1])."""

    def __init__(self, latent_dim=64, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden)   # 64 -> 256
        self.fc2 = nn.Linear(hidden, IMG_DIM)      # 256 -> 784

    def forward(self, z):
        # z -> Linear -> ReLU -> Linear -> Sigmoid (agar piksel di [0,1]) -> bentuk ulang jadi gambar 28x28
        return torch.sigmoid(self.fc2(F.relu(self.fc1(z)))).view(-1, 1, 28, 28)


class VanillaAE(nn.Module):
    """Autoencoder deterministik: z = f(x), x_hat = g(z). Loss: MSE."""

    def __init__(self, latent_dim=64, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(IMG_DIM, hidden)      # encoder layer 1: 784 -> 256
        self.fc2 = nn.Linear(hidden, latent_dim)   # encoder layer 2: 256 -> 64 (kode laten)
        self.decoder = MLPDecoder(latent_dim, hidden)

    def encode(self, x, return_hidden=False):
        h = F.relu(self.fc1(x.flatten(1)))   # flatten gambar (B,1,28,28) -> (B,784), lalu hidden 256
        z = self.fc2(h)                      # kode laten 64 (tanpa aktivasi / linear)
        # return_hidden=True juga mengembalikan h (dipakai Sparse AE & analisis sparsity)
        return (z, h) if return_hidden else z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, h = self.encode(x, return_hidden=True)
        # Output: (rekonstruksi, kode laten, aktivasi hidden)
        return self.decode(z), z, h


class SparseAE(VanillaAE):
    """Arsitektur identik dgn VanillaAE; perbedaannya pada loss:
    L = MSE(x, x_hat) + lambda * ||h||_1   (h = aktivasi hidden 256-d)."""

    def __init__(self, latent_dim=64, hidden=256, l1_lambda=3e-5):
        super().__init__(latent_dim, hidden)
        self.l1_lambda = l1_lambda   # bobot penalti sparsity

    def sparsity_penalty(self, h):
        # Jumlah |h| per gambar (norma L1), dirata-rata atas batch, dikali lambda
        return self.l1_lambda * h.abs().sum(dim=1).mean()


class VAE(nn.Module):
    """VAE: q(z|x) = N(mu, diag(sigma^2)), p(z) = N(0, I), p(x|z) = Bernoulli.
    Loss (negatif ELBO): BCE(x, x_hat) + KL(q(z|x) || N(0,I))."""

    def __init__(self, latent_dim=64, hidden=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(IMG_DIM, hidden)            # 784 -> 256
        self.fc_mu = nn.Linear(hidden, latent_dim)       # 256 -> 64 : rata-rata mu
        self.fc_logvar = nn.Linear(hidden, latent_dim)   # 256 -> 64 : log varians
        self.decoder = MLPDecoder(latent_dim, hidden)

    def encode(self, x):
        h = F.relu(self.fc1(x.flatten(1)))
        # Encoder mengeluarkan parameter distribusi, bukan satu titik z
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        # Reparameterization trick: z = mu + sigma * eps, eps ~ N(0, I)
        # sigma = exp(0.5 * logvar). Keacakan dipindah ke eps agar gradien tetap mengalir ke mu & logvar
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)   # ambil sampel z dari q(z|x)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss(x, x_hat, mu, logvar):
        """Dikembalikan per-sampel (dijumlah atas piksel/dimensi laten, dirata-rata atas batch)."""
        b = x.size(0)   # ukuran batch
        # BCE: kemiripan rekonstruksi (piksel dianggap Bernoulli)
        bce = F.binary_cross_entropy(x_hat, x, reduction="sum") / b
        # KL( N(mu, sigma^2) || N(0, I) ) dalam bentuk tertutup (closed form)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
        return bce + kl, bce, kl


# ===========================================================================
# 2. MLP Classifier (eksperimen batasan model diskriminatif)
# ===========================================================================
class MLPClassifier(nn.Module):
    """784 -> 256 -> 64 -> 10. Lapisan 784->256->64 identik secara struktural dgn encoder AE."""

    def __init__(self, hidden=256, latent_dim=64, n_classes=10, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(IMG_DIM, hidden)    # W1: (256, 784)
        self.fc2 = nn.Linear(hidden, latent_dim)  # W2: (64, 256)
        self.fc3 = nn.Linear(latent_dim, n_classes)  # W3: (10, 64)
        self.drop = nn.Dropout(dropout)   # dropout untuk regularisasi (mati saat eval)

    def features(self, x):
        """z_MLP in R^64 (aktivasi lapisan sebelum klasifikasi akhir)."""
        h = self.drop(F.relu(self.fc1(x.flatten(1))))
        return F.relu(self.fc2(h))

    def forward(self, x):
        # Output berupa logit 10 kelas (softmax ada di dalam cross_entropy)
        return self.fc3(self.drop(self.features(x)))


LatentDecoder = MLPDecoder  # decoder terpisah 64 -> 256 -> 784 untuk frozen latent decoding


# ===========================================================================
# 3. Vanilla GAN (fully connected)
# ===========================================================================
class MLPGenerator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        # Noise z (100) diperbesar bertahap 256 -> 512 -> 1024 -> 784 piksel
        # LeakyReLU(0.2): seperti ReLU tetapi nilai negatif dikali 0.2 (gradien tidak mati)
        # Tanh di akhir: output piksel di rentang [-1, 1]
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, IMG_DIM), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)   # (B,784) -> (B,1,28,28)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Gambar 784 -> 512 -> 256 -> 1 skor (logit "asli vs palsu")
        self.net = nn.Sequential(
            nn.Linear(IMG_DIM, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),  # logit; sigmoid ada di BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x.flatten(1)).squeeze(1)   # (B,1) -> (B,)


# ===========================================================================
# 4. DCGAN (Radford et al., 2016) untuk 28x28
# ===========================================================================
class DCGANGenerator(nn.Module):
    """z(100) -> 256x7x7 -> 128x14x14 -> 1x28x28. ConvTranspose + BatchNorm + ReLU, output Tanh."""

    def __init__(self, z_dim=100, ngf=64):
        super().__init__()
        self.z_dim = z_dim
        # ConvTranspose2d(in, out, kernel, stride, padding) = "konvolusi terbalik" untuk memperbesar gambar
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 4, 7, 1, 0, bias=False),   # 1x1 -> 7x7, 256 channel
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # 7x7 -> 14x14, 128 channel
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1),                   # 14x14 -> 28x28, 1 channel
            nn.Tanh(),                                                 # piksel di [-1, 1]
        )

    def forward(self, z):
        # z (B,100) dibentuk ulang jadi "gambar" 100 channel ukuran 1x1
        return self.net(z.view(z.size(0), -1, 1, 1))


class DCGANDiscriminator(nn.Module):
    """Strided conv (tanpa pooling) + BatchNorm + LeakyReLU; tanpa lapisan FC tersembunyi."""

    def __init__(self, ndf=64):
        super().__init__()
        # Conv dengan stride 2 memperkecil gambar (pengganti pooling)
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),                    # 14x14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),                         # 7x7
            nn.Conv2d(ndf * 2, 1, 7, 1, 0),                                           # 1x1 logit
        )

    def forward(self, x):
        return self.net(x).view(-1)   # (B,1,1,1) -> (B,)


def dcgan_weights_init(m):
    """Inisialisasi bobot N(0, 0.02) sesuai paper DCGAN."""
    name = m.__class__.__name__
    if "Conv" in name:
        # Bobot conv ~ N(0, 0.02), bias = 0
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif "BatchNorm" in name:
        # Skala BatchNorm ~ N(1, 0.02), geser = 0
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# ===========================================================================
# 5. DDPM (Ho et al., 2020) — U-Net kecil + noise schedule linear
# ===========================================================================
def timestep_embedding(t, dim):
    """Sinusoidal positional embedding untuk timestep t (B,) -> (B, dim)."""
    # Mengubah angka t (0..999) menjadi vektor sin/cos dengan berbagai frekuensi,
    # agar jaringan bisa "tahu" sedang berada di langkah noise ke berapa
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]     # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)   # (B, dim)


class ResBlock(nn.Module):
    # Blok residual: dua conv 3x3 + info timestep, ditambah jalur pintas (skip connection)
    def __init__(self, cin, cout, tdim, dropout=0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)              # normalisasi per grup channel
        self.c1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.t = nn.Linear(tdim, cout)              # proyeksi time embedding ke jumlah channel
        self.n2 = nn.GroupNorm(8, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1)
        # Jika jumlah channel berubah, skip connection butuh conv 1x1 agar ukurannya cocok
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))                    # norm -> SiLU -> conv
        h = h + self.t(F.silu(temb))[:, :, None, None]     # tambahkan info timestep ke tiap channel
        h = self.c2(self.drop(F.silu(self.n2(h))))         # norm -> SiLU -> dropout -> conv
        return h + self.skip(x)                            # residual: output + input


class AttnBlock(nn.Module):
    # Self-attention: tiap posisi piksel bisa "melihat" semua posisi lain (konteks global)
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)   # hasilkan query, key, value sekaligus
        self.proj = nn.Conv2d(ch, ch, 1)      # proyeksi output

    def forward(self, x):
        b, c, h, w = x.shape
        # Pisahkan menjadi q, k, v masing-masing (B, C, H*W)
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        # Attention atas H*W posisi (transpose -> (B, H*W, C))
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        # Kembalikan ke bentuk gambar, lalu tambahkan residual
        return x + self.proj(out.transpose(1, 2).reshape(b, c, h, w))


class UNet(nn.Module):
    """U-Net epsilon_theta(x_t, t): resolusi 28 -> 14 -> 7, self-attention pada bottleneck."""

    def __init__(self, ch=32, ch_mult=(1, 2, 4), num_res=2, dropout=0.1):
        super().__init__()
        self.ch = ch
        tdim = ch * 4
        # MLP untuk memproses time embedding
        self.temb = nn.Sequential(nn.Linear(ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.conv_in = nn.Conv2d(1, ch, 3, 1, 1)   # gambar 1 channel -> ch channel

        # ----- Jalur turun (encoder): perkecil resolusi, perbanyak channel -----
        self.down = nn.ModuleList()
        chans, cur = [ch], ch   # chans mencatat jumlah channel tiap output untuk skip connection
        for i, m in enumerate(ch_mult):
            for _ in range(num_res):
                self.down.append(ResBlock(cur, ch * m, tdim, dropout))
                cur = ch * m
                chans.append(cur)
            if i != len(ch_mult) - 1:
                self.down.append(nn.Conv2d(cur, cur, 3, 2, 1))  # downsample
                chans.append(cur)

        # ----- Bottleneck (resolusi terkecil 7x7) dengan self-attention -----
        self.mid1 = ResBlock(cur, cur, tdim, dropout)
        self.mid_attn = AttnBlock(cur)
        self.mid2 = ResBlock(cur, cur, tdim, dropout)

        # ----- Jalur naik (decoder): perbesar resolusi, gabungkan dengan skip dari jalur turun -----
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(ch_mult))):
            for _ in range(num_res + 1):
                # input = channel saat ini + channel skip connection (di-concat)
                self.up.append(ResBlock(cur + chans.pop(), ch * m, tdim, dropout))
                cur = ch * m
            if i != 0:
                # upsample x2 (nearest) lalu conv
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                             nn.Conv2d(cur, cur, 3, 1, 1)))
        # Output: prediksi noise, 1 channel (sama dengan ukuran gambar)
        self.out = nn.Sequential(nn.GroupNorm(8, cur), nn.SiLU(), nn.Conv2d(cur, 1, 3, 1, 1))

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.ch))   # embedding timestep
        h = self.conv_in(x)
        hs = [h]   # simpan semua output jalur turun untuk skip connection
        for layer in self.down:
            h = layer(h, temb) if isinstance(layer, ResBlock) else layer(h)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for layer in self.up:
            if isinstance(layer, ResBlock):
                # gabungkan (concat) dengan fitur jalur turun yang ukurannya sama
                h = layer(torch.cat([h, hs.pop()], dim=1), temb)
            else:
                h = layer(h)
        return self.out(h)


class DDPM(nn.Module):
    """Forward process q(x_t|x_0) = N(sqrt(a_bar_t) x_0, (1-a_bar_t) I);
    objektif sederhana: L = E || eps - eps_theta(x_t, t) ||^2."""

    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model, self.T = model, T
        betas = torch.linspace(beta_start, beta_end, T)   # jadwal noise linear: beta_1..beta_T
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, 0)              # alpha_bar_t = perkalian alpha_1..alpha_t
        # register_buffer: tensor ikut pindah ke GPU & tersimpan, tetapi bukan parameter yang dilatih
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0, t, noise):
        # Forward process: langsung lompat dari x0 ke x_t dalam satu langkah
        # x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, x0):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)   # timestep acak per gambar
        noise = torch.randn_like(x0)                                    # noise asli
        # U-Net menebak noise dari x_t; loss = MSE antara noise asli dan tebakan
        return F.mse_loss(self.model(self.q_sample(x0, t, noise), t), noise)

    @torch.no_grad()
    def sample(self, n, device, return_traj_at=None, amp=False):
        """Ancestral sampling x_T ~ N(0,I) -> x_0 (rentang [-1,1]).
        return_traj_at: daftar timestep yang ingin direkam untuk visualisasi trajektori."""
        x = torch.randn(n, 1, 28, 28, device=device)   # mulai dari noise murni x_T
        traj = {}
        for i in reversed(range(self.T)):              # t = 999, 998, ..., 0
            t = torch.full((n,), i, device=device, dtype=torch.long)
            # autocast fp16 (opsional) untuk mempercepat di GPU
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                eps = self.model(x, t)   # tebakan noise
            eps = eps.float()
            a, ab, b = self.alphas[i], self.alpha_bar[i], self.betas[i]
            # Rata-rata x_{t-1}: buang sebagian noise yang ditebak
            mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            # Tambahkan sedikit noise acak, kecuali di langkah terakhir (t=0)
            x = mean + b.sqrt() * torch.randn_like(x) if i > 0 else mean
            # Rekam gambar antara untuk visualisasi
            if return_traj_at is not None and i in return_traj_at:
                traj[i] = x.clamp(-1, 1).cpu()
        x = x.clamp(-1, 1)
        return (x, traj) if return_traj_at is not None else x


class EMA:
    """Exponential moving average bobot (dipakai untuk sampling DDPM)."""

    def __init__(self, model, decay=0.999):
        import copy
        self.decay = decay
        # Salinan model ("bayangan") yang bobotnya adalah rata-rata bergerak bobot asli
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)   # salinan tidak dilatih langsung

    @torch.no_grad()
    def update(self, model):
        # shadow = decay * shadow + (1 - decay) * bobot_sekarang  -> bobot lebih halus/stabil
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        # Buffer (mis. alpha_bar) cukup disalin apa adanya
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)
