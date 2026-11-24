"""
run_all.py — mengeksekusi seluruh notebook secara berurutan (output disimpan in-place).

    python run_all.py            # training penuh
    python run_all.py --quick    # uji pipeline cepat (1 epoch, sampel FID sedikit)
    python run_all.py 03 05      # hanya notebook dengan prefiks tertentu
"""
import os
import sys
import time

import nbformat                      # membaca/menulis file .ipynb
from nbclient import NotebookClient  # menjalankan notebook tanpa membuka Jupyter

# Urutan eksekusi penting: notebook berikutnya memakai checkpoint/hasil notebook sebelumnya
NOTEBOOKS = ["01_autoencoders.ipynb", "02_gans.ipynb", "03_ddpm.ipynb",
             "04_mlp_classifier_limits.ipynb", "05_evaluation.ipynb"]

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))   # folder proyek
    args = sys.argv[1:]                                 # argumen dari command line
    # --quick -> set env QUICK_RUN=1 sehingga common.QUICK = True di setiap notebook
    if "--quick" in args:
        os.environ["QUICK_RUN"] = "1"
        args.remove("--quick")
    # Tanpa argumen: jalankan semua. Dengan argumen (mis. "03"): hanya notebook berawalan itu
    todo = [nb for nb in NOTEBOOKS if not args or any(nb.startswith(a) for a in args)]
    for nb_name in todo:
        path = os.path.join(here, nb_name)
        nb = nbformat.read(path, as_version=4)   # baca notebook
        t0 = time.time()
        print(f"==> {nb_name}", flush=True)
        # Jalankan semua sel; timeout=None = tanpa batas waktu (training bisa lama)
        NotebookClient(nb, timeout=None, kernel_name="python3",
                       resources={"metadata": {"path": here}}).execute()
        nbformat.write(nb, path)                 # simpan notebook beserta output-nya
        print(f"    selesai dalam {(time.time() - t0) / 60:.1f} menit", flush=True)
