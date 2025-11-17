import argparse
from pathlib import Path
import cv2 as cv
import numpy as np

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"

PROTOTXT   = str(MODELS / "colorization_deploy_v2.prototxt")
CAFFEMODEL = str(MODELS / "colorization_release_v2.caffemodel")
PTS_NPY    = str(MODELS / "pts_in_hull.npy")

def preferred_dir(prefer: Path, fallback: Path) -> Path:
    return prefer if prefer.exists() else fallback

def colorize_bgr(img_bgr):
    net = cv.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

    # Inyectar centros ab (313x2) y re-peso
    pts = np.load(PTS_NPY).transpose().reshape(2,313,1,1).astype(np.float32)
    net.getLayer(net.getLayerId("class8_ab")).blobs    = [pts]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full((1,313), 2.606, np.float32)]

    # 1) Normaliza a [0,1] ANTES de convertir a Lab (clave)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)   # L en [0,100], a,b ~ [-128,127]
    L = img_lab[:, :, 0]

    # Entrada del modelo: 224x224 y centrado L-50 (pipeline Zhang)
    L_rs = cv.resize(L, (224, 224))
    L_rs -= 50
    blob = cv.dnn.blobFromImage(L_rs)

    # Forward
    net.setInput(blob)
    ab = net.forward()[0].transpose(1, 2, 0)           # H'×W'×2
    ab_up = cv.resize(ab, (img_bgr.shape[1], img_bgr.shape[0]))

    # 2) Reconstrucción: Lab float -> RGB float en [0,1]
    Lab_out = np.concatenate([L[:, :, None], ab_up], axis=2).astype(np.float32)
    out_rgb = cv.cvtColor(Lab_out, cv.COLOR_Lab2RGB)

    # 3) Escala a [0,255] ANTES de convertir a uint8 (clave)
    out_rgb = (np.clip(out_rgb, 0, 1) * 255.0).astype(np.uint8)
    out_bgr = cv.cvtColor(out_rgb, cv.COLOR_RGB2BGR)
    return out_bgr

if __name__ == "__main__":
    # Directorios por defecto:
    default_in  = preferred_dir(BASE / "imgs")
    default_out = preferred_dir(BASE / "imgs_out")

    ap = argparse.ArgumentParser(description="Colorización con OpenCV-DNN (Zhang 2016)")
    ap.add_argument("--in",  dest="indir",  type=str, default=str(default_in),
                    help="Carpeta de entrada (por defecto imgs)")
    ap.add_argument("--out", dest="outdir", type=str, default=str(default_out),
                    help="Carpeta de salida (por defecto imgs_out)")
    ap.add_argument("--img", dest="image",  type=str, default=None,
                    help="Ruta a una imagen concreta (si se usa, ignora --all)")
    ap.add_argument("--all", action="store_true",
                    help="Procesa todas las imágenes de la carpeta de entrada")
    args = ap.parse_args()

    in_dir  = Path(args.indir).resolve()
    out_dir = Path(args.outdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image and args.all:
        raise SystemExit("Usa --img o --all, pero no ambos.")

    if args.image:
        img_path = Path(args.image)
        img_bgr = cv.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")
        out = colorize_bgr(img_bgr)
        out_path = out_dir / f"{img_path.stem}_color.png"
        cv.imwrite(str(out_path), out)
        print(f"OK -> {out_path}")
    else:
        # --all (o sin --img): procesa toda la carpeta
        exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
        files = []
        for e in exts:
            files += list(in_dir.glob(e))
        if not files:
            raise SystemExit(f"No se encontraron imágenes en: {in_dir}")

        for p in files:
            img = cv.imread(str(p))
            if img is None:
                print(f"[AVISO] No se pudo leer: {p}")
                continue
            out = colorize_bgr(img)
            out_path = out_dir / f"{p.stem}_color.png"
            cv.imwrite(str(out_path), out)
            print(f"OK -> {out_path}")
