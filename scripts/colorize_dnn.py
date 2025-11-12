import cv2 as cv
import numpy as np
import os

PROTOTXT = "models/colorization_deploy_v2.prototxt"
CAFFEMODEL = "models/colorization_release_v2.caffemodel"  # local, pesado
PTS_NPY = "models/pts_in_hull.npy"

def colorize_bgr(img_bgr):
    net = cv.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

    # Inyectar centros ab (313x2) y re-peso
    pts = np.load(PTS_NPY).transpose().reshape(2,313,1,1).astype("float32")
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full((1,313), 2.606, np.float32)]

    # BGR -> Lab y extracción de L
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab).astype("float32")
    L = img_lab[:, :, 0]

    # Entrada del modelo: 224x224 y centrado L-50
    L_rs = cv.resize(L, (224, 224))
    L_rs -= 50
    blob = cv.dnn.blobFromImage(L_rs)

    # Forward
    net.setInput(blob)
    ab = net.forward()[0].transpose(1, 2, 0)
    ab_up = cv.resize(ab, (img_bgr.shape[1], img_bgr.shape[0]))

    # Reconstrucción Lab -> RGB -> BGR
    Lab_out = np.concatenate([L[:, :, None], ab_up], axis=2).astype("float32")
    out_rgb = cv.cvtColor(Lab_out, cv.COLOR_Lab2RGB)
    out_rgb = np.clip(out_rgb, 0, 255).astype("uint8")
    return cv.cvtColor(out_rgb, cv.COLOR_RGB2BGR)

if __name__ == "__main__":
    os.makedirs("data/results", exist_ok=True)
    img = cv.imread("data/raw/ejemplo.jpg")
    if img is None:
        raise FileNotFoundError("Pon una imagen de prueba en data/raw/ejemplo.jpg")
    out = colorize_bgr(img)
    cv.imwrite("data/results/ejemplo_color.png", out)
    print("OK -> data/results/ejemplo_color.png")
