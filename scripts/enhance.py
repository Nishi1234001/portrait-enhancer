import cv2
import numpy as np
import mediapipe as mp
import argparse

mp_selfie = mp.solutions.selfie_segmentation
mp_face = mp.solutions.face_detection

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

def unsharp_mask(img, strength=1.2, sigma=1.5):
    blurred = cv2.GaussianBlur(img, (0,0), sigma)
    return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)

def enhance_image(input_path, output_path):
    img = cv2.imread(input_path)
    h, w = img.shape[:2]

    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = seg.process(rgb)
        mask = (res.segmentation_mask > 0.5).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (21,21), 0)
        mask3 = np.repeat(mask[:, :, None], 3, axis=2)

    bg = cv2.GaussianBlur(img, (35,35), 0)
    merged = (img * mask3 + bg * (1 - mask3)).astype(np.uint8)

    den = cv2.fastNlMeansDenoisingColored(merged, None, 10, 10, 7, 21)
    clahe = apply_clahe(den)

    with mp_face.FaceDetection(0, 0.5) as fd:
        detections = fd.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        final = clahe.copy()
        if detections.detections:
            for d in detections.detections:
                b = d.location_data.relative_bounding_box
                x1, y1 = int(b.xmin*w), int(b.ymin*h)
                x2, y2 = int((b.xmin+b.width)*w), int((b.ymin+b.height)*h)
                face = final[y1:y2, x1:x2]
                if face.size > 0:
                    final[y1:y2, x1:x2] = unsharp_mask(face)

    cv2.imwrite(output_path, final)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()
    enhance_image(args.inp, args.out)
