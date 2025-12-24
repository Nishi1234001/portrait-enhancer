import cv2
import numpy as np

pairs = [
    ("raw8.jpg",  "result8.jpg",  "comparison8.jpg"),
    ("raw9.jpg",  "result9.jpg",  "comparison9.jpg"),
    ("raw10.jpg", "result10.jpg", "comparison10.jpg"),
    ("raw11.jpg", "result11.jpg", "comparison11.jpg"),
]

for raw, result, out in pairs:
    before = cv2.imread(f"demo_images/{raw}")
    after  = cv2.imread(f"outputs/{result}")

    if before is None or after is None:
        print(f"Skipping {raw} (file missing)")
        continue

    h = min(before.shape[0], after.shape[0])
    w = min(before.shape[1], after.shape[1])

    before = cv2.resize(before, (w, h))
    after  = cv2.resize(after, (w, h))

    comparison = np.hstack((before, after))
    cv2.imwrite(f"outputs/{out}", comparison)

    print(f"{out} created")
