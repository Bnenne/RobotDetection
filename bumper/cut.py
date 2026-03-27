import os, cv2

path = "../cropped"

images = os.listdir(path)

input_path = images[0]
output_dir = "outputs"

os.makedirs(output_dir, exist_ok=True)

img = cv2.imread(os.path.join(path, input_path))

print(os.path.join(path, input_path))

h, w = img.shape[:2]

cropped = [
    img,
    img[0:h//2, :],
    img[h//2:, :]
]

for i, img in enumerate(cropped):
    save_path = os.path.join(output_dir, f"{input_path.split('.')[0]}_{i}.png")
    cv2.imwrite(save_path, img)