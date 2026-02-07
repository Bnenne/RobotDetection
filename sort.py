import json, os

with open('reid.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

for item in data["items"]:
    if not item["annotations"]:
        continue
    team = int(item["annotations"][0]["attributes"]["Number"])

    if not os.path.exists(f"reid_dataset/train/{team}"):
        os.makedirs(f"reid_dataset/train/{team}")

    length = str(len(os.listdir(f"reid_dataset/train/{team}")))

    diff = 5 - len(length)

    for i in range(diff):
        length = "0" + length

    os.rename(f"cropped/{item['id']}.png", f"reid_dataset/train/{team}/{length}.png")