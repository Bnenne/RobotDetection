import os

for split in ["train", "val"]:
    label_dir = f"{split}/labels"
    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(label_dir, fname)
        with open(path) as f:
            lines = f.readlines()
        fixed = []
        for line in lines:
            parts = line.split()
            if parts:
                parts[0] = "0"  # remap class 1 → 0
                fixed.append(" ".join(parts) + "\n")
        with open(path, "w") as f:
            f.writelines(fixed)

print("Done")