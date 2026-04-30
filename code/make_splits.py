import os
import csv
import random

BASE = "data"
SEED = 42
random.seed(SEED)

datasets = {
    "dnabert2_100": f"{BASE}/promoter_binary_100bp.csv",
    "dnabert2_200": f"{BASE}/promoter_binary_200bp.csv",
    "dnabert2_500": f"{BASE}/promoter_binary_500bp.csv",
}

def stratified_split(rows, label_key="label"):
    class_0 = [r for r in rows if str(r[label_key]).strip() == "0"]
    class_1 = [r for r in rows if str(r[label_key]).strip() == "1"]

    random.shuffle(class_0)
    random.shuffle(class_1)

    def split_class(class_rows):
        n = len(class_rows)
        n_train = int(round(0.70 * n))
        n_dev = int(round(0.15 * n))
        train = class_rows[:n_train]
        dev = class_rows[n_train:n_train + n_dev]
        test = class_rows[n_train + n_dev:]
        return train, dev, test

    train0, dev0, test0 = split_class(class_0)
    train1, dev1, test1 = split_class(class_1)

    train = train0 + train1
    dev = dev0 + dev1
    test = test0 + test1

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    return train, dev, test

def read_csv_rows(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = str(row["sequence"]).strip().upper()
            label = str(row["label"]).strip()
            if seq and label in {"0", "1"}:
                rows.append({"sequence": seq, "label": label})
    return rows

def write_csv_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "label"])
        writer.writeheader()
        writer.writerows(rows)

for out_name, csv_path in datasets.items():
    rows = read_csv_rows(csv_path)
    train_rows, dev_rows, test_rows = stratified_split(rows)

    out_dir = os.path.join(BASE, out_name)
    os.makedirs(out_dir, exist_ok=True)

    write_csv_rows(os.path.join(out_dir, "train.csv"), train_rows)
    write_csv_rows(os.path.join(out_dir, "dev.csv"), dev_rows)
    write_csv_rows(os.path.join(out_dir, "test.csv"), test_rows)

    print(f"{out_name}:")
    print(f"  train = {len(train_rows)}")
    print(f"  dev   = {len(dev_rows)}")
    print(f"  test  = {len(test_rows)}")