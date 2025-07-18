import json
import random

def src_trg_json_spliter(path):
    src = []
    trg = []
    with open(path) as f:
        data = json.load(f)
    for record in data:
        random_val = random.randrange(len(record["text"]))
        src.append(record["text"][:random_val])
        trg.append(record["text"][random_val:])
    return src, trg
