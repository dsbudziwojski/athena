import json

def json_files_iter(paths):
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for record in data:
            yield record["text"]
