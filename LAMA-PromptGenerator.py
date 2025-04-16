import json
from pathlib import Path

# FILE PATHS
PROJECT_DIR = Path(__file__).parent
TRIPLES_FILE = PROJECT_DIR / 'data' / 'triples.jsonl'
OUTPUT_COMBINED = PROJECT_DIR / 'data' / 'LAMA-prompts.jsonl'

PROPERTY_NAMES = {
    'P2176': "drug used for treatment",
    'P2175': "medical condition treated",
    'P780': "symptoms",
    'P2293': "genetic association",
    'P4044': "therapeutic area"
}

PROPERTY_OBJ_TYPES = {
    'P2176': 'chemical compound',
    'P2175': 'disease',
    'P780': 'disease',
    'P2293': 'disease',
    'P4044': 'disease'
}


TEMPLATES = {
    "P2176": {
        "template": "The standard treatment for patients with [X] is a drug such as [Y].",
        "property": PROPERTY_NAMES["P2176"]
    },
    "P2175": {
        "template": "[X] has effects on diseases such as [Y].",
        "property": PROPERTY_NAMES["P2175"]
    },
    "P4044": {
        "template": "[X] cures diseases such as [Y].",
        "property": PROPERTY_NAMES["P4044"]
    },
    "P780": {
        "template": "[X] has symptoms such as [Y].",
        "property": PROPERTY_NAMES["P780"]
    },
    "P2293": {
        "template": "Gene [X] has a genetic association with diseases such as [Y].",
        "property": PROPERTY_NAMES["P2293"]
    }
}

# Helper functions
def flatten_aliases(alias_list):
    """Flattens nested alias list to a flat list of strings."""
    flat = []
    for item in alias_list:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat

def generate_cloze_entry(triple):
    """Creates a cloze-style prompt and answer entry from a triple."""
    relation = triple.get("predicate_id")
    if relation not in TEMPLATES:
        return None

    template_info = TEMPLATES[relation]
    template = template_info["template"]
    subject = triple.get("sub_label", "").strip()
    prompt = template.replace("[X]", subject).replace("[Y]", "[MASK]")

    expected = set()
    if "obj_labels" in triple:
        for lab in triple["obj_labels"]:
            expected.add(lab.strip().lower())
    if "obj_aliases" in triple:
        flat_aliases = flatten_aliases(triple["obj_aliases"])
        for a in flat_aliases:
            expected.add(a.strip().lower())

    return {
        "uuid": triple.get("uuid"),
        "predicate_id": relation,
        "cloze_prompt": prompt,
        "expected_answers": sorted(expected)
    }

# Main processing
def load_triples_from_file(filename):
    triples = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triples.append(json.loads(line))
    return triples

def process_triples(triples):
    combined_entries = []

    for triple in triples:
        entry = generate_cloze_entry(triple)
        if entry:
            combined_entries.append(entry)

    return combined_entries


def main():
    triples = load_triples_from_file(TRIPLES_FILE)

    combined_entries = process_triples(triples)

    OUTPUT_COMBINED.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_COMBINED, "w", encoding="utf-8") as f:
        for entry in combined_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved {len(combined_entries)} prompts and answer entries")

if __name__ == "__main__":
    main()
