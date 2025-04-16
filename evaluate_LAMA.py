import json
from pathlib import Path
from ollama import generate
from collections import defaultdict

PROJECT_DIR = Path(__file__).parent
LAMA_PROMPTS_FILE = PROJECT_DIR / 'data' / 'LAMA-prompts.jsonl'

def load_prompts():
    return [json.loads(line) for line in open(LAMA_PROMPTS_FILE)]


def match(response, expected):
    response_lower = response.strip().lower()
    return any(any(e in resp_part or resp_part in e
                   for resp_part in response_lower.split())
               for e in expected)


def evaluate_lama():
    prompts = load_prompts()
    results = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
    total = correct = 0

    for entry in prompts:
        response = generate('llama3.2', entry['cloze_prompt'])['response']

        if match(response, entry['expected_answers']):
            results[entry['predicate_id']]['correct'] += 1
            correct += 1
        else:
            results[entry['predicate_id']]['incorrect'] += 1
        total += 1

    print(f"Total Questions: {total}")
    print(f"Accuracy: {correct / total:.2%}\n")

    print("By Question Type:")
    for pred, counts in results.items():
        print(f"{pred}: {counts}")


if __name__ == "__main__":
    evaluate_lama()