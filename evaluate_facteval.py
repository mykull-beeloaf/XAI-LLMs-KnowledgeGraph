import json
from pathlib import Path
from ollama import generate
from collections import defaultdict

def match(response, expected):
    response_lower = response.strip().lower()
    return any(any(e in resp_part or resp_part in e
                   for resp_part in response_lower.split())
               for e in expected)


def is_abstained(response):
    return not response.strip() or "don't know" in response.lower()

def evaluate():
    questions_file = Path(__file__).parent / 'data' / 'facteval-prompts.jsonl'
    results = []

    # even though LLM-facteval-PromptGenerator produces around 600 entries, I will keep it at 50 prompts
    # due to the computational limitations of my local machine
    with open(questions_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            response = generate('llama3.2', entry['prompt'])['response']

            if is_abstained(response):
                result = 'abstained'
            elif entry['question_type'] == 'true_false':
                result = 'correct' if ('correct' in response.lower()) == (
                            'correct' in entry['expected_answers']) else 'incorrect'
            else:
                result = 'correct' if match(response, entry['expected_answers']) else 'incorrect'

            entry['response'] = response
            entry['result'] = result
            results.append(entry)

    metrics = defaultdict(lambda: defaultdict(int))
    total = len(results)

    for res in results:
        metrics['overall'][res['result']] += 1
        metrics[res['question_type']][res['result']] += 1
        metrics[res['context_type']][res['result']] += 1

    print(f"Total Questions: {total}")
    print(f"Accuracy: {metrics['overall']['correct'] / (total):.2%}")
    print("\nBy Question Type:")
    for q_type, counts in metrics.items():
        if q_type not in ['overall', 'relevant', 'irrelevant', 'anti_factual']:
            print(f"{q_type}: {counts}")

    print("\nBy Context Type:")
    for ctx_type in ['relevant', 'irrelevant', 'anti_factual']:
        print(f"{ctx_type}: {metrics[ctx_type]}")


if __name__ == "__main__":
    evaluate()