import json
import random
from pathlib import Path
from collections import defaultdict
from ollama import generate

PROJECT_DIR = Path(__file__).parent
TRIPLES_FILE = PROJECT_DIR / 'data' / 'triples-smaller.jsonl'
OUTPUT_QUESTIONS = PROJECT_DIR / 'data' / 'facteval-prompts.jsonl'

PROPERTY_NAMES = {
    'P2176': "drug used for treatment",
    'P2175': "medical condition treated",
    'P780': "symptoms",
    'P2293': "genetic association",
    'P4044': "therapeutic area"
}

CONTEXT_INJECTIONS = {
    'relevant': [
        "According to medical guidelines: ",
        "Based on established knowledge: "
    ],
    'irrelevant': [
        "In an unrelated context: ",
        "Some people believe: "
    ],
    'anti_factual': [
        "Contrary to medical advice: ",
        "Incorrectly stated: "
    ]
}

QUESTION_TEMPLATES = {
    'true_false': {
        'P780': "Is {object} a symptom of {subject}?",
        'P2176': "Is {subject} used to treat {object}?"
    },
    'short_answer': {
        'P780': "What are the symptoms of {subject}?",
        'P2176': "What drug treats {subject}?"
    }
}


def flatten_aliases(alias_list):
    return [item for sublist in alias_list for item in (sublist if isinstance(sublist, list) else [sublist])]


def generate_questions(triple, predicate_obj_map):
    entries = []
    entries.extend(generate_cloze(triple))
    entries.extend(generate_true_false(triple, predicate_obj_map))
    entries.extend(generate_short_answer(triple, predicate_obj_map))
    entries.extend(generate_llm_questions(triple))
    return entries


def generate_cloze(triple):
    cloze_template = "The standard treatment for {subject} is [MASK]."
    subject = triple['sub_label']
    prompt = cloze_template.format(subject=subject)

    entries = []
    for ctx_type, contexts in CONTEXT_INJECTIONS.items():
        for ctx in contexts:
            entries.append({
                "uuid": triple['uuid'],
                "predicate_id": triple['predicate_id'],
                "question_type": "cloze",
                "prompt": ctx + prompt,
                "expected_answers": get_expected_answers(triple),
                "context_type": ctx_type
            })
    return entries


def generate_true_false(triple, predicate_obj_map):
    entries = []
    pred = triple['predicate_id']
    if pred not in QUESTION_TEMPLATES['true_false']:
        return []

    try:
        obj_type = triple['obj_types'][0]
        all_objects = list(predicate_obj_map[pred][obj_type])
    except (KeyError, IndexError):
        return []

    correct_answers = get_expected_answers(triple)
    candidate_false = [o for o in all_objects if o not in correct_answers]

    if not candidate_false:
        candidate_false = ["unknown condition", "no known treatment"]

    true_prompt = QUESTION_TEMPLATES['true_false'][pred].format(
        subject=triple['sub_label'],
        object=triple['obj_labels'][0]
    )
    entries.append(create_entry(triple, true_prompt, ["correct"], "true_false", "relevant"))

    # False version with safety
    false_object = random.choice(candidate_false)
    false_prompt = QUESTION_TEMPLATES['true_false'][pred].format(
        subject=triple['sub_label'],
        object=false_object
    )
    entries.append(create_entry(triple, false_prompt, ["incorrect"], "true_false", "anti_factual"))

    return entries


def generate_short_answer(triple, predicate_obj_map):
    entries = []
    pred = triple['predicate_id']
    if pred not in QUESTION_TEMPLATES['short_answer']:
        return []

    base_prompt = QUESTION_TEMPLATES['short_answer'][pred].format(
        subject=triple['sub_label']
    )

    for ctx_type, contexts in CONTEXT_INJECTIONS.items():
        for ctx in contexts:
            entries.append(create_entry(
                triple,
                ctx + base_prompt,
                get_expected_answers(triple),
                "short_answer",
                ctx_type
            ))
    return entries


def generate_llm_questions(triple):
    prompt = """Generate a question where the answer is "{object}" about "{subject}" in biomedical context. Return question only:""".format(
        subject=triple['sub_label'],
        object=', '.join(triple['obj_labels'])
    )
    response = generate('llama3.2', prompt)['response']

    return [create_entry(
        triple,
        response,
        get_expected_answers(triple),
        "llm_generated",
        "no_context"
    )]


def create_entry(triple, prompt, answers, q_type, ctx_type):
    return {
        "uuid": triple['uuid'],
        "predicate_id": triple['predicate_id'],
        "question_type": q_type,
        "prompt": prompt,
        "expected_answers": answers,
        "context_type": ctx_type
    }


def get_expected_answers(triple):
    expected = set()
    expected.update([label.lower() for label in triple.get('obj_labels', [])])
    expected.update([alias.lower() for alias in flatten_aliases(triple.get('obj_aliases', []))])
    return sorted(expected)


def main():
    triples = []
    with open(TRIPLES_FILE, 'r') as f:
        triples = [json.loads(line) for line in f]

    predicate_obj_map = defaultdict(lambda: defaultdict(set))
    for t in triples:
        try:
            pred = t['predicate_id']
            obj_type = t['obj_types'][0] if t['obj_types'] else 'default'
            answers = get_expected_answers(t)
            predicate_obj_map[pred][obj_type].update(answers)
        except KeyError as e:
            print(f"Skipping malformed triple: {t.get('uuid', 'unknown')} - missing {e}")
            continue

    all_questions = []
    for triple in triples:
        all_questions.extend(generate_questions(triple, predicate_obj_map))

    with open(OUTPUT_QUESTIONS, 'w') as f:
        for q in all_questions:
            f.write(json.dumps(q) + '\n')


if __name__ == "__main__":
    main()