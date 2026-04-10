import json
import argparse
import os
import re
import string
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import List
import numpy as np
import torch
import random
import time
import sys
from tqdm import tqdm

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):  # Remove articles
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):  # Change multiple consecutive spaces into one
        return ' '.join(text.split())

    def remove_punc(text):  # Remove all punctuation marks
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):  # lower case
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            normalize_answer(pattern),
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(normalize_answer(text)) is not None


def exact_match_score(prediction, ground_truth):
    """ EM: ground_truth is a str """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def em_max_over_ground_truths(prediction, ground_truths):
    return max([regex_match(prediction, gt) for gt in ground_truths])


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)


def load_vllm_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if "lama" in args.model_path:  # llama / llama3
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        seed=args.seed
    )
    return llm, tokenizer


def chat_batch(questions_list: List[str], llm, tokenizer, gen_params) -> List[str]:
    messages_list = [[{"role": "user", "content": q}] for q in questions_list]
    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]

    outputs = llm.generate(prompts, sampling_params=gen_params)

    return [o.outputs[0].text for o in outputs]


def process_data(input_path, output_dir, llm, tokenizer, gen_params, batch_size=16):
    data = load_json(input_path)

    positive_q = []
    negative_q = []

    for item in tqdm(data):
        question = item['question']
        golden_answers = item['golden_answers']
        docs = item['docs'][:10]

        standard_in = [prompt_without_retrieval.format_map({"question": question})]

        direct_output = chat_batch(standard_in, llm, tokenizer, gen_params)[0]

        is_answer_correct = em_max_over_ground_truths(direct_output, golden_answers)

        pos_docs = []
        neg_docs = []

        q_batches, q_tmp, doc_tmp, doc_batches = [], [], [], []
        for doc in docs:
            doc_text = f"Passage #1 title:{doc['title']}\nPassage #1 text:{doc['text']}"
            retrieval_in = prompt_retrieval.format_map({"document": doc_text, "question": question})
            q_tmp.append(retrieval_in)
            doc_tmp.append(doc)
            if len(q_tmp) == batch_size:
                q_batches.append(q_tmp)
                doc_batches.append(doc_tmp)
                doc_tmp, q_tmp = [], []
        if q_tmp:
            q_batches.append(q_tmp)
            doc_batches.append(doc_tmp)

        for (in_batch, d_batch) in zip(q_batches, doc_batches):
            retrieval_out = chat_batch(in_batch, llm, tokenizer, gen_params)
            for output, doc in zip(retrieval_out, d_batch):
                doc['doc_response'] = output
                if em_max_over_ground_truths(output, golden_answers):
                    pos_docs.append(doc)
                else:
                    neg_docs.append(doc)

        result_item = {
            'question': question,
            'direct_response': direct_output,
            'golden_answers': golden_answers,
            'positive_doc': pos_docs,
            'negative_doc': neg_docs
        }

        if is_answer_correct:
            positive_q.append(result_item)
        else:
            negative_q.append(result_item)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    save_json(positive_q, os.path.join(output_dir, 'train_positive_q.jsonl'))
    save_json(negative_q, os.path.join(output_dir, 'train_negative_q.jsonl'))

    print(f"✅ Done. Saved {len(positive_q)} positive samples and {len(negative_q)} negative samples to: {output_dir}")


def print_parser(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self): pass


prompt_without_retrieval = (
    "Answer the following question based on your internal knowledge with one or few words without the source.\n"
    "Question:{question}"
)
prompt_retrieval = (
    "Given the following information: \n{document}\n"
    "Answer the following question based on the given information or your internal knowledge "
    "with one or few words without the source.\nQuestion:{question}"
)


def main():
    # ----------------------print log-----------------------
    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    t = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
    sys.stdout = Logger(os.path.join(log_dir, f'log{t}.txt'))
    # ------------------------------------------------------

    parser = argparse.ArgumentParser(description="Process RAG data into positive/negative sets based on model outputs.")
    parser.add_argument('--input_file', type=str, help='Path to input train.jsonl file')
    parser.add_argument('--output_dir', type=str, help='Directory to save output files')
    parser.add_argument('--model_path', type=str, help='Directory to load model')
    parser.add_argument('--seed', type=int, default='127')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    print_parser(args)
    seed_everything(args.seed)

    llm, tokenizer = load_vllm_model_tokenizer(args)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    gen_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        top_k=50,
        max_tokens=2048,
        stop_token_ids=terminators,
        n=1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
    )

    process_data(args.input_file, args.output_dir, llm, tokenizer, gen_params, args.batch_size)


if __name__ == '__main__':
    main()
