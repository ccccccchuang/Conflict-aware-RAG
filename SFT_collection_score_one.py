import json
import argparse
import os
import re
import string
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import numpy as np
import torch
import random
import time
import sys
from tqdm import tqdm
from utils import em_max_over_ground_truths, batch_compute_log_ppl


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if 'lama' in args.model_path:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tokenizer


def chat_batch(model, tokenizer, questions_list: List[str]) -> List[str]:
    messages_list = [[{"role": "user", "content": question}] for question in questions_list]
    texts = [tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    ) for message in messages_list]

    model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=8192,
                             padding_side='left').to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = [
        output_id[input_id.shape[-1]:] for input_id, output_id in zip(model_inputs['input_ids'], generated_ids)
    ]

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return responses


def build_sft_data(data, data_pos, model, tokenizer, s1_ratio=0, s2_ratio=0.6, s3_ratio=0.2, s4_ratio=0.2):
    total = len(data)
    s1_count = int(total * s1_ratio)
    s2_count = int(total * s2_ratio)
    s3_count = int(total * s3_ratio)
    s4_count = int(total * s4_ratio)

    s1, s2, s3, s4 = [], [], [], []

    for item in tqdm(data_pos):
        question = item['question']
        golden_answers = item['golden_answers']
        direct_response = item['direct_response'].strip()
        positives = item.get('positive_doc', [])
        negatives = item.get('negative_doc', [])

        # S3: multiple neg docs
        if len(s3) < s3_count:

            q_batch, chosen_resp = [], []
            for negative_doc in negatives:
                doc_one_text = f"Passage #1 title:{negative_doc['title']}\nPassage #1 text:{negative_doc['text']}"
                query = prompt_retrieval.format_map({"document": doc_one_text, "question": question})
                q_batch.append(query)
                chosen_resp.append(direct_response)
            relscore_neg = batch_compute_log_ppl(model, tokenizer, q_batch, chosen_resp)
            max_indices = np.argsort(relscore_neg)[::-1][:3]
            docs = [negatives[i] for i in max_indices]
            random.shuffle(docs)

            # docs = random.sample(negatives, min(3, len(negatives)))
            doc_text = "\n\n".join(
                [f"Passage #{i + 1} title:{p['title']}\nPassage #{i + 1} text:{p['text']}"
                 for i, p in enumerate(docs)]
            )
            prompt = prompt_retrieval.format_map({"document": doc_text, "question": question})
            s3.append({
                "instruction": prompt,
                "input": '',
                "output": direct_response
            })

    for item in tqdm(data):
        question = item['question']
        golden_answers = item['golden_answers']
        direct_response = item['direct_response'].strip()
        positives = item.get('positive_doc', [])
        negatives = item.get('negative_doc', [])

        # S4: multiple good docs
        if len(positives) >= 3 and len(s4) < s4_count:

            q_batch, chosen_resp = [], []
            for positive_doc in positives:
                pos_resp = positive_doc['doc_response'].strip()
                doc_one_text = f"Passage #1 title:{positive_doc['title']}\nPassage #1 text:{positive_doc['text']}"
                query = prompt_retrieval.format_map({"document": doc_one_text, "question": question})
                q_batch.append(query)
                chosen_resp.append(pos_resp)
            relscore_pos = batch_compute_log_ppl(model, tokenizer, q_batch, chosen_resp)
            max_indices = np.argsort(relscore_pos)[:3]
            docs = [positives[i] for i in max_indices]
            random.shuffle(docs)

            # docs = random.sample(positives, min(3, len(positives)))
            doc_text = "\n\n".join(
                [f"Passage #{i + 1} title:{p['title']}\nPassage #{i + 1} text:{p['text']}"
                 for i, p in enumerate(docs)]
            )
            prompt = prompt_retrieval.format_map({"document": doc_text, "question": question})
            retrieval_out = chat_batch(model, tokenizer, [prompt])
            if not em_max_over_ground_truths(retrieval_out[0].strip(), golden_answers)[0]:
                continue
            else:
                answer = retrieval_out[0].strip()

            s4.append({
                "instruction": prompt,
                "input": '',
                "output": answer
            })

        # S1: only 1 good doc
        elif len(positives) >= 1 and len(s1) < s1_count:
            pos_doc = positives[0]
            doc_text = f"Passage #1 title:{pos_doc['title']}\nPassage #1 text:{pos_doc['text']}"
            prompt = prompt_retrieval.format_map({"document": doc_text, "question": question})
            s1.append({
                "instruction": prompt,
                "input": '',
                "output": pos_doc['doc_response'].strip()
            })

        # S2: 1 good + 2 bad
        elif len(positives) >= 1 and len(negatives) >= 2 and len(s2) < s2_count:
            pos_doc = {}
            for positive_doc in positives:
                corr_answer = golden_answers[em_max_over_ground_truths(positive_doc['doc_response'].strip(), golden_answers)[1]]
                if em_max_over_ground_truths(positive_doc['title'] + ' ' + positive_doc['text'], corr_answer)[0]:
                    pos_doc = positive_doc
                    break
            if not pos_doc:
                pos_doc = random.sample(positives, 1)[0]

            # pos_doc = random.sample(positives, 1)[0]
            pos_response = pos_doc['doc_response'].strip()

            q_batch, chosen_resp, rejected_resp = [], [], []
            for negative_doc in negatives:
                neg_resp = negative_doc['doc_response'].strip()
                docs_one = [negative_doc] + [pos_doc]
                random.shuffle(docs_one)
                doc_one_text = f"Passage #1 title:{negative_doc['title']}\nPassage #1 text:{negative_doc['text']}"
                query = prompt_retrieval.format_map({"document": doc_one_text, "question": question})
                q_batch.append(query)
                chosen_resp.append(pos_response)

            relscore_neg = batch_compute_log_ppl(model, tokenizer, q_batch, chosen_resp)
            max_indices = np.argsort(relscore_neg)[::-1][:2]
            neg_docs = [negatives[i] for i in max_indices]

            # neg_docs = random.sample(negatives, 2)
            docs = [pos_doc] + neg_docs
            random.shuffle(docs)

            doc_text = "\n\n".join(
                [f"Passage #{i + 1} title:{p['title']}\nPassage #{i + 1} text:{p['text']}"
                 for i, p in enumerate(docs)]
            )
            prompt = prompt_retrieval.format_map({"document": doc_text, "question": question})
            s2.append({
                "instruction": prompt,
                "input": '',
                "output": pos_doc['doc_response'].strip()
            })

    final_data = s1 + s2 + s3 + s4
    s1_num, s2_num, s3_num, s4_num = len(s1), len(s2), len(s3), len(s4)
    return final_data, (s1_num, s2_num, s3_num, s4_num)


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def deal_data(data, k):
    temp_data = []
    for i in range(len(data)):
        if len(data[i]['positive_doc']) > 0:
            temp_data.append(data[i])
    temp_data = temp_data[:k]
    return temp_data


def deal_data_pos(data):
    temp_data = []
    for i in range(len(data)):
        if len(data[i]['negative_doc']) > 2:
            temp_data.append(data[i])
    return temp_data


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

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def print_s(s_num, data_name):
    s1, s2, s3, s4 = s_num
    print(data_name)
    print('s1_num:', s1)
    print('s2_num:', s2)
    print('s3_num:', s3)
    print('s4_num:', s4)
    print('data_num:', s1 + s2 + s3 + s4)
    print('')


prompt_retrieval = (
    "Given the following information: \n{document}\n"
    "Answer the following question based on the given information or your internal knowledge "
    "with one or few words without the source.\nQuestion:{question}"
)


def save_json(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)


def main():
    # ----------------------print log-----------------------
    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    t = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
    sys.stdout = Logger(os.path.join(log_dir, f'log{t}.txt'))
    # ------------------------------------------------------

    parser = argparse.ArgumentParser(description="Process RAG data into positive/negative sets based on model outputs.")
    parser.add_argument('--model_path', type=str, help='Directory to load model')
    parser.add_argument('--seed', type=int, default=127)
    parser.add_argument('--SFT_output_dir', type=str, default='data/one_sft', help='Directory to save data.')
    parser.add_argument('--input_pos', type=str, help='File for storing positive documents')
    parser.add_argument('--input_neg', type=str, help='File for storing negative documents')
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--data_num', type=int)
    args = parser.parse_args()

    print_parser(args)
    seed_everything(args.seed)

    model, tokenizer = load_model_tokenizer(args)

    data_pos = load_json(args.input_pos)
    data = load_json(args.input_neg)
    print(f'✅ Done load {args.data_name}!')
    data_pos = deal_data_pos(data_pos)
    data = deal_data(data, args.data_num)
    data, s = build_sft_data(data, data_pos, model, tokenizer)
    torch.cuda.empty_cache()
    s1, s2, s3, s4 = s
    os.makedirs(args.SFT_output_dir, exist_ok=True)
    save_json(data, os.path.join(args.SFT_output_dir, f'{args.data_name}_SFT.json'))
    print(f'S1:{s1} samples. S2: {s2} samples. S3: {s3} samples. S4: {s4} samples.')
    print(f"✅ Done. Saved {len(data)} samples to: {os.path.join(args.SFT_output_dir, f'{args.data_name}_SFT.json')}")


if __name__ == '__main__':
    with torch.no_grad():
        main()
