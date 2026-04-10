import os
import random, sys, time, json, argparse, re
from typing import List, Tuple
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

from evalution import get_evaluation

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser('Run Naive RAG with vLLM')
parser.add_argument('--data_file', type=str, help='Directory to load data')
parser.add_argument('--data_name', type=str)
parser.add_argument('--save_log_dir', type=str, default='log', help='Directory to save data')
parser.add_argument('--model_path', type=str, help='Directory to load model')
parser.add_argument('--reranker_path', type=str, help='Directory to load reranker')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--seed', type=int, default=127)
args = parser.parse_args()

seed_everything(args.seed)


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


log_dir = os.path.join(args.save_log_dir, args.data_name)
os.makedirs(log_dir, exist_ok=True)
t = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
sys.stdout = Logger(os.path.join(log_dir, f'log{t}.txt'))

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
if "lama" in args.model_path:  # llama / llama3
    tokenizer.pad_token = tokenizer.eos_token

llm = LLM(
    model=args.model_path,
    dtype="bfloat16",
    tensor_parallel_size=1,
    trust_remote_code=True,
    gpu_memory_utilization=0.6,
    seed=args.seed
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

gen_params = SamplingParams(
    temperature=0.6,
    top_p=0.9,
    top_k=50,
    max_tokens=128,
    stop_token_ids=terminators,
    n=1,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    repetition_penalty=1.0,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Reranker_train/ft_reranker/bge-rerank-base-ft_llama3_8B
# load BGE reranker
rerank_tokenizer = AutoTokenizer.from_pretrained(args.reranker_path)
rerank_model = AutoModelForSequenceClassification.from_pretrained(args.reranker_path).to(device)
rerank_model.eval()


def batch_rerank_docs_with_bge(items, rerank_from=20, batch_size=1):
    """
    Batched reranking using BGE.
    For each query, rerank its top `rerank_from` docs and keep top_k.
    """

    reranked_results = []
    for i in tqdm(range(0, len(items), batch_size)):
        batch_items = items[i: i + batch_size]
        # all_query_inputs = []
        all_pairs = []
        all_doc_inputs = []
        doc_counts = []

        for item in batch_items:
            query = item["question"]
            docs = item["docs"][:rerank_from]

            # query_inputs = [query] * len(docs)
            doc_inputs = [doc["title"] + "\n" + doc["text"] for doc in docs]  # [100]
            all_doc_inputs.extend(doc_inputs)  # [batch_size * 100]
            doc_counts.append(len(docs))

            all_pairs = all_pairs + [[query, doc] for doc in doc_inputs]  # [batch_size * 100, 2]

        inputs = rerank_tokenizer(
            all_pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = rerank_model(**inputs)
        sims = outputs.logits.view(-1).float()  # [batch_size * 100]
        idx = 0
        for item, count in zip(batch_items, doc_counts):
            docs = item["docs"][:count]
            scores = sims[idx:idx + count]

            for j, doc in enumerate(docs):
                doc["bge_score"] = scores[j].item()

            reranked_docs = sorted(docs, key=lambda x: x["bge_score"], reverse=True)
            item["docs"] = reranked_docs
            reranked_results.append(item)
            idx += count

    return reranked_results


def chat_batch(questions_list: List[str]) -> List[str]:
    messages_list = [[{"role": "user", "content": q}] for q in questions_list]
    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]

    outputs = llm.generate(prompts, sampling_params=gen_params)

    return [o.outputs[0].text for o in outputs]


prompt_without_retrieval = (
    "Answer the following question based on your internal knowledge with one or few words without the source.\n"
    "Question:{question}"
)
prompt_retrieval = (
    "Given the following information: \n{document}\n"
    "Answer the following question based on the given information or your internal knowledge "
    "with one or few words without the source.\nQuestion:{question}"
)


def get_input_batch(data: List[dict], retrieval: bool, K: int, batch_size: int
                    ) -> Tuple[List[List[str]], List[List[List[str]]]]:
    q_batches, ans_batches = [], []
    q_tmp, a_tmp = [], []
    if retrieval:
        rerank_data = batch_rerank_docs_with_bge(data)
    else:
        rerank_data = data
    for d in rerank_data:
        a_tmp.append(d["golden_answers"])

        if retrieval:
            doc_text = "\n\n".join(
                [f"Passage #{i + 1} title:{p['title']}\nPassage #{i + 1} text:{p['text']}"
                 for i, p in enumerate(d["docs"][:K])]
            )
            q_tmp.append(prompt_retrieval.format_map({"document": doc_text, "question": d["question"]}))
        else:
            q_tmp.append(prompt_without_retrieval.format_map({"question": d["question"]}))

        if len(q_tmp) == batch_size:
            q_batches.append(q_tmp)
            ans_batches.append(a_tmp)
            q_tmp, a_tmp = [], []
    if q_tmp:
        q_batches.append(q_tmp)
        ans_batches.append(a_tmp)
    return q_batches, ans_batches


def get_data_score(inputs_batch: List[List[str]], answers_batch: List[List[List[str]]]) -> Tuple[int, float, float]:
    num, reg_sum, f1_sum = 0, 0.0, 0.0
    for q_batch, a_batch in tqdm(zip(inputs_batch, answers_batch)):
        resp_batch = chat_batch(q_batch)
        for resp, gold in zip(resp_batch, a_batch):
            num += 1
            if num == 1:
                print(f"input:{q_batch[0]}\nresponse:{resp}\nanswer:{gold}")
            regex, f1 = get_evaluation("", resp, gold)
            reg_sum += regex
            f1_sum += f1
    return num, reg_sum, f1_sum


if __name__ == "__main__":
    print_parser(args)
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    retrieved_in, gold_batches = get_input_batch(data, True, args.k, args.batch_size)
    # standard_in, _ = get_input_batch(data, False, args.k, args.batch_size)

    n, ret_regex, ret_f1 = get_data_score(retrieved_in, gold_batches)
    # _, std_regex, std_f1 = get_data_score(standard_in, gold_batches)

    print(
        f"sample_num:{n}, retrieved_regex_score:{ret_regex / n * 100:.6f}, retrieved_f1_score:{ret_f1 / n * 100:.6f}"
    )
