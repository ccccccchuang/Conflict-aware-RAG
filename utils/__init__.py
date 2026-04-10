import re
import string
import numpy as np
from typing import List
import torch.nn.functional as F
import torch
from tqdm import tqdm


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
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
    matches = [regex_match(prediction, gt) for gt in ground_truths]
    max_index = matches.index(max(matches))
    return max(matches), max_index


def batch_compute_log_ppl(model, tokenizer, prompts: List[str], responses: List[str]):  # Calculate the PPL of the sentence
    device = model.device
    messages = [[{"role": "user", "content": sentence}] for sentence in prompts]
    texts = [tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    ) for message in messages]
    batch_texts = [p + a for p, a in zip(texts, responses)]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
                       padding_side='left').to(device)
    input_ids = inputs["input_ids"]  # shape: [B, L]
    attention_mask = inputs["attention_mask"]  # shape: [B, L]

    answer_ids = tokenizer(responses, return_tensors="pt", padding=True, truncation=True,
                           add_special_tokens=False).input_ids.to(device)
    answer_lens = (answer_ids != tokenizer.pad_token_id).sum(dim=1)  # shape: [B]

    labels = input_ids.clone()
    labels[:] = -100
    for i, ans_len in enumerate(answer_lens):
        labels[i, -ans_len:] = input_ids[i, -ans_len:]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [B, L, V]

    # shift logits & labels
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'
    ).view(shift_labels.size())  # [B, L-1]

    losses = []
    for i, ans_len in enumerate(answer_lens):
        loss_i = loss[i, -ans_len:]
        avg_loss = loss_i.mean().item()
        ppl = np.exp(avg_loss)
        log_ppl = np.log(ppl)
        losses.append(log_ppl)

    return losses


def batch_compute_all_log_ppl(model, tokenizer, query_prompt_w_contexts: List[str], response_w_contexts: List[str],
                              response_wo_contexts: List[str]):
    log_ppl_w_contexts = batch_compute_log_ppl(model, tokenizer, query_prompt_w_contexts, response_w_contexts)
    log_ppl_wo_contexts = batch_compute_log_ppl(model, tokenizer, query_prompt_w_contexts, response_wo_contexts)
    return log_ppl_w_contexts, log_ppl_wo_contexts


def batch_compute_relscore(batch_c_log_ppl: List[float], batch_r_log_ppl: List[float], dif_mean_std, conf_mean_std,
                           alpha=0.5):
    relscore_list = []
    dif_mean, dif_std = dif_mean_std
    conf_mean, conf_std = conf_mean_std
    for ppl_c, ppl_r in zip(batch_c_log_ppl, batch_r_log_ppl):
        dif_normal = (abs(ppl_c - ppl_r) - dif_mean) / (dif_std + 1e-6)
        conf_normal = (ppl_c - conf_mean) / (conf_std + 1e-6)
        temp_relscore = (1-alpha) * conf_normal - alpha * dif_normal
        relscore_list.append(temp_relscore)
    return relscore_list


def cal_mean_std(all_batch_w_log_ppl, all_batch_wo_log_ppl):
    flat_w_log_ppl = [item for batch in all_batch_w_log_ppl for item in batch]
    flat_wo_log_ppl = [item for batch in all_batch_wo_log_ppl for item in batch]
    all_dif_log_ppl, all_conf_log_ppl = [], []
    for w_log_ppl, wo_log_ppl in zip(flat_w_log_ppl, flat_wo_log_ppl):
        all_dif_log_ppl.append(abs(w_log_ppl - wo_log_ppl))
        all_conf_log_ppl.append(w_log_ppl)
    all_dif_mean = np.mean(all_dif_log_ppl)
    all_dif_std = np.std(all_dif_log_ppl)
    all_conf_mean = np.mean(all_conf_log_ppl)
    all_conf_std = np.std(all_conf_log_ppl)
    return (all_dif_mean, all_dif_std), (all_conf_mean, all_conf_std)

def cal_all_batch_relscore(model, tokenizer, q_batches, chosen_batches, rejected_batches):
    all_batch_chosen_log_ppl, all_batch_rejected_log_ppl = [], []
    for ques_batch, chosen_batch, rejected_batch in tqdm(zip(q_batches, chosen_batches, rejected_batches)):
        chosen_log_ppl = batch_compute_log_ppl(model, tokenizer, ques_batch, chosen_batch)
        rejected_log_ppl = batch_compute_log_ppl(model, tokenizer, ques_batch, rejected_batch)
        all_batch_chosen_log_ppl.append(chosen_log_ppl)
        all_batch_rejected_log_ppl.append(rejected_log_ppl)
    all_dif_mean_std, all_conf_mean_std = cal_mean_std(all_batch_chosen_log_ppl, all_batch_rejected_log_ppl)
    all_batch_relscore = []
    for batch_w_log_ppl, batch_wo_log_ppl in zip(all_batch_chosen_log_ppl, all_batch_rejected_log_ppl):
        batch_relscore = batch_compute_relscore(batch_w_log_ppl, batch_wo_log_ppl, all_dif_mean_std, all_conf_mean_std)
        all_batch_relscore.append(batch_relscore)
    return all_batch_chosen_log_ppl, all_batch_rejected_log_ppl, all_batch_relscore