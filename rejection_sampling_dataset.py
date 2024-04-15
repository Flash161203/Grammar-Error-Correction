import numpy as np
import editdistance
from preprocessing_functions import get_raw_splits, apply_prompt_template
from typing import List, Tuple
from transformers import AutoTokenizer
from tqdm import tqdm
# taken from https://github.com/huggingface/trl/pull/902/files
# modified from https://arxiv.org/pdf/2309.06657.pdf
def conduct_rejection_sampling(
    response_candidates: List[str], response_rewards: List[float], num_samples: int, beta: float
) -> Tuple[List[str], List[float]]:
    """Conducts statistical rejection sampling.
    Args:
        response_candidates: response candidates from sft policy
        response_rewards: response rewards.
        num_samples: number of samples to sub-sample.
        beta: beta parameter in KL-constrained reward maximization objective.
    Returns:
        accepted: Accepted rejection sampled sequences from the optimal policy.
        rewards: the rewards associated to the accepted samples.
    """
    candidates = {c: r for c, r in zip(response_candidates, response_rewards)}
    accepted = []
    rewards = []
    while len(accepted) < num_samples:
        max_reward = max(candidates.values())
        to_remove = []
        for c, r in candidates.items():
            u = np.random.uniform()
            if u >= np.exp((r - max_reward) / beta):
                continue
            accepted.append(c)
            rewards.append(r)
            to_remove.append(c)
            if len(accepted) == num_samples:
                break
        for c in to_remove:
            candidates.pop(c)
    return accepted, rewards

def tournament_ranking(responses: List[str], rewards: List[float]):
    """Conducts tournament ranking. Starts from n responses and constructs n/2 pairs. 
    
    Args:
        responses: accecpted candidates from rejection sampling.
        rewards: response rewards.
        
    Returns:
        chosen: chosen samples.
        rejected: rejected samples.
    """
    sorted_responses = [response for _, response in sorted(zip(rewards, responses), reverse=True)]

    chosen = [sorted_responses[i] for i in range(0, len(responses), 2)]
    rejected =[sorted_responses[i] for i in range(1, len(responses), 2)]

    return chosen, rejected

seed = 0
num_examples = 8
num_samples_from_model = 16
raw_datasets = get_raw_splits()
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'unk_token': '[UNK]'})
raw_datasets = raw_datasets.filter(lambda x: len(tokenizer.encode(x["text"])) <= 400)
print(raw_datasets)
# sample 500 examples from the dataset
samples = raw_datasets["train"].shuffle(seed=seed).select(range(num_examples))
# apply prompt, filter by size
# print(samples)
original_texts = samples["text"]

# print(f'samples {len(samples["text"])}, {samples["text"]}')
from inference_with_lora import correct_text
out = []
for i in tqdm(range(num_examples//2)):
    slice = original_texts[2*i: 2*i + 2]
    assert len(slice) <= 2
    out += correct_text(slice, k=num_samples_from_model, apply_template=True, batched=True, bsize=len(slice))
# print(out)
print(len(out), len(out[0]))
# collect responses as list[list[str]]
responses = []
for corrections in out:
    response = [correction["generated_text"] for correction in corrections]
    responses.append(response)

print(len(responses), len(responses[0]))
print(responses[4][0])
print(samples["text"][4])
out = responses
reward: list[list[int]] = [[editdistance.eval(original_text, response) for response in responses] for original_text, responses in zip(original_texts, out)]
print(len(reward), len(reward[0]))
dpo_dataset_dict = {"prompt": [], "chosen": [], "rejected": []}
for i in range(num_examples):
    accepted, rewards = conduct_rejection_sampling(responses[i], reward[i], 5, beta=0.7)

    accepted = [(i, accepted[i]) for i in range(len(accepted))]
    # print(accepted[-1], len(rewards))
    
    candidate_rewards = rewards
    while len(accepted) > 1:
        chosen, rejected = tournament_ranking(accepted, candidate_rewards)
        # print(chosen)
        for j in zip(chosen, rejected):
            dpo_dataset_dict["prompt"].append(samples[i]["text"])
            dpo_dataset_dict["chosen"].append(j[0][1])
            dpo_dataset_dict["rejected"].append(j[1][1])
        accepted = chosen
        candidate_rewards = [rewards[idx] for idx, _ in chosen]
    
from datasets import Dataset
print(dpo_dataset_dict.keys())
ds = Dataset.from_dict(dpo_dataset_dict)
from pprint import pprint
pprint(ds[5])
ds.save_to_disk('rejection_sampling_dataset')