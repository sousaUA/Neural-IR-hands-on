import os
from transformers.trainer_utils import set_seed
from transformers import TrainingArguments
import yaml
import json

def setup_wandb(dataset):
    os.environ["WANDB_API_KEY"] = open(".api").read().strip()
    os.environ["WANDB_PROJECT"] = f"REBUTAL EACL Synthethic train - {dataset} data"
    os.environ["WANDB_LOG_MODEL"]="false"
    os.environ["WANDB_ENTITY"] = "bitua"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"


def _load_flat_config(path):
    assert path is not None, "`path` cannot be none"
    with open(path) as fp:
        config = yaml.safe_load(fp)

    return _flatten(config)


def _flatten(d):
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                items.extend(_flatten(v).items())
            elif isinstance(v, list):
                for x in v:
                    items.extend(_flatten(x).items())
            else:
                try:
                    items.append((k, eval(v)))
                except (NameError, TypeError):
                    items.append((k, v))
    else:
        raise ValueError(f"Found leaf value ({repr(d)}) that is not a dictionary. Please convert it to a dictionary.")
    return dict(items)

def create_config(base_config_path="bert_trainer_config.yaml", **update_config):
    #assert isinstance(update_config, dict), "update_config config must be a dictionary."
    print(f"Combining values supplied as `keywords arguments` with base config from {base_config_path}" if update_config else f"Using base config from {base_config_path}")

    base_config = _load_flat_config(base_config_path)
    base_config.update(update_config)
    return TrainingArguments(**base_config)


class EmptyEncodeBatch:
    def __init__(self):
        self.input_ids = []
        self.attention_mask = []
        self.token_type_ids = []


def split_chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def load_rank_data(bm25_rank_path, at=1000, qrels=None):

    dataset = {}
    with open(bm25_rank_path) as f:
        for line in f:
            q_data = json.loads(line)
            if qrels:
                if q_data["id"] not in qrels:
                    continue
            dataset[q_data["id"]] = {"documents": q_data["documents"][:at], "question": q_data["question"]}
    return dataset