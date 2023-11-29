import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from itertools import chain
import torch
import random

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        embedded = self.embeddings(context)
        # in cbow the input embeddings are averaged
        aggregated = torch.mean(embedded, dim=1)
        out = self.linear(aggregated)
        return out
    
"""
class CBOWDataset(Dataset):
    def __init__(self, collection, tokenizer, window_size=2, max_elements_in_men=1_000_000):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_elements_in_men = max_elements_in_men
        
        # build pairs context, target word pairs
        self._context_target_pairs_iter = chain.from_iterable(map(self._create_context_target_pairs, tokenizer.iter_tokenizer(collection())))
        
        self.pairs = []
        for _ in range(max_elements_in_men):
            self.pairs.append(next(self._context_target_pairs_iter))
            
    def _create_context_target_pairs(self, tokens):
        pairs = []
        for i in range(self.window_size, len(tokens) - self.window_size):
            context = [tokens[i - j - 1] for j in range(self.window_size)] + [tokens[i + j + 1] for j in range(self.window_size)]
            target = tokens[i]
            pairs.append((context, target))
            
        return pairs

    def __len__(self):
        return self.tokenizer.total_number_tokens

    def __getitem__(self, idx):
        # convert index to 
        modular_idx = idx % len(self.pairs)
        context, target = self.pairs[modular_idx]
        context_indices = [self.tokenizer.token_to_id[word] for word in context]
        target_index = self.tokenizer.token_to_id[target]
        
        # take out the already processed element and add a new one
        del self.pairs[modular_idx]
        self.pairs.append(next(self._context_target_pairs_iter))
        
        return torch.tensor(context_indices), torch.tensor(target_index)
"""
class CBOWDatasetIterable(IterableDataset):
    def __init__(self, collection_iterable, tokenizer, window_size=2, max_elements_in_men=1_000_000):
        self.collection_iterable = collection_iterable
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_elements_in_men = max_elements_in_men
    
    def _create_context_target_pairs(self, tokens):
        pairs = []
        for i in range(self.window_size, len(tokens) - self.window_size):
            context = [tokens[i - j - 1] for j in range(self.window_size)] + [tokens[i + j + 1] for j in range(self.window_size)]
            target = tokens[i]
            pairs.append((context, target))
            
        return pairs
    
    def __iter__(self):
        # build pairs context, target word pairs
        self._context_target_pairs_iter = chain.from_iterable(map(self._create_context_target_pairs, self.tokenizer.iter_subsampling_tokenizer(self.collection_iterable())))
        
        pairs = []
        try:
            for _ in range(self.max_elements_in_men):
                pairs.append(next(self._context_target_pairs_iter))
        except:
            pass
        
        try:
            while True:
                try:
                    #print("sample from main loop")
                    idx = random.randint(0, len(pairs)-1)
                    context, target = pairs[idx]

                    context_indices = [self.tokenizer.token_to_id[word] for word in context]
                    target_index = self.tokenizer.token_to_id[target]
                    
                    # take out the already processed element and add a new one
                    pairs[idx] = next(self._context_target_pairs_iter)                    
                    yield torch.tensor(context_indices), torch.tensor(target_index)
                except StopIteration as e:
                    break
            while len(pairs) > 0:
                #print("sample from remiander")
                context, target = pairs.pop()
                context_indices = [self.tokenizer.token_to_id[word] for word in context]
                target_index = self.tokenizer.token_to_id[target]
                yield torch.tensor(context_indices), torch.tensor(target_index)
        except GeneratorExit:
            pass



        