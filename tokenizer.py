import re
from tqdm import tqdm
from itertools import chain
import json
from collections import defaultdict
import random
import math

class SimpleTokenizer:
    
    def __init__(self):
        self.regex = "[a-z-]{3,}"
        #self.regex = "[a-z]+"
        #self.regex = "[\w-]+"
        self.vocab_size = 0
        self.total_number_tokens = 0
        self.ignore_list = set()
        self.token_freq = defaultdict(int)
    
    @property
    def id_to_token(self):
        if not hasattr(self, "_id_to_token"):
            self._id_to_token = {v:k for k,v in self.token_to_id.items()}
        return self._id_to_token
    
    def tokenize(self, text):
        return [token for token in re.findall(self.regex, text.lower()) if token not in self.ignore_list]
    
    def iter_tokenizer(self, collection):
        return map(self.tokenize, collection)
    
    def iter_subsampling_tokenizer(self, collection):
        def subsampling_text(text):
            output_tokens = []
            for token in self.tokenize(text):
                rand = random.random()
                prob = 1-math.sqrt(10e-5/(self.token_freq[token]/self.total_number_tokens))
                if rand>prob:
                    output_tokens.append(token)
            return output_tokens
        return map(subsampling_text, collection)
    
    def build_tokenizer(self, collection, min_freq=5):
        vocab = set()
        for article_tokens in self.iter_tokenizer(collection):
            for token in article_tokens:
                self.token_freq[token] += 1
                vocab.add(token)                    
                    
        self.token_freq = dict(self.token_freq)
        # remove tokens with freq < min_freq
        for token, freq in self.token_freq.items():
            if freq < min_freq:
                #print("add min freq", token, freq)
                self.ignore_list.add(token)
                
        # rebuild token_to_id to have a sequential idx...
        # not the must efficient implementation
        self.token_to_id = {}
        for token in vocab:
            if token not in self.ignore_list:
                self.token_to_id[token] = len(self.token_to_id)
            #del self.token_freq[ignore_token]
        
        for ignore_token in self.ignore_list:
            del self.token_freq[ignore_token]
        
        self.vocab_size = len(self.token_to_id)
        
        self.total_number_tokens = sum(self.token_freq.values())
        
    def save(self, filepath):
        data = {
            'regex': self.regex,
            'token_to_id': self.token_to_id,
            'vocab_size': self.vocab_size,
            'total_number_tokens': self.total_number_tokens,
            'ignore_list': list(self.ignore_list),
            'token_freq': self.token_freq,
        }
        with open(filepath, 'w') as file:
            json.dump(data, file)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        tokenizer = cls()
        tokenizer.regex = data['regex']
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.total_number_tokens = data['total_number_tokens']
        tokenizer.ignore_list = set(data['ignore_list'])
        tokenizer.token_freq = data['token_freq']
        return tokenizer