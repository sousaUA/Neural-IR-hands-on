import torch
import json
import random
from transformers import AutoTokenizer
from collections import defaultdict
from functools import partial
from typing import Union
from utils import split_chunks

class IteratorPartialInitializer(type):
    def __getitem__(iterator, sampler):
        return IteratorSamplerCombiner(iterator, sampler=sampler)

class IteratorSamplerCombiner:

    def __init__(self, iterator, sampler):
        self.iterator = iterator
        self.sampler = sampler

    def get_n_samples(self, dataset):
        return self.iterator.get_n_samples(dataset)

    def __call__(self, *args, **kwargs):
        iterator = self.iterator(*args, **kwargs)
        iterator.add_sampler(self.sampler(*args, **kwargs))
        return iterator

class BioASQPointwiseIterator(metaclass=IteratorPartialInitializer):
    def __init__(self, tokenizer, expected_number_of_samples, slice_dataset, epoch, max_length, *args, **kwargs):
        self.tokenizer = tokenizer
        self.expected_number_of_samples = expected_number_of_samples
        self.slice_dataset = slice_dataset
        self.index = 0
        self.epoch = epoch
        self.max_length=max_length

    def add_sampler(self, sampler):
        self.sampler = sampler

    @staticmethod
    def get_n_samples(dataset):
        return sum([len(x["pos_docs"]) for x in dataset.values()]) * 2

    def _tokenize(self, q_text, doc_text, label=None):
        #, truncation=True, max_length=self.max_length) # TODO FIX THIS VALUE TO BE HANDLE BY THE MODEL
        if label is not None:
            # training
            inputs = self.tokenizer(q_text, doc_text)
            inputs["label_ids"] = int(label)
            return inputs
            #return inputs | {"label_ids": int(label)}
        else:
            # inference
            inputs = self.tokenizer(q_text, doc_text, truncation=True, max_length=self.max_length)
            return inputs

    def __next__(self):
        # spot criteria
        if self.index>=self.expected_number_of_samples:
            raise StopIteration

        label = not self.index%2

        while True:
            q_id, q_text = self.sampler.choose_question(self.index, self.epoch)

            doc_text = self.sampler.choose_positive_doc(self.index, self.epoch, q_id) if label else self.sampler.choose_negative_doc(self.index, self.epoch, q_id)

            inputs = self._tokenize(q_text, doc_text, label)
            if len(inputs["input_ids"])<=self.max_length:
                break


        self.index+=1

        return inputs

class BioASQPairwiseIterator(BioASQPointwiseIterator):

    @staticmethod
    def get_n_samples(dataset):
        return sum([len(x["pos_docs"]) for x in dataset.values()])

    def __next__(self):
        # spot criteria
        if self.index>self.expected_number_of_samples:
            raise StopIteration

        q_id, q_text = self.sampler.choose_question(self.index, self.epoch)

        # choose
        doc_pos_text = self.sampler.choose_positive_doc(self.index, self.epoch, q_id)
        doc_neg_text = self.sampler.choose_negative_doc(self.index, self.epoch, q_id)
        pos_inputs = self.tokenizer(q_text, doc_pos_text, truncation=True, max_length=self.max_length)
        neg_inputs = self.tokenizer(q_text, doc_neg_text, truncation=True, max_length=self.max_length)
        self.index+=1

        return {"pos_doc":pos_inputs,"neg_doc":neg_inputs}

class RankingIterator(BioASQPointwiseIterator):

    @staticmethod
    def get_n_samples(dataset):
        return sum([len(x["pos_docs"]) for x in dataset.values()]) + sum([len(x["neg_docs"]) for x in dataset.values()])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        def document_iterator_func():
            for q_id in self.slice_dataset:
                q_text = self.slice_dataset[q_id]["question"]
                for doc in self.slice_dataset[q_id]["neg_docs"]:
                    #yield self.tokenizer(q_text, doc["text"], truncation=True, max_length=512, padding="max_length") | {"id":q_id, "doc_id":doc["id"]}
                    tok = self._tokenize(q_text, doc["text"])
                    tok["id"] = q_id
                    tok["doc_id"] = doc["id"]
                    yield tok#self._tokenize(q_text, doc["text"]) | {"id":q_id, "doc_id":doc["id"]}
                for doc in self.slice_dataset[q_id]["pos_docs"]:
                    #yield self.tokenizer(q_text, doc["text"], truncation=True, max_length=512, padding="max_length") | {"id":q_id, "doc_id":doc["id"]}
                    tok = self._tokenize(q_text, doc["text"])# | {"id":q_id, "doc_id":doc["id"]}
                    tok["id"] = q_id
                    tok["doc_id"] = doc["id"]
                    yield tok


        self.iterator = iter(document_iterator_func())

    def __next__(self):
        return next(self.iterator)

class InferenceRankingIterator(RankingIterator):

    @staticmethod
    def get_n_samples(dataset):
        return sum([len(x["documents"]) for x in dataset.values()])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        def document_iterator_func():
            for q_id in self.slice_dataset:
                q_text = self.slice_dataset[q_id]["question"]
                for doc in self.slice_dataset[q_id]["documents"]:
                    #yield self.tokenizer(q_text, doc["text"], truncation=True, max_length=512, padding="max_length") | {"id":q_id, "doc_id":doc["id"]}
                    tok = self._tokenize(q_text, doc["text"])
                    tok["id"] = q_id
                    tok["doc_id"] = doc["id"]
                    yield tok

        self.iterator = iter(document_iterator_func())

def create_training_dataset(positive_data_path,
                                    negative_data_path,
                                    collection_data_path,
                                    tokenizer,
                                    **kwargs):

#positive_data_path = "synthetic_questions_top_100_clean.jsonl"
#negative_data_path = "training_data_negatives_synthetic_ids.jsonl"

    _dataset = {}
    _qrels_dict = {}
    with open(positive_data_path) as f:
        for line in f:
            sample = json.loads(line)
            _dataset[sample["id"]] = {"pos_docs":sample["documents"],"question":sample["body"]}
            _qrels_dict[sample["id"]] = {docid:1 for docid in sample["documents"]}

    q_to_remove = []

    with open(negative_data_path) as f:
        for line in f:
            sample = json.loads(line)
            _dataset[sample["id"]]["neg_docs"] = sample["neg_docs"]
            if len(sample["neg_docs"])==0:
                q_to_remove.append(sample["id"])

    for qid in q_to_remove:
        if qid in _dataset:
            del _dataset[qid]

    _collection = {}
    with open(collection_data_path) as f:
        for line in f:
            sample = json.loads(line)
            _collection[sample["id"]] = sample["text"]

    return BioASQDataset(dataset=_dataset,
                        tokenizer=tokenizer,
                        qrels_dict=_qrels_dict,
                        collection=_collection,
                        **kwargs)

def create_bioASQ_datasets(positive_data_path,
                           negative_data_path,
                           tokenizer,
                           subsets=["train", "test"],
                           test_split_percentage:float=0.05,
                           **kwargs):

    if not isinstance(subsets,list):
        subsets = [subsets]

    assert len(subsets)>0

    # load data
    dataset = {"train":{}, "test":{}}
    qrels_dict = {"train":{}, "test":{}}

    _dataset = {}
    _qrels_dict = {}

    with open(positive_data_path) as f:
        for line in f:
            sample = json.loads(line)
            _dataset[sample["id"]] = {"pos_docs":sample["documents"],"question":sample["body"]}
            _qrels_dict[sample["id"]] = {doc["id"]:1 for doc in sample["documents"]}

    with open(negative_data_path) as f:
        for line in f:
            sample = json.loads(line)
            _dataset[sample["id"]]["neg_docs"] = sample["neg_docs"]

    question_ids = list(_qrels_dict.keys())
    split_index=round(len(question_ids)*test_split_percentage)

    test_queries_ids = question_ids[:split_index]
    train_queries_ids= question_ids[split_index:]

    for q_id in train_queries_ids:
        dataset["train"][q_id] = _dataset[q_id]
        qrels_dict["train"][q_id] = _qrels_dict[q_id]

    for q_id in test_queries_ids:
        dataset["test"][q_id] = _dataset[q_id]
        qrels_dict["test"][q_id] = _qrels_dict[q_id]

    del _dataset

    # split kwargs
    train_dataset_args = {"tokenizer":tokenizer}
    test_dataset_args = {"tokenizer":tokenizer, "iterator_class": RankingIterator}
    for k in kwargs:
        if k.startswith("train_"):
            train_dataset_args[k[6:]] = kwargs[k]
        elif k.startswith("test_"):
            test_dataset_args[k[5:]] = kwargs[k]

    train_dataset_args["dataset"] = dataset["train"]
    train_dataset_args["qrels_dict"] = qrels_dict["train"]

    test_dataset_args["dataset"] = dataset["test"]
    test_dataset_args["qrels_dict"] = qrels_dict["test"]

    to_return = []
    for subset in subsets:
        if subset=="train":
            to_return.append(BioASQDataset(**train_dataset_args))
        elif subset=="test":
            to_return.append(BioASQDataset(**test_dataset_args))
    if len(to_return)==1:
        return to_return[0]
    else:
        return to_return


class BioASQDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 dataset,
                 tokenizer,
                 iterator_class: Union[IteratorSamplerCombiner, RankingIterator],
                 qrels_dict = None,
                 collection = None,
                 max_length = -1,
                 max_questions:int=-1,
                 max_neg_docs:int=-1):
        """
        dataset: {query_id: {}}
        """
        super().__init__()
        self.dataset = {}
        self.epoch=-1

        self.tokenizer = tokenizer
        self.iterator_class = iterator_class

        if max_length==-1:
            self.max_length = tokenizer.model_max_length
        else:
            self.max_length = self.max_length

        queries_ids = list(dataset.keys())

        if max_questions!=-1:
            queries_ids = queries_ids[:max_questions]

        self.qrels_dict = {}

        for q_id in queries_ids:
            if max_neg_docs!=-1:
                dataset[q_id]["neg_docs"] = dataset[q_id]["neg_docs"][:max_neg_docs]

            self.dataset[q_id] = dataset[q_id]
            if qrels_dict:
                self.qrels_dict[q_id] = qrels_dict[q_id]

        del dataset
        del qrels_dict

        self.expected_number_of_samples = self.iterator_class.get_n_samples(self.dataset)

        self.collection = collection


        #exit()

    def get_n_questions(self):
        return len(self.dataset)

    def get_qrels(self):
        return self.qrels_dict

    def __len__(self):
        #worker_info = torch.utils.data.get_worker_info()
        #print("LEN_CALL!!!! WORKER_INFO!!!", worker_info)
        return self.expected_number_of_samples
        #worker_info = torch.utils.data.get_worker_info()
        #worker_info = torch.utils.data.get_worker_info()
        #if worker_info is None:
        #    return self.expected_number_of_samples
        #else:


    def __iter__(self):
        self.epoch += 1
        #worker_info = torch.utils.data.get_worker_info()

        worker_info = torch.utils.data.get_worker_info()
        #print("WORKER_INFO!!!", worker_info)
        if worker_info is None:  # single-process data loading, return the full iterator
            return self.iterator_class(slice_dataset=self.dataset,
                                       tokenizer=self.tokenizer,
                                       collection=self.collection,
                                       expected_number_of_samples=self.expected_number_of_samples,
                                       epoch=self.epoch,
                                       max_length=self.max_length)
        else:  # in a worker process
            # split workload
            q_ids = list(self.dataset.keys())


            this_worker_ids = list(split_chunks(q_ids, worker_info.num_workers))[worker_info.id]

            _dataset = {q_id:self.dataset[q_id] for q_id in this_worker_ids}


            return self.iterator_class(slice_dataset=_dataset,
                                       tokenizer=self.tokenizer,
                                       collection=self.collection,
                                       expected_number_of_samples=self.iterator_class.get_n_samples(_dataset),
                                       epoch=self.epoch,
                                       max_length=self.max_length)