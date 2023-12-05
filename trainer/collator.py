class RankingCollator:

    def __init__(self,
                 tokenizer,
                 model_inputs={"input_ids", "attention_mask", "token_type_ids"},
                 padding=True,
                 max_length=None):
        self.tokenizer = tokenizer
        self.model_inputs = model_inputs
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch):
        batch = {key: [i[key] for i in batch] for key in batch[0]}

        reminder_keys = set(batch.keys())-self.model_inputs
        #return {"inputs": self.tokenizer.pad({k:batch[k] for k in self.model_inputs},
        #                             padding=self.padding,
        #                             max_length=self.max_length,
        #                             return_tensors="pt")
        #        } | {k:batch[k] for k in reminder_keys}
        col = {"inputs": self.tokenizer.pad({k:batch[k] for k in self.model_inputs},
                                     padding=self.padding,
                                     max_length=self.max_length,
                                     return_tensors="pt")
                }
        col.update({k:batch[k] for k in reminder_keys})

        return col

