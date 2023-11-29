from tokenizer import SimpleTokenizer
import json
from tqdm import tqdm
from word2vec import CBOW, CBOWDatasetIterable
from torch.utils.data import DataLoader, Dataset

tokenizer = SimpleTokenizer.load("simple_tokenzer_updated.json")


def article_reader(json_line):
    article = json.loads(json_line)
    return article["title"] + " " + article["abstract"]

def collection_reader():
    with open("pubmed_2022_tiny.jsonl") as f:
        i=0
        for article in map(article_reader, f):
            #if i>1000:
            #    break
            yield article
            i+=1
        
dataset = CBOWDatasetIterable(collection_reader, tokenizer, window_size=2, max_elements_in_men=1_000_000)

data_loader = DataLoader(dataset, batch_size=512, pin_memory=True, num_workers=1)

def train_cbow(model, data_loader, epochs, learning_rate, device):
    import torch
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    loss = None
    
    c = 0
    model.train()
    for epoch in range(epochs):
        
        pbar = tqdm()
        for context, target in data_loader:
            optimizer.zero_grad()
            context = context.to(device)
            target = target.to(device)
            logits = model(context)
            loss = loss_function(logits, target)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            pbar.set_description(f'Epoch {epoch}, Loss: {loss:.4f}')
            pbar.update(1)
            
           
            #if not c%2000:
            #    print(f'Epoch {epoch}, Loss: {loss:.4f}')
            
            
        print("saving")
        torch.save(model.state_dict(), f"cbow_model_new_tok_e{epoch}.pt")
            
                
model = CBOW(tokenizer.vocab_size, 300)

train_cbow(model, data_loader, 30, 0.001, "cuda:1")

