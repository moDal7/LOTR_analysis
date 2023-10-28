import torch
from tqdm import tqdm
from classification_model import DistilBERTClass

class Training():
    def __init__(self, args, training_loader):
        self.epochs = args["epochs"]
        self.learning_rate = args["lr"]
        self.device = args["device"]
        self.model = DistilBERTClass().to(self.device)
        self.optimizer = torch.optim.Adam(params =  self.model.parameters(), lr=self.learning_rate)
        self.train_loader = training_loader

    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def train(self.epochs, self.train_loader):
        self.model.train()
        for epoch in range(self.epochs):
            for _,data in tqdm(enumerate(self.train_loader, 0)):
                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                targets = data['targets'].to(self.device, dtype = torch.float)

                outputs = self.model(ids, mask, token_type_ids)

                self.optimizer.zero_grad()

                loss = self.loss_fn(outputs, targets)
                
                if _%1000==0:
                    print(f'Epoch: {epoch}, Loss:  {loss.item()}')
                
                loss.backward()

                self.optimizer.step()

        return loss.item()