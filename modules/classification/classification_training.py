import torch
from classification_model import DistilBERTClass

class Training:
    def __init__(self, args):
        self.epochs = args["epochs"]
        self.learning_rate = args["lr"]
        self.device = args["device"]
        self.model = DistilBERTClass().to(self.device)
        self.optimizer = torch.optim.Adam(params =  self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
    
    def train(self, train_loader):

        self.model.train()
        loss = float()

        for epoch in range(self.epochs):
            loss = 0 
            for _, data in enumerate(train_loader, 0):

                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                targets = data['targets'].to(self.device, dtype = torch.float)
                
                outputs = self.model(ids, mask, token_type_ids)
                loss += self.loss_fn(outputs, targets)

                if _%1000==0:
                    print(f'Epoch: {epoch}, Loss:  {loss.item()}')
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss
    
    def validate(self, validate_loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y in validate_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x)
                loss += self.loss_fn(outputs, y)
                pred = torch.argmax(outputs, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss