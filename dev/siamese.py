import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

class LeNet(nn.Module):
    def __init__(self, numChannels):
        # call the parent constructor
        super(LeNet, self).__init__()
        # CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=10, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
  
    def forward(self, x):  
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        return x

class SiameseNetwork(nn.Module):
    def __init__(self):

        super().__init__()

        self.backbone = LeNet(3)

        # Get the number of features that are outputted by the last layer of backbone network.
        out_features = list(self.backbone.modules())[-1].out_features

        # Create an MLP (multi-layer perceptron) as the classification head. 
        # Classifies if provided combined feature vector of the 2 images represent same player or different.
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, patch1, patch2):
        # Pass the both images through the backbone network to get their seperate feature vectors
        feat1 = self.backbone(patch1)
        feat2 = self.backbone(patch2)
        
        # Multiply (element-wise) the feature vectors of the two images together, 
        # to generate a combined feature vector representing the similarity between the two.
        combined_features = feat1 * feat2

        # Pass the combined feature vector through classification head to get similarity value in the range of 0 to 1.
        output = self.cls_head(combined_features)
        return output
    
def train(model, train_loader, epochs=10, lr=1e-3, trials=1):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    learning_curves = [[] for i in range(trials)]
    
    for trial in range(trials):
        for epoch in range(epochs):
            # for batch_input, batch_target in train_loader:
            
            optimizer.zero_grad(set_to_none=True)
            batch_pred = model(batch_input[0], batch_input[1])
            loss = torch.linalg.norm(batch_pred - batch_target) # TODO: replace with better loss
            learning_curves[trial].append(float(loss.item()))
            loss.backward()
            optimizer.step()
                # if (i % 100 == 0):
                #     print("{: 3} {: 6}: {}".format(trial, i, loss.item()))
    
    
def main():
    # Initialize W&B
    wandb.init(project="siamese", mode='disabled')

    net = SiameseNetwork(3)
    
    # TODO: deal with loading the data for patches....
    # 50/50 same image, if same image then 1 as target
    # etc etc?
    # train(net, meh, 10, 1e-3, 2)



if __name__ == "__main__":
    main()