import numpy
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
n = 28 # pixel nxn
N = n*n

class Autoencoder(nn.Module):
    def __init__(self):
        #nxn,N 28x28, 784
        super().__init__()

        self.encoder = nn.Sequential(
                                      nn.Linear(N, 128), # N=784 --> 128
                                      nn.ReLU(),
                                      nn.Linear(128, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 12),
                                      nn.ReLU(),
                                      nn.Linear(12, 3)
                                     )

        self.decoder = nn.Sequential(
                                      nn.Linear(3, 12), # N=784 --> 128
                                      nn.ReLU(),
                                      nn.Linear(12, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, N),
                                      nn.Sigmoid() #output is in the format of [0,1]

                                     )

    def forward(self, x):

        x = self.encoder(x)
        # print(x.size())

        x = self.decoder(x)
        # print(x.size())
        # x = self.unflatten(x)
        # print(x.size())

        return x

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)


    model.train()

    for batch, (X,_) in enumerate(dataloader):
        X=X.reshape([-1,N])
        X = X.to(device)

        #Inference
        out = model(X)
        loss = loss_fn(out, X)

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch +1)*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0

    with torch.no_grad():
        for X, _ in dataloader:
            X=X.reshape([-1,N])

            X = X.to(device)
            out = model(X)
            test_loss += loss_fn(out, X).item()

        test_loss /= num_batches
        print(f"Test Error: \nAvg loss: {test_loss:>8f} \n")

    return X, out


def plot_results(num_samples = 9):

    pass





if __name__ == '__main__':


    #### Import and Initialize Data

    print(f'Using {device} device')

    transform = transforms.ToTensor()

    training_data = datasets.MNIST( root ='./data',
                            train = True,
                            download = True,
                            transform = transform)

    testing_data = datasets.MNIST( root ='./data',
                            train = True,
                            download = False,
                            transform = transform)

    train_data_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=64, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(dataset=testing_data, batch_size=64, shuffle=True)


    # dataiter = iter(data_loader)
    #
    # images, labels = next(dataiter) # new syntax is next(iter)
    #
    # print(torch.min(images), torch.max(images))

    #### Model

    model = Autoencoder().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr =.001, weight_decay =.00001 )


    #### Training Loop

    num_epochs = 5

    results=[]

    for epoch in range(num_epochs):
        print( f'Epoch {epoch+1}\n-----------------------------')
        train(train_data_loader, model, loss_fn, optimizer)
        img, recon = test(test_data_loader, model, loss_fn)
        results.append((epoch,img,recon))

    ### Plot Results

    figure = []
    for k in range(0,num_epochs):

        figure.append((k,plt.figure()))
        plt.gray()

        imgs = results[k][1].reshape([-1,28,28])
        recons = results[k][2].reshape([-1,28,28])

        imgs = imgs.cpu().detach().numpy()
        recons = recons.cpu().detach().numpy()

        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2,9,i+1)
            plt.imshow(item)

        for i, item in enumerate(recons):
            if i>=9:break
            plt.subplot(2,9,9+i+1)
            plt.imshow(item)


    plt.show()


    print('done')
