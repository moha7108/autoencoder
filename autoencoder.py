import time, datetime
str_format = '%Y%m%d%H%M%S'
readable_format = '%Y/%m/%d %H:%M:%S'

import numpy
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

data_dir = './data/'
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
n = 28 # pixel nxn
N = n*n

class Autoencoder(nn.Module):
    def __init__(self, name='autoencoder'):
        #nxn,N 28x28, 784
        super().__init__()

        self.name = name

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
        x = self.decoder(x)

        return x



    def save_model(self, epoch, optimizer=None, loss=None, path= f'./data/checkoint.pth'):
        '''This method saves the model parameters to a .pth file given a file path'''

        self.optimizer = optimizer if optimizer else self.optimizer

        torch.save({
                     'epoch': epoch,
                     'model_state_dict': self.state_dict(),
                     'optimizer_state_dict': self.optimizer.state_dict(),
                     'loss':loss
        }, path)

        return path


    def load_model(self, loss_fn, optimizer,  path):
        '''This method loads the model parameters to a .pth file given a file path'''

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        checkpoint = torch.load(path)

        self.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']

    def train_step(self, dataloader, loss_fn=None, optimizer=None):

        self.loss_fn = loss_fn if loss_fn else self.loss_fn
        self.optimizer = optimizer if optimizer else self.optimizer

        size = len(dataloader.dataset)
        ###### POTENTIAL DECORATOR FUNCTION
        self.train()

        for batch, (X,_) in enumerate(dataloader):
            X=X.reshape([-1,N])
            X = X.to(device)

            #Inference
            out = self(X)
            loss = self.loss_fn(out, X)

            #Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch +1)*len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test_step(self, dataloader, loss_fn =None):
        self.loss_fn = loss_fn if loss_fn else self.loss_fn

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval()
        test_loss, correct = 0,0

        ###### POTENTIAL DECORATOR FUNCTION
        with torch.no_grad():
            for X, _ in dataloader:
                X=X.reshape([-1,N])

                X = X.to(device)
                out = self(X)
                test_loss += self.loss_fn(out, X).item()

            test_loss /= num_batches
            print(f"Test Error: \nAvg loss: {test_loss:>8f} \n")

        return X, out


    def train_model(self, train_data_loader, test_data_loader, num_epochs = 5):

        results=[]

        for epoch in range(num_epochs):
            print( f'Epoch {epoch+1}\n-----------------------------')
            self.train_step(dataloader=train_data_loader)
            img, recon = self.test_step(dataloader=test_data_loader)
            results.append((epoch,img,recon))

        return results


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
    # model = torch.load('./data/20230227150927-autoencoder_model_weights.pth')
    # model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr =.001, weight_decay =.00001 )


    model.load_model(loss_fn, optimizer,  path='./data/20230227211127-autoencoder_checkpoint.pth')


    # model.load_weights('./data/20230227141826-autoencoder_model_weights.pth')


    #### Training Loop

    num_epochs = 2
    #
    # results=[]
    #
    # for epoch in range(num_epochs):
    #     print( f'Epoch {epoch+1}\n-----------------------------')
    #     model.train_step(dataloader=train_data_loader)
    #     img, recon = model.test_step(dataloader=test_data_loader)
    #     results.append((epoch,img,recon))

    results=model.train_model(train_data_loader, test_data_loader, num_epochs)

    ts = datetime.datetime.now().strftime(str_format)
    filepath = f'{data_dir}{ts}-{model.name}_checkpoint.pth'

    model.save_model(epoch=num_epochs, optimizer=optimizer, path=filepath)

    ### Plot Results

    figure = []
    for k in range(0,len(results)):

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
