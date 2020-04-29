import numpy as np
import time, copy, torch
import torch.nn.functional as F

from torch import nn, optim
from jupyterplot import ProgressPlot
from torch.utils.data import DataLoader


class Solver():
    def __init__(self, model, **kwargs):
        """
        Required arguments:
        - model: a torch.nn model object
        """

        self.model = model

        # Unpack keyword arguments
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.mode = {}
        self.mode['plot'] = kwargs.pop('plot', False)
        self.mode['verbose'] = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)

    def _reset(self):
        self.stats = {}
        self.stats['train'] = {
            x: np.zeros(self.num_epochs)
            for x in ['loss', 'acc']
        }
        self.stats['val'] = {
            x: np.zeros(self.num_epochs)
            for x in ['loss', 'acc']
        }

        # Enable GPU if available
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        if self.mode['verbose']:
            print(f'Using {self.device} for training.')

        self.best_val_acc = 0
        self.best_params = copy.deepcopy(self.model.state_dict())

        # Plotting setup
        if self.mode['plot']:
            self.pp = ProgressPlot(plot_names=['loss', 'acc'],
                                   line_names=['train', 'val'],
                                   x_lim=[0, self.num_epochs],
                                   y_lim=[0, 1])

    def _accuracy(self, preds, targets):
        # Convert to numpy since we don't want grad
        with torch.no_grad():
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
            return torch.sum(torch.eq(preds, targets)).item()

    def _plotStats(self, epoch):
        data = [[
            self.stats['train']['loss'][epoch],
            self.stats['val']['loss'][epoch]
        ], [
            self.stats['train']['acc'][epoch], self.stats['val']['acc'][epoch]
        ]]

        self.pp.update(data)

    def setModel(self, model):
        self.model = model

    def _early_stop(self, epoch, trend=3):
        if epoch < trend:
            return False

        cond1 = True
        for i in range(trend):
            val1 = self.stats['val']['loss'][epoch-i]
            val2 = self.stats['val']['loss'][epoch-i-1]
            cond1 = cond1 and (val1 > val2)

        # Future loss should not be larger than initial loss
        cond2 = self.stats['val']['loss'][epoch] > self.stats['val']['loss'][0] * 1.05
        
        if cond1 or cond2:
            print(f'Early stop activated @ epoch: {epoch}')
            return True
    
        return False

    def train(self, dataloaders, dataset_sizes):
        self._reset()

        loss_fn = nn.BCELoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)

        # Prepare model for running
        self.model.double()
        self.model.to(self.device)

        since = time.time()
        for epoch in range(self.num_epochs):
            for phase in ['train', 'val']:

                if phase == 'train':
                    self.model.train()
                if phase == 'val':
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        # Forward pass
                        y_preds = self.model(inputs)
                        loss = loss_fn(y_preds, labels.double())

                        if phase == 'train':
                            loss.backward()  # Calculate gradients
                            optimizer.step()  # Update weights

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += self._accuracy(y_preds, labels)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                # Deep copy the model
                if phase == 'val' and epoch_acc > self.best_val_acc:
                    self.best_val_acc = epoch_acc
                    self.best_params = copy.deepcopy(self.model.state_dict())

                if self.mode['verbose']:
                    print('Epoch: {}-{:6>0} Loss: {:.4f} Acc: {:.4f}'.format(
                        epoch + 1, phase, epoch_loss, epoch_acc))

                self.stats[phase]['loss'][epoch] = epoch_loss
                self.stats[phase]['acc'][epoch] = epoch_acc

            # Realtime results plotting
            if self.mode['plot']:
                self._plotStats(epoch)

            # Stop training if validaiton loss starts increasing
            if self._early_stop(epoch):
                break

        if self.mode['verbose']:
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val acc: {:4f}'.format(self.best_val_acc))

        # Makes the plot persist across notebook sessions
        if self.mode['plot']:
            self.pp.finalize()

        # Reset model to return with the best parameters
        self.model.load_state_dict(self.best_params)

        return self.model.eval()

    def eval(self, model, dataloader):

        self.model.eval()
        self.model.to(self.device)

        ytest = torch.empty((0, 1), device=self.device)
        ypred = torch.empty((0, 1), device=self.device)

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                preds = self.model(inputs)

            ytest = torch.cat((ytest, labels.float()), axis=0)
            ypred = torch.cat((ypred, preds.float()), axis=0)

        if torch.cuda.is_available():
            ytest = ytest.data.cpu().numpy()
            ypred = ypred.data.cpu().numpy()
        else:
            ytest = ytest.data.numpy()
            ypred = ypred.data.numpy()

        return ytest, ypred