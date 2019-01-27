from azureml.core.run import Run

run = Run.get_context()

class Learner:
    def __init__(self, model, optimizer, train_loader, test_loader, criterion, val_criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.val_criterion = val_criterion
        self.device = device

    def train(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            # data = model.features(data)
            output = self.model(data)
            preds = output.data.max(1, keepdim=True)[1]
            loss = self.criterion(output, target)
            # loss_prior = var_model.prior_loss() / len(train_loader.dataset)
            # loss += loss_prior
            loss.backward()
            self.optimizer.step()

            # statistics
            running_loss += loss.item() * data.size(0)
            running_corrects += preds.eq(target.data.view_as(preds)).sum().item()

        training_loss = running_loss / len(self.train_loader.dataset)
        training_accuracy = running_corrects / len(self.train_loader.dataset)

        print('Train Epoch: {} \tLoss: {:.6f}\tAcc: {}'.format(
            epoch, training_loss, training_accuracy))

            # if batch_idx % 10 == 0:
            #     print('Train Epoch: {} [({:.0f}%)]\tLoss: {:.6f}\tAcc: {}'.format(
            #         epoch,
            #         100. * batch_idx / len(train_loader),
            #         (running_loss / ((1+batch_idx) * BATCH_SIZE) ),
            #         (running_corrects / ((1+batch_idx) * BATCH_SIZE))))

        # log the loss to the Azure ML run
        run.log('loss', training_loss)
        run.log('acc', training_accuracy)

    def test(self):
        self.model.eval()
        test_loss = 0.
        test_accuracy = 0.
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            # output = [var_model(data).view(-1, 2, 1) for i in range(5)]
            # output = torch.mean(torch.cat(outputf, 2), 2)
            # sum up batch loss
            test_loss += self.val_criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.data.view_as(pred)).sum().item()

        test_accuracy /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))

        run.log('val_loss', test_loss)
        run.log('val_acc', test_accuracy)