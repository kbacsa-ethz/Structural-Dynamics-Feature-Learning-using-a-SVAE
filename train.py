# Import necessary libraries and modules
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


@torch.no_grad()
def weight_reset(m: nn.Module):
    # - check if the current module has reset_parameters & if it's callabed called it on m
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


@torch.no_grad()
def generate_noise(inputs):
    SNR = 10
    signal_power = torch.mean(inputs ** 2)
    signal_power_dB = 10 * torch.log10(signal_power)
    noise_dB = signal_power_dB - SNR
    noise_watt = 10 ** (noise_dB / 10)
    noise = torch.sqrt(noise_watt) * torch.randn(size=inputs.size()).to(inputs.device)
    return noise


# Define the Train class, which is used for training the model
class Train:
    def __init__(self,
                 model,
                 model_hyper,
                 learning_rate,
                 learning_decay,
                 weight_decay,
                 save_path,
                 comet_logger
                 ):
        # Determine if a CUDA-capable GPU is available; otherwise, use CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # For debugging purposes, you can switch to CPU
        #device = torch.device('cpu')
        print("Using device:", device)

        self.device = device
        self.model = model.to(self.device)
        self.model_hyper = model_hyper
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.weight_decay = weight_decay
        self.epoch = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.learning_decay)
        self.save_path = save_path
        self.comet_logger = comet_logger

    def get_data(self, sample):
        pass

    def training_loss(self, inputs):
        pass

    def calculate_loss(self, inputs):
        pass

    def restore(self, save_file):
        # Restore training state from a saved checkpoint file
        checkpoint = torch.load(save_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        print("Restoring training from epoch {}...".format(self.epoch))

    def reset(self):
        self.epoch = 0
        self.model.apply(fn=weight_reset)

    def train(self, n_epochs, phases, datasets, dataloaders):
        best_val = 1e6
        pbar = tqdm(initial=self.epoch, total=n_epochs)
        while self.epoch < n_epochs:
            #print('Epoch {}'.format(self.epoch))
            for phase in phases:
                running_dict = {}

                if phase == 'train':
                    #print("Training with learning rate: {}".format(self.scheduler.get_last_lr()[0]))
                    self.model.train()
                else:
                    self.model.eval()

                for i, sample in enumerate(dataloaders[phase]):

                    # Get data for the current batch
                    data = self.get_data(sample)

                    if phase == 'train':
                        self.optimizer.zero_grad()

                    # Calculate batch loss
                    batch_loss_dict = self.training_loss(data)

                    if phase == 'train':
                        batch_loss_dict['TOTAL_LOSS'].backward()
                        self.optimizer.step()

                    # Update running loss values
                    for loss_name, loss_value in batch_loss_dict.items():
                        if loss_name not in running_dict:
                            if loss_name == 'CONF_MATRIX':
                                running_dict[loss_name] = loss_value
                            else:
                                running_dict[loss_name] = loss_value.item() / dataloaders[phase].batch_size
                        else:
                            if loss_name == 'CONF_MATRIX':
                                running_dict[loss_name] += loss_value
                            else:
                                running_dict[loss_name] += loss_value.item() / dataloaders[phase].batch_size

                # Print and log loss values for the current phase
                for loss_name, loss_value in running_dict.items():
                    if loss_name == 'CONF_MATRIX':
                        #print('     Phase {}: confusion:'.format(phase))
                        confusion = loss_value / loss_value.sum(1)
                        #print(confusion)
                        self.comet_logger.log_confusion_matrix(matrix=confusion,
                                                               file_name="{}-confusion-matrix.json".format(phase))
                    else:
                        #print("     Phase {}: {}: {:.4E}".format(phase, loss_name, loss_value / (len(datasets[phase]))))
                        self.comet_logger.log_metric("{}_{}".format(phase, loss_name),
                                                     loss_value / len(datasets[phase]), step=self.epoch)

                # Save the model if it performs better on the validation set
                if phase == 'val':
                    if (running_dict['TOTAL_LOSS'] / len(datasets[phase])) <= best_val:
                        best_val = running_dict['TOTAL_LOSS'] / len(datasets[phase])
                        torch.save({
                            'epoch': self.epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'best_val': best_val,
                        },
                            self.save_path)

            # Increment the epoch count and update the learning rate
            self.epoch += 1
            pbar.update(1)
            self.scheduler.step(self.epoch)
        pbar.close()

    def test(self, dataset, dataloader):
        self.model.eval()
        running_dict = {}
        for i, sample in enumerate(dataloader):

            # Get data for the current batch
            data = self.get_data(sample)

            # Calculate batch loss
            batch_loss_dict = self.calculate_loss(data)

            # Update running loss values
            for loss_name, loss_value in batch_loss_dict.items():
                if loss_name not in running_dict:
                    if loss_name == 'CONF_MATRIX':
                        running_dict[loss_name] = loss_value
                    else:
                        running_dict[loss_name] = loss_value.item() / dataloader.batch_size
                else:
                    if loss_name == 'CONF_MATRIX':
                        running_dict[loss_name] += loss_value
                    else:
                        running_dict[loss_name] += loss_value.item() / dataloader.batch_size

        # normalize
        for loss_name, loss_value in running_dict.items():
            running_dict[loss_name] = loss_value / len(dataset)

        return running_dict


# Define TrainerAE class, which is a subclass of Train for autoencoders
class TrainerAE(Train):
    def __init__(self,
                 model,
                 model_hyper,
                 learning_rate,
                 learning_decay,
                 weight_decay,
                 save_path,
                 comet_logger
                 ):
        super(TrainerAE, self).__init__(model,
                                        model_hyper,
                                        learning_rate,
                                        learning_decay,
                                        weight_decay,
                                        save_path,
                                        comet_logger
                                        )

        # Define Mean Squared Error (MSE) loss for reconstruction
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def get_data(self, sample):
        # Extract data from the sample and convert it to the appropriate data type and device
        data = sample[self.model_hyper['target']].float()
        data = data.to(self.device)
        return data

    def calculate_loss(self, inputs):
        loss_dict = {}
        latent_features = self.model.encode(inputs + generate_noise(inputs).to(self.device))
        reconstruction = self.model.decode(latent_features)
        loss = self.criterion(reconstruction, inputs)
        loss_dict['TOTAL_LOSS'] = loss
        loss_dict['MSE_LOSS'] = loss
        return loss_dict

    def training_loss(self, inputs):
        return self.calculate_loss(inputs)


# Define TrainerVAE class, a subclass of TrainerAE for Variational Autoencoders (VAE)
class TrainerVAE(TrainerAE):
    def __init__(self,
                 model,
                 model_hyper,
                 learning_rate,
                 learning_decay,
                 weight_decay,
                 save_path,
                 comet_logger,
                 annealing_epochs
                 ):
        super(TrainerVAE, self).__init__(model,
                                         model_hyper,
                                         learning_rate,
                                         learning_decay,
                                         weight_decay,
                                         save_path,
                                         comet_logger
                                         )
        self.annealing_epochs = annealing_epochs

    def training_loss(self, inputs):
        loss_dict = {}
        mu, logvar = self.model.encode(inputs + generate_noise(inputs).to(self.device))
        if self.epoch < self.annealing_epochs:
            beta = 1e2 * self.epoch / self.annealing_epochs
        else:
            beta = 1e2

        reconstruction = self.model.decode(self.model.reparameterize(mu, logvar))
        mse_loss, kld_loss = self.model.loss_function(reconstruction, inputs, mu, logvar, self.criterion, beta)
        relative_error = (torch.abs(reconstruction - inputs) / inputs).mean()
        loss = mse_loss + kld_loss
        loss_dict['TOTAL_LOSS'] = loss
        loss_dict['MSE_LOSS'] = mse_loss
        loss_dict['KLD_LOSS'] = kld_loss
        loss_dict['RELATIVE_ERROR'] = relative_error
        return loss_dict

    def calculate_loss(self, inputs):
        loss_dict = {}
        self.model.eval()
        mu, logvar = self.model.encode(inputs + generate_noise(inputs).to(self.device))
        reconstruction = self.model.decode(mu)

        mse_loss, kld_loss = self.model.loss_function(reconstruction, inputs, mu, logvar, self.criterion, 1e2)
        loss = mse_loss + kld_loss
        loss_dict['TOTAL_LOSS'] = loss
        loss_dict['MSE_LOSS'] = mse_loss
        loss_dict['KLD_LOSS'] = kld_loss
        return loss_dict


# Define TrainerCVAE class, a subclass of TrainerVAE for Conditional Variational Autoencoders (CVAE)
class TrainerCVAE(TrainerVAE):
    def __init__(self,
                 model,
                 model_hyper,
                 learning_rate,
                 learning_decay,
                 weight_decay,
                 save_path,
                 comet_logger,
                 annealing_epochs,
                 n_classes,
                 class_weight
                 ):
        super(TrainerCVAE, self).__init__(model,
                                          model_hyper,
                                          learning_rate,
                                          learning_decay,
                                          weight_decay,
                                          save_path,
                                          comet_logger,
                                          annealing_epochs
                                          )
        self.n_classes = n_classes
        self.class_weight = class_weight

    def get_data(self, sample):
        data = sample[self.model_hyper['target']].float()
        labels = sample['label'].float()
        data = data.to(self.device)
        labels = labels.to(self.device)
        return data, labels

    def training_loss(self, input_tuple):
        inputs, labels = input_tuple
        loss_dict = {}

        # Convert labels to one-hot encoded format
        seq_len = inputs.size(1)
        labels = torch.nn.functional.one_hot(labels.long(), self.n_classes).float().unsqueeze(1).repeat((1, seq_len, 1))
        inputs_labels = torch.cat([inputs + generate_noise(inputs).to(self.device), labels], -1)

        mu, logvar = self.model.encode(inputs_labels)
        if self.epoch < self.annealing_epochs:
            beta = 1e2 * self.epoch / self.annealing_epochs
        else:
            beta = 1e2
        z = self.model.reparameterize(mu, logvar)
        reconstruction = self.model.decode(z, labels)
        mse_loss, kld_loss = self.model.loss_function(reconstruction, inputs, mu, logvar, self.criterion, beta)
        loss = mse_loss + kld_loss
        loss_dict['TOTAL_LOSS'] = loss
        loss_dict['MSE_LOSS'] = mse_loss
        loss_dict['KLD_LOSS'] = kld_loss
        return loss_dict

    def calculate_loss(self, input_tuple):
        loss_dict = {}
        inputs, labels = input_tuple

        # Convert labels to one-hot encoded format
        seq_len = inputs.size(1)

        labels = torch.nn.functional.one_hot(labels.long(), self.n_classes).float().unsqueeze(1).repeat((1, seq_len, 1))
        inputs_labels = torch.cat([inputs + generate_noise(inputs).to(self.device), labels], -1)
        mu, logvar = self.model.encode(inputs_labels)
        reconstruction = self.model.decode(mu)

        mse_loss, kld_loss = self.model.loss_function(reconstruction, inputs, mu, logvar, self.criterion, 1e2)
        loss = mse_loss + kld_loss
        loss_dict['TOTAL_LOSS'] = loss
        loss_dict['MSE_LOSS'] = mse_loss
        loss_dict['KLD_LOSS'] = kld_loss
        return loss_dict


# Define TrainerCVAE class, a subclass of TrainerVAE for Conditional Variational Autoencoders (CVAE)
class TrainerCVAEContinuous(TrainerVAE):
    def __init__(self,
                 model,
                 model_hyper,
                 learning_rate,
                 learning_decay,
                 weight_decay,
                 save_path,
                 comet_logger,
                 annealing_epochs,
                 n_classes,
                 class_weight
                 ):
        super(TrainerCVAEContinuous, self).__init__(model,
                                                    model_hyper,
                                                    learning_rate,
                                                    learning_decay,
                                                    weight_decay,
                                                    save_path,
                                                    comet_logger,
                                                    annealing_epochs
                                                    )
        self.n_classes = n_classes
        self.class_weight = class_weight

    def get_data(self, sample):
        data = sample[self.model_hyper['target']].float()
        labels = sample['flexibility'].float()
        data = data.to(self.device)
        labels = labels.to(self.device)
        return data, labels

    def training_loss(self, input_tuple):
        inputs, labels = input_tuple
        loss_dict = {}

        # Convert labels to one-hot encoded format
        seq_len = inputs.size(1)

        labels = labels.float().unsqueeze(1).unsqueeze(1).repeat((1, seq_len, 1))
        inputs_labels = torch.cat([inputs + generate_noise(inputs).to(self.device), labels], -1)
        mu, logvar = self.model.encode(inputs_labels)
        if self.epoch < self.annealing_epochs:
            beta = 1e2 * self.epoch / self.annealing_epochs
        else:
            beta = 1e2
        z = self.model.reparameterize(mu, logvar)
        reconstruction = self.model.decode(z, labels)
        mse_loss, kld_loss = self.model.loss_function(reconstruction, inputs, mu, logvar, self.criterion, beta)
        loss = mse_loss + kld_loss
        loss_dict['TOTAL_LOSS'] = loss
        loss_dict['MSE_LOSS'] = mse_loss
        loss_dict['KLD_LOSS'] = kld_loss
        return loss_dict

    def calculate_loss(self, input_tuple):
        loss_dict = {}
        inputs, labels = input_tuple

        # Convert labels to one-hot encoded format
        seq_len = inputs.size(1)

        labels = labels.float().unsqueeze(1).unsqueeze(1).repeat((1, seq_len, 1))
        inputs_labels = torch.cat([inputs + generate_noise(inputs).to(self.device), labels], -1)
        mu, logvar = self.model.encode(inputs_labels)
        reconstruction = self.model.decode(mu, labels)

        mse_loss, kld_loss = self.model.loss_function(reconstruction, inputs, mu, logvar, self.criterion, 1e2)

        relative_error = (torch.abs(reconstruction - inputs) / inputs).mean()
        loss = mse_loss + kld_loss
        loss_dict['TOTAL_LOSS'] = loss
        loss_dict['MSE_LOSS'] = mse_loss
        loss_dict['RELATIVE_ERROR'] = relative_error
        loss_dict['KLD_LOSS'] = kld_loss
        return loss_dict


# Define TrainerSVAE class, a subclass of TrainerVAE for Supervised Variational Autoencoders (SVAE)
class TrainerSVAE(TrainerVAE):
    def __init__(self,
                 model,
                 model_hyper,
                 learning_rate,
                 learning_decay,
                 weight_decay,
                 save_path,
                 comet_logger,
                 annealing_epochs,
                 n_classes,
                 class_weight,
                 class_criterion
                 ):
        super(TrainerSVAE, self).__init__(model,
                                          model_hyper,
                                          learning_rate,
                                          learning_decay,
                                          weight_decay,
                                          save_path,
                                          comet_logger,
                                          annealing_epochs
                                          )
        self.n_classes = n_classes
        self.class_weight = class_weight
        self.class_criterion = class_criterion

    def get_data(self, sample):
        data = sample[self.model_hyper['target']].float()
        labels = sample['label'].float()
        data = data.to(self.device)
        labels = labels.to(self.device)
        return data, labels

    def training_loss(self, input_tuple):
        inputs, labels = input_tuple
        loss_dict = {}

        # Convert labels to one-hot encoded format
        labels = torch.nn.functional.one_hot(labels.long(), self.n_classes).float()

        mu, logvar = self.model.encode(inputs + generate_noise(inputs).to(self.device))
        if self.epoch < self.annealing_epochs:
            beta = 1e2 * self.epoch / self.annealing_epochs
        else:
            beta = 1e2
        z = self.model.reparameterize(mu, logvar)
        reconstruction = self.model.decode(z)
        mse_loss, kld_loss = self.model.loss_function(reconstruction, inputs, mu, logvar, self.criterion, beta)
        relative_error = (torch.abs(reconstruction - inputs) / inputs).mean()

        # Classification
        classifier_inputs = torch.cat([mu, logvar], dim=-1)
        preds_labels_score = self.model.classifier(classifier_inputs)
        class_loss = self.class_weight * self.class_criterion(preds_labels_score, labels)

        # Accuracy monitoring
        pred_labels = torch.argmax(preds_labels_score.squeeze(), dim=-1, keepdim=True)
        ground_labels = torch.argmax(labels, dim=-1, keepdim=True)
        accuracy = torch.sum(pred_labels == ground_labels) * ground_labels.size(0)

        # Confusion matrix
        confusion_matrix = torch.zeros(self.n_classes, self.n_classes)
        for t, p in zip(ground_labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        loss = mse_loss + kld_loss + class_loss
        loss_dict['TOTAL_LOSS'] = loss
        loss_dict['MSE_LOSS'] = mse_loss
        loss_dict['KLD_LOSS'] = kld_loss
        loss_dict['RELATIVE_ERROR'] = relative_error
        loss_dict['CLASS_LOSS'] = class_loss
        loss_dict['ACCURACY'] = accuracy
        loss_dict['CONF_MATRIX'] = confusion_matrix
        return loss_dict

    def calculate_loss(self, input_tuple):
        loss_dict = {}
        inputs, labels = input_tuple

        mu, logvar = self.model.encode(inputs + generate_noise(inputs).to(self.device))
        reconstruction = self.model.decode(mu)

        mse_loss, kld_loss = self.model.loss_function(reconstruction, inputs, mu, logvar, self.criterion, 1e2)
        loss = mse_loss + kld_loss
        loss_dict['TOTAL_LOSS'] = loss
        loss_dict['MSE_LOSS'] = mse_loss
        loss_dict['KLD_LOSS'] = kld_loss
        return loss_dict


# Define TrainerSVAE class, a subclass of TrainerVAE for Supervised Variational Autoencoders (SVAE)
class TrainerClassifier(Train):
    def __init__(self,
                 model,
                 model_hyper,
                 learning_rate,
                 learning_decay,
                 weight_decay,
                 save_path,
                 comet_logger,
                 n_classes,
                 class_criterion
                 ):
        super(TrainerClassifier, self).__init__(model,
                                                model_hyper,
                                                learning_rate,
                                                learning_decay,
                                                weight_decay,
                                                save_path,
                                                comet_logger,
                                                )
        self.n_classes = n_classes
        self.class_criterion = class_criterion

    def get_data(self, sample):
        data = sample[self.model_hyper['target']].float()
        labels = sample['label'].float()
        data = data.to(self.device)
        labels = labels.to(self.device)
        return data, labels

    def calculate_loss(self, input_tuple):
        inputs, labels = input_tuple
        loss_dict = {}

        # Convert labels to one-hot encoded format
        labels = torch.nn.functional.one_hot(labels.long(), self.n_classes).float()
        preds_labels_score = self.model(inputs)
        class_loss = self.class_criterion(preds_labels_score, labels)

        # Accuracy monitoring
        pred_labels = torch.argmax(preds_labels_score.squeeze(), dim=-1, keepdim=True)
        ground_labels = torch.argmax(labels, dim=-1, keepdim=True)
        accuracy = torch.sum(pred_labels == ground_labels) * ground_labels.size(0)

        # Confusion matrix
        confusion_matrix = torch.zeros(self.n_classes, self.n_classes)
        for t, p in zip(ground_labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        loss = class_loss
        loss_dict['TOTAL_LOSS'] = loss
        loss_dict['CLASS_LOSS'] = class_loss
        loss_dict['ACCURACY'] = accuracy
        loss_dict['CONF_MATRIX'] = confusion_matrix
        return loss_dict
