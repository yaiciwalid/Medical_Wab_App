import sys
import time
import torch.nn as nn
import torchvision.models as models
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from app.preprocessing.pneumonia import Pneumonia
from PIL import Image


sys.path.append('../')


class PneumoniaModel:
    """PneuMoonia Model"""
    def __init__(self, load=False):

        self._preprocessClass = Pneumonia()
        PARAMETERS = {}
        PARAMETERS['model_name'] = 'resnet18'
        # PARAMETERS['model_saved'] = "app/models/trained_models"
        PARAMETERS['learning_rate'] = 0.05
        PARAMETERS['momentum'] = 0.9
        PARAMETERS['epochs'] = 100
        PARAMETERS['weight_decay'] = 0.0001
        PARAMETERS['batch_size'] = 128
        PARAMETERS['temperature'] = 0.07
        PARAMETERS['num_channels'] = 3
        PARAMETERS['dictionary_size'] = 8192
        PARAMETERS['num_workers'] = 1
        PARAMETERS['num_cores'] = 1
        PARAMETERS['log_steps'] = 20
        PARAMETERS['load_from_saved'] = False
        PARAMETERS['start_epoch'] = 1
        self.PARAMETERS = PARAMETERS

        self.load_trained_moco()
        self.classification_model = self.PneumoniaClassifierModel(self.encoder_model)
        self.init_tranformation()
        if load:
            # If we charge the trained model (no need for GPU)
            self.load_model()
            self.loss_function = nn.BCELoss()
            
        else:
            # If we decide to train the model (need GPU)
            torch.cuda.empty_cache()
            self.classification_model = self.classification_model.cuda(0)
            self.loss_function = nn.BCELoss().cuda()
            self.prepare_data()
        self.optimizer = torch.optim.Adam(self.classification_model.parameters())
        self.start_epoch = 0
        
    def load_trained_moco(self):
        """Load the trained Moco for using it as encoder"""
        pth = self._preprocessClass._pretrained_encoder_path
        self.encoder_model = self.BaseModel('resnet18', 128)
        if torch.cuda.is_available() is True:
            self.encoder_model.load_state_dict(torch.load(pth)['model_state_dict'])
        else:
            self.encoder_model.load_state_dict(torch.load(pth,
                                                          map_location=torch.device('cpu'))['model_state_dict'])

    def load_model(self):
        """Load the train classifier"""
        pth = self._preprocessClass._pretrained_classifier_path
        if torch.cuda.is_available() is True:
            self.classification_model.load_state_dict(torch.load(pth)['model_state_dict'])
        else:
            self.classification_model.load_state_dict(torch.load(pth,
                                                                 map_location=torch.device('cpu'))['model_state_dict'])

    def predict(self, image):
        """Predict from an X-ray Pneumonia or Not"""
        image_test = Image.open(image).convert('RGB')
        image_test_tensor = self.transformation(image_test).unsqueeze(0)
        # img = image_test_tensor.cuda(0)
        img = image_test_tensor
        self.classification_model.eval()
        pred = self.classification_model.forward(img)

        return (pred.item() > 0.5) * 1

    def train(self, num_epochs):
        """Train the model for num_epochs"""
        start = self.start_epoch
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_acc = 0.0
            epoch_acc = 0.0
            epoch_loss = 0.0

            nb_elts = 0
            self.classification_model.train()
            for i, bc in enumerate(self.class_train_loader):

                inputs, labels = bc[0], bc[1]

                # Load images and labels to GPU
                inputs = inputs.cuda(0)
                labels = labels.cuda(0)

                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.classification_model(inputs)
                loss = self.loss_function(outputs.flatten(), labels.float())

                # Backward pass and optimization step
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item()

                predictions = (outputs > 0.5).long().flatten()

                correct = (predictions == labels).sum().item()

                epoch_acc += correct
                running_acc += correct/predictions.size(0)
                nb_elts += predictions.size(0)
                if ((i+1) % 10 == 0):
                    print('Batch =({}) Acc =({})  Loss={:.5f} Time={}'.format(
                        i, running_acc/10, running_loss/10, time.asctime()), flush=True)
                    running_loss = 0.0
                    running_acc = 0.0
        epoch_acc = epoch_acc / nb_elts
        print('Epoch: '+str(start + epoch) + ' Acc: '+str(epoch_acc) + 'Loss: '+str(epoch_loss))

        # Evaluation mode for validation set
        self.classification_model.eval()
        val_acc = 0
        val_loss = 0
        for inputs, label in self.class_test_loader:
            inputs = inputs.cuda(0)
            label = label.cuda(0)

            target = self.classification_model(inputs).flatten()
            loss = self.loss_function(target, label.float())
            prediction = (target > 0.5).long().flatten()
            val_acc += (prediction == label).item() * 1
            val_loss += loss.item()
        print('Epoch: {} Val Acc ={:.2f}  Val Loss={:.5f}'.format(
                    epoch + start, val_acc/len(self.class_test_loader), val_loss))
        print('\n')

    def prepare_data(self):
        """Load and prepare data for training"""

        path_train = self._preprocessClass.get_list_folders(subset='train')
        path_validation = self._preprocessClass.get_list_folders(subset='val')
        self.class_train_dataset = datasets.ImageFolder(
                path_train,
                transforms.Compose(self.transformation)
                                                       )
        self.class_train_loader = torch.utils.data.DataLoader(
                self.class_train_dataset,
                batch_size=self.PARAMETERS['batch_size'],
                shuffle=True
                                                        )
        self.class_test_dataset = datasets.ImageFolder(
                path_validation,
                transforms.Compose(self.transformation)
                                                      )
        self.class_test_loader = torch.utils.data.DataLoader(
            self.class_test_dataset,
            shuffle=True
                                                    )

    def init_tranformation(self):
        """Initalize the transformation that we gonna use for moco"""
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.ToTensor(),
            normalize
                       ]
        self.transformation = transforms.Compose(augmentation)
        
    class BaseModel(nn.Module):
        """The BaseModel usused for training the Encoder"""
        def __init__(self, base_model_name, channels_out):
            super().__init__()

            if base_model_name == 'resnet50':
                model = models.resnet50(pretrained=False)
            elif base_model_name == 'resnet18':
                model = models.resnet18(pretrained=False)

            penultimate = model.fc.weight.shape[1]
            modules = list(model.children())[:-1]
            self.encoder = nn.Sequential(*modules)

            self.relu = nn.ReLU()
            self.fc = nn.Linear(penultimate, channels_out)

        def forward(self, x):
            x = self.encoder(x)
            x = x.view(x.size(0), -1)
            x = self.relu(x)
            x = self.fc(x)

            return x

    class PneumoniaClassifierModel(nn.Module):
        """The classifier Model"""
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            """Freeze the pre trained moco model"""
            for params in self.encoder.parameters():
                params.requires_grad = False

            encoder_sortie = self.encoder.fc.weight.shape[0]
            self.relu1 = nn.ReLU()
            self.fc = nn.Linear(encoder_sortie, 512)
            self.relu2 = nn.ReLU()
            self.fc2 = nn.Linear(512, 512)
            self.relu3 = nn.ReLU()
            self.fc3 = nn.Linear(512, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.encoder(x)
            x = self.relu1(x)
            x = self.fc(x)
            x = self.relu2(x)
            x = self.fc2(x)
            x = self.relu3(x)
            x = self.fc3(x)
            x = self.sigmoid(x)
            return x
