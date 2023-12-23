import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

class HMCNTrainer():
    def __init__(self, model, encode_mode, encoder, encoder_weights, criterion, optimizer, train_loader, val_loader, test_loader=None, device='cuda'):
        self.model = model
        self.encode_mode = encode_mode
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.phi = 0.25
        self.tao = 1

    def encode(self, x):
        if self.encode_mode.startswith('vit'):
            preprocessing = self.encoder_weights.DEFAULT.transforms()

            x = preprocessing(x)
            x = self.encoder._process_input(x)

            # Expand the class token to the full batch
            batch_class_token = self.encoder.class_token.expand(x.shape[0], -1, -1)
            feats = torch.cat([batch_class_token, x], dim=1)
            feats = self.encoder.encoder(feats)
            feats = feats[:, 0]

        elif self.encode_mode == 'cnn':
            encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-1]))

            feats = encoder(x)
            feats = feats.reshape((x.shape[0], 512))
            
        return feats

    def train_epoch(self):
        loss = 0.0
        running_loss = 0.0
        device = self.device
        phi = self.phi
        tao = self.tao

        for i, data in enumerate(self.train_loader):
            inputs, super_labels, sub_labels = data[0].to(device), data[1].to(device), data[3].to(device)
           
            novel_labels = [3, 87]
            labels = [super_labels, sub_labels]
            cs_super_labels_one_hot = F.one_hot(super_labels, num_classes=3).float()
            cs_sub_labels_one_hot = F.one_hot(sub_labels, num_classes=87).float()
            cs_labels_one_hot = [cs_super_labels_one_hot, cs_sub_labels_one_hot]

            self.optimizer.zero_grad()
            
            inputs = self.encode(inputs)
            global_output, local_outputs, close_set_outputs = self.model(inputs)

            close_set_loss = 0.0
            for j, (close_set_output, cs_label_one_hot) in enumerate(zip(close_set_outputs, cs_labels_one_hot)):
                close_set_loss += self.criterion(close_set_output, cs_label_one_hot)
                close_set_output = torch.sigmoid(close_set_output)
                dot_product_result = torch.sum(close_set_output * cs_label_one_hot, dim=1)
                score = dot_product_result.view(-1, 1)
                sorted_score, sorted_indices = torch.sort(score.flatten())

                atypical_indices = sorted_indices[:int(sorted_indices.size()[0] * phi)]
                labels[j][atypical_indices] = novel_labels[j]

            super_labels_one_hot = F.one_hot(labels[0], num_classes=4).float()
            sub_labels_one_hot = F.one_hot(labels[1], num_classes=88).float()
            all_labels_one_hot = torch.cat((super_labels_one_hot, sub_labels_one_hot), 1).float()

            loss = self.criterion(local_outputs[0], super_labels_one_hot) \
                 + self.criterion(local_outputs[1], sub_labels_one_hot) \
                 + self.criterion(global_output, all_labels_one_hot) \
                 + tao * close_set_loss

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if (i % 10 == 0) and i != 0:
                print(f'Batch {i} loss: {loss.item()}')
        
        print(f'Training loss: {running_loss/i:.3f}')

        return running_loss/i

    def validate_epoch(self, B=0):
        super_correct = 0
        sub_correct = 0
        B_super_correct = 0
        B_sub_correct = 0
        total = 0
        running_loss = 0.0
        device = self.device
        phi = self.phi
        tao = self.tao

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                inputs, super_labels, sub_labels = data[0].to(device), data[1].to(device), data[3].to(device)

                novel_labels = [3, 87]
                labels = [super_labels, sub_labels]
                cs_super_labels_one_hot = F.one_hot(super_labels, num_classes=3).float()
                cs_sub_labels_one_hot = F.one_hot(sub_labels, num_classes=87).float()
                cs_labels_one_hot = [cs_super_labels_one_hot, cs_sub_labels_one_hot]

                inputs = self.encode(inputs)
                global_output, local_outputs, close_set_outputs = self.model(inputs)

                close_set_loss = 0.0
                for j, (close_set_output, cs_label_one_hot) in enumerate(zip(close_set_outputs, cs_labels_one_hot)):
                    close_set_loss += self.criterion(close_set_output, cs_label_one_hot)
                    close_set_output = torch.sigmoid(close_set_output)
                    dot_product_result = torch.sum(close_set_output * cs_label_one_hot, dim=1)
                    score = dot_product_result.view(-1, 1)
                    sorted_score, sorted_indices = torch.sort(score.flatten())

                    atypical_indices = sorted_indices[:int(sorted_indices.size()[0] * phi)]
                    labels[j][atypical_indices] = novel_labels[j]

                super_labels_one_hot = F.one_hot(labels[0], num_classes=4).float()
                sub_labels_one_hot = F.one_hot(labels[1], num_classes=88).float()
                all_labels_one_hot = torch.cat((super_labels_one_hot, sub_labels_one_hot), 1).float()

                loss = self.criterion(local_outputs[0], super_labels_one_hot) \
                     + self.criterion(local_outputs[1], sub_labels_one_hot) \
                     + self.criterion(global_output, all_labels_one_hot) \
                     + close_set_loss
                
                global_output = torch.sigmoid(global_output)
                local_outputs = torch.sigmoid(torch.cat(local_outputs, 1))

                pred = B * global_output + (1 - B) * local_outputs

                _, super_predicted = torch.max(pred[:,0: 4].data, 1)
                _, sub_predicted = torch.max(pred[:,4:].data, 1)

                total += super_labels.size(0)
                super_correct += (super_predicted == labels[0]).sum().item()
                sub_correct += (sub_predicted == labels[1]).sum().item()
                running_loss += loss.item()

        print(f'Validation loss: {running_loss/i:.3f}')
        print(f'Validation superclass acc: {100 * super_correct / total:.2f} %')
        print(f'Validation subclass acc: {100 * sub_correct / total:.2f} %')

        return running_loss/i

    def test(self, test_model, B=0.5, save_to_csv=False, return_predictions=False):
        if not self.test_loader:
            raise NotImplementedError('test_loader not specified')
        
        self.model.eval()
        # Evaluate on test set, in this simple demo no special care is taken for novel/unseen classes
        test_predictions = {'ID': [], 'superclass_index': [], 'subclass_index': []}
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, img_name = data[0].to(self.device), data[1]

                inputs = self.encode(inputs)
                global_output, local_outputs, close_set_outputs = self.model(inputs)
                
                global_output = F.sigmoid(global_output)
                local_outputs = F.sigmoid(torch.cat(local_outputs, 1))

                pred = B * global_output + (1 - B) * local_outputs

                _, super_predicted = torch.max(pred[:,0: 4].data, 1)
                _, sub_predicted = torch.max(pred[:,4:].data, 1)

                test_predictions['ID'].append(img_name[0])
                test_predictions['superclass_index'].append(super_predicted.item())
                test_predictions['subclass_index'].append(sub_predicted.item())

        test_predictions = pd.DataFrame(data=test_predictions)

        if save_to_csv:
            super_pred = test_predictions[['ID', 'superclass_index']]
            super_pred = super_pred.rename({'superclass_index': 'Taregt'}, axis=1)

            sub_pred = test_predictions[['ID', 'subclass_index']]
            sub_pred = sub_pred.rename({'subclass_index': 'Taregt'}, axis=1)

            super_pred.to_csv(f'./results/{test_model}_sup_pred.csv', index=False)
            sub_pred.to_csv(f'./results/{test_model}_sub_pred.csv', index=False)

        if return_predictions:
            return test_predictions
