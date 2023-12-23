import torch
import torch.nn as nn
import torch.nn.functional as F


class HMCN(nn.Module):
    def __init__(self, args):
        super(HMCN, self).__init__()

        self.args = args

        in_dim = self.args.in_dim
        global_weight_dim = self.args.global_weight_dim
        total_levels = self.args.total_levels
        transition_weight_dim = self.args.transition_weight_dim
        total_classes_at_level = self.args.total_classes_at_level
        total_classes = sum(total_classes_at_level)

        self.global_layers = nn.ModuleList([])
        self.local_layers = nn.ModuleList([])
        self.close_set_layers = nn.ModuleList([])
        self.block_num = 16

        for idx in range(total_levels):
            if idx == 0:
                self.global_layers.append(
                    nn.ModuleList(
                        nn.Sequential(
                            nn.Linear(in_dim, global_weight_dim),
                            nn.ReLU(),
                            nn.BatchNorm1d(global_weight_dim),
                            nn.Dropout(p=0.5)
                    )))
            else:
                self.global_layers.append(
                    nn.ModuleList(
                        nn.Sequential(
                            nn.Linear(global_weight_dim + in_dim, global_weight_dim),
                            nn.ReLU(),
                            nn.BatchNorm1d(global_weight_dim),
                            nn.Dropout(p=0.5)
                    )))

            if self.args.deep_residual:
                for block in range(self.block_num):
                    self.global_layers[idx].append(
                        nn.Sequential(
                            nn.Linear(global_weight_dim, global_weight_dim),
                            nn.ReLU(),
                            nn.BatchNorm1d(global_weight_dim),
                            nn.Dropout(p=0.5)
                        ))

            self.local_layers.append(
                nn.Sequential(
                    nn.Linear(global_weight_dim, transition_weight_dim[idx]),
                    torch.nn.ReLU(),
                    nn.BatchNorm1d(transition_weight_dim[idx]),
                    nn.Linear(transition_weight_dim[idx], total_classes_at_level[idx])
                ))
            
            self.close_set_layers.append(
                nn.Sequential(
                    nn.Linear(global_weight_dim, transition_weight_dim[idx]),
                    torch.nn.ReLU(),
                    nn.BatchNorm1d(transition_weight_dim[idx]),
                    nn.Linear(transition_weight_dim[idx], total_classes_at_level[idx] - 1)
                ))
            

        self.final_layer = nn.Linear(global_weight_dim, total_classes)

        self.global_layers.apply(self._init_weight)
        self.local_layers.apply(self._init_weight)
        self.close_set_layers.apply(self._init_weight)
        self.final_layer.apply(self._init_weight)

    def _init_weight(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.1) 

    def forward(self, x):
        local_outputs = []
        close_set_outputs = []
        global_activation = x 

        for i, (local_layer, close_set_layer, global_layer) in enumerate(zip(self.local_layers, self.close_set_layers, self.global_layers)):
            global_activation = global_layer[0](global_activation)
            
            if self.args.deep_residual:
                for block in range(self.block_num):
                    if block % 2 == 0: 
                        residual = global_activation

                    global_activation = global_layer[block + 1](global_activation)
                
                    if (block + 1) % 2 == 0:
                        global_activation += residual


            local_outputs.append(local_layer(global_activation)) 
            close_set_outputs.append(close_set_layer(global_activation)) 

            if i < len(self.global_layers)-1:
                global_activation = torch.cat((global_activation, x), 1)

        global_output = self.final_layer(global_activation) 


        return global_output, local_outputs, close_set_outputs
