import torch
import torch.nn as nn

from graph_model.dynamic_gnn_with_mtgat_prune import DynamicMTGATPruneModel
from consts import GlobalConsts as gc

class NetMTGATAverageUnalignedConcatMHA(nn.Module):
    def __init__(self, num_gat_layers, use_transformer=False, use_prune=False, use_pe=False):
        super(NetMTGATAverageUnalignedConcatMHA, self).__init__()
        if use_transformer:
            raise NotImplementedError
            # if not use_prune:
            #     self.dgnn = DynamicGNNModelWithTransformerPadding(gc.config, concat=True, num_gat_layers=num_gat_layers)
            # else:
            #     self.dgnn = DynamicGNNModelWithTransformerPaddingPrune(gc.config, concat=True,
            #                                                            num_gat_layers=num_gat_layers)

        else:
            if not use_prune:
                raise NotImplementedError('Only pruned version is implemented now.')
            else:
                self.dgnn = DynamicMTGATPruneModel(gc.config, concat=True, num_gat_layers=num_gat_layers, use_pe=use_pe)
        
        label_dim = 1
        if gc.dataset == "mosei":
            label_dim = 7
        elif gc.dataset in ['iemocap', 'iemocap_unaligned']:
            label_dim = 8 # 2 x 4category
        self.finalW = nn.Sequential(
            nn.Linear(gc.config['graph_conv_out_dim'], gc.config['graph_conv_out_dim'] // 4),
            nn.ReLU(),
            # nn.Linear(gc.config['graph_conv_out_dim'] // 4, label_dim),
            nn.Linear(gc.config['graph_conv_out_dim'] // 4, gc.config['graph_conv_out_dim'] // 4),
            nn.ReLU(),
            nn.Linear(gc.config['graph_conv_out_dim'] // 4, label_dim),
        )

    def forward(self, **kwargs):

        state = self.dgnn(**kwargs)
        state = torch.stack([torch.mean(state_i, dim=0) for state_i in state], 0)
        return self.finalW(state).squeeze()

    def inference_return_layer_outputs(self, **kwargs):
        state, batch, nodes_rec, edge_indices_rec, edge_weights_rec, edge_types_rec = self.dgnn(**kwargs)
        state = torch.stack([torch.mean(state_i, dim=0) for state_i in state], 0)
        return self.finalW(state).squeeze(), batch, nodes_rec, edge_indices_rec, edge_weights_rec, edge_types_rec
