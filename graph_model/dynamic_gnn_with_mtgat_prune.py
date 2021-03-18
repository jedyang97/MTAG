## Torch Modules
import copy

import torch
import torch.nn as nn
from torch_geometric.data import Data as gData
from torch_geometric.data import Batch

import traceback

# pytorch geometric
import torch_geometric as pyg

## graph building model_utils
from graph_model.graph_builder import construct_time_aware_dynamic_graph
from graph_model.mtgat_conv import MTGATConv
from graph_model.model_utils import device
from graph_model.pooling import TopKEdgePooling, TopKPooling, RandomEdgePooling
from graph_model.transformer import PositionalEncoding


class DynamicMTGATPruneModel(nn.Module):

    def __init__(self, config, concat=True, num_gat_layers=1, use_pe=False):
        super(DynamicMTGATPruneModel, self).__init__()

        # object states
        self.config = config
        self.use_pe = use_pe
        # Graph definitions

        if concat:
            gat_out_channel = int(self.config['graph_conv_out_dim'] / self.config['gat_conv_num_heads'])
        else:
            gat_out_channel = int(self.config['graph_conv_out_dim'])

        self.gats = nn.ModuleList([])
        for l in range(num_gat_layers):
            if self.config['time_aware_edges'] and self.config['type_aware_edges']:
                num_edge_types = 27
            if self.config['time_aware_edges'] and not self.config['type_aware_edges']:
                num_edge_types = 3
            if not self.config['time_aware_edges'] and self.config['type_aware_edges']:
                num_edge_types = 9
            if not self.config['time_aware_edges'] and not self.config['type_aware_edges']:
                num_edge_types = 1

            self.gats.append(MTGATConv(in_channels=self.config['graph_conv_in_dim'],
                                       num_node_types=3,
                                       num_edge_types=num_edge_types,
                                       out_channels=gat_out_channel,
                                       heads=self.config['gat_conv_num_heads'],
                                       dropout=self.config['gnn_dropout'],
                                       concat=concat))

            if l in range(0, num_gat_layers):
                # self.gats.append(
                #     TopKPooling(
                #         in_channels=self.config['graph_conv_in_dim'],
                #         min_score=self.config['graph_prune_min_score']
                #     )
                # )
                if self.config['prune_type'] == 'topk':
                    self.gats.append(
                        TopKEdgePooling(
                            min_score=None,
                            percentage=self.config['prune_keep_p']
                        )
                    )
                elif self.config['prune_type'] == 'random':
                    self.gats.append(
                        RandomEdgePooling(
                            percentage=self.config['prune_keep_p']
                        )
                    )
                else:
                    raise NotImplementedError

        # Vision models
        if use_pe:
            self.vision_pe = PositionalEncoding(d_model=self.config['graph_conv_in_dim'])

        if self.config['use_conv1d']:
            self.vision_conv = nn.Sequential(
                nn.Conv1d(in_channels=self.config['vision_dim'],
                          out_channels=self.config['graph_conv_in_dim'],
                          kernel_size=3),
                nn.ReLU())
            vision_fc_in_dim = self.config['graph_conv_in_dim']
        else:
            vision_fc_in_dim = self.config['vision_dim']

        if self.config['use_ffn']:
            self.vision_fc = nn.Sequential(
                nn.Linear(in_features=vision_fc_in_dim, out_features=self.config['graph_conv_in_dim']),
                nn.ReLU(),
                nn.Linear(in_features=self.config['graph_conv_in_dim'], out_features=self.config['graph_conv_in_dim']),
                nn.ReLU()
            )
        else:
            self.vision_fc = nn.Sequential(
                nn.Linear(in_features=vision_fc_in_dim, out_features=self.config['graph_conv_in_dim']),
            )

        # Text models
        if use_pe:
            self.text_pe = PositionalEncoding(d_model=self.config['graph_conv_in_dim'])
        if self.config['use_ffn']:
            self.text_fc = nn.Sequential(
                nn.Linear(in_features=self.config['text_dim'], out_features=self.config['graph_conv_in_dim']),
                nn.ReLU(),
                nn.Linear(in_features=self.config['graph_conv_in_dim'], out_features=self.config['graph_conv_in_dim']),
                nn.ReLU()
            )
        else:
            self.text_fc = nn.Sequential(
                nn.Linear(in_features=self.config['text_dim'], out_features=self.config['graph_conv_in_dim']),
            )

        # Audio models
        if use_pe:
            self.audio_pe = PositionalEncoding(d_model=self.config['graph_conv_in_dim'])
        if self.config['use_ffn']:
            self.audio_fc = nn.Sequential(
                nn.Linear(in_features=self.config['audio_dim'], out_features=self.config['graph_conv_in_dim']),
                nn.ReLU(),
                nn.Linear(in_features=self.config['graph_conv_in_dim'], out_features=self.config['graph_conv_in_dim']),
                nn.ReLU()
            )
        else:
            self.audio_fc = nn.Sequential(
                nn.Linear(in_features=self.config['audio_dim'], out_features=self.config['graph_conv_in_dim']),
            )

        if self.config['graph_activation']:
            if self.config['graph_activation'] == 'lrelu':
                self.activation = nn.LeakyReLU(negative_slope=0.1)
            elif self.config['graph_activation'] == 'gelu':
                self.activation = nn.GELU()

    def data_to_graph_nodes(self, **kwargs):
        """
        TODO: Make this graph node construction dynamic
        Extract features from raw input dataset and format them into graph node embeddings. This might involve bringing
        different feature dims into a same size.
        :param x: raw input dataset, List [object, person, text, audio, scene], Shapes:
                    object: batch_size, num_frames, N_objects, 3, crop_size (=224), crop_size (=224)
                    person: batch_size, num_frames, N_person, 3, crop_size (=224), crop_size (=224)
                    text: batch_size, num_frames, 1, text_dim (=512?)
                    ...
        :return:
            nodes: Tensor
        """
        vision = kwargs.pop("vision", None)
        text = kwargs.pop("text", None)
        audio = kwargs.pop("audio", None)

        # import ipdb
        # ipdb.set_trace()
        if vision is not None:
            if self.config['use_conv1d']:
                # right pad vision
                vision = torch.cat((vision, torch.zeros((vision.shape[0], 2, vision.shape[2])).to(device)), dim=1)
                vision = self.vision_conv(vision.permute(0, 2, 1).contiguous())
                vision = vision.permute(0, 2, 1).contiguous()
            vision = self.vision_fc(vision)
            if self.use_pe:
                vision = self.vision_pe(vision.permute(1, 0, 2).contiguous())
                vision = vision.permute(1, 0, 2).contiguous()

        if text is not None:
            text = self.text_fc(text)
            if self.use_pe:
                text = self.text_pe(text.permute(1, 0, 2).contiguous())
                text = text.permute(1, 0, 2).contiguous()

        if audio is not None:
            audio = self.audio_fc(audio)
            if self.use_pe:
                audio = self.audio_pe(audio.permute(1, 0, 2).contiguous())
                audio = audio.permute(1, 0, 2).contiguous()

        return vision, text, audio

    def sequential_process(self, **kwargs):
        """
        Use sequential model (transformer) to encode sequential dataset (scene, audio, etc.)
        :param x: raw input dataset, List [object, person, text, audio, scene], Shapes:
                    object: batch_size, num_frames, N_objects, 3, crop_size (=224), crop_size (=224)
                    person: batch_size, num_frames, N_person, 3, crop_size (=224), crop_size (=224)
                    text: batch_size, num_frames, 1, text_dim (=512?)
                    audio: batch_size, num_frames, 1, audio_dim (=512?)
                    scene: batch_size, num_frames, scene_dim (=512?)
                    ...
        :param kwargs:
        :return:
        """
        vision = kwargs.pop("vision", None)
        text = kwargs.pop("text", None)
        audio = kwargs.pop("audio", None)

        processed_feat_dict = {}
        if vision is not None:
            processed_feat_dict['vision'] = vision
        if text is not None:
            processed_feat_dict['text'] = text
        if audio is not None:
            processed_feat_dict['audio'] = audio

        return processed_feat_dict

    def forward(self, vision, text, audio, v_mask, t_mask, a_mask):
        vision, text, audio = self.data_to_graph_nodes(vision=vision, text=text, audio=audio)
        # batch_x, edge_index_list = self.construct_full_graph_dynamic(vision, text, audio, v_mask, t_mask, a_mask)
        batch_x, batch_x_type, edge_index_list, batch_edge_types = \
            construct_time_aware_dynamic_graph(vision, text, audio, v_mask, t_mask, a_mask,
                                               all_to_all=self.config['use_all_to_all'],
                                               time_aware=self.config['time_aware_edges'],
                                               type_aware=self.config['type_aware_edges'])
        assert vision.shape[0] == text.shape[0] == audio.shape[0], "Batch sizes must be the same!"
        batch_size = vision.shape[0]
        try:
            l = [gData(x=batch_x[i], edge_index=edge_index_list[i],
                       x_type=batch_x_type[i], edge_type=batch_edge_types[i]) for i in range(batch_size)]
        except:
            import ipdb
            ipdb.set_trace()
        batch = Batch.from_data_list(l)
        context_summ = batch.x
        if self.config['return_layer_outputs']:
            nodes_rec, edge_indices_rec, edge_weights_rec, edge_types_rec = \
                [context_summ], [batch.edge_index], [None], [batch.edge_type]
        for module in self.gats:
            if type(module) == MTGATConv:
                # # combine list of dict
                # edge_type_dict = {}
                # for my_dict in batch.edge_type_dict:
                #     edge_type_dict = {**edge_type_dict, **my_dict}
                try:
                    gat_output, (ei, e_weights) = module(context_summ, edge_index=batch.edge_index,
                                                         x_type=batch.x_type, edge_type=batch.edge_type,
                                                         return_attention_weights=True)
                    if self.config['use_residual']:
                        context_summ = context_summ + gat_output
                    else:
                        context_summ = gat_output
                except:
                    print(traceback.print_exc())
                    ipdb.set_trace()

                if self.config['graph_activation']:
                    context_summ = self.activation(context_summ)

            elif type(module) == TopKPooling:
                context_summ = module(context_summ, edge_index=batch.edge_index)
            elif type(module) == TopKEdgePooling:
                ei, e_weights, kept_index = module(ei, e_weights, return_kept_index=True)
                batch.edge_index = ei
                batch.edge_type = batch.edge_type[kept_index]
                if self.config['return_layer_outputs']:
                    nodes_rec.append(context_summ)
                    edge_indices_rec.append(ei)
                    edge_weights_rec.append(e_weights)
                    edge_types_rec.append(batch.edge_type)
            elif type(module) == RandomEdgePooling:
                ei, e_weights, kept_index = module(ei, e_weights, return_kept_index=True)
                batch.edge_index = ei
                batch.edge_type = batch.edge_type[kept_index]
                if self.config['return_layer_outputs']:
                    nodes_rec.append(context_summ)
                    edge_indices_rec.append(ei)
                    edge_weights_rec.append(e_weights)
                    edge_types_rec.append(batch.edge_type)
        shapes = [bx.shape[0] for bx in batch_x]
        if self.config['remove_isolated']:
            _, _, mask0 = pyg.utils.isolated.remove_isolated_nodes(batch.edge_index)
            mask = torch.zeros(batch.x.shape[0]).to(device) == 0
            mask[:mask0.shape[0]] = mask0
        else:
            mask = torch.ones(batch.x.shape[0]) == 1

        node_features = []
        offset = 0
        for i, s in enumerate(shapes):
            mask_s = mask[offset:offset + s]
            node_features.append(context_summ[offset:offset + s][mask_s])
            offset += s

        if self.config['return_layer_outputs']:
            return node_features, batch, nodes_rec, edge_indices_rec, edge_weights_rec, edge_types_rec
        return node_features


if __name__ == "__main__":
    from consts import GlobalConsts as gc
    import pickle

    dataset = pickle.load(open("/home/username/MTGAT/dataset/cmu_mosi/mosi_data_noalign.pkl", 'rb'))
    vision = torch.from_numpy(dataset['train']['vision'][100:102]).to(device).float()
    text = torch.from_numpy(dataset['train']['text'][100:102]).to(device).float()
    audio = torch.from_numpy(dataset['train']['audio'][100:102]).to(device).float()
    vision_mask = vision.sum(-1) != 0
    text_mask = text.sum(-1) != 0
    audio_mask = audio.sum(-1) != 0
    # import ipdb
    # ipdb.set_trace()
    # vision = torch.randn((50, 20)).to(device)
    # text = torch.randn((15, 300)).to(device)
    # audio = torch.randn((25, 5)).to(device)
    my_config = copy.deepcopy(gc.config)
    my_config['graph_conv_in_dim'] = 256
    my_config['graph_conv_out_dim'] = 256
    # TODO: when pruning, edge_type also needs to be updated
    my_config['prune_keep_p'] = 0.8
    my_config['use_ffn'] = 1
    my_config['remove_isolated'] = 1
    my_config['graph_activation'] = None
    gnn = DynamicMTGATPruneModel(config=my_config, concat=True, num_gat_layers=4, use_pe=True).to(device)
    # import ipdb
    # ipdb.set_trace()
    out = gnn(vision, text, audio, vision_mask, text_mask, audio_mask)
    import ipdb

    ipdb.set_trace()
