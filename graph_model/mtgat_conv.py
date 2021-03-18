from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor
# from torch_geometric.nn.conv import MessagePassing
from graph_model.message_passing import MessagePassing
from torch_geometric.utils import softmax

from torch_geometric.nn.inits import glorot, zeros


class MTGATConv(MessagePassing):
    r""" Time-aware Multimodal Graph Attention

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, num_node_types: int, num_edge_types: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = False, bias: bool = True, **kwargs):
        super(MTGATConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops  # Setting to True is currently not supported

        if isinstance(in_channels, int):
            self.lin_l = Parameter(torch.Tensor(num_node_types, in_channels, heads * out_channels))
            self.lin_r = self.lin_l
        else:
            raise NotImplementedError
            # self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            # self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, num_edge_types, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, num_edge_types, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l)
        glorot(self.lin_r)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                x_type: Tensor, edge_type: Tensor,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Tensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""

        Args:
            x_type (Tensor): shape (num_nodes, ). Should be an integer (short/long) tensor.
            edge_type (Tensor): shape (num_edges, ). Should be an integer (short/long) tensor.
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        assert x_type.dtype in [torch.int8, torch.int16, torch.int32, torch.int64] , f'### x_type.dtype: {x_type.dtype}'
        assert edge_type.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]
        assert x_type.dim() == 1 and x_type.shape[0] == x.shape[0]
        assert edge_type.dim() == 1 and edge_type.shape[0] == edge_index.shape[1]

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `MTGATConv`.'
            node_type_specific_lin_l = self.lin_l[x_type]  # shape (num_nodes, in_channels, heads * out_channels)

            x_l = torch.einsum('ijk,ij->ik', node_type_specific_lin_l, x)  # output shape (num_nodes, heads * out_channels)
            x_r = x_l = x_l.view(-1, 1, H, C)  # output shape (num_nodes, 1, heads, out_channels) the 1 in dim1 is used for broadcasting

            # attn has shape (1, num_edge_types, heads, out_channels)
            alpha_l = (x_l * self.att_l).sum(dim=-1)  # output shape (num_nodes, num_edge_types, heads)
            alpha_r = (x_l * self.att_r).sum(dim=-1)  # output shape (num_nodes, num_edge_types, heads)
        else:
            raise NotImplementedError
            # x_l, x_r = x[0], x[1]
            # assert x[0].dim() == 2, 'Static graphs not supported in `TMGATConv`.'
            # x_l = self.lin_l(x_l).view(-1, H, C)
            # alpha_l = (x_l * self.att_l).sum(dim=-1)
            # if x_r is not None:
            #     x_r = self.lin_r(x_r).view(-1, H, C)
            #     alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None and x_l is not None
        assert alpha_l is not None and alpha_r is not None

        if self.add_self_loops:
            raise NotImplementedError('Self-loop is currenly not supported. Note that the graph passed in already contains self-loops.')
            # if isinstance(edge_index, Tensor):
            #     num_nodes = x_l.size(0)
            #     num_nodes = size[1] if size is not None else num_nodes
            #     num_nodes = x_r.size(0) if x_r is not None else num_nodes
            #     edge_index, _ = remove_self_loops(edge_index)
            #     edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            # elif isinstance(edge_index, SparseTensor):
            #     edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        # edge_type = lookup_edge_type_based_on_edge_index(edge_type_dict, edge_index)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r),
                             edge_type=edge_type,
                             size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                edge_type: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        num_edges, num_edge_types, num_heads = alpha.shape
        assert num_edges == edge_type.shape[0]

        alpha = alpha.reshape(-1, num_heads).index_select(dim=0, index=(torch.arange(0, num_edges).to(alpha.device) * num_edge_types + edge_type))

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        x_j = x_j.squeeze()  # x_j was (num_edges, 1, num_heads, out_channels/num_heads),
                             # alpha is (num_edges, num_heads)
                             # therefore, we squeeze x_j's second dim, and expand alpha's last dim.
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
