# noinspection PyTypeChecker
import torch
import operator
from functools import reduce

from torch import Tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# NOTE: This is deprecated


# noinspection PyTypeChecker
def build_temporal_clique(node_index_aggr, node_types_aggr, order=1):
    """
    Build the temporal fully-connected graph (clique)
    :param node_index_aggr: aggregation node as indices, Shape (num_frames, num_aggr_nodes_per_frame)
    :param node_types_aggr: aggregation node types as integers, Shape (num_frames, num_aggr_nodes_per_frame)
    :param order: the highest connectivity
    :return:
    """
    assert order > 0, "Expected order > 0 but got {}".format(order)
    temporal_edge_index = []
    temporal_edge_types = []
    # loop through each order level
    for odr in range(1, order + 1):
        # build clique at each level of order
        for start in range(odr):
            for frame in list(range(start, len(node_index_aggr), odr))[:-1]:
                node_index_curr = node_index_aggr[frame]  # node index in the current frame
                node_index_next = node_index_aggr[frame + odr]  # node index in the next frame

                node_types_curr = node_types_aggr[frame]  # node types in the current frame
                node_types_next = node_types_aggr[frame + odr]  # node types in the next frame

                # mesh the grids
                curr_inds, next_inds = torch.meshgrid(torch.arange(len(node_index_curr)),
                                                      torch.arange(len(node_index_next)))

                curr_index, next_index = node_index_curr[curr_inds.reshape(1, -1)], \
                                         node_index_next[next_inds.reshape(1, -1)]
                source_index, target_index = torch.cat((curr_index, next_index), dim=1), \
                                             torch.cat((next_index, curr_index), dim=1)

                curr_type, next_type = node_types_curr[curr_inds.reshape(1, -1)], \
                                       node_types_next[next_inds.reshape(1, -1)]
                source_type, target_type = torch.cat((curr_type, next_type), dim=1), \
                                           torch.cat((next_type, curr_type), dim=1)

                # build the edge index
                edge_index = torch.cat((source_index, target_index), dim=0)
                temporal_edge_index.append(edge_index)

                # build the edge types
                edge_types = torch.cat((source_type, target_type), dim=0)
                temporal_edge_types.append(edge_types)
    temporal_edge_index = torch.cat(temporal_edge_index, dim=1)
    temporal_edge_types = torch.cat(temporal_edge_types, dim=1)
    return temporal_edge_index, temporal_edge_types


# noinspection PyTypeChecker
def build_edges(node_index_aggr, node_types_aggr, order=2, num_aggr_nodes_per_frame=None):
    """
    Build the edge_index used for PyG GNN modules
    :param node_index_aggr: aggregation node as indices, Shape (num_frames, num_aggr_nodes_per_frame)
    :param node_types_aggr: aggregation node types as integers, Shape (num_frames, num_aggr_nodes_per_frame)
    :param order: order of connectivity in the temporal direction
    :param num_aggr_nodes_per_frame: number of aggregation nodes per frame, List
            e.g. [num_objects_aggr, num_person_aggr, num_text_aggr, num_audio_aggr, num_scene_aggr]
    :return:
        edge_index_aggr: aggregation edge index needed by PyG, Shape 2xE
        edge_types_aggr: aggregation edge types encoded as source_type -> target_type, Shape 2xE
    """

    #########################################################################
    # 1. build aggregation edges
    #########################################################################
    # 1.1. within frame edges
    frame_edge_index = []
    frame_edge_types = []
    for frame in range(len(node_index_aggr)):
        # 1.1.1. get the node index
        node_index_frame = node_index_aggr[frame]  # node index in the current frame
        source_inds, target_inds = torch.meshgrid(torch.arange(len(node_index_frame)),
                                                  torch.arange(len(node_index_frame)))
        source, target = node_index_frame[source_inds.reshape(1, -1)], node_index_frame[target_inds.reshape(1, -1)]
        # store the edge_index
        edge_index = torch.cat((source, target), dim=0)
        frame_edge_index.append(edge_index)

        # 1.1.2. get the node types
        node_types_aggr_frame = node_types_aggr[frame]  # node types in the current frame
        source_types, target_types = node_types_aggr_frame[source_inds.reshape(1, -1)], \
                                     node_types_aggr_frame[target_inds.reshape(1, -1)]
        edge_types = torch.cat((source_types, target_types), dim=0)
        frame_edge_types.append(edge_types)

    frame_edge_index = torch.cat(frame_edge_index, dim=1)
    frame_edge_types = torch.cat(frame_edge_types, dim=1)

    # 1.2. temporal edges
    temporal_edge_index, temporal_edge_types = build_temporal_clique(node_index_aggr, node_types_aggr, order=order)

    # 1.3. merge them
    edge_index_aggr = torch.cat((frame_edge_index, temporal_edge_index), dim=1)
    edge_types_aggr = torch.cat((frame_edge_types, temporal_edge_types), dim=1)

    return edge_index_aggr, edge_types_aggr

def build_node_index(num_aggr_nodes_per_frame, num_frames):
    """
    Mark each node in the graph with an index
    :param num_aggr_nodes_per_frame: number of aggregation nodes per frame, List
            e.g. [num_objects_aggr, num_person_aggr, num_text_aggr, num_audio_aggr, num_scene_aggr]
    :param num_frames: total number of frames
    :return:
        node_index_aggr: aggregation node as indices, Shape (num_frames, num_aggr_nodes_per_frame)
            e.g. [[object1_aggr, object2_aggr, ..., objectN_aggr, person1_aggr, ...],
                  [...]]
        node_types_aggr: aggregation node types as integers, Shape (num_frames, num_aggr_nodes_per_frame)
            e.g. [[0, 0, ..., 0, 1, ...],
                  [...]]
    """

    # build aggregation node types. TODO: now this is assuming graph structure is the same across time
    node_types_aggr_frame = reduce(operator.add, [[i] * n for i, n in enumerate(num_aggr_nodes_per_frame)])
    node_types_aggr = torch.tensor([node_types_aggr_frame] * num_frames).to(device)

    # build aggregation nodes
    total_aggr_nodes_per_frame = sum(num_aggr_nodes_per_frame)
    node_index_aggr = torch.zeros(num_frames, total_aggr_nodes_per_frame)
    for frame in range(num_frames):
        # build the node index
        offset = frame * total_aggr_nodes_per_frame
        node_index_aggr_frame = (torch.arange(0, total_aggr_nodes_per_frame) + offset).reshape(1, -1)
        node_index_aggr[frame] = node_index_aggr_frame

    num_aggr_nodes = num_frames * total_aggr_nodes_per_frame

    return node_index_aggr, node_types_aggr


def construct_edge_type_dict(edge_index: Tensor, edge_type: Tensor):
    assert edge_type.dim() == 1 and edge_index.shape[1] == edge_type.shape[0]
    edge_type_dict = {}
    for edge, this_type in zip(edge_index.split(1, dim=1), edge_type):
        edge_tuple = tuple(edge.T[0].tolist())
        edge_type_dict[edge_tuple] = int(this_type)

    return edge_type_dict

def lookup_edge_type_based_on_edge_index(edge_type_dict: dict, edge_index: Tensor):
    list_edge_type = []
    for edge in edge_index.split(1, dim=1):
        edge_tuple = tuple(edge.T[0].tolist())
        if edge_tuple not in edge_type_dict:
            import ipdb
            ipdb.set_trace()
        list_edge_type.append(edge_type_dict[edge_tuple])

    return torch.tensor(list_edge_type, dtype=edge_index.dtype, device=edge_index.device)
