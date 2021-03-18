import math

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def construct_full_graph_dynamic(vision, text, audio, v_msk, t_msk, a_msk):
    batch_x, edge_index_list = [], []
    for v, t, a, vm, tm, am in zip(vision, text, audio, v_msk, t_msk, a_msk):
        if torch.sum(vm) != 0:
            vision_node_index = torch.arange(v[vm].shape[0])
        else:
            vision_node_index = torch.tensor([]).long()

        if torch.sum(tm) != 0:
            text_node_index = torch.arange(t[tm].shape[0]) + len(vision_node_index)
        else:
            text_node_index = torch.tensor([]).long()

        if torch.sum(am) != 0:
            audio_node_index = torch.arange(a[am].shape[0]) + len(text_node_index) + len(vision_node_index)
        else:
            audio_node_index = torch.tensor([]).long()

        node_index = torch.cat((vision_node_index, text_node_index, audio_node_index))
        # Construct a full graph where each node is connected to every other node
        src, dst = torch.meshgrid(node_index, node_index)
        src, dst = src.flatten(), dst.flatten()
        edge_index = torch.stack((src, dst), 0)
        edge_index_list.append(edge_index.to(device))
        # Construct the node features
        node_features = torch.cat((v[vm], t[tm], a[am]), 0)
        batch_x.append(node_features)
        assert node_features.shape[0] == max(node_index) + 1
    return batch_x, edge_index_list


def construct_time_aware_dynamic_graph(vision, text, audio, v_msk, t_msk, a_msk, window_factor=2, all_to_all=True,
                                       time_aware=True, type_aware=True):
    batch_x, batch_x_type, batch_edge_index, batch_edge_types = [], [], [], []
    for v, t, a, vm, tm, am in zip(vision, text, audio, v_msk, t_msk, a_msk):
        if torch.sum(vm) != 0:
            vision_node_index = torch.arange(v[vm].shape[0])
        else:
            vision_node_index = torch.tensor([]).long()

        if torch.sum(tm) != 0:
            text_node_index = torch.arange(t[tm].shape[0]) + len(vision_node_index)
        else:
            text_node_index = torch.tensor([]).long()

        if torch.sum(am) != 0:
            audio_node_index = torch.arange(a[am].shape[0]) + len(text_node_index) + len(vision_node_index)
        else:
            audio_node_index = torch.tensor([]).long()

        if torch.sum(vm) == 0 and torch.sum(tm) == 0 and torch.sum(am) == 0:
            continue

        # Constructing node features and types
        node_features = torch.cat((v[vm], t[tm], a[am]), 0)
        node_types = torch.cat((torch.zeros(len(v[vm])), torch.zeros(len(t[tm])) + 1, torch.zeros(len(a[am])) + 2))
        batch_x.append(node_features.to(device))
        batch_x_type.append(node_types.long().to(device))

        # Constructing time aware dynamic graph edge index and types
        edge_index_list, edge_types = [], []
        edge_type_offset = 0
        # build uni-modal
        build_time_aware_dynamic_graph_uni_modal(vision_node_index, edge_index_list, edge_types, edge_type_offset,
                                                 all_to_all=all_to_all, time_aware=time_aware, type_aware=type_aware)
        if time_aware and type_aware:
            edge_type_offset += 3
        if not time_aware and type_aware:
            edge_type_offset += 1

        # text
        build_time_aware_dynamic_graph_uni_modal(text_node_index, edge_index_list, edge_types, edge_type_offset,
                                                 all_to_all=all_to_all, time_aware=time_aware, type_aware=type_aware)
        if time_aware and type_aware:
            edge_type_offset += 3
        if not time_aware and type_aware:
            edge_type_offset += 1

        # audio
        build_time_aware_dynamic_graph_uni_modal(audio_node_index, edge_index_list, edge_types, edge_type_offset,
                                                 all_to_all=all_to_all, time_aware=time_aware, type_aware=type_aware)

        if time_aware and type_aware:
            edge_type_offset += 3
        if not time_aware and type_aware:
            edge_type_offset += 1

        # build cross-modal
        # vision - text
        build_time_aware_dynamic_graph_cross_modal(vision_node_index, text_node_index, edge_index_list, edge_types,
                                                   edge_type_offset, time_aware=time_aware, type_aware=type_aware)
        if time_aware and type_aware:
            edge_type_offset += 6
        if not time_aware and type_aware:
            edge_type_offset += 2

        # vision - audio
        build_time_aware_dynamic_graph_cross_modal(vision_node_index, audio_node_index, edge_index_list, edge_types,
                                                   edge_type_offset, time_aware=time_aware, type_aware=type_aware)
        if time_aware and type_aware:
            edge_type_offset += 6
        if not time_aware and type_aware:
            edge_type_offset += 2

        # text - audio
        build_time_aware_dynamic_graph_cross_modal(text_node_index, audio_node_index, edge_index_list, edge_types,
                                                   edge_type_offset, time_aware=time_aware, type_aware=type_aware)


        try:
            edge_index = torch.cat(edge_index_list, 1)
        except:
            import ipdb
            ipdb.set_trace()

        edge_types = torch.cat(edge_types)
        batch_edge_index.append(edge_index.to(device))
        batch_edge_types.append(edge_types.long().to(device))
    return batch_x, batch_x_type, batch_edge_index, batch_edge_types


def empty_tensor_list(num=1):
    return [torch.empty(2, 0) for _ in range(num)]


def build_time_aware_dynamic_graph_uni_modal(seq, edge_index_list, edge_types, edge_type_offset, all_to_all=True,
                                             time_aware=True, type_aware=True):
    if len(seq) > 0:
        current = torch.stack((seq, seq))
        if len(seq) > 1:
            if all_to_all:
                f_src, f_tgt = seq[:-1], seq[1:]
                f_src, f_tgt = torch.meshgrid(f_src, f_tgt)
                future = torch.stack((f_src.flatten(), f_tgt.flatten()))
                future_mask = (future[1, :] - future[0, :]) > 0
                future = future[:, future_mask]

                p_src, p_tgt = seq[1:], seq[:-1]
                p_src, p_tgt = torch.meshgrid(p_src, p_tgt)
                past = torch.stack((p_src.flatten(), p_tgt.flatten()))
                past_mask = (past[1, :] - past[0, :]) < 0
                past = past[:, past_mask]
            else:
                future = torch.stack((seq[:-1], seq[1:]))
                past = torch.stack((seq[1:], seq[:-1]))

        else:
            future = torch.empty(2, 0)
            past = torch.empty(2, 0)
    else:
        current, past, future = empty_tensor_list(3)
    for i, ei in enumerate([current, past, future]):
        if ei.shape[1] != 0:
            edge_index_list.append(ei)
            if time_aware and type_aware:
                edge_types.append(i + edge_type_offset + torch.zeros(ei.shape[1]))
            if time_aware and not type_aware:
                edge_types.append(i + torch.zeros(ei.shape[1]))
            if not time_aware and type_aware:
                edge_types.append(edge_type_offset + torch.zeros(ei.shape[1]))
            if not time_aware and not type_aware:
                edge_types.append(torch.zeros(ei.shape[1]))


def build_time_aware_dynamic_graph_cross_modal(seq1, seq2, edge_index_list, edge_types, edge_type_offset,
                                               time_aware=True, type_aware=True):
    if len(seq1) > 0 and len(seq2) > 0:
        longer = seq1 if len(seq1) > len(seq2) else seq2
        shorter = seq1 if len(seq1) <= len(seq2) else seq2
        # ei_current, ei_current_reverse, ei_past, ei_past_reverse, ei_future, ei_future_reverse is, respectively:
        # current seq1 -> seq2
        # current seq2 -> seq1
        # past seq1 -> seq2
        # future seq2 -> seq1
        # future seq1 -> seq2
        # past seq2 -> seq1
        ei_current, ei_current_reverse, ei_past, ei_past_reverse, ei_future, ei_future_reverse = \
            pseudo_align_cross_modal(shorter, longer)
    else:
        ei_current, ei_current_reverse, ei_past, ei_past_reverse, ei_future, ei_future_reverse = \
            empty_tensor_list(6)

    for i, ei in enumerate([ei_current, ei_past, ei_future, ei_current_reverse, ei_future_reverse, ei_past_reverse]):
        # the edges are iterated in an order of
        # [outward, outward, outward, inward, inward, inward]
        # and at the same time
        # [current, past, future, current, past, future]
        if ei.shape[1] != 0:
            edge_index_list.append(ei)
            if time_aware and type_aware:
                edge_types.append(i + edge_type_offset + torch.zeros(ei.shape[1]))
            if time_aware and not type_aware:
                edge_types.append(i % 3 + torch.zeros(ei.shape[1]))
            if not time_aware and type_aware:
                edge_types.append(i // 3 + edge_type_offset + torch.zeros(ei.shape[1]))
            if not time_aware and not type_aware:
                edge_types.append(torch.zeros(ei.shape[1]))


def pseudo_align_cross_modal(seq, longest, window_factor=2):
    N, M = len(seq), len(longest)
    assert N <= M
    if M == 1:
        inds1 = torch.arange(1).reshape(1, 1)

    else:
        # build the pseudo-alignment
        if M // N == 1:
            double_nodes = M % N
            if N == M:
                double_nodes = 1
            # the first double_nodes nodes in seq will have a window size of 2, stride of 2
            stride = 2
            window = 2
            inds1 = torch.arange(window).repeat(double_nodes, 1)
            offset = torch.arange(double_nodes).reshape(-1, 1).repeat(1, window) * stride
            inds1 = inds1 + offset

            # the rest double nodes will have a window size of 2 and stride of 1
            stride = 1
            inds2 = torch.arange(window).repeat(N - double_nodes, 1)
            offset = torch.arange(N - double_nodes).reshape(-1, 1).repeat(1, window) * stride
            inds2 = inds2 + offset + torch.max(inds1)

            if N == M:
                inds1 = torch.cat((inds1, inds2[:-1]), 0)
                inds1 = torch.cat((inds1, inds1[-1].reshape(1, -1)), 0)
            else:
                inds1 = torch.cat((inds1, inds2))
        else:
            # To calculate the smallest window size and the corresponding stride, we have the following equation
            # (M - W) / S + 1 = N, where W is the window size, S is the stride size
            # so, W = M - (N - 1) * S
            # therefore, the corresponding S = M // (N - 1), and then we can compute the W
            if N > 1:
                stride = M // (N - 1)
                stride = math.floor((stride + 1) / window_factor)
                stride = max(stride, 2)
                window = M - (N - 1) * stride
                if window < 2:
                    window = 2
                    stride = (M - window) // (N - 1)
            else:
                stride = M
                window = M

            inds1 = torch.arange(window).repeat(len(seq), 1)
            offset = torch.arange(len(seq)).reshape(-1, 1).repeat(1, window) * stride
            inds1 = inds1 + offset

    nodes_within_view = longest[inds1]

    # build the edge_index for the current cross modality clique
    seq_t = seq.reshape(-1, 1).repeat(1, nodes_within_view.shape[1]).flatten()
    nodes_within_view_f = nodes_within_view.flatten()

    # bi-directional edges
    ei_current = torch.stack((seq_t, nodes_within_view_f))
    ei_current_reverse = torch.stack((nodes_within_view_f, seq_t))

    all_inds = torch.arange(M).repeat(len(seq), 1)

    # build the FUTURE edges for the current cross modality clique (from seq to longest),
    # the reverse order of which is the PAST edges from longest to seq
    ei_future = []
    future_cutoff, _ = torch.max(inds1, dim=1)
    future_node_mask = all_inds > future_cutoff.reshape(-1, 1)
    for node, f_mask in zip(seq, future_node_mask):
        tgt = longest[f_mask]
        src = node.repeat(len(tgt))
        ei_future_i = torch.stack((src, tgt))
        ei_future.append(ei_future_i)
    ei_future = torch.cat(ei_future, dim=-1)
    ei_future_reverse = torch.roll(ei_future, shifts=[1], dims=[0])
    # future_xm_edges.append(ei_future)
    # past_xm_edges.append(ei_future_reverse)

    # build the PAST edges for the current cross modality clique (from seq to longest),
    # the reverse order of which is the FUTURE edges from longest to seq
    ei_past = []
    past_cutoff, _ = torch.min(inds1, dim=1)
    past_node_mask = all_inds < past_cutoff.reshape(-1, 1)
    for node, p_mask in zip(seq, past_node_mask):
        tgt = longest[p_mask]
        src = node.repeat(len(tgt))
        ei_past_i = torch.stack((src, tgt))
        ei_past.append(ei_past_i)
    ei_past = torch.cat(ei_past, dim=-1)
    ei_past_reverse = torch.roll(ei_past, shifts=[1], dims=[0])
    # edge_types = ("O", "I", "O", "I", "O","I")
    # edge_times = ("C", "Cr", "P", "F", "F", "P")
    return ei_current, ei_current_reverse, ei_past, ei_past_reverse, ei_future, ei_future_reverse


if __name__ == "__main__":
    from visualizations import visualize_graph
    for i in range(1, 3):
        vision = torch.zeros(1, 4, 128)
        test = torch.ones(1, i, 128)
        audio = torch.ones(1, 2, 128)
        vision_mask = torch.ones(1, 4).long() == 1
        test_mask = torch.ones(1, i).long() == 1
        audio_mask = torch.ones(1, 2).long() == 1
        construct_time_aware_dynamic_graph(vision, test, audio, vision_mask, test_mask, audio_mask,
                                           time_aware=True, type_aware=True)

