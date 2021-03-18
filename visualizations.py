import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm


def visualize_graph(nodes_1, nodes_2, edge_index, edge_weights, edge_types, nodes_1_color="#EC7063",
                    nodes_2_color="#AF7AC5"):
    G = nx.Graph()
    for (u, v), w, t in zip(edge_index.T, edge_weights, edge_types):
        G.add_edge(u, v, weight=w, type=t)
    pos = nx.bipartite_layout(G, nodes_1, align='horizontal', scale=0.3)  # positions for all nodes

    # TODO: change the way past, current, future are defined.
    past = [(u, v) for (u, v, d) in G.edges(data=True) if d['type'] == 0]
    past_weights = np.array([d['weight'] for (u, v, d) in G.edges(data=True) if d['type'] == 0])
    past_weights /= np.max(past_weights)
    past_weights += 1

    current = [(u, v) for (u, v, d) in G.edges(data=True) if d['type'] == 1]
    current_weights = np.array([d['weight'] for (u, v, d) in G.edges(data=True) if d['type'] == 1])
    current_weights /= np.max(current_weights)
    current_weights += 1

    future = [(u, v) for (u, v, d) in G.edges(data=True) if d['type'] == 2]
    future_weights = np.array([d['weight'] for (u, v, d) in G.edges(data=True) if d['type'] == 2])
    future_weights /= np.max(future_weights)
    future_weights += 1

    nx.draw_networkx_nodes(
        G, pos, nodelist=np.concatenate((nodes_1, nodes_2)).astype(int), node_size=400,
        cmap=cm.get_cmap("summer"),
        node_color=[nodes_1_color] * len(nodes_1) + [nodes_2_color] * len(nodes_2)
    )

    for e, ew in zip(past, past_weights):
        nx.draw_networkx_edges(G, pos, edgelist=[e], width=ew ** 2.5, alpha=(ew - 1) / 2,
                               edge_cmap=cm.get_cmap('RdYlBu'), edge_color=[ew])

    for e, ew in zip(current, current_weights):
        nx.draw_networkx_edges(G, pos, edgelist=[e], width=ew ** 2.5, alpha=(ew - 1) / 2,
                               edge_cmap=cm.get_cmap('coolwarm'), edge_color=[ew])

    for e, ew in zip(future, future_weights):
        nx.draw_networkx_edges(G, pos, edgelist=[e], width=ew ** 2.5, alpha=(ew - 1) / 2,
                               edge_cmap=cm.get_cmap('summer'), edge_color=[ew])


def visualize_graph_dynamic(nodes_1, nodes_2, edge_index, edge_weights, edge_types,
                            nodes_1_color="#d7cee6", nodes_2_color="#becb95",
                            nodes_1_border="#7355a2", nodes_2_border="#7d9263",
                            linewidths=1.0, aspect_ratio=4/3, scale=1, threshold=None, node_size=400):
    G = nx.Graph()

    for (u, v), w, t in zip(edge_index.T, edge_weights, edge_types):
        G.add_edge(u, v, weight=w, type=t)
    pos = nx.bipartite_layout(G, nodes_1, align='horizontal', scale=scale, aspect_ratio=aspect_ratio)  # positions for all nodes

    cmap_list = ['coolwarm', 'summer', 'RdYlBu', ]

    nx.draw_networkx_nodes(
        G, pos, nodelist=np.concatenate((nodes_1, nodes_2)).astype(int), node_size=node_size,
        cmap=cm.get_cmap("summer"),
        node_color=[nodes_1_color] * len(nodes_1) + [nodes_2_color] * len(nodes_2),
        edgecolors=[nodes_1_border] * len(nodes_1) + [nodes_2_border] * len(nodes_2),
        linewidths=linewidths
    )

    unique_edges, unique_inds = np.unique(edge_types, return_index=True)
    edges = np.arange(len(unique_edges))
    type_dict = {ue: e for ue, e in zip(unique_edges, edges)}
    for i in range(len(edges)):
        if threshold:
            past = [(u, v) for (u, v, d) in G.edges(data=True) if type_dict[d['type']] == i and
                    d['weight'] >= threshold]
            past_weights = np.array([d['weight'] for (u, v, d) in G.edges(data=True) if type_dict[d['type']] == i and
                                     d['weight'] >= threshold])
        else:
            past = [(u, v) for (u, v, d) in G.edges(data=True) if type_dict[d['type']] == i]
            past_weights = np.array([d['weight'] for (u, v, d) in G.edges(data=True) if type_dict[d['type']] == i])
        past_weights /= np.max(past_weights)
        past_weights += 1

        if len(past) == 0 or len(past_weights) == 0:
            continue
        for e, ew in zip(past, past_weights):
            nx.draw_networkx_edges(G, pos, edgelist=[e], width=ew ** 2.5, alpha=(ew - 1) / 1.2,
                                   edge_cmap=cm.get_cmap(cmap_list[i % 3]), edge_color=[ew])


def visualize_graph_circular(nodes_1, nodes_2, nodes_3, edge_index, edge_weights, edge_types, 
                 nodes_1_color="#d7cee6", nodes_2_color="#becb95", nodes_3_color="#b3c5da",
                 nodes_1_border="#7355a2", nodes_2_border="#7d9263", nodes_3_border="#4e75a3",
                 linewidths=1.0, threshold=None, node_size=400, cmap='coolwarm'
                ):
    from collections import OrderedDict
    G = nx.Graph()
    
    # Align and draw the nodes first
    for node in np.concatenate((nodes_1, nodes_2, nodes_3)):
        G.add_node(node)

    pos = nx.circular_layout(G, scale=1)  # positions for all nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=np.concatenate((nodes_1, nodes_2, nodes_3)).astype(int), node_size=node_size,
        cmap=cm.get_cmap("summer"),
        node_color=[nodes_1_color] * len(nodes_1) + [nodes_2_color] * len(nodes_2) + [nodes_3_color] * len(nodes_3),
        edgecolors=[nodes_1_border] * len(nodes_1) + [nodes_2_border] * len(nodes_2) + [nodes_3_border] * len(nodes_3),
        linewidths=linewidths, arrowsize=100000000
        
    )
    for (u, v), w, t in zip(edge_index.T, edge_weights, edge_types):
        G.add_edge(u, v, weight=w, type=t)
        
    unique_edges, unique_inds = np.unique(edge_types, return_index=True)
    zero_indexed_edge_types = np.arange(len(unique_edges))
    type_dict = {ue: e for ue, e in zip(unique_edges, zero_indexed_edge_types)}
    for i in range(len(zero_indexed_edge_types)):
        if threshold:
            past = [(u, v) for (u, v, d) in G.edges(data=True) if type_dict[d['type']] == i and
                    d['weight'] >= threshold]
            past_weights = np.array([d['weight'] for (u, v, d) in G.edges(data=True) if type_dict[d['type']] == i and
                                     d['weight'] >= threshold])
        else:
            past = [(u, v) for (u, v, d) in G.edges(data=True) if type_dict[d['type']] == i]
            past_weights = np.array([d['weight'] for (u, v, d) in G.edges(data=True) if type_dict[d['type']] == i])
        if len(past) == 0 or len(past_weights) == 0:
            continue
            
        past_weights /= np.max(past_weights)
        past_weights += 1
        
        for e, ew in zip(past, past_weights):
            nx.draw_networkx_edges(G, pos, edgelist=[e], width=ew**2.5, alpha=(ew-1)/1.2, edge_cmap=cm.get_cmap(cmap), edge_color=[ew])