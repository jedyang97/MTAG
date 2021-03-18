![](overview.png)

MTAG (Modal-Temporal Attention Graph) is a GNN-based machine learning framework that can learn fusion and alignment for unaligned multimodal sequences.

Our code is written as an extension to the awesome [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) library. Users are encouraged to read their [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) and documentations to understand the basics.

Our main contributions include:
 - A [graph builder](https://github.com/jedyang97/MTAG/blob/main/graph_model/graph_builder.py) to construct graphs with modal and temporal edges.
 - A new GNN convolution operation called [MTGATConv](https://github.com/jedyang97/MTAG/blob/main/graph_model/mtgat_conv.py) that uses distinct attentions for edges with distinct modality and temporal ordering. It also transforms each node based on its modality type. It is like a combination of [RGCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.RGCNConv) and [GATConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv) with an efficient implementation. We hope this operation can be inlcuded into [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) as a standard operation.
 - A [TopK pooling](https://github.com/jedyang97/MTAG/blob/main/graph_model/pooling.py) operation to prune edges with low attention weights.
## Installation

Please refer to the `requirement.txt` for setup.

## Dataset Preperation
Download the [MOSI](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/cmu-mosei/seq_length_50/mosei_senti_data_noalign.pkl) and [IEMOCAP](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/iemocap/seq_length_50/iemocap_data_noalign.pkl) unaligned sequence data and put them into a desired folder (.e.g. ```<dataroot>```). Then specify in ```run.sh``` the folder containing the data of the desired dataset. For example:


```
python main.py \
...
--dataroot <dataroot>
...
```    

## Running Example

```
bash run.sh
```

To visualize the edges:
```
jupyter notebook construct_edge_type_dict
```

## Citation

```
@article{Yang2020MTAG,
  title={MTGAT: Multimodal Temporal Graph Attention Networks for Unaligned Human Multimodal Language Sequences},
  author={Jianing Yang and Yongxin Wang and Ruitao Yi and Yuying Zhu and Azaan Rehman and Amir Zadeh and Soujanya Poria and Louis-Philippe Morency},
  journal={ArXiv},
  year={2020},
  volume={abs/2010.11985}
}
```
