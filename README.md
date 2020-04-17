# Reading List
[The long list from MIT](https://people.csail.mit.edu/jshun/graph.shtml)

[Other List](https://github.com/Qingfeng-Yao/Readinglist)

[GNN papers](https://github.com/thunlp/GNNPapers)

[NN-on-Silicon](https://github.com/fengbintu/Neural-Networks-on-Silicon)

## Papers on Graph Mining

[Scaling Up Subgraph Query Processing with Efficient Subgraph Matching](mining/ICDE19-ScalingUpSubgraphQueryProcessing.pdf) ICDE'19

[Efficient Parallel Subgraph Enumeration on a Single Machine](mining/ICDE19-LIGHT.pdf) ICDE'19

[Fast and Robust Distributed Subgraph Enumeration](mining/VLDB19-FastRobustDistributedSubgraphEnumeration.pdf) VLDB'19

[Peregrine: A Pattern-Aware Graph Mining System](mining/Eurosys20-Peregrine.pdf) Eurosys'20

[EmptyHeaded: A Relational Engine for Graph Processing](mining/EmptyHeaded.pdf) SIGMOD'16

[The Power of Pivoting for Exact Clique Counting](mining/Pivoter.pdf) WSDM'20

## Papers on Graph Learning
[Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs](learning/DGL.pdf) ICLR'19 [DGL website](https://www.dgl.ai/)

[Fast Graph Representation Learning with PyTorch Geometric](learning/PyG.pdf) ICLR'19 [PyG website](https://github.com/rusty1s/pytorch_geometric)

[Improving the Accuracy, Scalability, and Performance of Graph Neural Networks with Roc](learning/Roc.pdf) MLSys'20 [Roc website](https://github.com/flexflow/FlexFlow)

[NeuGraph: Parallel Deep Neural Network Computation on Large Graphs](learning/NeuGraph.pdf) USENIX ATC'19 [NeuGraph website](https://www.microsoft.com/en-us/research/publication/neugraph-parallel-deep-neural-network-computation-on-large-graphs/)

[Semi-Supervised Classification with Graph Convolutional Networks](learning/GCN.pdf) ICLR'17 [GCN website](https://github.com/tkipf/gcn)

[Hierarchical Graph Representation Learning with Differentiable Pooling](learning/diffpool.pdf) NeurIPS'18 [diffpool website](https://github.com/RexYing/diffpool)

[Inductive Representation Learning on Large Graphs](learning/GraphSAGE.pdf) NIPS'17 [GraphSAGE website](http://snap.stanford.edu/graphsage/)

[Stochastic Training of Graph Convolutional Networks with Variance Reduction](http://proceedings.mlr.press/v80/chen18p/chen18p.pdf) ICML'18 [S-GCN](https://github.com/thu-ml/stochastic_gcn)

[Adaptive Sampling Towards Fast Graph Representation Learning](https://papers.nips.cc/paper/7707-adaptive-sampling-towards-fast-graph-representation-learning.pdf) NIPS'18 [AS-GCN website](https://github.com/huangwb/AS-GCN)

[Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/pdf/1905.07953.pdf) KDD'19 [ClusterGCN website](https://github.com/google-research/google-research/tree/master/cluster_gcn)

[FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling](https://openreview.net/pdf?id=rytstxWAW) ICLR'18 [FastGCN website](https://github.com/matenure/FastGCN)

[Large-Scale Learnable Graph Convolutional Networks](https://arxiv.org/pdf/1808.03965.pdf) KDD'18 [LGCN website](https://github.com/HongyangGao/LGCN)

[DeepWalk: Online Learning of Social Representations](learning/DeepWalk.pdf) KDD'14

[node2vec: Scalable Feature Learning for Networks](learning/node2vec.pdf) KDD'16

[GraphVite: A High-Performance CPU-GPU Hybrid System for Node Embedding](https://arxiv.org/pdf/1903.00757.pdf) WWW'19 [GraphVite website](https://graphvite.io/)

## Papers on Hardware Acceleration
[The TrieJax Architecture: Accelerating Graph Operations Through Relational Joins](https://arxiv.org/pdf/1905.08021.pdf) ASPLOS'20

[HyGCN: A GCN Accelerator with Hybrid Architecture](https://arxiv.org/pdf/2001.02514.pdf) HPCA'20

[Exploiting Locality in Graph Analytics through Hardware-Accelerated Traversal Scheduling](http://people.csail.mit.edu/sanchez/papers/2018.hats.micro.pdf) MICRO'18

[UWB-GCN: Accelerating Graph Convolutional Networks through Runtime Workload Rebalancing](https://arxiv.org/pdf/1908.10834.pdf)

[Graphicionado: A High-Performance and Energy Efficient Accelerator for Graph Analytics](https://mrmgroup.cs.princeton.edu/papers/taejun_micro16.pdf) MICRO'16

[GraphR: Accelerating Graph Processing Using ReRAM](https://arxiv.org/pdf/1708.06248.pdf) HPCA'18

[GraphQ: Scalable PIM-Based Graph Processing](http://alchem.usc.edu/portal/static/download/graphq.pdf) MICRO'19

[GraphP: Reducing Communication of PIM-based Graph Processing with Efficient Data Partition](http://csl.stanford.edu/~christos/publications/2018.graphp.hpca.pdf) HPCA'18

[GraphSAR: A Sparsity-Aware Processing-in-Memory Architecture for Large-Scale Graph Processing on ReRAMs](https://dl.acm.org/doi/pdf/10.1145/3287624.3287637) ASPDAC'19

[Alleviating Irregularity in Graph Analytics Acceleration: a Hardware/Software Co-Design Approach](https://web.ece.ucsb.edu/~iakgun/files/MICRO2019.pdf) MICRO'19

[GraphABCD: Scaling Out Graph Analytics with Asynchronous Block Coordinate Descent]() ISCA'20

[GaaS-X: Graph Analytics Accelerator Supporting Sparse Data Representation Using Crossbar Architectures]() ISCA'20

[POSTER: Domain-Specialized Cache Management for Graph Analytics](http://www.faldupriyank.com/papers/GRASP_PACT19.pdf) PACT'19 [code](https://github.com/ease-lab/grasp)

[Domain-Specialized Cache Management for Graph Analytics](https://www.research.ed.ac.uk/portal/files/131011069/Domain_Specialized_Cache_FALDU_DOA06112019_AFV.pdf) HPCA'20

[Q100: The architecture and design of a database processing unit](https://dl.acm.org/doi/pdf/10.1145/2654822.2541961) ASPLOS'14

[Analysis and Optimization of the Memory Hierarchy for Graph Processing Workloads](https://seal.ece.ucsb.edu/sites/default/files/publications/hpca-2019-abanti.pdf) HPCA'19

[SCU: a GPU stream compaction unit for graph processing](http://personals.ac.upc.edu/asegura/publications/isca2019.pdf) ISCA'19

[Balancing Memory Accesses for Energy-Efficient Graph Analytics Accelerators](https://ieeexplore.ieee.org/abstract/document/8824832) ISLPED'19

[Energy Efficient Architecture for Graph Analytics Accelerators](https://www.cs.virginia.edu/~smk9u/CS6501F16/p166-ozdal.pdf) ISCA'16

[GraphH: A Processing-in-Memory Architecture for Large-Scale Graph Processing](https://cseweb.ucsd.edu/~jzhao/files/GraphH-tcad.pdf) TCADICS


