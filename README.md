# Reading List
[The long list from MIT](https://people.csail.mit.edu/jshun/graph.shtml)

[Other List](https://github.com/Qingfeng-Yao/Readinglist)

[GNN papers](https://github.com/thunlp/GNNPapers)

[NN-on-Silicon](https://github.com/fengbintu/Neural-Networks-on-Silicon)

## Papers on Graph Sampling Accelerators

[Graph Sampling with Fast Random Walker on HBM-enabled FPGA Accelerators](sampling/FPGA_Random_Walker.pdf) FPL'21

## Papers on Graph Mining Accelerators

[IntersectX: An Accelerator for Graph Mining](https://arxiv.org/pdf/2012.10848.pdf)

[A Locality-Aware Energy-Efficient Accelerator for Graph Mining Applications](https://www.microarch.org/micro53/papers/738300a895.pdf) MICRO'20

[The TrieJax Architecture: Accelerating Graph Operations Through Relational Joins](https://arxiv.org/pdf/1905.08021.pdf) ASPLOS'20

## Papers on Graph Learning Accelerators

[A Collection](https://github.com/zjjzby/GNN-hardware-acceleration-paper)

[I-GCN: A Graph Convolutional Network Accelerator with Runtime Locality Enhancement through Islandization](learning/I-GCN.pdf) MICRO'21

[Computing Graph Neural Networks: A Survey from Algorithms to Accelerators](https://arxiv.org/pdf/2010.00130.pdf)

[GCNAX: A Flexible and Energy-efficient Accelerator for Graph Convolutional Neural Networks](learning/GCNAX.pdf) HPCA'21

[Hardware Acceleration of Graph Neural Networks](http://rakeshk.crhc.illinois.edu/dac20.pdf) DAC'20

[AWB-GCN: A Graph Convolutional Network Accelerator with Runtime Workload Rebalancing](https://www.microarch.org/micro53/papers/738300a922.pdf) MICRO'20

[HyGCN: A GCN Accelerator with Hybrid Architecture](https://arxiv.org/pdf/2001.02514.pdf) HPCA'20

[GraphACT: Accelerating GCN Training on CPU-FPGA Heterogeneous Platforms](https://arxiv.org/pdf/2001.02498.pdf) FPGA'20

[A Taxonomy for Classification and Comparison of Dataflows for GNN Accelerators](https://arxiv.org/pdf/2103.07977.pdf)

[Architectural Implication of Graph Neural Networks](http://www.cs.sjtu.edu.cn/~leng-jw/resources/Files/zhang20cal-gnn.pdf)

[GReTA: Hardware Optimized Graph Processing for GNNs](https://sing.stanford.edu/site/publications/greta-recoml20.pdf)

[GRIP: A Graph Neural Network Accelerator Architecture](https://arxiv.org/pdf/2007.13828.pdf)

## Papers on Graph Sampling Frameworks

[Skywalker: Efficient Alias-method-based Graph Sampling and Random Walk on GPUs](sampling/Skywalker.pdf) PACT'21 [Skywalker website]()

[KnightKing: A Fast Distributed Graph RandomWalk Engine](sampling/KnightKing.pdf) SOSP'19 [KnightKing website](https://github.com/KnightKingWalk/KnightKing)

[Accelerating Graph Sampling for Graph Machine Learning using GPUs](sampling/NextDoor.pdf) EuroSys'21 [NextDoor website](https://github.com/plasma-umass/NextDoor)

[C-SAW: A Framework for Graph Sampling and Random Walk on GPUs](sampling/C-SAW.pdf) SC'20 [C-SAW website](https://github.com/concept-inversion/C-SAW)

[ThunderRW: An In-Memory Graph RandomWalk Engine](sampling/ThunderRW.pdf) VLDB'21 [ThunderRW website](https://github.com/Xtra-Computing/ThunderRW)

[Memory-Aware Framework for Efficient Second-Order Random Walk on Large Graphs](https://shaoyx.github.io/files/main.pdf) SIGMOD'20

## Papers on Graph Mining Systems

[First: Fast interactive attributed subgraph matching](https://idvxlab.com/papers/2017KDD_First_Du.pdf) KDD'17

[In-Memory Subgraph Matching: An In-depth Study](https://dl.acm.org/doi/pdf/10.1145/3318464.3380581) SIGMOD'2020 [website](https://github.com/RapidsAtHKUST/SubgraphMatching)

[Efficient Subgraph Matching: Harmonizing Dynamic Programming, Adaptive Matching Order, and Failing Set Together](mining/DAF.pdf) SIGMOD'19 [DAF website](https://github.com/SNUCSE-CTA/DAF)

[Scaling Up Subgraph Query Processing with Efficient Subgraph Matching](mining/ICDE19-ScalingUpSubgraphQueryProcessing.pdf) ICDE'19

[Efficient Parallel Subgraph Enumeration on a Single Machine](mining/ICDE19-LIGHT.pdf) ICDE'19

[Fast and Robust Distributed Subgraph Enumeration](mining/VLDB19-FastRobustDistributedSubgraphEnumeration.pdf) VLDB'19

[Peregrine: A Pattern-Aware Graph Mining System](mining/Eurosys20-Peregrine.pdf) Eurosys'20

[Optimizing Subgraph Queries by Combining Binary and Worst-Case Optimal Joins](http://www.vldb.org/pvldb/vol12/p1692-mhedhbi.pdf) VLDB'19

[EmptyHeaded: A Relational Engine for Graph Processing](mining/EmptyHeaded.pdf) SIGMOD'16

[The Power of Pivoting for Exact Clique Counting](mining/Pivoter.pdf) WSDM'20

[DUALSIM: Parallel Subgraph Enumeration in a Massive Graph on a Single Machine](https://www.ntu.edu.sg/home/assourav/papers/SIGMOD-16-DualSim.pdf) SIGMOD'16

[AutoMine](https://dl.acm.org/doi/abs/10.1145/3341301.3359633) SOSP'19

[Arabesque](http://arabesque.qcri.org/) SOSP'2015

[RStream](https://www.usenix.org/system/files/osdi18-wang.pdf) OSDI'18

[GraphPi](https://arxiv.org/abs/2009.10955) [website](https://github.com/thu-pacman/GraphPi) SC'20

## Papers on Graph Learning Systems
[Understanding and Bridging the Gaps in Current GNN Performance Optimizations]() PPoPP'21

[Dorylus: Affordable, Scalable, and Accurate GNN Training over Billion-Edge Graphs]() OSDI'21

[GNNAdvisor: An Efficient Runtime System for GNN Acceleration on GPUs](https://arxiv.org/pdf/2006.06608.pdf) OSDI'21

[Rubik: A Hierarchical Architecture for Efficient Graph Learning](https://arxiv.org/pdf/2009.12495.pdf) TCAD'21

[fuseGNN: Accelerating Graph Convolutional Neural Network Training on GPGPU](https://seal.ece.ucsb.edu/sites/default/files/publications/fusegcn_camera_ready_.pdf) ICCAD'20

[Deep Graph Library Optimizations for Intel(R) x86 Architecture](https://arxiv.org/pdf/2007.06354.pdf)

[Improving the Accuracy, Scalability, and Performance of Graph Neural Networks with Roc](learning/Roc.pdf) MLSys'20 [Roc website](https://github.com/jiazhihao/ROC)

[FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems](https://www.csl.cornell.edu/~zhiruz/pdfs/featgraph-sc2020.pdf) SC'20 [FeatGraph website](https://github.com/amazon-research/FeatGraph)

[GraphSAINT: Graph Sampling Based Inductive Learning Method](https://openreview.net/pdf?id=BJe8pkHFwS) ICLR'20 [GraphSAINT website](https://github.com/GraphSAINT/GraphSAINT)

[Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs](learning/DGL.pdf) ICLR'19 [DGL website](https://www.dgl.ai/)

[Fast Graph Representation Learning with PyTorch Geometric](learning/PyG.pdf) ICLR'19 [PyG website](https://github.com/rusty1s/pytorch_geometric)

[NeuGraph: Parallel Deep Neural Network Computation on Large Graphs](learning/NeuGraph.pdf) USENIX ATC'19 [NeuGraph website](https://www.microsoft.com/en-us/research/publication/neugraph-parallel-deep-neural-network-computation-on-large-graphs/)

[CAGNET](https://arxiv.org/pdf/2005.03300.pdf) [website](https://github.com/PASSIONLab/CAGNET) SC'20

[AGL: a scalable system for industrial-purpose graph machine learning](http://www.vldb.org/pvldb/vol13/p3125-zhang.pdf) VLDB'20

[Accurate, Efficient and Scalable Graph Embedding](https://arxiv.org/pdf/1810.11899.pdf) IPDPS'19

## Papers on Graph Learning Algorithms
[Semi-Supervised Classification with Graph Convolutional Networks](learning/GCN.pdf) ICLR'17 [GCN website](https://github.com/tkipf/gcn)

[The Logical Expressiveness of Graph Neural Networks](https://openreview.net/pdf?id=r1lZ7AEKvB)

[How Powerful are Graph Neural Networks?](https://openreview.net/pdf?id=ryGs6iA5Km) ICLR'19 [GIN website](https://github.com/weihua916/powerful-gnns)

[Hierarchical Graph Representation Learning with Differentiable Pooling](learning/diffpool.pdf) NeurIPS'18 [diffpool website](https://github.com/RexYing/diffpool)

[Inductive Representation Learning on Large Graphs](learning/GraphSAGE.pdf) NIPS'17 [GraphSAGE website](http://snap.stanford.edu/graphsage/)

[Stochastic Training of Graph Convolutional Networks with Variance Reduction](http://proceedings.mlr.press/v80/chen18p/chen18p.pdf) ICML'18 [S-GCN](https://github.com/thu-ml/stochastic_gcn)

[Adaptive Sampling Towards Fast Graph Representation Learning](https://papers.nips.cc/paper/7707-adaptive-sampling-towards-fast-graph-representation-learning.pdf) NIPS'18 [AS-GCN website](https://github.com/huangwb/AS-GCN)

[Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/pdf/1905.07953.pdf) KDD'19 [ClusterGCN website](https://github.com/google-research/google-research/tree/master/cluster_gcn)

[FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling](https://openreview.net/pdf?id=rytstxWAW) ICLR'18 [FastGCN website](https://github.com/matenure/FastGCN)

[Large-Scale Learnable Graph Convolutional Networks](https://arxiv.org/pdf/1808.03965.pdf) KDD'18 [LGCN website](https://github.com/HongyangGao/LGCN)

[Representation Learning on Graphs with Jumping Knowledge Networks](http://proceedings.mlr.press/v80/xu18c/xu18c.pdf) KDD'18 [JK-net code](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/jumping_knowledge.html#JumpingKnowledge)

[DeepWalk: Online Learning of Social Representations](learning/DeepWalk.pdf) KDD'14

[node2vec: Scalable Feature Learning for Networks](learning/node2vec.pdf) KDD'16

[GraphVite: A High-Performance CPU-GPU Hybrid System for Node Embedding](https://arxiv.org/pdf/1903.00757.pdf) WWW'19 [GraphVite website](https://graphvite.io/)

## Papers on Graph Analytics Accelerators
[DepGraph: A Dependency-Driven Accelerator for Efficient Iterative Graph Processing]() HPCA'21

[P-OPT: Practical Optimal Cache Replacement for Graph Analytics](https://brandonlucia.com/pubs/POPT_HPCA21_CameraReady.pdf) HPCA'21

[ThunderGP: HLS-based Graph Processing Framework on FPGAs](https://www.comp.nus.edu.sg/~wongwf/papers/FPGA2021.pdf) FPGA'21

[HitGraph: High-throughput Graph Processing Framework on FPGA](https://par.nsf.gov/servlets/purl/10125453) TPDS'19

[Exploiting Locality in Graph Analytics through Hardware-Accelerated Traversal Scheduling](http://people.csail.mit.edu/sanchez/papers/2018.hats.micro.pdf) MICRO'18

[UWB-GCN: Accelerating Graph Convolutional Networks through Runtime Workload Rebalancing](https://arxiv.org/pdf/1908.10834.pdf)

[Graphicionado: A High-Performance and Energy Efficient Accelerator for Graph Analytics](https://mrmgroup.cs.princeton.edu/papers/taejun_micro16.pdf) MICRO'16

[GraphR: Accelerating Graph Processing Using ReRAM](https://arxiv.org/pdf/1708.06248.pdf) HPCA'18

[GraphQ: Scalable PIM-Based Graph Processing](http://alchem.usc.edu/portal/static/download/graphq.pdf) MICRO'19

[GraphP: Reducing Communication of PIM-based Graph Processing with Efficient Data Partition](http://csl.stanford.edu/~christos/publications/2018.graphp.hpca.pdf) HPCA'18

[GraphSAR: A Sparsity-Aware Processing-in-Memory Architecture for Large-Scale Graph Processing on ReRAMs](https://dl.acm.org/doi/pdf/10.1145/3287624.3287637) ASPDAC'19

[Alleviating Irregularity in Graph Analytics Acceleration: a Hardware/Software Co-Design Approach](https://web.ece.ucsb.edu/~iakgun/files/MICRO2019.pdf) MICRO'19

[GraphABCD: Scaling Out Graph Analytics with Asynchronous Block Coordinate Descent](hardware/GraphABCD.pdf) ISCA'20

[GaaS-X: Graph Analytics Accelerator Supporting Sparse Data Representation Using Crossbar Architectures](hardware/GaaS-X.pdf) ISCA'20

[POSTER: Domain-Specialized Cache Management for Graph Analytics](http://www.faldupriyank.com/papers/GRASP_PACT19.pdf) PACT'19 [code](https://github.com/ease-lab/grasp)

[Domain-Specialized Cache Management for Graph Analytics](https://www.research.ed.ac.uk/portal/files/131011069/Domain_Specialized_Cache_FALDU_DOA06112019_AFV.pdf) HPCA'20

[Q100: The architecture and design of a database processing unit](https://dl.acm.org/doi/pdf/10.1145/2654822.2541961) ASPLOS'14

[Analysis and Optimization of the Memory Hierarchy for Graph Processing Workloads](https://seal.ece.ucsb.edu/sites/default/files/publications/hpca-2019-abanti.pdf) HPCA'19

[SCU: a GPU stream compaction unit for graph processing](http://personals.ac.upc.edu/asegura/publications/isca2019.pdf) ISCA'19

[Balancing Memory Accesses for Energy-Efficient Graph Analytics Accelerators](https://ieeexplore.ieee.org/abstract/document/8824832) ISLPED'19

[Energy Efficient Architecture for Graph Analytics Accelerators](https://www.cs.virginia.edu/~smk9u/CS6501F16/p166-ozdal.pdf) ISCA'16

[GraphH: A Processing-in-Memory Architecture for Large-Scale Graph Processing](https://cseweb.ucsd.edu/~jzhao/files/GraphH-tcad.pdf) TCADICS

[GraFBoost: Using accelerated flash storage for external graph analytics](https://people.csail.mit.edu/wjun/papers/isca2018-camera.pdf) ISCA'18

## Papers on Graph Analytics Systems

[Galois](https://github.com/IntelligentSoftwareSystems/Galois)

[Ligra](https://github.com/jshun/ligra)

[PowerGraph](https://github.com/jegonzal/PowerGraph)

## Survey Papers
[Introduction to Graph Neural Networks](https://www.morganclaypool.com/doi/10.2200/S00980ED1V01Y202001AIM045) Book

[The Ubiquity of Large Graphs and Surprising Challenges of Graph Processing](http://www.vldb.org/pvldb/vol11/p420-sahu.pdf) VLDB'18

[Link prediction in complex networks: A survey](https://arxiv.org/pdf/1010.0725.pdf)

[Survey on social community detection](https://hal.archives-ouvertes.fr/hal-00804234/file/Survey-on-Social-Community-Detection-V2.pdf)

[Practice of Streaming Processing of Dynamic Graphs: Concepts, Models, and Systems](https://arxiv.org/pdf/1912.12740.pdf)
