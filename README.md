# Reading List

There are five categories of graph algorithms: 

(1) Graph analytics, e.g., PageRank, SSSP, BFS, betweenness centrality;

(2) Graph pattern mining (GPM), e.g., k-clique listing, motif counting, graph querying, frequent subgraph mining;

(3) Graph machine learning (GML), e.g., graph embedding, DeepWalk, Node2Vec, graph neural networks;

(4) Graph sampling, e.g., random walk.

(5) Graph clustering, partitioning and coarsening.

We group papers at different levels of the system stack: (1) algorithms; (2) software frameworks/systems; (3) hardware acclerators.

[The long list from MIT](https://people.csail.mit.edu/jshun/graph.shtml)

[Other List](https://github.com/Qingfeng-Yao/Readinglist)

[GNN papers](https://github.com/thunlp/GNNPapers)

[NN-on-Silicon](https://github.com/fengbintu/Neural-Networks-on-Silicon)

## Getting Started ##

Papers are listed in the following order:

* [Graph Sampling Accelerators](#graph-sampling-accelerators)
* [Graph Mining Accelerators](#graph-mining-accelerators)
* [Graph Learning Accelerators](#graph-learning-accelerators)
* [Graph Sampling Frameworks](#graph-sampling-frameworks)
* [Graph Mining Systems](#graph-mining-systems)
* [Graph Learning Systems](#graph-learning-systems)
* [Graph Learning Algorithms](#graph-learning-algorithms)
* [Graph Analytics Accelerators](#graph-analytics-accelerators)
* [Graph Analytics Systems](#graph-analytics-systems)
* [Graph Mining Algorithms](#graph-mining-algorithms)
* [Graph Querying Systems](#graph-querying-systems)
* [Graph Clustering and Coarsening](#graph-clustering-and-coarsening)
* [Sparse Linear Algebra](#sparse-linear-algebra)
* [Survey Papers and Books](#survey-papers-and-books)

### Graph Sampling Accelerators ###

[Graph Sampling with Fast Random Walker on HBM-enabled FPGA Accelerators](sampling/FPGA_Random_Walker.pdf) FPL'21

### Graph Mining Accelerators ###

[GraphMineSuite](http://vldb.org/pvldb/vol14/p1922-besta.pdf) VLDB 2021

[FINGERS: Exploiting Fine-Grained Parallelism in Graph Mining Accelerators](https://dl.acm.org/doi/10.1145/3503222.3507730) ASPLOS 2022

[SparseCore: Stream ISA and Processor Specialization for Sparse Computation](http://alchem.usc.edu/portal/static/download/sparsecore.pdf) ASPLOS'22

[SISA: Set-Centric Instruction Set Architecture for Graph Mining on Processing-in-Memory Systems](https://dl.acm.org/doi/pdf/10.1145/3466752.3480133?casa_token=QZo9UWJa8L0AAAAA:vpXSFO2SDjxctb9rf5hy8-sQSy8HQQWm2z5F5tuI259McPVMWJhvADSZ03RHOzhSy0h9dMlqAKyo) MICRO'21

[IntersectX: An Accelerator for Graph Mining](https://arxiv.org/pdf/2012.10848.pdf)

[A Locality-Aware Energy-Efficient Accelerator for Graph Mining Applications](https://www.microarch.org/micro53/papers/738300a895.pdf) MICRO'20

[The TrieJax Architecture: Accelerating Graph Operations Through Relational Joins](https://arxiv.org/pdf/1905.08021.pdf) ASPLOS'20

### Graph Learning Accelerators ###

[A Collection](https://github.com/zjjzby/GNN-hardware-acceleration-paper)

[Computing Graph Neural Networks: A Survey from Algorithms to Accelerators](https://arxiv.org/pdf/2010.00130.pdf)

[Survey on Graph Neural Network Acceleration: An Algorithmic Perspective](https://arxiv.org/pdf/2202.04822.pdf)

[ReGNN: A Redundancy-Eliminated Graph Neural Networks Accelerator]() HPCA 2022

[LISA: Graph Neural Network Based Portable Mapping on Spatial Accelerators]() HPCA 2022

[GCoD: Graph Convolutional Network Acceleration via Dedicated Algorithm and Accelerator Co-Design]() HPCA 2022

[Accelerating Graph Convolutional Networks Using Crossbar-based Processing-In-Memory Architectures]() HPCA 2022

[I-GCN: A Graph Convolutional Network Accelerator with Runtime Locality Enhancement through Islandization](learning/I-GCN.pdf) MICRO'21

[Crossbar based Processing in Memory Accelerator Architecture for Graph Convolutional Networks](learning/PIM-GCN.pdf) ICCAD'21

[A Deep Dive Into Understanding The Random Walk-Based Temporal Graph Learning](learning/RandomWalk-GCN.pdf) IISWC'21

[GCNear: A Hybrid Architecture for Efficient GCN Training with Near-Memory Processing](https://arxiv.org/pdf/2111.00680.pdf) 

[GCNAX: A Flexible and Energy-efficient Accelerator for Graph Convolutional Neural Networks](learning/GCNAX.pdf) HPCA'21

[Hardware Acceleration of Graph Neural Networks](http://rakeshk.crhc.illinois.edu/dac20.pdf) DAC'20

[AWB-GCN: A Graph Convolutional Network Accelerator with Runtime Workload Rebalancing](https://www.microarch.org/micro53/papers/738300a922.pdf) MICRO'20

[HyGCN: A GCN Accelerator with Hybrid Architecture](https://arxiv.org/pdf/2001.02514.pdf) HPCA'20

[GraphACT: Accelerating GCN Training on CPU-FPGA Heterogeneous Platforms](https://arxiv.org/pdf/2001.02498.pdf) FPGA'20

[A Taxonomy for Classification and Comparison of Dataflows for GNN Accelerators](https://arxiv.org/pdf/2103.07977.pdf)

[Architectural Implication of Graph Neural Networks](http://www.cs.sjtu.edu.cn/~leng-jw/resources/Files/zhang20cal-gnn.pdf)

[GReTA: Hardware Optimized Graph Processing for GNNs](https://sing.stanford.edu/site/publications/greta-recoml20.pdf)

[GRIP: A Graph Neural Network Accelerator Architecture](https://arxiv.org/pdf/2007.13828.pdf)

### Graph Sampling Frameworks ###

[FlashWalker: An In-Storage Accelerator for Graph Random Walks]() IPDPS 2022

[Random Walks on Huge Graphs at Cache Efficiency](http://madsys.cs.tsinghua.edu.cn/publications/sosp21-yang.pdf) SOSP'21 [FlashMob website](https://github.com/flashmobwalk/flashmob)

[Skywalker: Efficient Alias-method-based Graph Sampling and Random Walk on GPUs](sampling/Skywalker.pdf) PACT'21 [Skywalker website](https://github.com/wpybtw/Skywalker)

[KnightKing: A Fast Distributed Graph RandomWalk Engine](sampling/KnightKing.pdf) SOSP'19 [KnightKing website](https://github.com/KnightKingWalk/KnightKing)

[Accelerating Graph Sampling for Graph Machine Learning using GPUs](sampling/NextDoor.pdf) EuroSys'21 [NextDoor website](https://github.com/plasma-umass/NextDoor)

[C-SAW: A Framework for Graph Sampling and Random Walk on GPUs](sampling/C-SAW.pdf) SC'20 [C-SAW website](https://github.com/concept-inversion/C-SAW)

[ThunderRW: An In-Memory Graph RandomWalk Engine](sampling/ThunderRW.pdf) VLDB'21 [ThunderRW website](https://github.com/Xtra-Computing/ThunderRW)

[Memory-Aware Framework for Efficient Second-Order Random Walk on Large Graphs](https://shaoyx.github.io/files/main.pdf) SIGMOD'20

[GraphWalker: An I/O-Efficient and Resource-Friendly Graph Analytic System for Fast and Scalable Random Walks](https://www.usenix.org/system/files/atc20-wang-rui.pdf) ATC'20

### Graph Mining Systems ###

[Mnemonic: A Parallel Subgraph Matching System for Streaming Graphs]() IPDPS 2022

[Tesseract: distributed, general graph pattern mining on evolving graphs](https://dl.acm.org/doi/abs/10.1145/3447786.3456253) EuroSys'21

[SumPA: Efficient Pattern-Centric Graph Mining with Pattern Abstraction](https://ieeexplore.ieee.org/abstract/document/9563022) PACT'21

[RapidMatch: A Holistic Approach to Subgraph Query Processing](https://www.comp.nus.edu.sg/~hebs/pub/rapidmatch-vldb21.pdf) VLDB 2021 [Website](https://github.com/RapidsAtHKUST/RapidMatch)

[GPU-Accelerated Subgraph Enumeration on Partitioned Graphs](https://dl.acm.org/doi/abs/10.1145/3318464.3389699?casa_token=6YJkJ4c7b_UAAAAA:JbNWDytqd6kY8hdktAp0FQsXGTFTaWQxAih16Q-lJZd_qzKlE3TV06HOB1brW9ThFqllWR9FqRY). SIGMOD 2020. [Bib entry](https://scholar.googleusercontent.com/scholar.bib?q=info:F6fuEJ0tqPIJ:scholar.google.com/&output=citation&scisdr=CgXYWi02EL6ftXcuxRA:AAGBfm0AAAAAX5Ar3RD-F_5o4Eu-2ejzNDHMIG7taZup&scisig=AAGBfm0AAAAAX5Ar3b8tdj-daz05wgRPHlYYWPf_O-GH&scisf=4&ct=citation&cd=-1&hl=en) [Slides](https://drive.google.com/file/d/1tPbdvbMZMaK21U-XfLG0QREgIA97RM84/view?usp=sharing) [Video](https://drive.google.com/file/d/1DnlntHNPt5HLKAgJyx4_JnSmdRn7APZy/view?usp=sharing) [Website](https://github.com/guowentian/SubgraphMatchGPU)

[Exploiting Reuse for GPU Subgraph Enumeration](https://ieeexplore.ieee.org/document/9247538). TKDE 2020

[First: Fast interactive attributed subgraph matching](https://idvxlab.com/papers/2017KDD_First_Du.pdf) KDD'17

[In-Memory Subgraph Matching: An In-depth Study](https://dl.acm.org/doi/pdf/10.1145/3318464.3380581) SIGMOD'2020 [website](https://github.com/RapidsAtHKUST/SubgraphMatching)

[Efficient Subgraph Matching: Harmonizing Dynamic Programming, Adaptive Matching Order, and Failing Set Together](mining/DAF.pdf) SIGMOD'19 [DAF website](https://github.com/SNUCSE-CTA/DAF)

[Scaling Up Subgraph Query Processing with Efficient Subgraph Matching](mining/ICDE19-ScalingUpSubgraphQueryProcessing.pdf) ICDE'19

[Efficient Parallel Subgraph Enumeration on a Single Machine](mining/ICDE19-LIGHT.pdf) ICDE'19

[Fast and Robust Distributed Subgraph Enumeration](mining/VLDB19-FastRobustDistributedSubgraphEnumeration.pdf) VLDB'19

[Peregrine: A Pattern-Aware Graph Mining System](mining/Eurosys20-Peregrine.pdf) Eurosys'20

[Optimizing Subgraph Queries by Combining Binary and Worst-Case Optimal Joins](http://www.vldb.org/pvldb/vol12/p1692-mhedhbi.pdf) VLDB'19

[The Power of Pivoting for Exact Clique Counting](mining/Pivoter.pdf) WSDM'20

[DUALSIM: Parallel Subgraph Enumeration in a Massive Graph on a Single Machine](https://www.ntu.edu.sg/home/assourav/papers/SIGMOD-16-DualSim.pdf) SIGMOD'16

[AutoMine](https://dl.acm.org/doi/abs/10.1145/3341301.3359633) SOSP'19

[Arabesque](http://arabesque.qcri.org/) SOSP'2015

[RStream](https://www.usenix.org/system/files/osdi18-wang.pdf) OSDI'18

[GraphPi](https://arxiv.org/abs/2009.10955) [website](https://github.com/thu-pacman/GraphPi) SC'20

### Graph Learning Systems ###

[Distributed Hybrid CPU and GPU training for Graph Neural Networks on Billion-Scale Heterogeneous Graphs](https://arxiv.org/pdf/2112.15345.pdf)

[Marius++: Large-Scale Training of Graph Neural Networks on a Single Machine](https://arxiv.org/abs/2202.02365)

[Learn Locally, Correct Globally: A Distributed Algorithm for Training Graph Neural Networks](https://arxiv.org/pdf/2111.08202.pdf) ICLR'22

[QGTC: Accelerating Quantized Graph Neural Networks via GPU Tensor Core](https://arxiv.org/pdf/2111.09547.pdf) PPoPP'22

[Understanding and Bridging the Gaps in Current GNN Performance Optimizations](https://dl.acm.org/doi/10.1145/3437801.3441585) PPoPP'21

[P3: Distributed Deep Graph Learning at Scale](https://www.usenix.org/conference/osdi21/presentation/gandhi) OSDI 2021

[Dorylus: Affordable, Scalable, and Accurate GNN Training over Billion-Edge Graphs](https://www.usenix.org/conference/osdi21/presentation/thorpe) OSDI 2021

[GNNAdvisor: An Efficient Runtime System for GNN Acceleration on GPUs](https://www.usenix.org/conference/osdi21/presentation/wang-yuke) OSDI 2021

[Marius: Learning Massive Graph Embeddings on a Single Machine](https://www.usenix.org/conference/osdi21/presentation/mohoney) OSDI 2021

[Large Graph Convolutional Network Training with GPU-Oriented Data Communication Architecture](http://vldb.org/pvldb/vol14/p2087-min.pdf) VLDB 2021

[Grain: Improving Data Efficiency of Graph Neural Networks via Diversified Influence Maximization](http://vldb.org/pvldb/vol14/p2473-zhang.pdf) VLDB 2021

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

### Graph Learning Algorithms ###

[Global Neighbor Sampling for Mixed CPU-GPU Training on Giant Graphs (GNS)](https://arxiv.org/abs/2106.06150) KDD 2021 [GNS website](https://github.com/jadadong/GNS)

[GCN meets GPU: Decoupling “When to Sample” from “How to Sample”](https://proceedings.neurips.cc/paper/2020/file/d714d2c5a796d5814c565d78dd16188d-Paper.pdf) NeurIPS 2020 [LazyGCN website](https://github.com/MortezaRamezani/lazygcn)

[Count-GNN: Graph Neural Networks for Subgraph Isomorphism Counting](https://openreview.net/forum?id=_MO2xzOZXv) (GNN for GPM)

[Can Graph Neural Networks Count Substructures?](https://proceedings.neurips.cc/paper/2020/file/75877cb75154206c4e65e76b88a12712-Paper.pdf) (GNN for GPM)

[Graph Transformer Networks](https://proceedings.neurips.cc/paper/2019/file/9d63484abb477c97640154d40595a3bb-Paper.pdf) (GPM for GNN)

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

### Graph Analytics Accelerators ###

[Hardware-Accelerated Hypergraph Processing with Chain-Driven Scheduling]() HPCA 2022

[ScalaGraph: A Scalable Accelerator for Massively Parallel Graph Processing]() HPCA 2022

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

### Graph Analytics Systems ###

[Galois](https://github.com/IntelligentSoftwareSystems/Galois)

[Ligra](https://github.com/jshun/ligra)

[PowerGraph](https://github.com/jegonzal/PowerGraph)

[GraphScope: A Unified Engine For Big Graph Processing](http://vldb.org/pvldb/vol14/p2879-qian.pdf) VLDB 2021

[Automating Incremental Graph Processing with Flexible Memoization](http://vldb.org/pvldb/vol14/p1613-gong.pdf) VLDB 2021

[EMOGI: Efficient Memory-access for Out-of-memory Graph-traversal in GPUs](http://vldb.org/pvldb/vol14/p114-min.pdf) VLDB 2021

### Graph Mining Algorithms ###

[LOTUS: Locality Optimizing Triangle Counting](https://blogs.qub.ac.uk/graphprocessing/wp-content/uploads/sites/300/2022/02/LOTUS_TC_Authors_Copy.pdf) PPoPP 2022

[Efficient Streaming Subgraph Isomorphism with Graph Neural Networks](http://vldb.org/pvldb/vol14/p730-duong.pdf) VLDB 2021

[On Analyzing Graphs with Motif-Paths](http://vldb.org/pvldb/vol14/p1111-li.pdf) VLDB 2021

[Symmetric Continuous Subgraph Matching with Bidirectional Dynamic Programming](http://vldb.org/pvldb/vol14/p1298-han.pdf) VLDB 2021

[Real-time Twitter Recommendation: Online Motif Detection in Large Dynamic Graphs](http://www.vldb.org/pvldb/vol7/p1379-lin.pdf) VLDB 2014

### Graph Querying Systems ###

[Mycelium: Large-Scale Distributed Graph Queries with Differential Privacy](https://www.cis.upenn.edu/~sga001/papers/mycelium-sosp21.pdf) SOSP 2021

[Columnar Storage and List-based Processing for Graph Database Management Systems](http://vldb.org/pvldb/vol14/p2491-gupta.pdf) VLDB 2021

[EmptyHeaded: A Relational Engine for Graph Processing](mining/EmptyHeaded.pdf) SIGMOD'16

[Graphflow](http://graphflow.io/)

[Schemaless and Structureless Graph Querying](https://dl.acm.org/doi/pdf/10.14778/2732286.2732293) VLDB 2014

[GraphDB: Modeling and Querying Graphs in Databases](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.52.3865&rep=rep1&type=pdf) VLDB 1994

### Graph Clustering and Coarsening ###

[Scaling Up Graph Neural Networks Via Graph Coarsening](https://arxiv.org/pdf/2106.05150.pdf)

### Sparse Linear Algebra ###

[TileSpGEMM: A Tiled Algorithm for Parallel Sparse General Matrix-Matrix Multiplication on GPUs](https://github.com/SuperScientificSoftwareLaboratory/TileSpGEMM) PPoPP 2022

### Survey Papers and Books ###

[Introduction to Graph Neural Networks](https://www.morganclaypool.com/doi/10.2200/S00980ED1V01Y202001AIM045) Book

[The Ubiquity of Large Graphs and Surprising Challenges of Graph Processing](http://www.vldb.org/pvldb/vol11/p420-sahu.pdf) VLDB'18

[Link prediction in complex networks: A survey](https://arxiv.org/pdf/1010.0725.pdf)

[Survey on social community detection](https://hal.archives-ouvertes.fr/hal-00804234/file/Survey-on-Social-Community-Detection-V2.pdf)

[Practice of Streaming Processing of Dynamic Graphs: Concepts, Models, and Systems](https://arxiv.org/pdf/1912.12740.pdf)

[Programming Massively Parallel Processors](https://safari.ethz.ch/architecture/fall2019/lib/exe/fetch.php?media=2013_programming_massively_parallel_processors_a_hands-on_approach_2nd.pdf)
