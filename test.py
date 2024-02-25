#!/usr/bin/env python
# coding: utf-8

# # Causal Inference for Policy Evaluation in Networks
# 
# ### IDEAL Get Ready For Research Workshop
# 
# Mentors: Shishir Adhikari (sadhik9@uic.edu) and Ahmed Sayeed Faruk (afaruk2@uic.edu)
# 
# 
# Hey, welcome to IDEAL Get Ready for Research Workshop. Thank you for your interest on this research topic.
# 
# ## Outline
# 
# The [presentation slides](https://docs.google.com/presentation/d/1SHgSqjK5fT9axS2WN-ODRhpgMMowtpPEiRuFiASNVI8/edit?usp=sharing) introduced the background on causal inference in networks, treatment policies, types of network effects, and individual network effect (INE) estimation problem. Please feel free to discuss among yourselves or ask us clarification questions about these concepts.
# 
# In this notebook, we will guide you with the following steps.
# 1. Causal Reasoning
#   - Analysis of assumptions and estimands (equations) for INE estimation from experimental data
# 2. Dataset preparation
#   - Network Experiment Design (A/B test) with node-level randomization
#     - Synthetic data generation for model development and testing
# 3. Estimator
#   - **Graph neural networks (GNN)** for representational learning of contexts $Z_i$ as mapping of node attributes, edge attributes, and network structure
#   - Summarizing peer exposure
#   - Counterfactual prediction neural network (CPNN)
# 4. Training
#   - Optimizing loss function and regularization
#   - Train GNN and CPNN (including hyperparameters tuning/model selection)
# 5. Inference
#   - Use trained model to estimate unit-level effects and average effects for given policies
# 6. Evaluation
#  - Evaluation metrics for average and heterogeneous (individual) effects
# 7. Discussion
#   - Related research problems
#     - Analysis from real-world observational social network data: Understanding the Dynamics between Vaping and Cannabis Legalization Using Twitter Opinions. [Adhikari, Uppal, Mermelstein, Berger-Wolf, and Zheleva, 2021](https://ojs.aaai.org/index.php/ICWSM/article/view/18037) 
#     - Inferring causal effects under heterogeneous peer influence. [Adhikari and Zheleva, 2023](https://arxiv.org/pdf/2305.17479.pdf)
#     - Estimating causal effects in networks with cluster-based bandits. [Faruk, Sulskis, and Zheleva, 2022](#)
#   - Next steps
#     - Learn best/optimal policy using INE estimator
#     - Learn policy on the fly (recommendations) and maximize rewards (clicks or sales)
#     - What happens when treatment affects network connections?
# 
# 
# 
# 
# 
# 
# 
# 

# ## 1. Causal Reasoning
# 
# >**Please feel free to skip the mathematical details and see the takeaways at the end.**
# 
# We defined individual network effects (INE) in terms of potential outcomes (or counterfactual outcomes), i.e., 
# 
# \begin{equation}
# INE(\pi_i, \pi_{-i}, \pi'_i, \pi'_{-i}, Z_i) = E[Y_i(T_i=\pi_i, T_{-i}=\pi_{-i}) - Y_i(T_i=\pi'_i, T_{-i}=\pi'_{-i}) | Z_i]\tag{1},
# \end{equation}
# 
# where $<\pi_i, \pi_{-i}>$ and $<\pi'_i, \pi'_{-i}>$ are alternative policies and $Y_i(T_i=\pi_i, T_{-i}=\pi_{-i})|Z_i$ is the potential outcome for units with contexts $Z_i$ if we could assign $<\pi_i, \pi_{-i}>$ treatment.
# 
# **Assumption 1: Network and its attributes are measured before treatment and not affected by the treatment.**
# 
# Due to Assumption 1, we can express counterfactuals in Equation 1 as do-expression (Pearl 2009) where $do(.)$ denotes interventions. 
# \begin{equation}
# INE(\pi_i, \pi_{-i}, \pi'_i, \pi'_{-i}, Z_i) = E[Y_i|do(T_i=\pi_i, T_{-i}=\pi_{-i}),Z_i] - E[Y_i|do(T_i=\pi'_i, T_{-i}=\pi'_{-i}),Z_i]\tag{2},
# \end{equation}
# 
# We can represent assumptions using a graphical (causal) model as shown in Figure below.
# <img src="https://drive.google.com/uc?export=view&id=1qhfLATr3j1s0mkUdCzeMbtxvpW14DI1Q" height=400/>
# 
# Here, $Z_i$ in square box means it is observed and it could affect outcome but not treatment because treatments are randomized. The bidirectional dotted line joining unit's treatment $T_i$ and others' treatment $T_{-i}$ suggests the assignments could be correlated. We denote unit's measured outcome by $Y_i$ and other units' outcomes by $Y_{-i}$. The latent (unmeasured) time-lagged outcomes are denoted by $Y_i^L$ and $Y_{-i}^L$. The blue edges represent spillover effects and purple edges show contagion effects.
# 
# 
# For experimental data, we can estimate INE as follows using Pearl's second rule of do-calculus (Pearl 2009; Bareinboim and Pearl 2016):
# 
# \begin{equation}
# INE(\pi_i, \pi_{-i}, \pi'_i, \pi'_{-i}, Z_i)=E[Y_i|T_i=\pi_i, T_{-i}=\pi_{-i},Z_i] - E[Y_i|T_i=\pi'_i, T_{-i}=\pi'_{-i},Z_i] \tag{3}
# \end{equation}
# 
# **Takeaways**
# 
# Using machine learning, we can learn two models for predicting counterfactual outcomes under treatment $\hat{Y_i(1)}$ and counterfactual outcomes under control $\hat{Y_i(0)}$ for peer exposure $\phi(T_{-i})$ and contexts $\phi(Z_i)$:
#   - $\hat{Y_i(1)} = f_1(\phi(T_{-i}),\phi(Z_i);\Theta_1)$ for individuals in the treatment group and
#   - $\hat{Y_i(0)} = f_0(\phi(T_{-i}), \phi(Z_i);\Theta_0)$ for individuals in the control group.
# 
# where, $\phi$ maps raw data (network, attributes, and treatment) to feature vectors, and $\Theta_{1}$ and $\Theta_{0}$ are learnable parameters.
# 
# Depending on the value of $T_i$, we can estimate $E[Y_i|T_i=\pi_i, T_{-i}=\pi_{-i},Z_i]$ and $E[Y_i|T_i=\pi_i, T_{-i}=\pi_{-i},Z_i]$ in Equation (3) using models $f_1$ and $f_0$.
# 
# 
# **Feature mapping**: We need to encode node attribute, edge attribute, and network structure features $\phi(Z_i)$. So, we use graph neural network (GNN) for it.
# 
# **Exposure mapping**: We need to encode peer exposure, i.e., treatment of neighbors. So, we use fraction of treated friends for it.
# 
# **Counterfactual prediction**: We need to learn $f_0$ and $f_1$. So, we use neural network for it. We will use regression for continuous outcome and classification for categorical outcome.
# 
# Then, we estimate individual network effect (INE) for given policies $<\pi_i, \pi_{-i}>$ vs $<\pi'_i, \pi'_{-i}>$.
# 
# \begin{equation}
# \begin{split}
# Y_i(\pi_i, \pi_{-i}) = &f_1(\phi(\pi_{-i}), \phi(Z_i)) \text{ if } \pi_i = 1\\
# Y_i(\pi_i, \pi_{-i}) = &f_0(\phi(\pi_{-i}), \phi(Z_i)) \text{ if } \pi_i = 0\\
# \end{split}\tag{4}
# \end{equation}
# <br>
# 
# \begin{equation}
# \begin{split}
# Y_i(\pi'_i, \pi'_{-i}) = &f_1(\phi(\pi'_{-i}), \phi(Z_i)) \text{ if } \pi'_i = 1\\
# Y_i(\pi'_i, \pi'_{-i}) = &f_0(\phi(\pi'_{-i}), \phi(Z_i)) \text{ if } \pi'_i = 0\\
# \end{split}\tag{5}
# \end{equation}
# <br>
# 
# \begin{equation}
# \tau_i = INE(\pi_i, \pi_{-i}, \pi'_i, \pi'_{-i}, Z_i) = Y_i(\pi_i, \pi_{-i}) - Y_i(\pi'_i, \pi'_{-i}) \tag{6}
# \end{equation}
# <br>
# 
# The following figure summarizes empirical estimation of individual network effects (INE) using graph neural network (GNN) and counterfactual prediction. This end-to-end framework of estimating INE is inspired by the Treatment Agnostic Representation Network (TARNet) (Shalit et al., 2017) architecture.
# 
# <img src="https://drive.google.com/uc?export=view&id=1u1985mKo2fugKWkNrBZ2zuwkfelkSQvi" width=600/>
# 
# 
# We will explore more about practically estimating INE later. Next, we will see how to collect data and visualize data for exploratory research.
# 
# **References**
# 
# Pearl J. Causality. Cambridge university press. 2009.
# 
# Bareinboim E, Pearl J. Causal inference and the data-fusion problem. Proceedings of the National Academy of Sciences. 2016
# 
# Shalit U, Johansson FD, Sontag D. Estimating individual treatment effect: generalization bounds and algorithms. In International Conference on Machine Learning. 2017.

# ## Dataset preparation
# 
# We need to test causal inference models with synthetic or semi-synthetic data because true causal effects in real-world data is unknown. Here we will generate popular random networks that simulate some real-world phenomena, visualize the networks, and assign treatments to each node randomly.
# 
# We will use a popular python library `networkx` for synthetic network generation.

# Although `networkx` allows us to visualize the graph using `nx.draw` function, it is preferable to use interactive visualizations for large networks. We will use the [`pyVis`](https://pyvis.readthedocs.io/en/latest/index.html) library for visualization. The `networkx` library was available by default. But we need to install `pyVis` first.

# In[1]:


get_ipython().system('pip install pyvis')


# In[2]:


import networkx as nx # Networkx library for generating networks and other analysis
from pyvis.network import Network # For interactive visualization
from IPython.core.display import display, HTML # For displaying on Colab notebook


# ## Barabasi-Albert (BA) preferential attachment network
# 
# First, let us explore Barabasi-Albert networks that simulate preferential attachment phenomena. These graphs simulate web and social networks where some units act as hubs with high degree while most of the units have low degree. These networks capture preferential attachment phenomena where units tend to connect with popular unit or influencer.

# In[3]:


num_nodes = 100 # Number of units/individuals (Use smaller values for visualization purpose. Otherwise it may not load.)
m = 5 # attachment parameter controls how many existing nodes should a new node connects to. 1<=m< num_nodes
seed = 0 # for reproducibility (which is very important for research)
ba_G = nx.barabasi_albert_graph(num_nodes, m, seed)


# You can explore different values of `m` or different seeds and try visualizing the network with display options.

# In[4]:


degrees = [val for (node, val) in ba_G.degree()] # Get degrees to plot node size based on degree
ba_net = Network(notebook=True,cdn_resources='remote', neighborhood_highlight=True)
for i,node in enumerate(ba_G.nodes):
  ba_net.add_node(node, label=node, value=degrees[i])
ba_net.add_edges(ba_G.edges)
ba_net.toggle_physics(False)
ba_net.show_buttons(filter_=['physics'])
ba_net.show('ba_net.html')
display(HTML('ba_net.html'))


# ### Watts-Strogatz (WS) small-world network
# 
# This network model captures small-world phenomena characterized by high clustering and short path lengths between nodes. These networks simulate real-world contact networks where individuals are mostly in touch with nearest neighbors.

# In[5]:


num_nodes=100
k = 10 # Each node is connected to k-nearest neighbors
p = 0.1 # random rewiring probability
seed = 1
ws_G = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p, seed=seed)


# In[6]:


ws_net = Network(notebook=True,cdn_resources='remote', neighborhood_highlight=True)
for i,node in enumerate(ws_G.nodes):
  ws_net.add_node(node, label=node, value=node) # node size controlled by id (notice similar size nodes connected for lower p)
ws_net.add_edges(ws_G.edges)
ws_net.toggle_physics(False)
ws_net.show_buttons(filter_=['physics'])
ws_net.show('ws_net.html')
display(HTML('ws_net.html'))


# ### Erdos-Renyi (ER) random network

# Try generating Erdos-Renyi (ER) random graph (network). ER graphs connects two edges with given probability. We need to input the number of nodes and probability of edge between two nodes to generate ER networks.

# In[7]:


# Use nx.erdos_renyi_graph(n=num_nodes, p=edge_prob) function


# In[8]:


# copy paste previous code to visualize and modify accordingly
# Plot size of node based on degree. Any observation on degree distribution?


# ### Node-level randomization
# 
# Let us randomly assign binary treatments to nodes in a BA network and visualize it. We will use `numpy` library for generating data. 

# In[143]:


import numpy as np
num_nodes = 100
m = 1
seed = 0
ba_G = nx.barabasi_albert_graph(n=num_nodes, m=m, seed=seed)
np.random.seed(seed)
treatments = np.random.binomial(1, p=0.5, size=len(ba_G)).tolist() # Equivalent to coin toss for each node


# In[144]:


degrees = [val for (node, val) in ba_G.degree()] # Get degrees to plot node size based on degree
ba_net = Network(notebook=True,cdn_resources='remote', neighborhood_highlight=True)
for i,node in enumerate(ba_G.nodes):
  ba_net.add_node(node, title=str(treatments[i]), value=degrees[i], group=treatments[i])# group based on treatment assignment
ba_net.add_edges(ba_G.edges)
ba_net.toggle_physics(False)
ba_net.show_buttons(filter_=['physics'])
ba_net.show('ba_net.html')
display(HTML('ba_net.html'))


# ## Generating node attributes
# 
# Here, we see how we can generate features from different probability distributions.

# In[145]:


num_nodes = 3000 # Using larger network for preparing dataset (smaller networks for visualizatoin)
m = 1
seed = 0
ba_G = nx.barabasi_albert_graph(n=num_nodes, m=m, seed=seed)
np.random.seed(seed)
N = len(ba_G)
np.random.seed(seed)
treatments = np.random.binomial(1, p=0.5, size=N).tolist() # Equivalent to coin toss for each node
# Binary attribute drawn from Bernoulli distribution with parameter p (e.g., married or unmarried)
attr1 = np.random.binomial(1, p=0.6, size=N).reshape(N,1)
# Beta distribution for polarity (continuous between 0 and 1)
attr2 = np.random.beta(0.7, 0.7, size=N).reshape(N,1)
# Dirichlet distribution to get probs that sum to 1
probs = np.random.dirichlet([2,5,5])
# Categorical distribution (e.g., Gender: other, male , and female)
indices = np.random.choice(range(len(probs)), p=probs, size=N)
attr3 = np.zeros((N, len(probs))) # convert to one-hot format
attr3[range(N), indices]=1
attr3 = attr3.reshape(N, len(probs))
# Nominal distribution (e.g., age groups)
probs = np.random.dirichlet([5,5,5,5])
attr4 = np.random.choice(range(len(probs)), p=probs, size=N).reshape(N,1) # No need of one-hot
# Uniform distribution (continuous)
attr5 = np.random.uniform(1,5,size=N).reshape(N,1)
# Normal distribution (continuous)
attr6 = np.random.normal(loc=0.0, scale=1.0, size=N).reshape(N,1)
node_attrs = np.hstack((attr1, attr2, attr3, attr4, attr5, attr6)) # Node attributes


# You can visualize distribution of each attribute by changing the parameters.

# In[146]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(attr2) # change attr2 to other variables and see the distribution


# ### Generating edge attributes

# In[147]:


M = len(ba_G.edges)
# For simplicity consider uniform edge weights (i.e. only one edge attribute)
weight = np.random.uniform(0,10, size=M)
weight


# ## Understanding peer exposure
# 
# **How can we summarize treatment of other units?**

# In[148]:


# Let's calculate Peer exposure: Fraction of treated neighbors
peer_exposure = []
treatments = np.array(treatments)
for node in ba_G.nodes:
  neighbors = ba_G[node]
  neighbors_T = treatments[neighbors]
  neighbors_T = np.mean(neighbors_T)
  peer_exposure.append(neighbors_T)

peer_exposure = np.array(peer_exposure).reshape(N,1)
treatments = np.array(treatments).reshape(N,1)


# In[149]:


import pandas as pd
features = ['married', 'polarity', 'other', 'male', 'female', 'age_group', 'priors', 'salary']
cols = features + ['treatments', 'peer_exposure']
data = np.hstack((node_attrs, treatments, peer_exposure))
df = pd.DataFrame(data, columns=cols)
df


# In[150]:


sns.histplot(x='peer_exposure', data=df)


# **Some questions to think?**
# 
# >Does this imbalance in distribution affect our estimation?
# 
# >How can we solve it?

# ### Analysis of assumptions
# 
# Another important step in research is to analyze the dataset/experiment settings to see whether our assumptions are (mostly) statisfied.
# 
# Let's get an idea if the positivity assumption holds.
# 
# 

# In[151]:


sns.histplot(x='peer_exposure', data=df, hue='treatments')


# **Question: What is your conclusion about voilation of positivity based on above graph?**
# 
# 

# How about positivity for specific subpopulation?

# In[152]:


sns.histplot(x='peer_exposure', data=df[df['male']==1], hue='treatments')
plt.title('Male')
plt.show()
sns.histplot(x='peer_exposure', data=df[df['female']==1], hue='treatments')
plt.title('Female')
plt.show()
sns.histplot(x='peer_exposure', data=df[df['other']==1], hue='treatments')
plt.title('Other')
plt.show()


# ### Generating underlying outcomes

# In[153]:


# scaling and preventing divide by 0
def scale(vals, start=0, end=1):
  width = end - start
  vals = (vals - vals.min()) / max(vals.max() - vals.min(), 1)
  return vals*width + start

def agg_peers(G, vals):
  result = list(map(lambda x: np.mean(vals[G[x]]), G.nodes)) # getting mean values of peer features where G is graph
  return np.array(result)

seed = 0
np.random.seed(seed)

def get_outcomes(df, G, lagged_outcomes, exposure=None, seed=0):
  np.random.seed(seed)
  outcomes = 5 # constant coefficient (infection due to current environment condition)
  tau_T = -2 # tau_T is the effect due to unit's own treatment (treatment reduces infection)
  outcomes += tau_T*df['treatments']

  tau_N = -1 # tau_N is the effect due to neighors' treatment (peers' treatment further reduces infection)
  if exposure is None:
    peer_exposure = agg_peers(G, df['treatments'])
  else:
    peer_exposure = exposure

  outcomes += tau_N*peer_exposure

  outcomes += tau_N*peer_exposure*df['treatments'] # if a unit and neighbors' both treated, then additional protection

  ################################################################

  # outcomes += 1*scale(df['age_group']) # higher age group more vulnerable

  # outcomes += -0.5*scale(df['age_group'])*df['treatments'] # treatment more effective for higher age group

  # outcomes += 1*scale(df['priors']) # units with more prior health concern are more vulnerable
  # outcomes += 0.5*(1-scale(df['priors']))*df['treatments'] # treatment less effective for units with prior concern (e.g., low immunity)

  # outcomes += 1*(1-df['treatments'])*scale(agg_peers(G, lagged_outcomes)) # untreated individuals are vulnerable to contagious infection from peers
  ########################################################
  outcomes += np.random.normal(0, 0.1) # noise
  return outcomes

def simulate_outcome(df, G, timesteps=1, exposure=None, seed=0):
  lagged_outcomes = np.zeros(len(ba_G))
  outcomes = lagged_outcomes
  for i in range(timesteps):
    outcomes = get_outcomes(df, G, lagged_outcomes, exposure=exposure)
    lagged_outcomes = outcomes
  # outcomes = scale(outcomes, 0, 1)
  return outcomes

outcomes = simulate_outcome(df, ba_G, timesteps=1) # consider only one time step for now
outcomes


# In[154]:


ba_net = Network(notebook=True,cdn_resources='remote', neighborhood_highlight=True)
for i,node in enumerate(ba_G.nodes):
  ba_net.add_node(node, title=str(int(treatments[i])), value=outcomes[i], group=int(treatments[i]))# group based on treatment assignment and size based on outcome
ba_net.add_edges(ba_G.edges)
ba_net.toggle_physics(False)
ba_net.show_buttons(filter_=['physics'])
ba_net.show('ba_net.html')
display(HTML('ba_net.html'))


# ## Estimating causal effects
# 
# Now, we have measured outcomes. First, let us estimate how would be the value of average treatment effect if we had ignored network structure.

# In[155]:


df['Y'] = outcomes
display(df)
ate = df[df['treatments']==1]['Y'].mean() - df[df['treatments']==0]['Y'].mean()
print(f"ATE = {ate}")


# This estimation of average individual effect or average total effect is biased due to network interference. Let us first implement GNN-based INE estimator using pytorch library. `pytorch` is already installed in Google colab. We will install `torch_geometric` and supporting libraries needed for using graph neural networks layers.

# In[20]:


import torch

# !pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
get_ipython().system('pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html')
get_ipython().system('pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html')
get_ipython().system('pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html')
get_ipython().system('pip install git+https://github.com/pyg-team/pytorch_geometric.git')


# In[156]:


import torch.nn.functional as F
from torch_geometric.nn import GCNConv # Using Graph Convolution Network layer
from torch_sparse import SparseTensor


# ## Model
# Let's recall our model and implement it.
# 
# <img src="https://drive.google.com/uc?export=view&id=1u1985mKo2fugKWkNrBZ2zuwkfelkSQvi" height=400/>

# In[163]:


class INE_Estimator(torch.nn.Module):
  def __init__(self, num_features, num_hidden, in_layers=1, dropout=0, clip=0):
    super(INE_Estimator, self).__init__()
    self.hdim = num_hidden
    self.dropout = dropout
    self.in_layers = in_layers
    # define GNN
    self.gnn = torch.nn.ModuleList([GCNConv(num_features, num_hidden)])
    for i in range(in_layers-1):
      self.gnn.append(GCNConv(num_hidden, num_hidden))
    
    # define CPNN with common embedding layer followed by batch normalization
    self.embedding = torch.nn.Linear(self.hdim+1, self.hdim)
    # Then counterfactual predictors for treatment and control
    self.out_t0 = torch.nn.Linear(self.hdim, 1)
    self.out_t1 = torch.nn.Linear(self.hdim, 1)

    if clip > 0:
      modules = [self.embedding, self.out_t0, self.out_t1]
      for module in modules:
        for p in module.parameters():
          p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))


  def forward(self, A, features, treatments, exposure=None):
    edge_index = A.nonzero().t()
    # convert to sparse tensor for reproducibility. adj_t need for directed graphs
    adj_t = SparseTensor(row=edge_index[0], col=edge_index[1]).t()
    # pass through gnn
    h = F.relu(self.gnn[0](features, adj_t))
    h = F.dropout(h, self.dropout, self.training)
    for i in range(1, self.in_layers):
      h = F.relu(self.gnn[i](h, adj_t))
      h = F.dropout(h, self.dropout, self.training)
    
    # define exposure
    deg = A.sum(dim=1) + 1e-8
    if exposure is not None:
      exposure = exposure.to(A.device)
    else:
      exposure = (A*treatments).sum(dim=1).div(deg)
      exposure = exposure.view(-1,1)  

    # concat
    rep = torch.cat([h, exposure], dim=1)  

    # pass through embedding layer
    emb = F.relu(self.embedding(rep))

    # pass through predictors
    y0_pred = self.out_t0(emb).view(-1)
    y1_pred = self.out_t1(emb).view(-1)

    return y0_pred, y1_pred, emb


# ## Training and Hyperparameter tuning
# 
# We will use `ray` library to tune hyperparameter and select best configuration. We will divide units in network to 80% training set and 20% validation set.

# In[164]:


get_ipython().system('pip install ray')


# In[165]:


from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import os
from ray import tune
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['RAY_PICKLE_VERBOSE_DEBUG']='0'
os.environ['TUNE_MAX_PENDING_TRIALS_PG']='8'
os.environ['RAY_DISABLE_MEMORY_MONITOR']='1'


# In[166]:


def tune_and_train_gnn_model(config, A=None, Z=None, T=None,
                   Y=None, checkpoint_dir=None, enable_tune=True):
  model = INE_Estimator(Z.shape[1],
                        num_hidden=config.get('f_hid', 32),
                        in_layers=config.get('in_layers', 1),
                        dropout=config.get('dropout', 0.),
                        clip=config.get('clip', 0),
                      )
  # print(model)
  model = model.to(Z.device)
  epochs = config.get('epochs', 300)
  l2_lambda = config.get('l2_lambda', 1e-5)
  reg_lambda = config.get('reg_lambda', 0.1)
  lr = config.get('lr', 1e-1)
  optimizer = torch.optim.Adam(model.parameters(),
                                lr=lr,
                                weight_decay=l2_lambda) 
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                        step_size=config.get('lr_step_size',50),
                                        gamma=config.get('lr_step_gamma', 0.5))
  
  criterion = torch.nn.MSELoss()  # Define loss criterion.
  train_index, val_index = train_test_split(np.arange(Z.shape[0]), 
                                            test_size=config.get('num_val', 0.2),
                                            stratify=T.cpu().numpy(),
                                            random_state=9)
  data = Data(x=Z,y=Y, train_index=train_index, val_index=val_index).to(A.device)
  best_loss = float('inf')
  # early stopping
  eval_interval = config.get('eval_interval', 100)
  max_patience = config.get('max_patience', 10)
  patience = 0
  for epoch in range(epochs):
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      y0_pred, y1_pred, rep = model(A, Z, T)  # Perform a single forward pass.
      y_pred = torch.where(T > 0, y1_pred, y0_pred)
      pred_loss = criterion(y_pred[data.train_index], Y[data.train_index])  # Compute the loss solely based on the training nodes.
      # Regularization to smooth variance of predicted effects
      if epoch >= 150:
          smooth_coeff = torch.exp(-3*torch.var(y1_pred-y0_pred)).detach()
      else:
          smooth_coeff = 0.
      loss = pred_loss + reg_lambda*smooth_coeff*torch.var(y1_pred-y0_pred)
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      scheduler.step() # Update learning rate scheduler
      with torch.no_grad():
        model.eval()
        y0_pred, y1_pred, rep = model(A, Z, T)
        y_pred = torch.where(T > 0, y1_pred, y0_pred)
        loss_val = criterion(y_pred[data.val_index], Y[data.val_index])
        if loss_val < best_loss and epoch > 50:
            best_loss = loss_val
            patience = 0
        else:
            patience += 1

      if epoch % eval_interval == 0:
          if enable_tune:
              # print(f'Epoch: {epoch}, Train loss: {loss.item()}, Val loss: {loss_val.item()}')
              tune.report(loss=loss_val.item())
          else:
              print(f'Epoch: {epoch}, Train loss: {loss.item()}, Val loss: {loss_val.item()}')

      if patience == max_patience and epoch>149:
          break
  if not enable_tune:
      print(f'Epoch: {epoch}, Train loss: {loss.item()}, Val loss: {loss_val.item()}')
      return model
  else:
      tune.report(loss=loss_val.item())
      print(f'Epoch: {epoch}, Train loss: {loss.item()}, Val loss: {loss_val.item()}')
      with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
          path = os.path.join(checkpoint_dir, "checkpoint")
          torch.save(model.state_dict(), path)


# In[167]:


def train(A, Z, T, Y, epochs=300, lr=1e-1, lr_decay=0.5,
          lr_step=100, weight_decay=1e-5, num_val=0.2,
          f_hid=32, in_layers=1, max_patience=300, dropout=0.,
          clip=2., verbose=0, exp_name='ideal-workshop-ine', reg=1):
    reg_params = reg*np.array([0.1, 1.0])
    reg_params = np.unique(reg_params)
    params = {
        'reg_lambda': tune.grid_search(reg_params.tolist()),
        'lr':lr,
        'l2_lambda': weight_decay,
        'lr_step_size':lr_step,
        'lr_step_gamma':lr_decay,
        'clip': clip,
        'max_patience':max_patience, 'epochs':epochs,
        'in_layers':in_layers,
        'f_hid': f_hid, 'dropout': dropout,
        'num_val':num_val
    }
    num_samples=1
    time_budget_s = 3000
    trials = tune.run(tune.with_parameters(tune_and_train_gnn_model,
                                         A=A, Z=Z, T=T, Y=Y),
                    config=params,
                    metric="loss",
                    mode="min",
                    resources_per_trial={'cpu':1, 'gpu':0.3, 'memory':2*1024*1024*1024},
                    time_budget_s=time_budget_s,
                    verbose=verbose,
                    num_samples=num_samples,
                    name=exp_name,
                    max_concurrent_trials=4,
                   )
    return trials


# Our implementation of INE Estimator and training module looks ready. Let us prepare input data for training the model.

# In[168]:


# Get adjacency matrix from network
def adjacency_matrix(G):
  N = len(G)
  A = torch.zeros((N, N))
  indices = torch.LongTensor(list(G.edges))
  A[indices[:,0], indices[:,1]] = 1.
  A[indices[:,1], indices[:,0]] = 1.
  return A

A = torch.FloatTensor(adjacency_matrix(ba_G))
T = torch.FloatTensor(df['treatments'].values)
Y = torch.FloatTensor(df['Y'].values)
print(features)
Z = torch.FloatTensor(df[features].values)

device = 'cuda'
A, T, Y, Z = A.to(device), T.to(device), Y.to(device), Z.to(device)
A.shape, T.shape, Y.shape, Z.shape


# In[169]:


trials = train(A, Z, T, Y, reg=1)


# In[170]:


# best_model = tune_and_train_gnn_model(config={}, A=A, Z=Z,T=T, Y=Y, enable_tune=False)


# In[171]:


best_trial = trials.get_best_trial('loss', 'min', "last")
best_cfg = best_trial.config
print(best_cfg)
checkpoint_value = getattr(best_trial.checkpoint, "dir_or_data", None) or best_trial.checkpoint.value
checkpoint_path = os.path.join(checkpoint_value, "checkpoint")
model_state = torch.load(checkpoint_path)
best_model = INE_Estimator(Z.shape[1],
                      best_trial.config.get('f_hid', 32),
                      in_layers=best_trial.config.get('in_layers', 1),
                      dropout=best_trial.config.get('dropout', 0.),
                    )
best_model = best_model.to(Y.device)
# print(best_model)
best_model.load_state_dict(model_state)


# ## Evaluate policy

# In[172]:


eval_df = df.copy()
eval_df['treatments'] = 1.
ite_1 = simulate_outcome(eval_df, ba_G, exposure=np.zeros(len(ba_G)))
eval_df['treatments'] = 0.
ite_0 = simulate_outcome(eval_df, ba_G, exposure=np.zeros(len(ba_G)))

true_total_effects = ite_1 - ite_0


# In[175]:


print(f'True average insulated individual effects = {true_total_effects.mean()}')


# In[176]:


with torch.no_grad():
  ones = torch.ones_like(T).to(T.device)
  zeros = torch.zeros_like(T).to(T.device)
  _, y_all1, _ = best_model(A,Z,ones, exposure=zeros.view(-1,1))
  y_all0, _, _ = best_model(A,Z,zeros, exposure=zeros.view(-1,1))


# In[179]:


pred_total_effects = y_all1 - y_all0
print(f'Predicted average individual effects = {pred_total_effects.mean()}')


# In[180]:


ate = df[df['treatments']==1]['Y'].mean() - df[df['treatments']==0]['Y'].mean()
print(f"Baseline average effects ignoring network = {ate}")

