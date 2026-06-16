# The JEPA framework

The scope of this document is to provide an introduction to the framework of the *Joint-Embedding Predictive Architecture* (JEPA), covering at first the seminal ideas, and then the functional components of the state-of-the-art models. \
Finally, we will present our proposal for a new, stable and end-to-end trainable JEPA model.


## Contents
<!-- TOC -->
* [The JEPA framework](#the-jepa-framework)
  * [Contents](#contents)
  * [Contrastive Learning](#contrastive-learning)
  * [JEPA architecture](#jepa-architecture)
  * [AV-JEPA](#av-jepa)
  * [References](#references)
<!-- TOC -->



## Contrastive Learning
Contrastive learning is a self-supervised learning technique whose aim is to train models able to distinguish between similar and dissimilar data inputs. 

The general framework consists of three main steps. \
In the first step, a raw data sample (the **anchor point**) is process by data-augmentation methods; since the resulting new data will be in some way "similar" to the anchor, they are called **positive examples**. All the remaining data, instead, consists of **negative** (or dissimilar) **examples**. \
The second step is an *encoding* of the anchor to a higher-dimensional space. A *projection head* then projects this embedded representation to a low-dimensional space. This step is done for the anchor and for all the positive and negative examples. \
Finally, in the third step, a **contrastive loss** computes the distances between similar and dissimilar data in this latent space.

The encoder is usually a ResNet (in *SimCLR*, *MoCo*, *BYOL*) or recently a Vision Transformer (*DINO*). 

The way positive and negative examples are chosen is extremely important. 
Usually, the number of negatives provided to the network is much higher than the number of positives. \
In many architectures, a large number of negative sample embeddings is stored into a "bank". The loss is computed w.r.t. to this batch of embeddings. However, since the updates also regard the encoder, the stored embeddings will become soon useless. \
To prevent this, the MoCo architecture defines a momentum encoder, i.e., a copy of the data encoder. The momentum one is responsible for the embeddings of negatives, and its weights $\theta_k$ using an exponential smoothing w.r.t. the data encoder weights $\theta_q$:
$$
    \theta_k = \alpha \theta_k + (1-\alpha) \theta_q
$$

The definition of the loss is mostly case-dependent. \
Some common solutions are:
- **Noise-contrastive estimation (NCE)**
$$
    \mathcal{L}(\theta) = \sum_{i=1}^n \log \frac{p_\theta(y_i | x)}{p_\theta(y_i | x) + kp_n(y_i|x)} + \sum_{j=1}^m \log \frac{p_n(y_j | x)}{p_\theta(y_j | x) + kp_n(y_j|x)}
$$
    where $k$ is the number of negative samples, $p_n$ their distribution, $n$ the number of positives and $p_\theta$ their distribution. \
    This loss reduces the problem to a binary classification (similar/dissimilar).
- **Info Noise-contrastive estimation (NCE)**
$$
    \mathcal{L}(\theta) = -\log \frac{\exp(\frac{q \cdot k^+}{\tau})}{\sum_{i=0}^K \exp{\frac{q \cdot k_i}{\tau}} } 
$$
    where $q$ is the embedding of the ancor, $k^+$ is the embedding of the (unique) positive sample and $K$ is the set of the (multiple) negatives. \
    Its goal is to maximize the mutual information between positive samples pairs while minimizing the mutual information between negative samples pairs. 
- **Cross-entropy loss**
$$
    \mathcal{L} = -\log \frac{1}{\mathbf{B}} \sum_j^\mathbf{B} \sum_i^n y_i \log y_j
$$
    where $\mathbf{B}$ is the batch size.


## Distillation Learning


## JEPA architecture

## AV-JEPA

## References
