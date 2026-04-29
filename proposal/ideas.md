Experiments:
-We could try to predict the accuracy gap between architectures (like) $\Delta =  \text{acc}{\text{FAGCN}} - \text{acc}{\text{MLP}}$  from the SLP.
- Can try to find a high-pass only model. This may be a bit artificial. But we could have a GCN model where the propagation operator is flipped, i.e. 
    - GCN: $\hat{A} = \tilde{D}^{-1/2}(A + I)\tilde{D}^{-1/2}, \qquad h(\lambda) = (1 - \lambda)^L$
    - GCN-HF: $L = I - \hat{A}, \qquad h(\lambda) = \lambda^L$
    - This could be justified with this paper "Interpreting and Unifying Graph Neural Networks with an Optimization Framework (https://dl.acm.org/doi/epdf/10.1145/3442381.3449953)" where they state in the abstract "we further develop two novel objective func-tions considering adjustable graph kernels showing low-pass orhigh-pass filtering capabilities respectively."
- If we think is what is happening is that FAGCN can just approximate what GCN does for homophilic graphs, then we could write out some theory based on how FAGCN works on how it can adapt to just use low-frequency signal in our write-up.
- Try to see if our claim "a linear spectral label profile indicates no graph–label alignment, so an MLP should work best" holds true.
- Instead of thinking about classification over architectures, we could do regression on the accuracy gap between architectures, based on the SLP.
- Question for Kayden: Why do we think that the SLP will outperform the adjusted homophily metric, and outperform in what way?
- Certainly we want to try this on the big datasets once we validate in these torch_geometric.datasets what we want to run on bigger graphs. And on SBM? Also we currently haven
- Ege: probably in spectral-profling-gnns/spectral we run the other set of experiments for spectral gnns. ANd we need to do the Chebyshev polynomial ideas.

Draft write-up structure:
- We need a 3 pages x 4 page/person final write up. Keep in mind. 
- 

Presentations (when 5/12 comes around), approx 5m per group member.
- One for how to structure this is to have:
- Our project is about using spectral information of input graphs to GNNs as a cheap approx ... (20s)
- 5 minute intro on how GNNs work, how message-passing works for node-classification, the metric we introduce and what the intuition is for this metric, why spectral information should be helpful. Some examples of our CDF graphs for different datasets and how this captures what it is supposed to be capturing. Some results easing the worries about induced subgraph vs. full graph (Dries)
- 5 minutes on results on results with spatial GNNs (Kayden), potentially a slide on the theory of how an FAGCN collapses to a GCN for homophilic data, 
- 5 minutes on results with spectral GNNs (Ege)
- The extra ideas we tried (APPNP for example). Explaining that we are working on cleaning this up for a workshop submission. (Lia)
- 3 take-aways from our project (Lia)