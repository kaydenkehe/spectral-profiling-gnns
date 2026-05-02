Experiments:
-We could try to predict the accuracy gap between architectures (like) $\Delta =  \text{acc}{\text{FAGCN}} - \text{acc}{\text{MLP}}$  from the SLP.
- Can try to find a high-pass only model. This may be a bit artificial. But we could have a GCN model where the propagation operator is flipped, i.e. 
    - GCN: $\hat{A} = \tilde{D}^{-1/2}(A + I)\tilde{D}^{-1/2}, \qquad h(\lambda) = (1 - \lambda)^L$
    - GCN-HF: $L = I - \hat{A}, \qquad h(\lambda) = \lambda^L$
    - This could be justified with this paper "Interpreting and Unifying Graph Neural Networks with an Optimization Framework (https://dl.acm.org/doi/epdf/10.1145/3442381.3449953)" where they state in the abstract "we further develop two novel objective func-tions considering adjustable graph kernels showing low-pass orhigh-pass filtering capabilities respectively."
    - Run shows that indeed the heterophilic graphs are now getting higher classification accuracy ?
- If we think is what is happening is that FAGCN can just approximate what GCN does for homophilic graphs, then we could write out some theory based on how FAGCN works on how it can adapt to just use low-frequency signal in our write-up.
- Try to see if our claim "a linear spectral label profile indicates no graph–label alignment, so an MLP should work best" holds true.
- Instead of thinking about classification over architectures, we could do regression on the accuracy gap between architectures, based on the SLP.
- Question for Kayden: Why do we think that the SLP will outperform the adjusted homophily metric for predicting architecture (pursuant to whatsapp discussion on 4/29)
- Certainly we want to try this on the big datasets once we validate in these torch_geometric.datasets what we want to run on bigger graphs. And on SBM. 
    - On SBM: We can try to create block graphs where we want to compare FAGCN to MLP Or we can try creating graphs where we expect to have a clean MLP vs. HFGCN vs GCN.
    - We want to create synthetic examples that can validate RQ1, graphs where what architecture is optimal depends on shape of the SLP not just the h. Claude has some ideas on this, and Im sure we can too.
    - basically we can generate homophilic
- Also we currently haven't done what we said in the proposal about 'induced subgraph of training points'. (Kayden working on this now.)
- This paper (https://openreview.net/pdf?id=m7PIJWOdlY) that we cite for RQ1 proposes a new metric for homophily. Another way to spin our SLP is that some features of SLP are a novel, more fine-grained homophily metric. We would have to think about how to justify that claim exactly.
- Ege: probably in spectral-profling-gnns/spectral we run the other set of experiments for spectral gnns. ANd we need to do the Chebyshev polynomial ideas (maybe just bins for now).
- I don't quite get the transductive message passing idea. 

Draft write-up structure:
- We need a 3 pages x 4 page/person final write up. Keep in mind. 
- We should try a section on how the high-pass vs low-pass filter idea works theoretically maybe?
- Introduction, unlike prposal where we just talk about the value of finding better GNN architecture based on spectral info, cheaply, and introduce the SLP. Our contributions are X Y Z
- Background: GNNs as spectral filters, homophily metrics, diagnostic based arch selection.
- SLP section: defn and show-off of graphs computing it and comparing it to h on some datasets. The theory work as a little theorem from the google doc on it being richer than homophily metric. Then computation and limitation of it O(n^3) for eigendecomposition, but still doable. ANd a note on robustness with slp_comparison.png
- RQ1 and Experiments on spatial gnns
- experiments on spatial gnns using SBMs
- Discussion 1: pareto domination of FAGCN, limitation of 13 graphs, then idea of HFGCN vs GCN.
- RQ2 and Experiments spectral GNN with basis selection idea.
- Discussion 2 of those experiments

Presentations (when 5/12 comes around), approx 5m per group member.
- One for how to structure this is to have:
- Our project is about using spectral information of input graphs to GNNs as a cheap approx ... (20s)
- 5 minute intro on how GNNs work, how message-passing works for node-classification, the metric we introduce and what the intuition is for this metric, why spectral information should be helpful. Some examples of our CDF graphs for different datasets and how this captures what it is supposed to be capturing. Some results easing the worries about induced subgraph vs. full graph (Dries)
- 5 minutes on results on results with spatial GNNs (Kayden), potentially a slide on the theory of how an FAGCN collapses to a GCN for homophilic data, 
- 5 minutes on results with spectral GNNs (Ege)
- The extra ideas we tried (APPNP for example). Explaining that we are working on cleaning this up for a workshop submission. (Lia)
- 3 take-aways from our project (Lia)