import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_sparse
from torch import FloatTensor
from copy import deepcopy
from pathlib import Path
import numpy as np

# standard GCN setup

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))
        self.dropout = dropout 
        

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class H2GCN(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            hidden_dim: int,
            class_dim: int,
            k: int = 2,
            dropout: float = 0.5,
            use_relu: bool = True
    ):
        super(H2GCN, self).__init__()
        self.dropout = dropout
        self.k = k
        self.act = F.relu if use_relu else lambda x: x
        self.use_relu = use_relu
        self.w_embed = nn.Parameter(
            torch.zeros(size=(feat_dim, hidden_dim)),
            requires_grad=True
        )
        self.w_classify = nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, class_dim)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w_classify)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

#     @classmethod
#     def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
#         n = adj.size(0)
#         d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
#         d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
#         d_tiled = torch.sparse_coo_tensor(
#             indices=[list(range(n)), list(range(n))],
#             values=d_diag,
#             size=(n, n)
#         )
#         return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    
    @classmethod
    def _adj_norm(cls, adj):
        csp = adj.coalesce()
        idx = csp.indices()
        val = csp.values()

        deg = torch.sparse.sum(adj, dim=1).to_dense()      # length-n, includes zeros
        d_inv_sqrt = deg.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0            # isolated nodes -> 0

        new_val = val * d_inv_sqrt[idx[0]] * d_inv_sqrt[idx[1]]
        return torch.sparse_coo_tensor(idx, new_val, csp.shape).coalesce()


    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).to(device)
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._indicator(self._spspmm(adj, adj)) - adj - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)

    def forward(self, x, edge_index): 
        if not self.initialized:
            n = x.size(0)
            adj = torch.sparse_coo_tensor(
                indices=edge_index,
                values=torch.ones(edge_index.size(1), device=edge_index.device),
                size=(n, n),
            ).coalesce() # coalesce to sum duplicate edges
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [self.act(torch.mm(x, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(torch.cat([r1, r2], dim=1))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        return torch.mm(r_final, self.w_classify)

# load 80/10/10 train, val, test masks from spatial/masks/

def load_masks(name):
    p = Path(__file__).resolve().parent / 'masks' / f'{name}.pt'
    m = torch.load(p)
    return m['train_mask'], m['val_mask'], m['test_mask']

# train one model. val early stopping. return test acc

def fit(model, data, masks, lr=1e-2, weight_decay=5e-4, max_epochs=300, patience=20, device='cuda'):
    tr_mask, va_mask, te_mask = (m.to(device) for m in masks)
    model = model.to(device)
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_va = -np.inf
    best_state = None
    patience_counter = 0
    val_history = []

    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[tr_mask], data.y[tr_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            va_acc = (out.argmax(dim=1)[va_mask] == data.y[va_mask]).float().mean().item()
            val_history.append(va_acc)
            
        if va_acc > best_va:
            best_va = va_acc
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        te_acc = (out.argmax(dim=1)[te_mask] == data.y[te_mask]).float().mean().item()

    return te_acc, val_history

# run for all datasets

def train_sweep(datasets, depths=(1, 2), hidden_dims=(64,), dropouts=(0.0, 0.25, 0.5),
                weight_decays=(0, 1e-5, 5e-4), use_relu=(True, False), n_runs=3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for name, d in datasets:
        g = d[0]
        masks = load_masks(name)
        results[name] = {}
        for depth in depths:
            for hidden_dim in hidden_dims:
                for dropout in dropouts:
                    for wd in weight_decays:
                        for relu_flag in use_relu:
                            accs = []
                            val_histories = []

                            for i in range(n_runs):
                                torch.manual_seed(i)
                                np.random.seed(i)

                                model = H2GCN(
                                    feat_dim=g.x.size(1),
                                    hidden_dim=hidden_dim,
                                    class_dim=d.num_classes,
                                    k=depth,
                                    dropout=dropout,
                                    use_relu=relu_flag,
                                )

                                acc, val_history = fit(model, g, masks, weight_decay=wd, device=device)
                                accs.append(acc)
                                val_histories.append(val_history)

                            results[name][(depth, hidden_dim, dropout, wd, relu_flag)] = (accs, val_histories)

    return results


