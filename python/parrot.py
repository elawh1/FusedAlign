import time
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import time
import psutil
import os
import numpy as np
import threading


from PlanetAlign.data import Dataset
from PlanetAlign.utils import get_anchor_pairs, get_batch_rwr_scores, get_normalized_neg_exp_dist
from PlanetAlign.metrics import hits_ks_scores, mrr_score
from .base_model import BaseModel


def write_to_file(var, var_name, out_dir=None):
    """
    Saves a numeric tensor/array-like to <var_name>.txt (optionally in out_dir)
    in COLUMN-MAJOR (Fortran-style) order — suitable for direct reading into
    C++ std::vector<double> that stores column-major matrices.
    """
    import os, numpy as np, torch

    # move torch tensor -> numpy double
    if isinstance(var, torch.Tensor):
        var = var.detach().cpu().numpy()
    var = np.asarray(var, dtype=np.float64)

    # create folder
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"{var_name}.txt")
    else:
        filename = f"{var_name}.txt"

    # flatten in column-major order (order='F')
    flat_col_major = var.flatten(order='F')

    # save one element per line (easiest for C++ fscanf / ifstream)
    np.savetxt(filename, flat_col_major, fmt="%.17g")

    print(f"Saved variable '{var_name}' in column-major order to {filename}")

def save_for_cpp(graph1, graph2, anchor_links, test_pairs, use_attr, out_dir="outputs-douban"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Convert edge indices or attributes to flat numeric arrays
    np.savetxt(f"{out_dir}/graph1.txt", graph1.edge_index.T.cpu().numpy(), fmt="%.10f")
    np.savetxt(f"{out_dir}/graph2.txt", graph2.edge_index.T.cpu().numpy(), fmt="%.10f")
    np.savetxt(f"{out_dir}/anchor_links.txt", np.array(anchor_links, dtype=float), fmt="%.10f")
    np.savetxt(f"{out_dir}/test_pairs.txt", np.array(test_pairs, dtype=float), fmt="%.10f")

    if use_attr :
        np.savetxt(f"{out_dir}/graph1_feat.txt", graph1.x.T.cpu().numpy(), fmt="%.10f")
        np.savetxt(f"{out_dir}/graph2_feat.txt", graph2.x.T.cpu().numpy(), fmt="%.10f")
        
def compute_errors(a: torch.Tensor, b: torch.Tensor):
    """
    a, b: PyTorch tensors of the same shape
    returns: (max_abs_err, max_rel_err)
    """
    diff = torch.abs(a - b)
    max_abs_err = diff.max().item()

    denom = torch.maximum(torch.maximum(a.abs(), b.abs()), torch.tensor(1e-12))
    rel = diff / denom
    max_rel_err = rel.max().item()

    return max_abs_err, max_rel_err
    
class PARROT(BaseModel):
    """OT-based method PARROT for pairwise network alignment.
    PARROT is proposed by the paper "`PARROT: Position-Aware Regularized Optimal Transport for Network Alignment. <https://dl.acm.org/doi/abs/10.1145/3543507.3583357>`_"
    in WWW 2023.

    Parameters
    ----------
    alpha: float, optional
        The hyparameter balancing the costs computed by RWR and node attributes. Default is 0.5.
    rwr_restart_prob: float, optional
        The restart probability for the random walk with restart (RWR). Default is 0.15.
    gamma: float, optional
        The discount factor of RWR. Default is 0.75.
    lambda_p: float, optional
        The weight of the proximal point term. Default is 5e-4.
    lambda_e: float, optional
        The weight of the edge consistency term. Default is 1e-5.
    lambda_n: float, optional
        The weight of the neighborhood consistency term. Default is 5e-3.
    lambda_a: float, optional
        The weight of the alignment preference term. Default is 5e-4.
    dtype: torch.dtype, optional
        Data type of the tensors, choose from torch.float32 or torch.float64. Default is torch.float32.
    """

    def __init__(self,
                 alpha: float = 0.5,
                 rwr_restart_prob: float = 0.15,
                 gamma: float = 0.75,
                 lambda_p: float = 5e-4,
                 lambda_e: float = 1e-5,
                 lambda_n: float = 5e-3,
                 lambda_a: float = 5e-4,
                 dtype: torch.dtype = torch.float32):
        super(PARROT, self).__init__(dtype=dtype)

        self.alpha = alpha
        self.rwr_restart_prob = rwr_restart_prob
        self.gamma = gamma
        self.lambda_p = lambda_p
        self.lambda_e = lambda_e
        self.lambda_n = lambda_n
        self.lambda_a = lambda_a

    def train(self,
              dataset: Dataset,
              gid1: int,
              gid2: int,
              outputs="douban",
              use_attr: bool = True,
              max_iters_sep_rwr: int = 100,
              max_iters_prod_rwr: int = 50,
              inner_iters: int = 5,
              outer_iters: int = 10,
              save_log: bool = True,
              verbose: bool = True):
        """
        Parameters
        ----------
        dataset : Dataset
            The dataset containing graphs to be aligned and the training/test data.
        gid1 : int
            The graph id of the first graph to be aligned.
        gid2 : int
            The graph id of the second graph to be aligned.
        use_attr : bool, optional
            Flag for using attributes. Default is True.
        max_iters_sep_rwr : int, optional
            Maximum number of iterations for separate RWR. Default is 100.
        max_iters_prod_rwr : int, optional
            Maximum number of iterations for product RWR. Default is 50.
        inner_iters : int, optional
            Number of inner iterations for the proximal point optimization. Default is 5.
        outer_iters : int, optional
            Number of outer iterations for the proximal point optimization. Default is 10.
        save_log : bool, optional
            Flag for saving the logs. Default is True.
        verbose : bool, optional
            Flag for printing the logs. Default is True.
        """

        self.check_inputs(dataset, (gid1, gid2), plain_method=False, use_attr=use_attr, pairwise=True, supervised=True)

        # thread info
        print("number of threads from threading : ", threading.active_count())
        print("number cores : ", os.cpu_count())
        print("OMP # threads : ", os.environ.get("OMP_NUM_THREADS"))
        print("MKL # threads : ", os.environ.get("MKL_NUM_THREADS"))

        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        
        print("PyTorch intra-op threads:", torch.get_num_threads())
        print("PyTorch inter-op threads:", torch.get_num_interop_threads())
        
        
        logger = self.init_training_logger(dataset, use_attr, additional_headers=['memory', 'infer_time'], save_log=save_log)
        process = psutil.Process(os.getpid())

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        anchor_links = get_anchor_pairs(dataset.train_data, gid1, gid2)
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)
        print(graph1)
        print(hasattr(graph1, "x"), graph1.x)
        save_for_cpp(graph1, graph2, anchor_links, test_pairs, use_attr)
        # torch.save(graph1.edge_index.cpu(), "outputs/graph1_edges.txt")
        # np.savetxt("outputs/graph1_edges.txt", graph1.edge_index.cpu().T.numpy(), fmt="%d")
        # torch.save(graph2.edge_index.cpu(), "outputs/graph2_edges.txt")
        # np.savetxt("outputs/graph2_edges.txt", graph2.edge_index.cpu().T.numpy(), fmt="%d")
        # np.savetxt("outputs/anchor_links.txt", anchor_links.cpu().numpy(), fmt="%d")
        
    
        # Compute transport cost matrices
        inf_t0 = time.time()
        cross_dist, intra_dist1, intra_dist2 = self.get_transport_cost(graph1, graph2, anchor_links, max_iters_sep_rwr, max_iters_prod_rwr, use_attr, verbose, outputs)
        cost_time = time.time() - inf_t0

        # write_to_file(cross_dist, "cross_dist", out_dir=outputs)
        # write_to_file(intra_dist1, "intra_dist1", out_dir=outputs)
        # write_to_file(intra_dist2, "intra_dist2", out_dir=outputs)

        # Constraint proximal point optimization
        S, logger = self.con_prox_pt_opt(graph1, graph2, cross_dist, intra_dist1, intra_dist2, inner_iters, outer_iters, anchor_links, test_pairs, logger, cost_time, process, verbose, outputs)
        end_time = time.time() - inf_t0
        print('total time to run algorithm in seconds: ', end_time)
        return S, logger
    
    def con_prox_pt_opt(self, graph1, graph2, cross_dist, intra_dist1, intra_dist2, inner_iters, outer_iters, anchor_links, test_pairs, logger, cost_time, process, verbose, outputs):
        n1, n2 = graph1.num_nodes, graph2.num_nodes
        infer_time = cost_time
        # print('intra_dist1 max {:.6f} intra_dist1 min {:.6f} '.format(torch.max(intra_dist1), torch.min(intra_dist1)))
        # max_abs, max_rel = compute_errors(intra_dist1) 
        # print('intra_dist1 max {:.6f} intra_dist1 min {:.6f} '.format(max_abs, max_rel))
        print('intra_dist2 max {:.6f} intra_dist2 min {:.6f} '.format(torch.max(intra_dist2), torch.min(intra_dist2)))
        # Normalize adjacency matrices
        adj1 = to_dense_adj(graph1.edge_index, max_num_nodes=graph1.num_nodes).squeeze().to(self.dtype).to(self.device)
        adj2 = to_dense_adj(graph2.edge_index, max_num_nodes=graph2.num_nodes).squeeze().to(self.dtype).to(self.device)
        adj1[torch.where(~adj1.sum(1).bool())] = torch.ones(n1, dtype=self.dtype).to(self.device)
        adj2[torch.where(~adj2.sum(1).bool())] = torch.ones(n2, dtype=self.dtype).to(self.device)
        row_norm_adj1 = F.normalize(adj1, p=1, dim=1)
        row_norm_adj2 = F.normalize(adj2, p=1, dim=1)
        # write_to_file(row_norm_adj1, "row_norm_adj1", out_dir=outputs)
        # write_to_file(row_norm_adj2, "row_norm_adj2", out_dir=outputs)

        # Constraint proximal point iterations
        lambda_total = self.lambda_n + self.lambda_a + self.lambda_p
        lambda_e = self.lambda_e * n1 * n2

        one_vec_n1 = torch.ones((n1, 1), dtype=self.dtype).to(self.device)
        one_vec_n2 = torch.ones((n2, 1), dtype=self.dtype).to(self.device)

        a = one_vec_n1 / n1
        b = one_vec_n2.T / n2
        r = one_vec_n1 / n1
        c = one_vec_n2.T / n2

        S = torch.ones((n1, n2), dtype=self.dtype).to(self.device) / (n1 * n2)
        H = torch.zeros((n1, n2), dtype=self.dtype).to(self.device) + 1e-6
        H[anchor_links[:, 0], anchor_links[:, 1]] = 1

        def mina(H_in, epsilon):
            in_a = torch.ones((n1, 1), dtype=self.dtype).to(self.device) / n1
            return -epsilon * torch.log(torch.sum(in_a * torch.exp(-H_in / epsilon), dim=0, keepdim=True))

        def minb(H_in, epsilon):
            in_b = torch.ones((1, n2), dtype=self.dtype).to(self.device) / n2
            return -epsilon * torch.log(torch.sum(in_b * torch.exp(-H_in / epsilon), dim=1, keepdim=True))

        def minaa(H_in, epsilon):
            return mina(H_in - torch.min(H_in, dim=0).values.view(1, -1), epsilon) + torch.min(H_in, dim=0).values.view(
                1, -1)

        def minbb(H_in, epsilon):
            return minb(H_in - torch.min(H_in, dim=1).values.view(-1, 1), epsilon) + torch.min(H_in, dim=1).values.view(
                -1, 1)

        print('intra_dist1 max {:.6f} intra_dist1 min {:.6f} '.format(torch.max(intra_dist1), torch.min(intra_dist1)))
        print('intra_dist2 max {:.6f} intra_dist2 min {:.6f} '.format(torch.max(intra_dist2), torch.min(intra_dist2)))
        L_fixed = (intra_dist1 ** 2) @ r @ one_vec_n2.T + one_vec_n1 @ c @ (intra_dist2 ** 2).T
        print('L_fix max {:.6f} L_fix min {:.6f} '.format(torch.max(L_fixed), torch.min(L_fixed)))
        # write_to_file(L_fixed, "L_fixed.txt", out_dir=outputs)
        logs = {}

        C_old = None
        if verbose:
            print('Starting constraint proximal point iteration')
        for i in range(outer_iters):
            if verbose:
                print(f'Iteration {i + 1}/{outer_iters}:', end=" ")
            t0 = time.time()
            logs[i] = {}
            S_old = torch.clone(S)

            L = L_fixed - 2 * intra_dist1 @ S @ intra_dist2.T
            print('outer iter {} L max {:.6f} L min {:.6f}'.format(i, torch.max(L), torch.min(L)))
            # write_to_file(L, "L_outer_iter_"+str(i), out_dir=outputs)
            C = cross_dist + lambda_e * L - self.lambda_n * torch.log(
                row_norm_adj1.T @ S @ row_norm_adj2) - self.lambda_a * torch.log(H)
            print('outer iter {} C max {:.6f} C min {:.6f}'.format(i, torch.max(C), torch.min(C)))
            # write_to_file(C, "C_outer_iter_"+str(i), out_dir=outputs)
            if C_old is None:
                C_old = C
            else:
                W_old = torch.sum(S * C_old)
                W = torch.sum(S * C)
                if W <= W_old:
                    C_old = C
                else:
                    C = C_old

            Cost = C - self.lambda_p * torch.log(S)
            if verbose:
                print('sinkhorn iteration', end=" ")
            for j in range(inner_iters):
                a = minaa(Cost - b, lambda_total)
                b = minbb(Cost - a, lambda_total)
                if verbose:
                    print(j + 1, end=" ")
                    
            print('outer iter {} a max {:.6f} a min {:.6f}'.format(i, torch.max(a), torch.min(a)))
            # write_to_file(a, "a_outer_iter_"+str(i), out_dir=outputs)
            print('outer iter {} b max {:.6f} b min {:.6f}'.format(i, torch.max(b), torch.min(b)))
            # write_to_file(b, "b_outer_iter_"+str(i), out_dir=outputs)
            print('outer iter {} (a + b - Cost) max {:.6f} (a + b - Cost) min {:.6f}'.format(i, torch.max((a + b - Cost)), torch.min((a + b - Cost))))
            print('outer iter {} (a + b - Cost) / lambda_total max {:.6f} (a + b - Cost) / lambda_total min {:.6f}'.format(i, torch.max((a + b - Cost) / lambda_total), torch.min((a + b - Cost) / lambda_total)))

            big_S = r * torch.exp((a + b - Cost) / lambda_total) * c
            print('outer iter {} bigger S max {:.6f} bigger S min {:.6f}'.format(i, torch.max(big_S), torch.min(big_S)))
            S = 0.05 * S_old + 0.95 * r * torch.exp((a + b - Cost) / lambda_total) * c
            print('outer iter {} S max {:.6f} S min {:.6f}'.format(i, torch.max(S), torch.min(S)))

            diff_S = torch.sum(torch.abs(S - S_old))
            t1 = time.time()
            infer_time += t1 - t0
            if verbose:
                print('done, time spent: {:.2f}s'.format(t1 - t0))

            hits = hits_ks_scores(S, test_pairs, mode='mean')
            mrr = mrr_score(S, test_pairs, mode='mean')
            mem_gb = process.memory_info().rss / 1024 ** 3
            logger.log(epoch=i+1,
                       loss=diff_S.item(),
                       epoch_time=t1-t0,
                       hits=hits,
                       mrr=mrr,
                       memory=round(mem_gb, 4),
                       infer_time=round(infer_time, 4),
                       verbose=verbose)

        # write_to_file(S, "S", out_dir=outputs)
        return S, logger

    def get_transport_cost(self, graph1, graph2, anchor_links, max_iters_sep_rwr, max_iters_prod_rwr, use_attr, verbose, outputs):
        # Compute position-aware cross-network transport cost
        if verbose:
            print('Computing separate RWR scores ...', end=" ")
        t0 = time.time()
        t0_0 = time.time()
        rwr_emb1 = get_batch_rwr_scores(graph1, anchor_links[:, 0],
                                        restart_prob=self.rwr_restart_prob,
                                        max_iters=max_iters_sep_rwr,
                                        connect_isolated=True,
                                        device=self.device).to(self.dtype)
        t0_1 = time.time()
        print('rwr_emb1 computed in ', t0_1-t0_0, ' seconds')
        # write_to_file(rwr_emb1, "rwr_emb1", out_dir=outputs)
        t1_0 = time.time()
        rwr_emb2 = get_batch_rwr_scores(graph2, anchor_links[:, 1],
                                        restart_prob=self.rwr_restart_prob,
                                        max_iters=max_iters_sep_rwr,
                                        connect_isolated=True,
                                        device=self.device).to(self.dtype)
        t1_1 = time.time()
        print('rwr_emb2 computed', t1_1-t1_0, ' seconds')
        # write_to_file(rwr_emb2, "rwr_emb2", out_dir=outputs)
        
        t1 = time.time()
        if verbose:
            print(f'done, time spent: {t1 - t0:.2f}s')

        cross_rwr_dist = get_normalized_neg_exp_dist(rwr_emb1, rwr_emb2, device=self.device).to(self.dtype)
        if use_attr:
            cross_attr_dist = get_normalized_neg_exp_dist(graph1.x, graph2.x, device=self.device).to(self.dtype)
        else:
            cross_attr_dist = get_normalized_neg_exp_dist(rwr_emb1, rwr_emb2, device=self.device).to(self.dtype)
        cross_node_dist = self.alpha * cross_rwr_dist + cross_attr_dist
        cross_node_dist[anchor_links[:, 0], anchor_links[:, 1]] = 0

        adj1 = to_dense_adj(graph1.edge_index, max_num_nodes=graph1.num_nodes).squeeze().to(self.dtype).to(self.device)
        adj2 = to_dense_adj(graph2.edge_index, max_num_nodes=graph2.num_nodes).squeeze().to(self.dtype).to(self.device)
        adj1[torch.where(~adj1.sum(1).bool())] = torch.ones(graph1.num_nodes, dtype=self.dtype).to(self.device)
        adj2[torch.where(~adj2.sum(1).bool())] = torch.ones(graph2.num_nodes, dtype=self.dtype).to(self.device)

        if verbose:
            print('Computing product RWR scores ...', end=" ")
        t0 = time.time()
        cross_dist = self.get_product_rwr_mat(adj1, adj2, in_cross_dist=cross_node_dist, max_iters=max_iters_prod_rwr)
        t1 = time.time()
        if verbose:
            print(f'done, time spent: {t1 - t0:.2f}s')

        # Compute intra-network transport cost
        if use_attr:
            intra_dist1 = adj1 * get_normalized_neg_exp_dist(graph1.x, graph1.x, device=self.device).to(self.dtype)
            intra_dist2 = adj2 * get_normalized_neg_exp_dist(graph2.x, graph2.x, device=self.device).to(self.dtype)
        else:
            intra_dist1 = adj1 * get_normalized_neg_exp_dist(rwr_emb1, rwr_emb1, device=self.device).to(self.dtype)
            intra_dist2 = adj2 * get_normalized_neg_exp_dist(rwr_emb2, rwr_emb2, device=self.device).to(self.dtype)

        return cross_dist, intra_dist1, intra_dist2

    def get_product_rwr_mat(self, adj1, adj2, in_cross_dist, max_iters=50, tol=1e-2):
        row_norm_adj1 = F.normalize(adj1.to(self.device), p=1, dim=1)
        row_norm_adj2 = F.normalize(adj2.to(self.device), p=1, dim=1)

        out_cross_dist = torch.zeros(in_cross_dist.shape).to(self.device).to(self.dtype)
        for i in range(max_iters):
            out_cross_dist_old = torch.clone(out_cross_dist)
            out_cross_dist = ((1 + self.rwr_restart_prob) * in_cross_dist +
                              (1 - self.rwr_restart_prob) * self.gamma * row_norm_adj1 @ out_cross_dist @ row_norm_adj2.T)
            if torch.max(torch.abs(out_cross_dist - out_cross_dist_old)) < tol:
                break

        # TODO: Why? Remove this line
        out_cross_dist = (1 - self.gamma) * out_cross_dist
        return out_cross_dist

    
    
