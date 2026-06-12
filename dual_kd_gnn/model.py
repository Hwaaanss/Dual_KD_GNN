from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def masked_mean_pool(x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
    if pad_mask is None:
        return x.mean(dim=1)
    valid = (~pad_mask).unsqueeze(-1).type_as(x)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (x * valid).sum(dim=1) / denom


def masked_mse(student: torch.Tensor, teacher: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
    diff = (student - teacher).pow(2)
    if pad_mask is None:
        return diff.mean()
    valid = (~pad_mask).unsqueeze(-1).type_as(diff)
    denom = (valid.sum() * diff.size(-1)).clamp_min(1.0)
    return (diff * valid).sum() / denom


def masked_cosine_distance(
    student: torch.Tensor,
    teacher: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    student_n = F.normalize(student, dim=-1, eps=1e-8)
    teacher_n = F.normalize(teacher, dim=-1, eps=1e-8)
    cos = (student_n * teacher_n).sum(dim=-1)
    dist = 1.0 - cos
    if pad_mask is None:
        return dist.mean()
    valid = (~pad_mask).type_as(dist)
    denom = valid.sum().clamp_min(1.0)
    return (dist * valid).sum() / denom


class GNNEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.edge_enc = nn.Linear(edge_dim, hidden_dim)
        self.edge_weight_proj = nn.Linear(hidden_dim, 1)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.res_projs = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = node_dim if layer_idx == 0 else hidden_dim
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.res_projs.append(
                nn.Linear(in_dim, hidden_dim, bias=False) if in_dim != hidden_dim else nn.Identity()
            )
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor):
        if edge_attr.size(0) > 0:
            edge_emb = self.edge_enc(edge_attr)
            edge_weight = torch.sigmoid(self.edge_weight_proj(edge_emb)).squeeze(-1)
        else:
            edge_weight = None

        for layer_idx, (conv, bn, res_proj) in enumerate(zip(self.convs, self.bns, self.res_projs)):
            residual = res_proj(x)
            x = bn(conv(x, edge_index, edge_weight=edge_weight)) + residual
            if layer_idx < self.num_layers - 1:
                x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        x = self.norm1(x + self.drop1(attn_out))
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        ff_out = self.ff(x)
        x = self.norm2(x + self.drop2(ff_out))
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x


class BranchTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pad_mask)
        x = self.out_norm(x)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x


class InteractionTensorHead(nn.Module):
    """Interaction tensor head with optional codebook-shared low-rank factors.

    When ``num_prototypes > 0`` (and low-rank is enabled) the per-class factor
    ``U_k`` is reparameterized as a (possibly sparse) combination of ``M`` shared
    prototypes ``C_m`` via learned assignment logits ``theta_{k,m}``:

        U_k = sum_m alpha_{k,m} * C_m,    alpha_k = route(theta_k / tau)

    The routing function ``route`` is selected by ``assignment_mode``:
      - ``"hard"`` -> Gumbel-softmax with Straight-Through estimator
      - ``"soft"`` -> plain softmax
      - ``"sparse"`` -> top-k of softmax (renormalized)

    A diversity regularizer ``diversity_loss`` discourages codebook collapse by
    penalizing pairwise absolute cosine similarity between prototypes.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        rank: int = 32,
        use_bias: bool = True,
        symmetric: bool = True,
        proj_dim: int = 0,
        num_prototypes: int = 6,
        assignment_mode: str = "hard",
        tau: float = 1.0,
        diversity_weight: float = 0.01,
        codebook_init: str = "orthogonal",
        codebook_init_scale: float = 0.1,
        topk: int = 2,
    ) -> None:
        super().__init__()
        if assignment_mode not in {"hard", "soft", "sparse"}:
            raise ValueError(f"Unknown assignment_mode: {assignment_mode}")
        if codebook_init not in {"orthogonal", "random"}:
            raise ValueError(f"Unknown codebook_init: {codebook_init}")

        self.symmetric = symmetric
        self.use_low_rank = rank > 0
        self.num_classes = num_classes
        self.rank = rank
        self.num_prototypes = int(num_prototypes)
        self.assignment_mode = assignment_mode
        self.diversity_weight = float(diversity_weight)
        self.topk = int(topk)
        self.tau = float(tau)

        if proj_dim > 0:
            self.proj = nn.Sequential(nn.Linear(d_model, proj_dim), nn.ReLU())
            effective_dim = proj_dim
        else:
            self.proj = None
            effective_dim = d_model
        self.effective_dim = effective_dim

        self._use_codebook = self.use_low_rank and self.num_prototypes > 0

        if self._use_codebook:
            self.codebook_u = nn.Parameter(
                self._init_codebook(self.num_prototypes, effective_dim, rank, codebook_init, codebook_init_scale)
            )
            self.assignment_logits_u = nn.Parameter(torch.zeros(num_classes, self.num_prototypes))
            nn.init.normal_(self.assignment_logits_u, std=1.0)
            if not symmetric:
                self.codebook_v = nn.Parameter(
                    self._init_codebook(self.num_prototypes, effective_dim, rank, codebook_init, codebook_init_scale)
                )
                self.assignment_logits_v = nn.Parameter(torch.zeros(num_classes, self.num_prototypes))
                nn.init.normal_(self.assignment_logits_v, std=1.0)
        else:
            if self.use_low_rank:
                self.U = nn.Parameter(torch.randn(num_classes, effective_dim, rank) * 0.01)
                if not symmetric:
                    self.V = nn.Parameter(torch.randn(num_classes, effective_dim, rank) * 0.01)
            else:
                self.A = nn.Parameter(torch.randn(num_classes, effective_dim, effective_dim) * 0.01)

        self.linear_residual = nn.Linear(effective_dim, num_classes, bias=False)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_classes))
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _init_codebook(num_prototypes: int, d: int, r: int, init: str, scale: float) -> torch.Tensor:
        codebook = torch.empty(num_prototypes, d, r)
        if init == "orthogonal":
            for m in range(num_prototypes):
                nn.init.orthogonal_(codebook[m])
            codebook.mul_(scale)
        else:
            codebook.normal_(0.0, 0.01)
        return codebook

    @property
    def use_codebook(self) -> bool:
        return self._use_codebook

    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)

    def _compute_assignment(self, logits: torch.Tensor) -> torch.Tensor:
        tau = max(self.tau, 1e-3)
        if self.training:
            if self.assignment_mode == "soft":
                return F.softmax(logits / tau, dim=-1)
            if self.assignment_mode == "hard":
                return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
            soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            k = min(self.topk, soft.size(-1))
            topk_vals, topk_idx = soft.topk(k, dim=-1)
            mask = torch.zeros_like(soft).scatter_(-1, topk_idx, topk_vals)
            return mask / mask.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if self.assignment_mode == "soft":
            return F.softmax(logits / tau, dim=-1)
        if self.assignment_mode == "sparse":
            soft = F.softmax(logits / tau, dim=-1)
            k = min(self.topk, soft.size(-1))
            topk_vals, topk_idx = soft.topk(k, dim=-1)
            mask = torch.zeros_like(soft).scatter_(-1, topk_idx, topk_vals)
            return mask / mask.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        idx = logits.argmax(dim=-1)
        return F.one_hot(idx, num_classes=logits.size(-1)).to(logits.dtype)

    @staticmethod
    def _codebook_project(z: torch.Tensor, codebook: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # z: [B, d], codebook: [M, d, r], alpha: [K, M] -> [B, K, r]
        y = torch.einsum("bd,mdr->bmr", z, codebook)
        return torch.einsum("km,bmr->bkr", alpha, y)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            z = self.proj(z)
        if self._use_codebook:
            alpha_u = self._compute_assignment(self.assignment_logits_u)
            u_proj = self._codebook_project(z, self.codebook_u, alpha_u)
            if self.symmetric:
                logits = (u_proj ** 2).sum(dim=-1)
            else:
                alpha_v = self._compute_assignment(self.assignment_logits_v)
                v_proj = self._codebook_project(z, self.codebook_v, alpha_v)
                logits = (u_proj * v_proj).sum(dim=-1)
        elif self.use_low_rank:
            if self.symmetric:
                z_proj = torch.einsum("bd,kdr->bkr", z, self.U)
                logits = (z_proj ** 2).sum(dim=-1)
            else:
                z_u = torch.einsum("bd,kdr->bkr", z, self.U)
                z_v = torch.einsum("bd,kdr->bkr", z, self.V)
                logits = (z_u * z_v).sum(dim=-1)
        else:
            az = torch.einsum("kdf,bf->bkd", self.A, z)
            logits = (z.unsqueeze(1) * az).sum(dim=-1)
        logits = logits + self.linear_residual(z)
        if self.bias is not None:
            logits = logits + self.bias
        return logits

    @staticmethod
    def _pairwise_abs_cos(codebook: torch.Tensor) -> torch.Tensor:
        m = codebook.size(0)
        if m <= 1:
            return codebook.new_zeros(())
        flat = codebook.reshape(m, -1)
        normed = F.normalize(flat, dim=-1)
        sim = normed @ normed.t()
        off_diag = sim - torch.eye(m, device=sim.device, dtype=sim.dtype)
        return off_diag.abs().sum() / (m * (m - 1))

    def diversity_loss(self) -> torch.Tensor:
        zero = self.linear_residual.weight.new_zeros(())
        if not self._use_codebook:
            return zero
        loss = self._pairwise_abs_cos(self.codebook_u)
        if not self.symmetric:
            loss = 0.5 * (loss + self._pairwise_abs_cos(self.codebook_v))
        return loss

    @torch.no_grad()
    def get_assignment_probabilities(self) -> torch.Tensor:
        """Return the soft assignment matrix alpha (K x M) for analysis/visualization."""
        if not self._use_codebook:
            return self.linear_residual.weight.new_zeros((self.num_classes, 0))
        tau = max(self.tau, 1e-3)
        return F.softmax(self.assignment_logits_u / tau, dim=-1)


class DoubleGCNTransformerModel(nn.Module):
    supports_stagewise_teacher = True

    def __init__(
        self,
        chem_dim: int = 19,
        phys_dim: int = 5,
        edge_dim: int = 7,
        gnn_hidden: int = 256,
        gnn_layers: int = 3,
        gnn_dropout: float = 0.4,
        nhead: int = 4,
        tf_layers: int = 2,
        dim_ff: int = 512,
        tf_dropout: float = 0.3,
        num_classes: int = 12,
        ih_rank: int = 32,
        ih_use_bias: bool = True,
        ih_symmetric: bool = True,
        ih_proj_dim: int = 0,
        ih_num_prototypes: int = 6,
        ih_assignment_mode: str = "hard",
        ih_tau_init: float = 1.0,
        ih_tau_final: float = 0.1,
        ih_diversity_weight: float = 0.01,
        ih_codebook_init: str = "orthogonal",
        ih_topk: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = gnn_hidden
        self.ih_tau_init = float(ih_tau_init)
        self.ih_tau_final = float(ih_tau_final)

        self.gnn_c = GNNEncoder(chem_dim, edge_dim, gnn_hidden, gnn_layers, gnn_dropout)
        self.gnn_p = GNNEncoder(phys_dim, edge_dim, gnn_hidden, gnn_layers, gnn_dropout)

        self.teacher_gnn_c = copy.deepcopy(self.gnn_c)
        self.teacher_gnn_p = copy.deepcopy(self.gnn_p)
        self._freeze_teacher_parameters()
        self.set_teacher_eval()

        self.input_norm_c = nn.LayerNorm(gnn_hidden)
        self.input_norm_p = nn.LayerNorm(gnn_hidden)
        self.input_drop = nn.Dropout(tf_dropout)

        self.phys_encoder = BranchTransformerEncoder(gnn_hidden, nhead, dim_ff, tf_layers, tf_dropout)
        self.chem_encoder = BranchTransformerEncoder(gnn_hidden, nhead, dim_ff, tf_layers, tf_dropout)

        self.concat_dim = gnn_hidden * 2
        self.concat_norm = nn.LayerNorm(self.concat_dim)
        self.classifier = InteractionTensorHead(
            d_model=self.concat_dim,
            num_classes=num_classes,
            rank=ih_rank,
            use_bias=ih_use_bias,
            symmetric=ih_symmetric,
            proj_dim=ih_proj_dim,
            num_prototypes=ih_num_prototypes,
            assignment_mode=ih_assignment_mode,
            tau=ih_tau_init,
            diversity_weight=ih_diversity_weight,
            codebook_init=ih_codebook_init,
            topk=ih_topk,
        )

        # BYOL-style asymmetric predictors for cross-modal EMA distillation
        # (Plan 4). Student-only; no EMA copy. Applied node-wise on the
        # student GCN outputs, with stop-gradient on EMA teacher targets.
        self.predictor_c2p = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden),
            nn.BatchNorm1d(gnn_hidden),
            nn.GELU(),
            nn.Linear(gnn_hidden, gnn_hidden),
        )
        self.predictor_p2c = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden),
            nn.BatchNorm1d(gnn_hidden),
            nn.GELU(),
            nn.Linear(gnn_hidden, gnn_hidden),
        )

        self.sync_teachers()

    def _freeze_teacher_parameters(self) -> None:
        for param in self.teacher_gnn_c.parameters():
            param.requires_grad = False
        for param in self.teacher_gnn_p.parameters():
            param.requires_grad = False

    def set_teacher_eval(self) -> None:
        self.teacher_gnn_c.eval()
        self.teacher_gnn_p.eval()

    @torch.no_grad()
    def sync_teachers(self) -> None:
        self.teacher_gnn_c.load_state_dict(self.gnn_c.state_dict())
        self.teacher_gnn_p.load_state_dict(self.gnn_p.state_dict())
        self._freeze_teacher_parameters()
        self.set_teacher_eval()

    @torch.no_grad()
    def _ema_update_module(self, teacher: nn.Module, student: nn.Module, ema_decay: float) -> None:
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(ema_decay).add_(student_param.data, alpha=1.0 - ema_decay)
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            if teacher_buffer.dtype.is_floating_point:
                teacher_buffer.data.mul_(ema_decay).add_(student_buffer.data, alpha=1.0 - ema_decay)
            else:
                teacher_buffer.data.copy_(student_buffer.data)

    @torch.no_grad()
    def update_teachers(self, ema_decay: float = 0.99) -> None:
        self._ema_update_module(self.teacher_gnn_c, self.gnn_c, ema_decay)
        self._ema_update_module(self.teacher_gnn_p, self.gnn_p, ema_decay)
        self.set_teacher_eval()

    def auxiliary_loss(self) -> torch.Tensor:
        """Diversity regularization on the classifier codebook (zero when disabled)."""
        if not self.classifier.use_codebook or self.classifier.diversity_weight <= 0.0:
            return self.classifier.linear_residual.weight.new_zeros(())
        return self.classifier.diversity_weight * self.classifier.diversity_loss()

    def step_classifier_schedule(self, stage: str, epoch_idx: int, total_epochs: int) -> None:
        """Anneal classifier sampling temperature linearly over stage-2 epochs."""
        if stage != "stage2" or not self.classifier.use_codebook:
            return
        if total_epochs <= 1:
            self.classifier.set_tau(self.ih_tau_final)
            return
        progress = max(0.0, min(1.0, float(epoch_idx) / float(total_epochs - 1)))
        tau = self.ih_tau_init + (self.ih_tau_final - self.ih_tau_init) * progress
        self.classifier.set_tau(tau)

    def get_gcn_pretrain_parameters(self):
        params = []
        params += list(self.gnn_c.parameters())
        params += list(self.gnn_p.parameters())
        params += list(self.input_norm_c.parameters())
        params += list(self.input_norm_p.parameters())
        params += list(self.concat_norm.parameters())
        params += list(self.classifier.parameters())
        params += list(self.predictor_c2p.parameters())
        params += list(self.predictor_p2c.parameters())
        return params

    def get_gcn_pretrain_param_groups(self, default_weight_decay: float, predictor_weight_decay: float = 1e-4):
        main_params = []
        main_params += list(self.gnn_c.parameters())
        main_params += list(self.gnn_p.parameters())
        main_params += list(self.input_norm_c.parameters())
        main_params += list(self.input_norm_p.parameters())
        main_params += list(self.concat_norm.parameters())
        main_params += list(self.classifier.parameters())
        predictor_params = []
        predictor_params += list(self.predictor_c2p.parameters())
        predictor_params += list(self.predictor_p2c.parameters())
        return [
            {"params": main_params, "weight_decay": float(default_weight_decay)},
            {"params": predictor_params, "weight_decay": float(predictor_weight_decay)},
        ]

    def get_transformer_parameters(self):
        params = []
        params += list(self.input_norm_c.parameters())
        params += list(self.input_norm_p.parameters())
        params += list(self.phys_encoder.parameters())
        params += list(self.chem_encoder.parameters())
        params += list(self.concat_norm.parameters())
        params += list(self.classifier.parameters())
        return params

    def _run_student_gcn(self, data):
        chem_nodes = self.gnn_c(data.x_chem, data.edge_index, data.edge_attr, data.batch)
        phys_nodes = self.gnn_p(data.x_phys, data.edge_index, data.edge_attr, data.batch)
        return chem_nodes, phys_nodes

    @torch.no_grad()
    def _run_frozen_student_gcn(self, data):
        prev_chem_training = self.gnn_c.training
        prev_phys_training = self.gnn_p.training
        self.gnn_c.eval()
        self.gnn_p.eval()
        try:
            return self._run_student_gcn(data)
        finally:
            self.gnn_c.train(prev_chem_training)
            self.gnn_p.train(prev_phys_training)

    @torch.no_grad()
    def _run_teacher_gcn(self, data):
        self.set_teacher_eval()
        chem_nodes = self.teacher_gnn_c(data.x_chem, data.edge_index, data.edge_attr, data.batch)
        phys_nodes = self.teacher_gnn_p(data.x_phys, data.edge_index, data.edge_attr, data.batch)
        return chem_nodes, phys_nodes

    def _apply_node_predictor(
        self,
        predictor: nn.Module,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B, N, d] (zero on padded positions). BN runs only on valid nodes
        # to keep its running stats clean.
        valid = ~pad_mask
        if valid.any():
            y_valid = predictor(x[valid])
            out = torch.zeros_like(x)
            out[valid] = y_valid
            return out
        return torch.zeros_like(x)

    def _pad_dual_sequences(self, chem_nodes: torch.Tensor, phys_nodes: torch.Tensor, batch: torch.Tensor):
        batch_size = int(batch.max().item()) + 1
        seqs_c, seqs_p = [], []
        for graph_idx in range(batch_size):
            node_mask = batch == graph_idx
            seqs_c.append(chem_nodes[node_mask])
            seqs_p.append(phys_nodes[node_mask])

        max_len = max(seq.size(0) for seq in seqs_c)
        device = chem_nodes.device
        padded_c = torch.zeros(batch_size, max_len, self.d_model, device=device)
        padded_p = torch.zeros(batch_size, max_len, self.d_model, device=device)
        pad_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)

        for graph_idx, (seq_c, seq_p) in enumerate(zip(seqs_c, seqs_p)):
            seq_len = seq_c.size(0)
            padded_c[graph_idx, :seq_len] = seq_c
            padded_p[graph_idx, :seq_len] = seq_p
            pad_mask[graph_idx, :seq_len] = False

        return padded_c, padded_p, pad_mask

    def forward_gcn_pretrain(self, data):
        teacher_edge_dropout = 0.0
        student_edge_dropout = 0.1

        student_data = copy.copy(data)
        student_edge_index = data.edge_index.clone()
        student_edge_attr = data.edge_attr.clone() if getattr(data, "edge_attr", None) is not None else None
        dropout_before_edges = int(student_edge_index.size(1))

        if dropout_before_edges > 0:
            keep_mask = torch.rand(dropout_before_edges, device=student_edge_index.device) >= student_edge_dropout
            student_edge_index = student_edge_index[:, keep_mask]
            if student_edge_attr is not None and student_edge_attr.size(0) == dropout_before_edges:
                student_edge_attr = student_edge_attr[keep_mask]

        student_data.edge_index = student_edge_index
        if student_edge_attr is not None:
            student_data.edge_attr = student_edge_attr

        teacher_chem, teacher_phys = self._run_teacher_gcn(data)
        student_chem, student_phys = self._run_student_gcn(student_data)

        student_c, student_p, pad_mask = self._pad_dual_sequences(student_chem, student_phys, data.batch)
        teacher_c, teacher_p, _ = self._pad_dual_sequences(teacher_chem, teacher_phys, data.batch)

        student_c = student_c.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        student_p = student_p.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        teacher_c = teacher_c.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        teacher_p = teacher_p.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        student_c_cls = self.input_drop(self.input_norm_c(student_c))
        student_p_cls = self.input_drop(self.input_norm_p(student_p))
        student_fused_seq = torch.cat([student_p_cls, student_c_cls], dim=-1)
        student_fused_seq = student_fused_seq.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        student_fused_seq = self.concat_norm(student_fused_seq)
        student_fused_seq = student_fused_seq.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        student_graph_repr = masked_mean_pool(student_fused_seq, pad_mask)
        student_logits = self.classifier(student_graph_repr)

        # Cross-modal asymmetric predictors (Plan 4): student-only, node-wise.
        student_c_to_p = self._apply_node_predictor(self.predictor_c2p, student_c, pad_mask)
        student_p_to_c = self._apply_node_predictor(self.predictor_p2c, student_p, pad_mask)

        return {
            "student_phys_seq": student_p,
            "student_chem_seq": student_c,
            "student_logits": student_logits,
            "student_chem_to_phys": student_c_to_p,
            "student_phys_to_chem": student_p_to_c,
            "teacher_phys_seq": teacher_p.detach(),
            "teacher_chem_seq": teacher_c.detach(),
            "pad_mask": pad_mask,
            "debug_info": {
                "teacher_edge_dropout": teacher_edge_dropout,
                "student_edge_dropout": student_edge_dropout,
                "teacher_num_edges": int(data.edge_index.size(1)),
                "student_num_edges": int(student_data.edge_index.size(1)),
                "dropout_before_edges": dropout_before_edges,
                "dropout_after_edges": int(student_data.edge_index.size(1)),
                "teacher_edge_index_shape": tuple(data.edge_index.shape),
                "student_edge_index_shape": tuple(student_data.edge_index.shape),
                "student_uses_separate_edge_tensor": student_data.edge_index.data_ptr() != data.edge_index.data_ptr(),
            },
        }

    def compute_distill_loss(self, stage_out):
        loss_phys = masked_mse(
            stage_out["student_phys_seq"],
            stage_out["teacher_phys_seq"],
            stage_out["pad_mask"],
        )
        loss_chem = masked_mse(
            stage_out["student_chem_seq"],
            stage_out["teacher_chem_seq"],
            stage_out["pad_mask"],
        )
        return 0.5 * (loss_phys + loss_chem)

    def compute_cross_distill_loss(self, stage_out):
        # Predictor outputs target the OPPOSITE-branch EMA teacher embeddings.
        # Teacher tensors are already detached in stage_out.
        loss_c_to_p = masked_cosine_distance(
            stage_out["student_chem_to_phys"],
            stage_out["teacher_phys_seq"],
            stage_out["pad_mask"],
        )
        loss_p_to_c = masked_cosine_distance(
            stage_out["student_phys_to_chem"],
            stage_out["teacher_chem_seq"],
            stage_out["pad_mask"],
        )
        return 0.5 * (loss_c_to_p + loss_p_to_c)

    def forward(self, data):
        student_chem, student_phys = self._run_frozen_student_gcn(data)
        padded_c, padded_p, pad_mask = self._pad_dual_sequences(student_chem, student_phys, data.batch)

        padded_c = self.input_drop(self.input_norm_c(padded_c))
        padded_p = self.input_drop(self.input_norm_p(padded_p))

        attn_pad_mask = pad_mask if padded_c.device.type != "mps" else None
        phys_seq = self.phys_encoder(padded_p, attn_pad_mask)
        chem_seq = self.chem_encoder(padded_c, attn_pad_mask)

        fused_seq = torch.cat([phys_seq, chem_seq], dim=-1)
        fused_seq = fused_seq.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        fused_seq = self.concat_norm(fused_seq)
        fused_seq = fused_seq.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        graph_repr = masked_mean_pool(fused_seq, pad_mask)
        return self.classifier(graph_repr)
