from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from common.metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        val_dataset,
        device: torch.device,
        batch_size: int = 64,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        num_epochs: int = 100,
        patience: int = 15,
        num_classes: int = 12,
        gcn_pretrain_epochs: int | None = None,
        transformer_epochs: int | None = None,
        pretrain_lr: float | None = None,
        transformer_lr: float | None = None,
        ema_decay: float = 0.99,
        distill_weight: float = 0.1,
        epoch_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.patience = patience
        self.epoch_callback = epoch_callback
        self.loader_num_workers = 2
        self.loader_kwargs = self._build_loader_kwargs()
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.loader_num_workers,
            **self.loader_kwargs,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.loader_num_workers,
            **self.loader_kwargs,
        )
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.use_stagewise_teacher = bool(getattr(self.model, "supports_stagewise_teacher", False))
        self.gcn_pretrain_epochs = gcn_pretrain_epochs if gcn_pretrain_epochs is not None else num_epochs
        self.transformer_epochs = transformer_epochs if transformer_epochs is not None else num_epochs
        self.pretrain_lr = pretrain_lr if pretrain_lr is not None else lr
        self.transformer_lr = transformer_lr if transformer_lr is not None else lr
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.distill_weight = distill_weight
        self._pretrain_debug_logged = False

        self.optimizer = None
        self.scheduler = None
        self.pretrain_optimizer = None
        self.pretrain_scheduler = None

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_aucs: list[float] = []
        self.val_aucs: list[float] = []
        self.pretrain_train_losses: list[float] = []
        self.pretrain_val_losses: list[float] = []
        self.pretrain_train_aucs: list[float] = []
        self.pretrain_val_aucs: list[float] = []
        self.pretrain_cls_losses: list[float] = []
        self.pretrain_distill_losses: list[float] = []
        self.pretrain_val_cls_losses: list[float] = []
        self.pretrain_val_distill_losses: list[float] = []
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}

        if not self.use_stagewise_teacher:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
            )

    def _build_loader_kwargs(self) -> dict[str, object]:
        loader_kwargs: dict[str, object] = {
            "pin_memory": self.device.type == "cuda",
        }
        if self.loader_num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        return loader_kwargs

    def _forward(self, data):
        outputs = self.model(data)
        return outputs[0] if isinstance(outputs, tuple) else outputs

    def _set_model_mode(self, train: bool = True) -> None:
        self.model.train() if train else self.model.eval()
        if hasattr(self.model, "set_teacher_eval"):
            self.model.set_teacher_eval()

    def _compute_task_loss(self, outputs: torch.Tensor, targets: torch.Tensor):
        if targets.dim() == 1:
            targets = targets.view(-1, self.num_classes)
        loss_matrix = self.criterion(outputs, targets)
        mask = (targets != -1).float()
        loss = (loss_matrix * mask).sum() / mask.sum().clamp(min=1.0)
        return loss, targets

    def _notify_epoch(self, event: dict[str, object]) -> None:
        if self.epoch_callback is not None:
            self.epoch_callback(event)

    def _run_epoch(self, loader, train: bool = True):
        self._set_model_mode(train)
        total_loss = 0.0
        all_outputs, all_targets = [], []
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for data in loader:
                data = data.to(self.device)
                outputs = self._forward(data)
                loss, targets = self._compute_task_loss(outputs, data.y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                total_loss += float(loss.item())
                all_outputs.append(outputs.detach())
                all_targets.append(targets.detach())

        average_loss = total_loss / max(len(loader), 1)
        metrics = compute_metrics(torch.cat(all_outputs), torch.cat(all_targets), self.num_classes)
        return average_loss, float(metrics["roc_auc"])

    def _run_pretrain_epoch(self, loader, train: bool = True):
        self._set_model_mode(train)
        total_cls = 0.0
        total_distill = 0.0
        total_applied = 0.0
        all_outputs, all_targets = [], []
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for data in loader:
                data = data.to(self.device)
                stage_out = self.model.forward_gcn_pretrain(data)
                cls_loss, targets = self._compute_task_loss(stage_out["student_logits"], data.y)
                distill_loss = self.model.compute_distill_loss(stage_out)
                loss = cls_loss + (self.distill_weight * distill_loss)

                if train and not self._pretrain_debug_logged:
                    debug_info = stage_out.get("debug_info", {})
                    print("    [Stage 1 Debug - first step only]")
                    print(f"      teacher_edge_dropout: {debug_info.get('teacher_edge_dropout', 0.0):.1f}")
                    print(f"      student_edge_dropout: {debug_info.get('student_edge_dropout', 0.1):.1f}")
                    print(f"      teacher_edges: {debug_info.get('teacher_num_edges')}")
                    print(f"      student_edges: {debug_info.get('student_num_edges')}")
                    print(f"      task_loss_type: BCEWithLogitsLoss")
                    print(f"      distill_weight: {self.distill_weight:.4f}")
                    print(f"      raw_cls_loss: {cls_loss.item():.6f}")
                    print(f"      raw_distill_loss: {distill_loss.item():.6f}")
                    print(f"      total_loss: {loss.item():.6f}")
                    self._pretrain_debug_logged = True

                if train:
                    self.pretrain_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.get_gcn_pretrain_parameters(), 1.0)
                    self.pretrain_optimizer.step()
                    self.model.update_teachers(self.ema_decay)

                total_cls += float(cls_loss.item())
                total_distill += float(distill_loss.item())
                total_applied += float(loss.item())
                all_outputs.append(stage_out["student_logits"].detach())
                all_targets.append(targets.detach())

        average_cls = total_cls / max(len(loader), 1)
        average_distill = total_distill / max(len(loader), 1)
        average_loss = total_applied / max(len(loader), 1)
        metrics = compute_metrics(torch.cat(all_outputs), torch.cat(all_targets), self.num_classes)
        return average_loss, float(metrics["roc_auc"]), average_cls, average_distill

    def _run_transformer_epoch(self, loader, train: bool = True):
        self._set_model_mode(train)
        total_loss = 0.0
        all_outputs, all_targets = [], []
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for data in loader:
                data = data.to(self.device)
                outputs = self._forward(data)
                loss, targets = self._compute_task_loss(outputs, data.y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.get_transformer_parameters(), 1.0)
                    self.optimizer.step()
                total_loss += float(loss.item())
                all_outputs.append(outputs.detach())
                all_targets.append(targets.detach())

        average_loss = total_loss / max(len(loader), 1)
        metrics = compute_metrics(torch.cat(all_outputs), torch.cat(all_targets), self.num_classes)
        return average_loss, float(metrics["roc_auc"])

    def _train_standard(self):
        patience_count = 0
        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_auc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_auc = self._run_epoch(self.val_loader, train=False)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_aucs.append(train_auc)
            self.val_aucs.append(val_auc)
            self.scheduler.step(val_auc)
            print(
                f"  Epoch {epoch}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}"
            )
            self._notify_epoch(
                {
                    "phase": "train",
                    "epoch": epoch,
                    "global_epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_metric": train_auc,
                    "val_metric": val_auc,
                }
            )

            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                self.best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= self.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        print(f"  Best Val AUC: {self.best_val_auc:.4f} (epoch {self.best_epoch})")
        return self.best_val_auc

    def _train_stagewise_teacher(self):
        print("  [Stage 1/2] Student GCN minimizes task BCE + lambda * online EMA teacher KD loss")
        self.model.sync_teachers()
        self.pretrain_optimizer = torch.optim.Adam(
            self.model.get_gcn_pretrain_parameters(),
            lr=self.pretrain_lr,
            weight_decay=self.weight_decay,
        )
        self.pretrain_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.pretrain_optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        for epoch in range(1, self.gcn_pretrain_epochs + 1):
            train_loss, train_auc, train_cls, train_dist = self._run_pretrain_epoch(self.train_loader, train=True)
            val_loss, val_auc, val_cls, val_dist = self._run_pretrain_epoch(self.val_loader, train=False)
            self.pretrain_train_losses.append(train_loss)
            self.pretrain_val_losses.append(val_loss)
            self.pretrain_train_aucs.append(train_auc)
            self.pretrain_val_aucs.append(val_auc)
            self.pretrain_cls_losses.append(train_cls)
            self.pretrain_distill_losses.append(train_dist)
            self.pretrain_val_cls_losses.append(val_cls)
            self.pretrain_val_distill_losses.append(val_dist)
            self.pretrain_scheduler.step(val_loss)
            print(
                f"    GCN Epoch {epoch}/{self.gcn_pretrain_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train BCE: {train_cls:.4f} | Val BCE: {val_cls:.4f} | "
                f"Train KD: {train_dist:.4f} | Val KD: {val_dist:.4f} | "
                f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}"
            )
            self._notify_epoch(
                {
                    "phase": "stage1_gcn_kd",
                    "epoch": epoch,
                    "global_epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_metric": train_auc,
                    "val_metric": val_auc,
                    "train_distill_loss": train_dist,
                    "val_distill_loss": val_dist,
                }
            )

        print("  [Stage 2/2] Frozen student GCN -> transformer encoders")
        self.optimizer = torch.optim.Adam(
            self.model.get_transformer_parameters(),
            lr=self.transformer_lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )

        patience_count = 0
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}

        for epoch in range(1, self.transformer_epochs + 1):
            train_loss, train_auc = self._run_transformer_epoch(self.train_loader, train=True)
            val_loss, val_auc = self._run_transformer_epoch(self.val_loader, train=False)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_aucs.append(train_auc)
            self.val_aucs.append(val_auc)
            self.scheduler.step(val_auc)
            print(
                f"    TF Epoch {epoch}/{self.transformer_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}"
            )
            self._notify_epoch(
                {
                    "phase": "stage2_transformer",
                    "epoch": epoch,
                    "global_epoch": self.gcn_pretrain_epochs + epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_metric": train_auc,
                    "val_metric": val_auc,
                }
            )

            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                self.best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= self.patience:
                print(f"  Early stopping at transformer epoch {epoch}")
                break

        print(f"  Best Final Val AUC: {self.best_val_auc:.4f} (transformer epoch {self.best_epoch})")
        return self.best_val_auc

    def train(self):
        if self.use_stagewise_teacher:
            return self._train_stagewise_teacher()
        return self._train_standard()

    def evaluate(self, test_dataset, batch_size: int = 64):
        self.model.load_state_dict(self.best_state)
        self.model.to(self.device)
        loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.loader_num_workers,
            **self.loader_kwargs,
        )
        if self.use_stagewise_teacher:
            _, test_auc = self._run_transformer_epoch(loader, train=False)
        else:
            _, test_auc = self._run_epoch(loader, train=False)
        return test_auc

    def build_history_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        global_epoch = 0

        if self.use_stagewise_teacher:
            for epoch, (train_loss, val_loss, train_auc, val_auc, train_dist, val_dist) in enumerate(
                zip(
                    self.pretrain_train_losses,
                    self.pretrain_val_losses,
                    self.pretrain_train_aucs,
                    self.pretrain_val_aucs,
                    self.pretrain_distill_losses,
                    self.pretrain_val_distill_losses,
                ),
                start=1,
            ):
                global_epoch += 1
                rows.append(
                    {
                        "phase": "stage1_gcn_kd",
                        "epoch": epoch,
                        "global_epoch": global_epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_metric": train_auc,
                        "val_metric": val_auc,
                        "train_distill_loss": train_dist,
                        "val_distill_loss": val_dist,
                    }
                )

        for epoch, (train_loss, val_loss, train_auc, val_auc) in enumerate(
            zip(self.train_losses, self.val_losses, self.train_aucs, self.val_aucs),
            start=1,
        ):
            global_epoch += 1
            rows.append(
                {
                    "phase": "stage2_transformer" if self.use_stagewise_teacher else "train",
                    "epoch": epoch,
                    "global_epoch": global_epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_metric": train_auc,
                    "val_metric": val_auc,
                    "train_distill_loss": math.nan,
                    "val_distill_loss": math.nan,
                }
            )

        return rows
