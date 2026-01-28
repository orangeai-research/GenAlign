# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
RFBM3: RF-Enhanced BM3
Integrates Rectified Flow module to enhance collaborative filtering embeddings
"""

import torch
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from models.bm3 import BM3
from models.rf_modules import RFEmbeddingGenerator


class RFBM3(BM3):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_rf = config["use_rf"] if "use_rf" in config else True

        if self.use_rf:
            # Initialize RF generator with consistent parameters
            self.rf_generator = RFEmbeddingGenerator(
                embedding_dim=self.embedding_dim,
                hidden_dim=config["rf_hidden_dim"] if "rf_hidden_dim" in config else 128,
                n_layers=config["rf_n_layers"] if "rf_n_layers" in config else 2,
                dropout=config["rf_dropout"] if "rf_dropout" in config else 0.1,
                learning_rate=config["rf_learning_rate"] if "rf_learning_rate" in config else 0.0001,
                sampling_steps=config["rf_sampling_steps"] if "rf_sampling_steps" in config else 10,
                warmup_epochs=config["rf_warmup_epochs"] if "rf_warmup_epochs" in config else 5,
                train_mix_ratio=config["rf_mix_ratio"] if "rf_mix_ratio" in config else 0.1,
                inference_mix_ratio=config["rf_inference_mix_ratio"] if "rf_inference_mix_ratio" in config else 0.2,
                contrast_temp=config["rf_contrast_temp"] if "rf_contrast_temp" in config else 0.2,
                contrast_weight=config["rf_loss_weight"] if "rf_loss_weight" in config else 1.0,
            )
            self._rf_logged_this_epoch = False

    def set_epoch(self, epoch):
        """Set current epoch for RF generator."""
        if self.use_rf:
            self.rf_generator.set_epoch(epoch)
            self._rf_logged_this_epoch = False

    def forward(self):
        h = self.item_id_embedding.weight

        # Original BM3 GCN
        ego_embeddings = torch.cat((self.user_embedding.weight,
                                    self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)

        # all_embeddings: (n_users + n_items, embedding_dim)
        all_embeddings_ori = all_embeddings.clone()

        u_g_embeddings, i_g_embeddings_ori = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )

        # ===== RF Enhancement for both users and items =====
        rf_outputs = None

        if self.use_rf:
            # Prepare multimodal conditions for items
            t_feat_online = self.text_trs(self.text_embedding.weight) if self.t_feat is not None else None
            v_feat_online = self.image_trs(self.image_embedding.weight) if self.v_feat is not None else None

            # Extend conditions to user+item space
            # For users, we use the item features aggregated via interaction matrix
            item_conditions = []
            full_conditions = []
            if v_feat_online is not None:
                item_conditions.append(v_feat_online)
                # Aggregate item visual features to users via R matrix
                user_v_feat = torch.sparse.mm(self.R, v_feat_online) if hasattr(self, 'R') else torch.zeros(self.n_users, v_feat_online.shape[1]).to(v_feat_online.device)
                full_v_feat = torch.cat([user_v_feat, v_feat_online], dim=0)
                full_conditions.append(full_v_feat)
            if t_feat_online is not None:
                item_conditions.append(t_feat_online)
                # Aggregate item text features to users via R matrix
                user_t_feat = torch.sparse.mm(self.R, t_feat_online) if hasattr(self, 'R') else torch.zeros(self.n_users, t_feat_online.shape[1]).to(t_feat_online.device)
                full_t_feat = torch.cat([user_t_feat, t_feat_online], dim=0)
                full_conditions.append(full_t_feat)

            if len(full_conditions) > 0 and self.training:
                # RF training with full user+item embeddings
                loss_dict = self.rf_generator.compute_loss_and_step(
                    target_embeds=all_embeddings_ori.detach(),
                    conditions=[c.detach() for c in full_conditions],
                )

                if not self._rf_logged_this_epoch:
                    print(f"  [RF Train] epoch={self.rf_generator.current_epoch}, "
                          f"rf_loss={loss_dict['rf_loss']:.6f}")
                    self._rf_logged_this_epoch = True

                # Generate RF embeddings for full user+item space
                rf_embeds = self.rf_generator.generate(full_conditions)

                # Mix embeddings
                all_embeddings_mixed = self.rf_generator.mix_embeddings(
                    all_embeddings_ori, rf_embeds.detach(), training=True
                )

                u_g_embeddings, i_g_embeddings = torch.split(
                    all_embeddings_mixed, [self.n_users, self.n_items], dim=0
                )

                # Store rf_outputs for cl_loss in calculate_loss
                rf_outputs = {
                    "rf_embeds": rf_embeds,
                    "target_embeds": all_embeddings_ori,
                }

                # Residual connection for items only
                i_g_embeddings = i_g_embeddings + h
                return u_g_embeddings, i_g_embeddings, rf_outputs

            elif len(full_conditions) > 0 and not self.training:
                # Inference mode
                with torch.no_grad():
                    rf_embeds = self.rf_generator.generate(full_conditions)
                    all_embeddings_mixed = self.rf_generator.mix_embeddings(
                        all_embeddings_ori, rf_embeds, training=False
                    )
                    u_g_embeddings, i_g_embeddings = torch.split(
                        all_embeddings_mixed, [self.n_users, self.n_items], dim=0
                    )

        # Residual connection for items
        i_g_embeddings = i_g_embeddings_ori + h

        return u_g_embeddings, i_g_embeddings, None

    def calculate_loss(self, interactions):
        # online network (RFBM3 forward returns 3 values)
        u_online_ori, i_online_ori, rf_outputs = self.forward()
        t_feat_online, v_feat_online = None, None
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)

        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            if self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)

            if self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)

        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)

        users, items = interactions[0], interactions[1]
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        if self.t_feat is not None:
            t_feat_online = self.predictor(t_feat_online)
            t_feat_online = t_feat_online[items, :]
            t_feat_target = t_feat_target[items, :]
            loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        if self.v_feat is not None:
            v_feat_online = self.predictor(v_feat_online)
            v_feat_online = v_feat_online[items, :]
            v_feat_target = v_feat_target[items, :]
            loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()

        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        total_loss = (loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + \
                     self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean()

        # RF contrastive loss (cl_loss) - split into users and items
        if self.use_rf and rf_outputs is not None:
            rf_embeds = rf_outputs["rf_embeds"]
            target_embeds = rf_outputs["target_embeds"]

            rf_users, rf_items = torch.split(rf_embeds, [self.n_users, self.n_items], dim=0)
            target_users, target_items = torch.split(target_embeds, [self.n_users, self.n_items], dim=0)

            rf_cl_loss = self.rf_generator._infonce_loss(rf_items[items], target_items[items], 0.2) + \
                         self.rf_generator._infonce_loss(rf_users[users], target_users[users], 0.2)

            total_loss = total_loss + self.rf_generator.contrast_weight * rf_cl_loss

        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, i_online, _ = self.forward()
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui