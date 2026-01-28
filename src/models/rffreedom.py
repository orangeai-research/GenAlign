# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
RFFREEDOM: RF-Enhanced FREEDOM
Integrates Rectified Flow module to enhance collaborative filtering embeddings
"""

import torch
import torch.nn.functional as F

from models.freedom import FREEDOM
from models.rf_modules import RFEmbeddingGenerator


class RFFREEDOM(FREEDOM):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_rf = config["use_rf"] if "use_rf" in config else True

        if self.use_rf:
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

    def forward(self, adj):
        # 1. Multimodal feature aggregation (through mm_adj)
        h = self.item_id_embedding.weight
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        # 2. Collaborative graph convolution
        ego_embeddings = torch.cat((self.user_embedding.weight,
                                    self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        all_embeddings_ori = all_embeddings.clone()

        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings,
                                                      [self.n_users, self.n_items], dim=0)
        i_g_embeddings_ori = i_g_embeddings.clone()

        # ===== RF Enhancement for both users and items =====
        rf_outputs = None

        if self.use_rf and self.training:
            # Prepare conditions for items
            image_feats = self.image_trs(self.image_embedding.weight) if self.v_feat is not None else None
            text_feats = self.text_trs(self.text_embedding.weight) if self.t_feat is not None else None

            # Extend conditions to user+item space
            full_conditions = []
            if image_feats is not None:
                # Aggregate item visual features to users
                user_image_feats = torch.sparse.mm(self.R, image_feats) if hasattr(self, 'R') else torch.zeros(self.n_users, image_feats.shape[1]).to(image_feats.device)
                full_image_feats = torch.cat([user_image_feats, image_feats], dim=0)
                full_conditions.append(full_image_feats)
            if text_feats is not None:
                # Aggregate item text features to users
                user_text_feats = torch.sparse.mm(self.R, text_feats) if hasattr(self, 'R') else torch.zeros(self.n_users, text_feats.shape[1]).to(text_feats.device)
                full_text_feats = torch.cat([user_text_feats, text_feats], dim=0)
                full_conditions.append(full_text_feats)

            if len(full_conditions) > 0:
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

        elif self.use_rf and not self.training:
            # Inference mode
            with torch.no_grad():
                image_feats = self.image_trs(self.image_embedding.weight) if self.v_feat is not None else None
                text_feats = self.text_trs(self.text_embedding.weight) if self.t_feat is not None else None

                full_conditions = []
                if image_feats is not None:
                    user_image_feats = torch.sparse.mm(self.R, image_feats) if hasattr(self, 'R') else torch.zeros(self.n_users, image_feats.shape[1]).to(image_feats.device)
                    full_image_feats = torch.cat([user_image_feats, image_feats], dim=0)
                    full_conditions.append(full_image_feats)
                if text_feats is not None:
                    user_text_feats = torch.sparse.mm(self.R, text_feats) if hasattr(self, 'R') else torch.zeros(self.n_users, text_feats.shape[1]).to(text_feats.device)
                    full_text_feats = torch.cat([user_text_feats, text_feats], dim=0)
                    full_conditions.append(full_text_feats)

                if len(full_conditions) > 0:
                    rf_embeds = self.rf_generator.generate(full_conditions)
                    all_embeddings_mixed = self.rf_generator.mix_embeddings(
                        all_embeddings_ori, rf_embeds, training=False
                    )
                    u_g_embeddings, i_g_embeddings = torch.split(
                        all_embeddings_mixed, [self.n_users, self.n_items], dim=0
                    )

        # Fuse multimodal features
        return u_g_embeddings, i_g_embeddings + h, rf_outputs

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, rf_outputs = self.forward(self.masked_adj)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        mf_v_loss, mf_t_loss = 0.0, 0.0
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])

        total_loss = batch_mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss)

        # RF contrastive loss (cl_loss) - split into users and items
        if self.use_rf and rf_outputs is not None:
            rf_embeds = rf_outputs["rf_embeds"]
            target_embeds = rf_outputs["target_embeds"]

            rf_users, rf_items = torch.split(rf_embeds, [self.n_users, self.n_items], dim=0)
            target_users, target_items = torch.split(target_embeds, [self.n_users, self.n_items], dim=0)

            rf_cl_loss = self.rf_generator._infonce_loss(rf_items[pos_items], target_items[pos_items], 0.2) + \
                         self.rf_generator._infonce_loss(rf_users[users], target_users[users], 0.2)

            total_loss = total_loss + self.rf_generator.contrast_weight * rf_cl_loss

        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e, _ = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
