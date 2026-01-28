# Rectified Flow for GUME: RF-GUME
# Integrating Rectified Flow into GUME for generating extended_id_embeds

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gume import GUME
from models.rf_modules import RFEmbeddingGenerator


class RFGUME(GUME):
    """
    GUME with Rectified Flow

    This is a refactored version that uses the pluggable RF module.
    The RF components are now fully decoupled and can be reused in other models.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # RF相关参数
        self.use_rf = config["use_rf"] if "use_rf" in config else True

        if self.use_rf:
            # 初始化RF生成器
            self.rf_generator = RFEmbeddingGenerator(
                # 必需参数
                embedding_dim=self.embedding_dim,

                # 可选架构参数
                hidden_dim=config["rf_hidden_dim"] if "rf_hidden_dim" in config else 256,
                n_layers=config["rf_n_layers"] if "rf_n_layers" in config else 2,
                dropout=config["rf_dropout"] if "rf_dropout" in config else 0.1,

                # 可选训练参数
                learning_rate=config["rf_learning_rate"] if "rf_learning_rate" in config else 0.001,
                sampling_steps=config["rf_sampling_steps"] if "rf_sampling_steps" in config else 10,

                # 可选高级参数（先验指导）
                user_guidance_scale=config["user_guidance_scale"] if "user_guidance_scale" in config else 0.2,
                guidance_decay_power=config["guidance_decay_power"] if "guidance_decay_power" in config else 2.0,
                cosine_guidance_scale=config["cosine_guidance_scale"] if "cosine_guidance_scale" in config else 0.1,
                cosine_decay_power=config["cosine_decay_power"] if "cosine_decay_power" in config else 2.0,

                # 可选策略参数
                warmup_epochs=config["rf_warmup_epochs"] if "rf_warmup_epochs" in config else 0,
                train_mix_ratio=config["rf_mix_ratio"] if "rf_mix_ratio" in config else 0.5,
                inference_mix_ratio=config["rf_inference_mix_ratio"] if "rf_inference_mix_ratio" in config else 0.5,
                contrast_temp=config["rf_contrast_temp"] if "rf_contrast_temp" in config else 0.2,
                contrast_weight=config["rf_loss_weight"] if "rf_loss_weight" in config else 1.0,
            )

            # 用于控制每个epoch只打印一次日志
            self._rf_logged_this_epoch = False

            print(
                f"RF-GUME initialized with pluggable RF module:\n"
                f"  Architecture: hidden_dim={self.rf_generator.hidden_dim}, "
                f"n_layers={self.rf_generator.n_layers}\n"
                f"  Training: lr={self.rf_generator.learning_rate}, "
                f"sampling_steps={self.rf_generator.sampling_steps}, "
                f"warmup_epochs={self.rf_generator.warmup_epochs}\n"
                f"  Mix ratios: train={self.rf_generator.train_mix_ratio}, "
                f"inference={self.rf_generator.inference_mix_ratio}\n"
                f"  User guidance: scale={self.rf_generator.user_guidance_scale}, "
                f"decay={self.rf_generator.guidance_decay_power}\n"
                f"  Cosine guidance: scale={self.rf_generator.cosine_guidance_scale}, "
                f"decay={self.rf_generator.cosine_decay_power}"
            )

    def set_epoch(self, epoch):
        """由 trainer 调用，更新当前 epoch"""
        if self.use_rf:
            self.rf_generator.set_epoch(epoch)
            self._rf_logged_this_epoch = False  # 重置日志标记

    def is_rf_active(self):
        """检查 RF 是否已激活"""
        return self.use_rf

    def forward(self, adj, train=False):
        """
        使用RF生成extended_id_embeds的前向传播

        Args:
            adj: 邻接矩阵
            train: 是否训练模式

        Returns:
            all_embeds: 最终的嵌入
            rf_outputs: RF相关的输出（用于计算损失）
            other_outputs: 其他输出
        """
        # ===== 原始GUME的多模态编码 =====
        image_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.image_space_trans(self.image_embedding.weight),
        )
        text_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.text_space_trans(self.text_embedding.weight),
        )

        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight

        # 原始的extended_id_embeds（用作RF的目标）
        extended_id_embeds_target = self.conv_ui(adj, user_embeds, item_embeds)

        # ===== 显式的多模态特征（作为RF的条件） =====
        # Image modality
        explicit_image_item = self.conv_ii(self.image_original_adj, image_item_embeds)
        explicit_image_user = torch.sparse.mm(self.R, explicit_image_item)
        explicit_image_embeds = torch.cat(
            [explicit_image_user, explicit_image_item], dim=0
        )

        # Text modality
        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R, explicit_text_item)
        explicit_text_embeds = torch.cat(
            [explicit_text_user, explicit_text_item], dim=0
        )

        # ===== 使用RF生成extended_id_embeds =====
        if self.use_rf and train:
            # ===== 训练模式：RF独立训练 =====

            # 计算用户先验（用于RF指导）
            # Z_u: 用户特定的多模态兴趣表示
            Z_u = explicit_image_embeds[:self.n_users] + explicit_text_embeds[:self.n_users]

            # Z_hat: 通用兴趣表示（所有用户的平均值）
            Z_hat = Z_u.mean(dim=0, keepdim=True)

            # 用户先验: 独特的用户兴趣
            user_prior = Z_u - Z_hat  # shape: (n_users, embedding_dim)

            # 对于物品，不使用个性化指导（零指导）
            item_prior = torch.zeros(self.n_items, self.embedding_dim).to(Z_u.device)

            # 合并用户和物品先验
            full_prior = torch.cat([user_prior, item_prior], dim=0)

            # 使用RF生成器计算损失并更新
            loss_dict = self.rf_generator.compute_loss_and_step(
                target_embeds=extended_id_embeds_target.detach(),
                conditions=[explicit_image_embeds.detach(), explicit_text_embeds.detach()],
                user_prior=full_prior.detach(),
                epoch=self.rf_generator.current_epoch,
            )

            # 打印RF训练信息（每个epoch只打印一次）
            if not self._rf_logged_this_epoch:
                print(
                    f"  [RF Train] epoch={self.rf_generator.current_epoch}, "
                    f"rf_loss={loss_dict['rf_loss']:.6f}"
                )
                self._rf_logged_this_epoch = True

            # 生成RF embeddings
            rf_embeds = self.rf_generator.generate(
                [explicit_image_embeds, explicit_text_embeds]
            )

            # 混合原始和RF生成的embeddings
            extended_id_embeds = self.rf_generator.mix_embeddings(
                extended_id_embeds_target,
                rf_embeds.detach(),
                training=True,
                epoch=self.rf_generator.current_epoch,
            )

        elif self.use_rf and not train:
            # ===== 推理模式：使用RF生成并混合 =====
            with torch.no_grad():
                rf_embeds = self.rf_generator.generate(
                    [explicit_image_embeds, explicit_text_embeds]
                )
                extended_id_embeds = self.rf_generator.mix_embeddings(
                    extended_id_embeds_target,
                    rf_embeds,
                    training=False,
                    epoch=self.rf_generator.current_epoch,
                )
        else:
            # 不使用RF，保持原始GUME行为
            extended_id_embeds = extended_id_embeds_target

        # ===== 继续GUME的其余部分 =====
        extended_image_embeds = self.conv_ui(
            adj, self.extended_image_user.weight, explicit_image_item
        )
        extended_text_embeds = self.conv_ui(
            adj, self.extended_text_user.weight, explicit_text_item
        )
        extended_it_embeds = (extended_image_embeds + extended_text_embeds) / 2

        # Attributes Separation for Better Integration
        image_weights, text_weights = torch.split(
            self.softmax(
                torch.cat(
                    [
                        self.separate_coarse(explicit_image_embeds),
                        self.separate_coarse(explicit_text_embeds),
                    ],
                    dim=-1,
                )
            ),
            1,
            dim=-1,
        )
        coarse_grained_embeds = (
            image_weights * explicit_image_embeds + text_weights * explicit_text_embeds
        )

        fine_grained_image = torch.multiply(
            self.image_behavior(extended_id_embeds),
            (explicit_image_embeds - coarse_grained_embeds),
        )
        fine_grained_text = torch.multiply(
            self.text_behavior(extended_id_embeds),
            (explicit_text_embeds - coarse_grained_embeds),
        )
        integration_embeds = (
            fine_grained_image + fine_grained_text + coarse_grained_embeds
        ) / 3

        all_embeds = extended_id_embeds + integration_embeds

        if train and self.use_rf:
            other_outputs = {
                "integration_embeds": integration_embeds,
                "extended_id_embeds": extended_id_embeds,
                "extended_it_embeds": extended_it_embeds,
                "explicit_image_embeds": explicit_image_embeds,
                "explicit_text_embeds": explicit_text_embeds,
                "rf_embeds": rf_embeds,
                "extended_id_embeds_target": extended_id_embeds_target,
            }

            return all_embeds, other_outputs
        elif train and not self.use_rf:
            return (
                all_embeds,
                (integration_embeds, extended_id_embeds, extended_it_embeds),
                (explicit_image_embeds, explicit_text_embeds),
            )

        return all_embeds

    def calculate_loss(self, interaction):
        """
        计算总损失（RF损失已在forward中独立计算和反向传播）

        Args:
            interaction: 交互数据

        Returns:
            total_loss: 总损失
        """
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # 前向传播（RF损失已在forward中独立计算和反向传播）
        if self.is_rf_active():
            embeds_1, other_outputs = self.forward(self.norm_adj, train=True)

            integration_embeds = other_outputs["integration_embeds"]
            extended_id_embeds = other_outputs["extended_id_embeds"]
            extended_it_embeds = other_outputs["extended_it_embeds"]
            explicit_image_embeds = other_outputs["explicit_image_embeds"]
            explicit_text_embeds = other_outputs["explicit_text_embeds"]
        else:
            embeds_1, embeds_2, embeds_3 = self.forward(self.norm_adj, train=True)
            integration_embeds, extended_id_embeds, extended_it_embeds = embeds_2
            explicit_image_embeds, explicit_text_embeds = embeds_3

        users_embeddings, items_embeddings = torch.split(
            embeds_1, [self.n_users, self.n_items], dim=0
        )

        # ===== 原始GUME损失 =====
        u_g_embeddings = users_embeddings[users]
        pos_i_g_embeddings = items_embeddings[pos_items]
        neg_i_g_embeddings = items_embeddings[neg_items]

        vt_loss = self.vt_loss * self.align_vt(
            explicit_image_embeds, explicit_text_embeds
        )

        integration_users, integration_items = torch.split(
            integration_embeds, [self.n_users, self.n_items], dim=0
        )
        extended_id_user, extended_id_items = torch.split(
            extended_id_embeds, [self.n_users, self.n_items], dim=0
        )
        bpr_loss, reg_loss_1 = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )

        bm_loss = self.bm_loss * (
            self.InfoNCE(
                integration_users[users], extended_id_user[users], self.bm_temp
            )
            + self.InfoNCE(
                integration_items[pos_items], extended_id_items[pos_items], self.bm_temp
            )
        )

        al_loss = vt_loss + bm_loss

        extended_it_user, extended_it_items = torch.split(
            extended_it_embeds, [self.n_users, self.n_items], dim=0
        )

        c_loss = self.InfoNCE(
            extended_it_user[users], integration_users[users], self.um_temp
        )
        noise_loss_1 = self.cal_noise_loss(users, integration_users, self.um_temp)
        noise_loss_2 = self.cal_noise_loss(users, extended_it_user, self.um_temp)
        um_loss = self.um_loss * (c_loss + noise_loss_1 + noise_loss_2)

        reg_loss_2 = (
            self.reg_weight_2
            * self.sq_sum(extended_it_items[pos_items])
            / self.batch_size
        )
        reg_loss = reg_loss_1 + reg_loss_2

        # RF contrastive loss (cl_loss)
        rf_cl_loss = 0.0
        if self.is_rf_active():
            rf_embeds = other_outputs["rf_embeds"]
            target_embeds = other_outputs["extended_id_embeds_target"]

            rf_users, rf_items = torch.split(rf_embeds, [self.n_users, self.n_items], dim=0)
            target_users, target_items = torch.split(target_embeds, [self.n_users, self.n_items], dim=0)

            # cl_loss with pos_items and users indices (using rf_generator._infonce_loss)
            rf_cl_loss = self.rf_generator._infonce_loss(rf_items[pos_items], target_items[pos_items], 0.2) + \
                         self.rf_generator._infonce_loss(rf_users[users], target_users[users], 0.2)

        return bpr_loss + al_loss + um_loss + reg_loss + self.rf_generator.contrast_weight * rf_cl_loss

    def full_sort_predict(self, interaction):
        """
        预测（推理模式）
        """
        user = interaction[0]

        if self.use_rf:
            all_embeds = self.forward(self.norm_adj, train=False)
        else:
            all_embeds = self.forward(self.norm_adj)

        restore_user_e, restore_item_e = torch.split(
            all_embeds, [self.n_users, self.n_items], dim=0
        )
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
