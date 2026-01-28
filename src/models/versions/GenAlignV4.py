import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
from collections import defaultdict
import math
from scipy.sparse import lil_matrix
import random
import json


class RectifiedFlow(nn.Module):
    '''
    ersion 2
    Rectified Flow for Cross-Modal Alignment
    Args:
        dim: dimension of the input and output
        hidden_dim: dimension of the hidden layer
    Returns:
        output: the output of the Rectified Flow
    '''
    def __init__(self, dim, hidden_dim=None):
        super(RectifiedFlow, self).__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)

    def compute_loss(self, x0, x1):
        batch_size = x0.size(0)
        t = torch.rand(batch_size, 1, device=x0.device)
        z_t = t * x1 + (1 - t) * x0
        v_target = x1 - x0 # 目标传输向量v_target：图像到文本的真实差值（最优传输方向）跨模态语义gap
        v_pred = self.forward(z_t, t) # 模型预测的 t 时刻传输方向
        loss = F.mse_loss(v_pred, v_target)
        return loss

class ConditionalRectifiedFlow(nn.Module):
    """
    支持条件输入的 Rectified Flow
    用于: Noise (x0) -> Agent (t=0.5) -> ID (x1), Conditioned on Multimodal Features
    """
    def __init__(self, dim, hidden_dim=None):
        super(ConditionalRectifiedFlow, self).__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2  # 稍微增大隐藏层，因为输入维度变大了
            
        # 输入: x(dim) + t(1) + condition(dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim) # 输出仅为 velocity (维度与 x 一致)
        )
    
    def forward(self, x, t, condition):
        # 将 噪声/中间态、时间、多模态条件 拼接
        xtc = torch.cat([x, t, condition], dim=-1)
        return self.net(xtc)

    def compute_loss(self, target_id, condition_feat):
        """
        计算 Flow Matching Loss
        Args:
            target_id (x1): 目标 ID Embedding (Sparse Signal)
            condition_feat (c): 多模态特征 (Dense Signal)
        """
        batch_size = target_id.size(0)
        
        # 1. 定义起点 x0: 高斯噪声
        x0 = torch.randn_like(target_id)
        
        # 2. 定义终点 x1: 真实的协同 ID
        x1 = target_id
        
        # 3. 采样时间 t
        t = torch.rand(batch_size, 1, device=x0.device)
        
        # 4. 构造插值路径 (Interpolation)
        z_t = t * x1 + (1 - t) * x0
        
        # 5. 计算目标速度 (Target Velocity)
        v_target = x1 - x0
        
        # 6. 模型预测速度 (Predicted Velocity)
        # 关键：模型必须看到 condition 才能知道往哪走
        v_pred = self.forward(z_t, t, condition_feat)
        
        # 7. MSE Loss
        loss = F.mse_loss(v_pred, v_target)
        return loss

    def generate_agent_modality(self, condition_feat):
        """
        推理阶段：生成 t=0.5 的 Agent Modality
        """
        batch_size = condition_feat.size(0)
        
        # 1. 起点：采样高斯噪声
        z_0 = torch.randn_like(condition_feat)
        
        # 2. 时间：设定为 t=0
        t_zero = torch.zeros(batch_size, 1, device=condition_feat.device)
        
        # 3. 预测初始方向
        v_pred = self.forward(z_0, t_zero, condition_feat)
        
        # 4. 欧拉一步：走到 t=0.5
        # z_{0.5} = z_0 + 0.5 * v
        agent_modality = z_0 + 0.5 * v_pred
        
        return agent_modality

class GenAlignGUME(GeneralRecommender):
    def __init__(self, config, dataset):
        super(GenAlignGUME, self).__init__(config, dataset)
        self.sparse = True
        self.bm_loss = config['bm_loss']
        self.um_loss = config['um_loss']
        self.vt_loss = config['vt_loss']
        self.reg_weight_1 = config['reg_weight_1']
        self.reg_weight_2 = config['reg_weight_2']
        self.bm_temp = config['bm_temp']
        self.um_temp = config['um_temp']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.rf_weight = config['rf_weight'] if 'rf_weight' in config else 0.1
        
        self.agent_weight = config['agent_weight'] if 'agent_weight' in config else 0.2
        self.agent_loss_weight = config['agent_loss_weight'] if 'agent_loss_weight' in config else 0.1
        self.agent_warmup_epochs = config['agent_warmup_epochs'] if 'agent_warmup_epochs' in config else 0
        self.current_epoch = 0

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        self.extended_image_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_image_user.weight)
        
        self.extended_text_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_text_user.weight)

        # self.dataset_path = os.path.abspath(os.getcwd()+config['data_path'] + config['dataset'])
        self.dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.data_name = config['dataset']

        image_adj_file = os.path.join(self.dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(self.dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        #  Enhancing User-Item Graph
        self.inter = self.find_inter(self.image_original_adj, self.text_original_adj)
        self.ii_adj = self.add_edge(self.inter)
        self.norm_adj = self.get_adj_mat(self.ii_adj.tolil())
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)
        
        
        self.image_reduce_dim = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.image_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.image_space_trans = nn.Sequential(
            self.image_reduce_dim,
            self.image_trans_dim
        )
        
        self.text_reduce_dim = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.text_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_space_trans = nn.Sequential(
            self.text_reduce_dim,
            self.text_trans_dim
        )
        
        self.separate_coarse = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )
        
        self.softmax = nn.Softmax(dim=-1)
                
        self.image_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.tau = 0.5
        
        # 1. 模态内对齐模型 (Intra-modal: Image <-> Text)
        self.rf_intra_model = RectifiedFlow(self.embedding_dim).to(self.device)
        
        # 2. Agent 对齐模型 (Inter-modal: Noise -> ID | Condition=Content)
        self.rf_agent_model = ConditionalRectifiedFlow(self.embedding_dim).to(self.device)

    def get_dynamic_agent_weights(self):
        """
        Calculate dynamic weights for Agent Modality based on warmup schedule.
        Returns:
            (curr_agent_weight, curr_agent_loss_weight)
        """
        if self.agent_warmup_epochs > 0:
            if self.current_epoch < self.agent_warmup_epochs:
                ratio = self.current_epoch / self.agent_warmup_epochs
                return self.agent_weight * ratio, self.agent_loss_weight * ratio
        return self.agent_weight, self.agent_loss_weight
    
    def find_inter(self, image_adj, text_adj):
        inter_file = os.path.join(self.dataset_path, 'inter.json')
        if os.path.exists(inter_file):
            with open(inter_file) as f:
                inter = json.load(f)
        else:
            j = 0
            inter = defaultdict(list)
            img_sim = []
            txt_sim = []
            for i in range(0,len(image_adj._indices()[0])):
                img_id = image_adj._indices()[0][i]
                txt_id = text_adj._indices()[0][i]
                assert img_id == txt_id
                id = img_id.item()
                img_sim.append(image_adj._indices()[1][j].item())
                txt_sim.append(text_adj._indices()[1][j].item())
                
                if len(img_sim)==10 and len(txt_sim)==10:
                    it_inter = list(set(img_sim) & set(txt_sim))
                    inter[id] = [v for v in it_inter if v != id]
                    img_sim = []
                    txt_sim = []
                
                j += 1
            
            with open(inter_file, "w") as f:
                json.dump(inter, f)
        
        return inter

    def add_edge(self, inter):
        sim_rows = []
        sim_cols = []
        for id, vs in inter.items():
            if len(vs) == 0:
                continue
            for v in vs:
                sim_rows.append(int(id))
                sim_cols.append(v)
        
        sim_rows = torch.tensor(sim_rows)
        sim_cols = torch.tensor(sim_cols)
        sim_values = [1]*len(sim_rows)

        item_adj = sp.coo_matrix((sim_values, (sim_rows, sim_cols)), shape=(self.n_items,self.n_items), dtype=np.int64)
        return item_adj
    
    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self, item_adj):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T

        adj_mat[self.n_users:, self.n_users:] = item_adj
        
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def conv_ui(self, adj, user_embeds, item_embeds):
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        
        return all_embeddings

    def conv_ii(self, ii_adj, single_modal):
        for i in range(self.n_layers):
            single_modal = torch.sparse.mm(ii_adj, single_modal)
        return single_modal

    def forward(self, adj, train=False):
        #  Encoding Multiple Modalities

        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.image_space_trans(self.image_embedding.weight))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.text_space_trans(self.text_embedding.weight))

        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight

        extended_id_embeds = self.conv_ui(adj, user_embeds, item_embeds)
        
        explicit_image_item = self.conv_ii(self.image_original_adj, image_item_embeds)
        explicit_image_user = torch.sparse.mm(self.R, explicit_image_item)
        explicit_image_embeds = torch.cat([explicit_image_user, explicit_image_item], dim=0)
        
        extended_image_embeds = self.conv_ui(adj, self.extended_image_user.weight, explicit_image_item) 

        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R, explicit_text_item)
        explicit_text_embeds = torch.cat([explicit_text_user, explicit_text_item], dim=0)
        
        extended_text_embeds = self.conv_ui(adj, self.extended_text_user.weight, explicit_text_item)

        extended_it_embeds = (extended_image_embeds + extended_text_embeds) / 2
        
        # Attributes Separation for Better Integration
        image_weights, text_weights = torch.split(
            self.softmax(
                torch.cat([
                    self.separate_coarse(explicit_image_embeds),
                    self.separate_coarse(explicit_text_embeds)
                ], dim=-1)
            ),
            1,
            dim=-1
        )
        coarse_grained_embeds = image_weights * explicit_image_embeds + text_weights * explicit_text_embeds
                
        fine_grained_image = torch.multiply(self.image_behavior(extended_id_embeds), (explicit_image_embeds - coarse_grained_embeds))
        fine_grained_text = torch.multiply(self.text_behavior(extended_id_embeds), (explicit_text_embeds - coarse_grained_embeds))
        integration_embeds = (fine_grained_image + fine_grained_text + coarse_grained_embeds) / 3

        all_embeds = extended_id_embeds + integration_embeds

        if train:
            return all_embeds, (integration_embeds, extended_id_embeds, extended_it_embeds), (explicit_image_embeds, explicit_text_embeds)

        return all_embeds

    def sq_sum(self, emb):
        return 1. / 2 * (emb ** 2).sum()
    
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = (self.sq_sum(users) + self.sq_sum(pos_items) + self.sq_sum(neg_items)) / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        reg_loss = self.reg_weight_1 * regularizer

        return mf_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        embeds_1, embeds_2, embeds_3 = self.forward(self.norm_adj, train=True)
        users_embeddings, items_embeddings = torch.split(embeds_1, [self.n_users, self.n_items], dim=0)
        
        integration_embeds, extended_id_embeds, extended_it_embeds = embeds_2
        explicit_image_embeds, explicit_text_embeds = embeds_3
 

        u_g_embeddings = users_embeddings[users]
        pos_i_g_embeddings = items_embeddings[pos_items]
        neg_i_g_embeddings = items_embeddings[neg_items]

        vt_loss = self.vt_loss * self.align_vt(explicit_image_embeds, explicit_text_embeds)
        
        integration_users, integration_items = torch.split(integration_embeds, [self.n_users, self.n_items], dim=0)
        extended_id_user, extended_id_items = torch.split(extended_id_embeds, [self.n_users, self.n_items], dim=0)
        bpr_loss, reg_loss_1 = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,neg_i_g_embeddings)
        
        bm_loss = self.bm_loss * (self.InfoNCE(integration_users[users], extended_id_user[users], self.bm_temp) + self.InfoNCE(integration_items[pos_items], extended_id_items[pos_items], self.bm_temp))
        
        al_loss = vt_loss + bm_loss
        
        extended_it_user, extended_it_items = torch.split(extended_it_embeds, [self.n_users, self.n_items], dim=0)

        # Enhancing User Modality Representation
        c_loss = self.InfoNCE(extended_it_user[users], integration_users[users], self.um_temp)
        noise_loss_1 = self.cal_noise_loss(users, integration_users, self.um_temp)
        noise_loss_2 = self.cal_noise_loss(users, extended_it_user, self.um_temp)
        um_loss = self.um_loss * (c_loss + noise_loss_1 + noise_loss_2)
        
        reg_loss_2 = self.reg_weight_2 * self.sq_sum(extended_it_items[pos_items]) / self.batch_size
        reg_loss = reg_loss_1 + reg_loss_2
        
        # ===================================================== 
        #  Task A: 模态内对齐 (Intra-Modality Alignment) 
        #  Image -> Text (或双向) 
        # ===================================================== 
        unique_items = torch.unique(torch.cat([pos_items, neg_items])) 
        batch_img = explicit_image_embeds[unique_items] 
        batch_txt = explicit_text_embeds[unique_items] 
        
        # 计算 Image -> Text 的流损失 
        loss_intra_flow = self.rf_weight * self.rf_intra_model.compute_loss(batch_img, batch_txt) 
        
        # ===================================================== 
        #  Task B: Agent Modality 对齐 (Generative Alignment) 
        #  Noise -> ID (Conditioned on Multimodal Content) 
        # ===================================================== 
        # 目标: 让 ID Embedding (Target) 
        batch_target_id = extended_id_embeds[unique_items].detach() # 建议 detach ID，让 Flow 去追 ID 
        
        # 条件: 多模态融合特征 (Condition) 
        batch_condition = integration_embeds[unique_items].detach() 
        
        # 计算 Conditional Flow Loss 
        loss_agent_flow = self.agent_loss_weight * self.rf_agent_model.compute_loss( 
            target_id=batch_target_id, 
            condition_feat=batch_condition 
        ) 
        
        return bpr_loss + al_loss + um_loss + reg_loss + loss_intra_flow + loss_agent_flow
    
    
    def cal_noise_loss(self, id, emb, temp):

        def add_perturbation(x):
            random_noise = torch.rand_like(x).to(self.device)
            x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
            return x

        emb_view1 = add_perturbation(emb)
        emb_view2 = add_perturbation(emb)
        emb_loss = self.InfoNCE(emb_view1[id], emb_view2[id], temp)

        return emb_loss
    
    def align_vt(self,embed1, embed2):
        emb1_var, emb1_mean = torch.var(embed1), torch.mean(embed1)
        emb2_var, emb2_mean = torch.var(embed2), torch.mean(embed2)
        
        vt_loss = (torch.abs(emb1_var - emb2_var) + torch.abs(emb1_mean - emb2_mean)).mean()
        
        return vt_loss
    
    def full_sort_predict(self, interaction):
        """ 
        推理阶段：利用 Agent Modality 增强推荐 
        """ 
        user = interaction[0] 

        # 1. 获取基础特征 (Use train=True to get intermediate embeddings)
        # We are in eval mode so gradients are off anyway (from Trainer)
        all_embeds, embeds_2, embeds_3 = self.forward(self.norm_adj, train=True)
        
        # Unpack integration_embeds
        integration_embeds, extended_id_embeds, extended_it_embeds = embeds_2
        
        # Split into user and item parts
        restore_user_e, restore_item_e = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)
        integration_users, integration_items = torch.split(integration_embeds, [self.n_users, self.n_items], dim=0)
        
        # 2. 获取多模态条件特征 (Condition) - 使用与训练一致的 integration_items
        condition_feat = integration_items
        
        # 3. 生成 Agent Modality (t=0.5) !!! 核心 !!! 
        # 输入: 多模态特征作为条件 
        # 输出: 位于 噪声 和 ID 中间的 Agent 向量 
        with torch.no_grad(): 
            agent_modality = self.rf_agent_model.generate_agent_modality(condition_feat) 
        
        # 4. 增强物品表示 
        # 最终 Item = 原始 GCN Item + α * Agent Modality 
        # Agent Modality 补全了稀疏 ID 缺失的语义信息，且分布比纯多模态特征更接近协同空间 
        curr_agent_weight, _ = self.get_dynamic_agent_weights()
        final_item_embeddings = restore_item_e + curr_agent_weight * agent_modality 
        
        # 5. 预测 
        u_embeddings = restore_user_e[user] 
        scores = torch.matmul(u_embeddings, final_item_embeddings.transpose(0, 1)) 
        
        return scores
