import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import sys

from tqdm import tqdm

from codes.run_llama import run_llama

from codes.model.GCNModel import GNNModel, ContrastiveLoss, MergeNetwork, MLP
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from codes.BaseModel import BaseModel
from codes.model.text_prototypes import CrossAttentionModel, TransposedLinear

import time
import numpy as np

from sklearn import metrics
from codes.utils import dicts_to_embeddings, compute_batch_hop, compute_zero_WL


class AnomalyLLM(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    spy_tag = True

    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config, args):
        super(AnomalyLLM, self).__init__(config, args)
        self.args = args

        self.config = config
        self.transformer = BaseModel(config)
        self.gcn_model = GNNModel(config.hidden_size, 128, config.hidden_size)
        self.merge_network = MergeNetwork(config.hidden_size, 4, 4096)
        self.criterion = ContrastiveLoss(1.0)
        self.text_model = CrossAttentionModel(embed_dim=4096, num_heads=4)
        self.w_model = TransposedLinear(in_features=677, out_features=100)
        self.mlp = MLP(input_dim=4096, hidden_dim1=1024, hidden_dim2=512, output_dim=1)


        self.weight_decay = config.weight_decay
        self.init_weights()
        self.batch_hop_dicts = None
        self.hop_embeddings = None
        self.int_embeddings = None
        self.time_embeddings = None

    def forward(self, init_pos_ids, hop_dis_ids, time_dis_ids, len_pos, state, idx=None):

        if len_pos == 0:
            index = self.config.k
            outputs = self.transformer(init_pos_ids, hop_dis_ids, time_dis_ids)
            anchor_1 = self.extract_subtensor(outputs[0][:], 0, 2, index + 1)
            anchor_2 = self.extract_subtensor(outputs[0][:], 0, 2 * index + 4, 3 * index + 3)
            positive_1 = self.extract_subtensor(outputs[0][:], 1, index + 2, 2 * index + 1)
            positive_2 = self.extract_subtensor(outputs[0][:], 1, 3 * index + 4, 4 * index + 3)

            anchor, positive = torch.cat((anchor_1, anchor_2), dim=0), torch.cat((positive_1, positive_2), dim=0)
            adjacency_matrix = torch.zeros(index + 1, index + 1)
            for i in range(1, index + 1):
                adjacency_matrix[0, i] = 1.0
                adjacency_matrix[i, 0] = 1.0

            asequence_output = []

            for x in anchor:
                out = self.gcn_model(x, adjacency_matrix)
                asequence_output.append(out)
            asequence_output = torch.stack(asequence_output, dim=0)

            psequence_output = []
            for x in positive:
                out = self.gcn_model(x, adjacency_matrix)
                psequence_output.append(out)
            psequence_output = torch.stack(psequence_output, dim=0)
            len_pos = int(asequence_output.size()[0] / 2)
            achor_combined = torch.cat((asequence_output[:len_pos], asequence_output[len_pos:]), dim=1)
            positive_combined = torch.cat((psequence_output[:len_pos], psequence_output[len_pos:]), dim=1)
            edge_representation = self.merge_network(achor_combined, positive_combined)
            if state != "pretraining":
                normal_edge = self.merge_network(achor_combined, positive_combined)
                with open(r'./AnomalyLLM/data/text_prototype_embeddings.pkl', 'rb') as file:
                    text_embeddings = pickle.load(file)

                text_prototype_embeddings = self.w_model(text_embeddings)

                edge_representation = normal_edge
                output = self.text_model(edge_representation.unsqueeze(1), text_prototype_embeddings.unsqueeze(1),
                                        text_prototype_embeddings.unsqueeze(1))
                # llama_normal_edge = run_llama(normal_edge, "pre-train-normal", False)
                list_of_tensors = [t for t in output]
                result_list = [a + b for a, b in zip(list_of_tensors, normal_edge)]
                nsequence_output = None

                llama_normal_edge = run_llama(result_list, "few-shot", True)
                edge_representation = torch.mean(torch.stack(llama_normal_edge), dim=1)  # 输出大小为 (2000, 4096)

            output = self.mlp(edge_representation)
        
            return output

        else:
            index = self.config.k
            outputs = self.transformer(init_pos_ids, hop_dis_ids, time_dis_ids)
            anchor_1 = self.extract_subtensor(outputs[0][:len_pos], 0, 2, index + 1)
            anchor_2 = self.extract_subtensor(outputs[0][:len_pos], 0, 2 * index + 4, 3 * index + 3)
            positive_1 = self.extract_subtensor(outputs[0][:len_pos], 1, index + 2, 2 * index + 1)
            positive_2 = self.extract_subtensor(outputs[0][:len_pos], 1, 3 * index + 4, 4 * index + 3)
            nagivate_1 = self.extract_subtensor(outputs[0][len_pos:], 1, index + 2, 2 * index + 1)
            nagivate_2 = self.extract_subtensor(outputs[0][len_pos:], 1, 3 * index + 4, 4 * index + 3)

            anchor, positive, nagivate = torch.cat((anchor_1, anchor_2), dim=0), torch.cat((positive_1, positive_2),
                                                                                           dim=0), torch.cat(
                (nagivate_1, nagivate_2), dim=0)

            adjacency_matrix = torch.zeros(index + 1, index + 1)
            for i in range(1, index + 1):
                adjacency_matrix[0, i] = 1.0
                adjacency_matrix[i, 0] = 1.0

            asequence_output = []
            for x in anchor:
                out = self.gcn_model(x, adjacency_matrix)
                asequence_output.append(out)
            asequence_output = torch.stack(asequence_output, dim=0)

            psequence_output = []
            for x in positive:
                out = self.gcn_model(x, adjacency_matrix)
                psequence_output.append(out)
            psequence_output = torch.stack(psequence_output, dim=0)

            nsequence_output = []
            for x in nagivate:
                out = self.gcn_model(x, adjacency_matrix)
                nsequence_output.append(out)
            nsequence_output = torch.stack(nsequence_output, dim=0)

            achor_combined = torch.cat((asequence_output[:len_pos], asequence_output[len_pos:]), dim=1)
            positive_combined = torch.cat((psequence_output[:len_pos], psequence_output[len_pos:]), dim=1)
            negative_combined = torch.cat((nsequence_output[:len_pos], nsequence_output[len_pos:]), dim=1)

            normal_edge = self.merge_network(achor_combined, positive_combined)
            anomaly_edge = self.merge_network(achor_combined, negative_combined)
            edge_representation = torch.cat((normal_edge, anomaly_edge), dim=0)

            if state != "pretraining":
                with open(r'./AnomalyLLM/data/text_prototype_embeddings.pkl', 'rb') as file:
                    text_embeddings = pickle.load(file)
        
                edge_representation = torch.stack(normal_edge).float()
                text_prototype_embeddings = self.w_model(text_embeddings)

                normal_output = self.text_model(edge_representation.unsqueeze(1), text_prototype_embeddings.unsqueeze(1), text_prototype_embeddings.unsqueeze(1))
                list_of_tensors = [t for t in normal_output]
                result_list = [a + b for a, b in zip(list_of_tensors, normal_edge)]
                llama_normal_edge = run_llama(result_list, "few-shot", True)

                llama_normal_edge = torch.stack(llama_normal_edge)

                anomaly_edge = []
                for i in anomaly_edge_1:
                    ii = i
                    ii = ii.repeat(8)
                    ii = torch.stack([ii])
                    anomaly_edge.append(ii.squeeze())
                anomaly_edge_representation = torch.stack(anomaly_edge).float()
                anomaly_output = self.text_model(anomaly_edge_representation.unsqueeze(1), text_prototype_embeddings.unsqueeze(1), text_prototype_embeddings.unsqueeze(1))
                # llama_normal_edge = run_llama(normal_edge, "pre-train-normal", False)
                anomaly_list_of_tensors = [t for t in anomaly_output]
                anomaly_result_list = [a + b for a, b in zip(anomaly_list_of_tensors, anomaly_edge)]

                # llama_anomaly_edge = run_llama(anomaly_edge, "pre-train-anomalous", False)
                llama_anomaly_edge = run_llama(anomaly_result_list, "few-shot", True)
                llama_anomaly_edge = torch.stack(llama_anomaly_edge)
                
                edge_representation = torch.cat((llama_normal_edge, llama_anomaly_edge), dim=0)
                edge_representation = torch.mean(edge_representation, dim=1)  # 输出大小为 (2000, 4096)

            output = self.mlp(edge_representation)
            # output = edge_representation

        return output, asequence_output, psequence_output, nsequence_output

    def aggregate_graph_features(self, graph_features):
        graph_features_mean = torch.mean(graph_features, dim=1)
        return graph_features_mean

    def batch_cut(self, idx_list):
        batch_list = []
        for i in range(0, len(idx_list), self.config.batch_size):
            batch_list.append(idx_list[i:i + self.config.batch_size])
        return batch_list

    def extract_subtensor(self, input_tensor, f, g, h):

        subtensor_f = torch.index_select(input_tensor, dim=1, index=torch.tensor([f]))

        subtensor_gh = input_tensor[:, g - 1:h, :]

        result_tensor = torch.cat([subtensor_f, subtensor_gh],
                                  dim=1)

        return result_tensor

    def evaluate(self, trues, preds):
        aucs = {}
        aucss = []
        for snap in range(len(self.data['few_shot_test'])):
            auc = metrics.roc_auc_score(trues[snap], preds[snap])
            aucs[snap] = auc
            aucss.append(auc)
            print("Snap: %02d | AUC: %.4f" % (snap, auc))

        trues_full = np.hstack(trues)
        preds_full = np.hstack(preds)
        auc_full = metrics.roc_auc_score(trues_full, preds_full)
        # cr = sm.classification_report(trues_full, (preds_full >= 0.5).astype(int))
        # print('-------------------------------------------')
        print('TOTAL AUC:{:.4f}'.format(auc_full))
        # print(cr)
        print('-------------------------------------------')
        return auc_full

    def generate_embedding(self, edges):
        num_snap = len(edges)
        # WL_dict = compute_WL(self.data['idx'], np.vstack(edges[:7]))
        WL_dict = compute_zero_WL(self.data['idx'], np.vstack(edges[:7]))
        batch_hop_dicts = compute_batch_hop(self.data['idx'], edges, num_snap, self.data['S'], self.config.k,
                                            self.config.window_size)
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
            dicts_to_embeddings(self.data['X'], batch_hop_dicts, WL_dict, num_snap)

        return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings, batch_hop_dicts

    def generate_negative_edges(self, snap_edge, num_nodes, snap_id):
        negative_edge = []
        seen_edges = set() 
        for edge in snap_edge:
            start_node, _ = edge
            end_node = random.randint(1, num_nodes - 1)  
            new_edge = [start_node, end_node]

            while tuple(new_edge) in seen_edges:
                end_node = random.randint(1, num_nodes - 1)  
                new_edge = [start_node, end_node]

            negative_edge.append(new_edge)
            seen_edges.add(tuple(new_edge))

        return negative_edge

    def negative_sampling(self, edges):
        negative_edges = []
        snap_id = 0
        node_list = self.data['idx']
        num_node = node_list.shape[0]

        for snap_edge in edges:
            snap_id = snap_id + 1
            num_edge = snap_edge.shape[0]

            negative_edge = self.generate_negative_edges(snap_edge, num_node, snap_id)

            negative_edges.append(negative_edge)
        return negative_edges

    def pretraining(self, max_epoch):

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        raw_embeddings, wl_embeddings, self.hop_embeddings, self.int_embeddings, self.time_embeddings, self.batch_hop_dicts = self.generate_embedding(
            self.data['edges'])
        self.data['raw_embeddings'] = None
        state = "pretraining"
        ns_function = self.negative_sampling

        for epoch in range(max_epoch):

            t_epoch_begin = time.time()
            negatives = ns_function(self.data['edges'][0:max(self.data['snap_train']) + 1])
            raw_embeddings_neg, wl_embeddings_neg, hop_embeddings_neg, int_embeddings_neg, \
                time_embeddings_neg, batch_hop_dicts_neg = self.generate_embedding(negatives)
            self.train()
            loss_train = 0
            for snap in self.data['snap_train']:

                if wl_embeddings[snap] is None:
                    continue
                int_embedding_pos = self.int_embeddings[snap]
                hop_embedding_pos = self.hop_embeddings[snap]
                time_embedding_pos = self.time_embeddings[snap]
                y_pos = self.data['y'][snap].float()

                int_embedding_neg = int_embeddings_neg[snap]
                hop_embedding_neg = hop_embeddings_neg[snap]
                time_embedding_neg = time_embeddings_neg[snap]
                y_neg = torch.ones(int_embedding_neg.size()[0])

                int_embedding = torch.vstack((int_embedding_pos, int_embedding_neg))
                hop_embedding = torch.vstack((hop_embedding_pos, hop_embedding_neg))
                time_embedding = torch.vstack((time_embedding_pos, time_embedding_neg))

                len_pos = y_pos.size()[0]
                y = torch.hstack((y_pos, y_neg))

                optimizer.zero_grad()

                output, anchor_representation, positive_representation, negative_representation = self.forward(
                    int_embedding, hop_embedding, time_embedding, len_pos, state)

                output = output.squeeze()

                loss_c = self.criterion(anchor_representation, positive_representation, negative_representation)

                loss_r = F.binary_cross_entropy_with_logits(output, y)
                loss = loss_r + loss_c
                loss.backward()
                optimizer.step()

                loss_train += loss.detach().item()

            loss_train /= len(self.data['snap_train']) - self.config.window_size + 1
            print('Epoch: {}, loss:{:.4f}, Time: {:.4f}s'.format(epoch + 1, loss_train, time.time() - t_epoch_begin))


    def alignment(self, max_epoch):

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        raw_embeddings, wl_embeddings, self.hop_embeddings, self.int_embeddings, self.time_embeddings, self.batch_hop_dicts = self.generate_embedding(
            self.data['edges'])
        self.data['raw_embeddings'] = None
        state = "alignment"

        ns_function = self.negative_sampling

        for epoch in range(max_epoch):

            t_epoch_begin = time.time()
            checkpoint_file = r"./data/pre_train.pth"
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Load model successfully")
            negatives = ns_function(self.data['edges'][0:max(self.data['snap_train']) + 1])
            raw_embeddings_neg, wl_embeddings_neg, hop_embeddings_neg, int_embeddings_neg, \
                time_embeddings_neg, batch_hop_dicts_neg = self.generate_embedding(negatives)
            self.train()
            loss_train = 0
            for snap in self.data['snap_train']:

                if wl_embeddings[snap] is None:
                    continue
                int_embedding_pos = self.int_embeddings[snap]
                hop_embedding_pos = self.hop_embeddings[snap]
                time_embedding_pos = self.time_embeddings[snap]
                y_pos = self.data['y'][snap].float()

                int_embedding_neg = int_embeddings_neg[snap]
                hop_embedding_neg = hop_embeddings_neg[snap]
                time_embedding_neg = time_embeddings_neg[snap]
                y_neg = torch.ones(int_embedding_neg.size()[0])

                int_embedding = torch.vstack((int_embedding_pos, int_embedding_neg))
                hop_embedding = torch.vstack((hop_embedding_pos, hop_embedding_neg))
                time_embedding = torch.vstack((time_embedding_pos, time_embedding_neg))

                len_pos = y_pos.size()[0]
                y = torch.hstack((y_pos, y_neg))

                optimizer.zero_grad()

                output, anchor_representation, positive_representation, negative_representation = self.forward(
                    int_embedding, hop_embedding, time_embedding, len_pos, state)

                output = output.squeeze()

                loss_c = self.criterion(anchor_representation, positive_representation, negative_representation)

                loss_r = F.binary_cross_entropy_with_logits(output, y)
                loss = loss_r + loss_c
                loss.backward()
                optimizer.step()

                loss_train += loss.detach().item()

            loss_train /= len(self.data['snap_train']) - self.config.window_size + 1
            print('Epoch: {}, loss:{:.4f}, Time: {:.4f}s'.format(epoch + 1, loss_train, time.time() - t_epoch_begin))

            self.eval()
            preds = []
            few_shot_test = []
            for i in self.data['few_shot_test']:
                few_shot_test.append(self.data['y'][i])
            ii = 0
            for snap in self.data['few_shot_test']:
                int_embedding = self.int_embeddings[snap]
                hop_embedding = self.hop_embeddings[snap]
                time_embedding = self.time_embeddings[snap]
                len_pos = 0
                with torch.no_grad():
                    output = self.forward(int_embedding, hop_embedding, time_embedding, len_pos, state)
                    output = torch.sigmoid(output)
                pred = output.squeeze().numpy()
                preds.append(pred)
                # wwww = self.data['y'][min(self.data['few_shot_test']):max(self.data['few_shot_test']) + 1]
                auc = metrics.roc_auc_score(few_shot_test[ii], pred)
                ii = ii+1
                print("AUC: %.4f" % auc)
            trues_full = np.hstack(few_shot_test)
            preds_full = np.hstack(preds)
            auc_full = metrics.roc_auc_score(trues_full, preds_full)
            # cr = sm.classification_report(trues_full, (preds_full >= 0.5).astype(int))
            # print('-------------------------------------------')
            print('TOTAL AUC:{:.4f}'.format(auc_full))

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'loss': loss_train,
            }, save_path)
            print(save_path)

    def evaluate(self, max_epoch):

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        raw_embeddings, wl_embeddings, self.hop_embeddings, self.int_embeddings, self.time_embeddings, self.batch_hop_dicts = self.generate_embedding(
            self.data['edges'])
        self.data['raw_embeddings'] = None

        ns_function = self.negative_sampling

        checkpoint_file = r"./data/Llama_uci_340.8466256048387097_model.pth"
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.eval()
        preds = []
        state = "evaluate"
        few_shot_test = []
        for i in self.data['few_shot_test']:
            few_shot_test.append(self.data['y'][i])
        ii = 0
        for snap in tqdm(self.data['few_shot_test'], bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            int_embedding = self.int_embeddings[snap]
            hop_embedding = self.hop_embeddings[snap]
            time_embedding = self.time_embeddings[snap]
            len_pos = 0
            with torch.no_grad():
                output = self.forward(int_embedding, hop_embedding, time_embedding, len_pos, state)
                output = torch.sigmoid(output)
            pred = output.squeeze().numpy()
            preds.append(pred)
            auc = metrics.roc_auc_score(few_shot_test[ii], pred)
            ii = ii+1
            print("AUC: %.4f" % auc)
        trues_full = np.hstack(few_shot_test)
        preds_full = np.hstack(preds)
        auc_full = metrics.roc_auc_score(trues_full, preds_full)
        print('TOTAL AUC:{:.4f}'.format(auc_full))


        save_path = 'Llama_uci_' + str(auc_full) + '_model.pth'
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss_train,
        }, save_path)
        print(save_path)
    def run(self, step):
        if step == "evaluate":
            self.evaluate(1)
            return self.learning_record_dict
        if step == "pretraining":
            self.pretraining(self.max_epoch)
            return self.learning_record_dict
        if step == "alignment":
            self.evaluate(1)
            return self.learning_record_dict
