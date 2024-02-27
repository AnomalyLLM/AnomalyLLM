import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from torch.nn import DataParallel
import warnings
import sys
from codes.conversation import conv_templates, SeparatorStyle

from tqdm import tqdm
from codes.model.MetaLlama import DyGLlamaForCausalLM
import time
import pickle
warnings.filterwarnings("ignore")
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

DEFAULT_EDGE_TOKEN = "edge"
DEFAULT_EDGE_PATCH_TOKEN = "vector"
DEFAULT_E_START_TOKEN = "start"
DEFAULT_E_END_TOKEN = "end"

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def run_llama(model, args, all_edge_1, label, add_example):
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # model = DyGLlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, use_cache=True,
    #                                             low_cpu_mem_usage=True, ).cuda()
    # Wrap your model with DataParallel to use multiple GPUs
    # model = DataParallel(model).cuda()
    emb_res = []
    aa = []
    edge_embs = []
    with open('./data/past_key_values_constant.pkl', 'rb') as file:
        past_key_values_constant = pickle.load(file)

    batch_size = 8


    for i in range(0, len(all_edge_1), batch_size):

        if label == "few-shot":
            qs = "As an AI trained in the few-shot learning approach, I have been provided with examples of both normal and anomaly edges. The anomalies are identified as Contextual Dissimilarity Anomalies, where we first utilize node2vec to obtain the representation of each node in the graph, and connect the pairs of nodes with the maximum Euclidean distance as anomaly edges. These examples serve as a reference for detecting similar patterns in new edges. Please note the following examples and their labels, indicating whether they are normal or anomaly: Example 1: <vector> Label: Normal  Example 2: <vector> Label: Anomaly Example 3: <vector> Label: Normal Example 4: <vector> Label: Normal Example 5: <vector> Label: Anomaly Example 6: <vector> Label:Anomaly Example 7: <vector> Label: Anomaly Example 8: <vector> Label: Anomaly Example 9: <vector> Label: Normal Example 10: <vector> Label: Anomaly (Note: All the above examples are anomaly and represent the same type of anomaly.) Based on the pattern in the examples and samples provided, classify the sentiment of the following new edge. If the new edge is similar to the example edges, it should be considered anomaly. If it is dissimilar, it should be considered normal. New Example: <vector> Label:"
            # qs = "As an AI trained in the few-shot learning approach, I have been provided with examples of both normal and anomalous edges.  The anomalies are identified as Contextual Dissimilarity Anomalies, where we first utilize node2vec to obtain the representation of each node in the graph, and connect the pairs of nodes with the maximum Euclidean distance as anomalous edges. These examples serve as a reference for detecting similar patterns in new edges. Please note the following examples and their labels, indicating whether they are normal or anomalous:\nExample 1:\n<<<vector>>>\nLabel: Nomal   \nExample 2:\n<<<vector>>>\nLabel: Anomalous\nExample 3:\n<<<vector>>>\nLabel: Nomal\nExample 4:\n<<<vector>>>\nLabel: Nomal\nExample 5:\n<<<vector>>>\nLabel: Anomalous\nExample 6:\n<<<vector>>>\nLabel:Anomalous\nExample 7:\n<<<vector>>>\nLabel: Anomalous\nExample 8:\n<<<vector>>>\nLabel: Anomalous\nExample 9:\n<<<vector>>>\nLabel: Nomal\nExample 10:\n<<<vector>>>\nLabel: Anomalous\n(Note: All the above examples are anomalous and represent the same type of anomaly.)\nBased on the pattern in the examples and samples provided, classify the sentiment of the following new edge. If the new edge is similar to the example edges, it should be considered anomalous. If it is dissimilar, it should be considered normal.\nNew Example:\n<<<vector>>>\nLabel:"
        conv_mode = "vicuna_v1_1"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
                conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids

        # print(input_ids)
        
        input_ids = torch.as_tensor(input_ids).cuda()

        # 计算当前切片的结束索引，确保不会超出列表的长度
        end_index = min(i + batch_size, len(all_edge_1))
        # 从all_edge_1中取出当前的切片，并将其转换为float类型和移动到CUDA设备0上
        current_slice = [x.float().cuda(0) for x in all_edge_1[i:end_index]]
        new_edge_embs = []
        for ind in range(len(current_slice)):
            updated_emb = edge_embs.copy()  
            updated_emb.append(current_slice[ind])
            new_edge_embs.append(updated_emb) 

        # Calculate the actual size of the current batch
        current_batch_size = end_index - i

        # print(input_ids.shape)

        input_ids = input_ids.repeat(current_batch_size, 1)

        input_ids_variable = input_ids[:, 308:317]  
        expanded_past_key_values = []

        for layer_past_key_value in past_key_values_constant:
            expanded_layer_past_key_value = (
                layer_past_key_value[0].repeat(current_batch_size, 1, 1, 1),
                layer_past_key_value[1].repeat(current_batch_size, 1, 1, 1)
            )
            expanded_past_key_values.append(expanded_layer_past_key_value)

        expanded_past_key_values = tuple(expanded_past_key_values)



        start = time.perf_counter()
        # print(input_ids.shape)
        # print(type(edge_embs))
        output_get_emb = model(
            input_ids_variable,
            output_hidden_states=True,
            edge_embedding=new_edge_embs,
            past_key_values=expanded_past_key_values,
            add_example=add_example,
        )
        # embedding = torch.zeros(1, 332 * 4096).reshape(1, 332, 4096)
        embedding = [t.detach().cpu() for t in output_get_emb.hidden_states][-1].float()
        
        aa.append(embedding)
        emb_res.append(embedding)

    return emb_res



if __name__ == "__main__":
    with open(r'../data/all_edge.pkl', 'rb') as file:
        your_data = pickle.load(file)
    your_data = your_data[:100]
    run_llama(your_data, "few-shot", True)
    