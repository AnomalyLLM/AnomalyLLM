import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from torch.nn import DataParallel
import warnings
# from conversation import conv_templates, SeparatorStyle

from tqdm import tqdm
from codes.model.MetaLlama import DyGLlamaForCausalLM
import os
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

def run_eval(args, all_edge_1, label, add_example):
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = DyGLlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, use_cache=True,
                                                low_cpu_mem_usage=True, ).cuda()
    # Wrap your model with DataParallel to use multiple GPUs
    # model = DataParallel(model).cuda()
    emb_res = []
    aa = []
    edge_embs = []
    dir_1 = './data/all_edge.pkl'
    with open(dir_1, 'rb') as f:
        all_edge = pickle.load(f)

    anormaly_edge = [all_edge[6356], all_edge[5662], all_edge[5506], all_edge[5865], all_edge[6035],
                     all_edge[6087], all_edge[6366]]
    normal_edge = [all_edge[666], all_edge[667], all_edge[668], all_edge[669], all_edge[670], all_edge[671],
                   all_edge[672]]

    edge_embs.append(normal_edge[0].float().cuda(0))
    edge_embs.append(anormaly_edge[0].float().cuda(0))
    edge_embs.append(normal_edge[1].float().cuda(0))
    edge_embs.append(normal_edge[2].float().cuda(0))
    edge_embs.append(anormaly_edge[1].float().cuda(0))
    edge_embs.append(anormaly_edge[2].float().cuda(0))
    edge_embs.append(anormaly_edge[3].float().cuda(0))
    edge_embs.append(anormaly_edge[4].float().cuda(0))
    edge_embs.append(normal_edge[3].float().cuda(0))
    edge_embs.append(anormaly_edge[5].float().cuda(0))

    batch_size = 1


    for i in tqdm(range(0, len(all_edge_1), batch_size)):

        if label == "pre-train":
            qs = "As an AI trained in the few-shot learning approach, I have been provided with examples of both normal and anomaly edges. The anomalies are identified as edges that did not exist before, randomly sampled. New Example: <vector> Label:"
        if label == "few-shot":
            qs = "As an AI trained in the few-shot learning approach, I have been provided with examples of both normal and anomaly edges. The anomalies are identified as Contextual Dissimilarity Anomalies, where we first utilize node2vec to obtain the representation of each node in the graph, and connect the pairs of nodes with the maximum Euclidean distance as anomaly edges. These examples serve as a reference for detecting similar patterns in new edges. Please note the following examples and their labels, indicating whether they are normal or anomaly: Example 1: <Edge> Label: Normal  Example 2: <Edge> Label: Anomaly Example 3: <Edge> Label: Normal Example 4: <Edge> Label: Normal Example 5: <Edge> Label: Anomaly Example 6: <Edge> Label:Anomaly Example 7: <Edge> Label: Anomaly Example 8: <Edge> Label: Anomaly Example 9: <Edge> Label: Normal Example 10: <Edge> Label: Anomaly (Note: All the above examples are anomaly and represent the same type of anomaly.) Based on the pattern in the examples and samples provided, classify the sentiment of the following new edge. If the new edge is similar to the example edges, it should be considered anomaly. If it is dissimilar, it should be considered normal. New Example: <vector> Label:"
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

        end_index = min(i + batch_size, len(all_edge_1))
        current_slice = [x.float().cuda(0) for x in all_edge_1[i:end_index]]
        new_edge_embs = []
        for i in range(len(current_slice)):
            updated_emb = edge_embs.copy()  
            updated_emb.append(current_slice[i])
            new_edge_embs.append(updated_emb) 
        input_ids = input_ids.repeat(batch_size, 1)
        start = time.perf_counter()

        output_get_emb = model(
            input_ids,
            output_hidden_states=True,
            edge_embedding=new_edge_embs,
            add_example=add_example,
        )
        # embedding = torch.zeros(1, 332 * 4096).reshape(1, 332, 4096)
        embedding = [t.detach().cpu() for t in output_get_emb.hidden_states][-1]
        # print(embedding.shape)
        # embedding = embedding[:, 341:352, :].reshape(11, 4096).float()
        aa.append(embedding)
        if i > 3:
            print("shitttttt")
            comparison = torch.eq(aa[i - 1], aa[i])
            not_equal = ~comparison.all(dim=2)
            index_of_not_equal = not_equal.nonzero(as_tuple=True)[1]
            print(index_of_not_equal)
        emb_res.append(embedding)

    return emb_res


def run_llama(all_edge_1, label, add_example):
    output_model = "./backbone/Vicuna-7b-v1.5"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=output_model)
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--prompting_file", type=str, default=datapath)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--graph_data_path", type=str, default=graph_data_path)

    parser.add_argument("--output_res_path", type=str, default=res_path)
    parser.add_argument("--num_gpus", type=int, default=num_gpus)

    parser.add_argument("--start_id", type=int, default=start_id)
    parser.add_argument("--end_id", type=int, default=end_id)

    args = parser.parse_args()

    return run_eval(args, all_edge_1, label, add_example)

if __name__ == "__main__":
    with open(r'./data/all_edge.pkl', 'rb') as file:
        your_data = pickle.load(file)
    your_data = your_data[:100]
    run_llama(your_data, "few-shot", True)
    