from typing import List, Optional, Tuple, Union
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, \
    LlamaConfig, LlamaModel, LlamaForCausalLM, \
    CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


DEFAULT_EDGE_TOKEN = "<edge>"
DEFAULT_EDGE_PATCH_TOKEN = "<e_patch>"
DEFAULT_E_START_TOKEN = "<e_start>"
DEFAULT_E_END_TOKEN = "<e_end>"


class DyGLlamaConfig(LlamaConfig):
    model_type = "DyGLlama"


class DyGLlamaModel(LlamaModel):
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig, edge_embedding: Optional[List[torch.Tensor]] = None, add_example: Optional[bool] = None):
        super(DyGLlamaModel, self).__init__(config)
        self.edge_embedding = edge_embedding
        self.add_example = add_example
        # self.cls_y = torch.nn.Linear(config.hidden_size, 1)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            edge_embedding: Optional[List[torch.Tensor]] = None,
            add_example: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        #
        if input_ids.shape[1] != 1:
            cur_input = 0
            new_input_embeds = []
            # print(input_ids.shape)
            # for i in range(len(edge_embedding)):
            #     print(len(edge_embedding[i]))

            start_token = 8111

            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                cur_edge_idx = 0

                graph_start_tokens = torch.where(cur_input_ids == start_token)[0]
                for graph_start_token_pos in graph_start_tokens:
                    cur_graph_features = edge_embedding[cur_input][cur_edge_idx].float()
                    cur_new_input_embeds = torch.cat((cur_input_embeds[:graph_start_token_pos],
                                                      cur_graph_features,
                                                      cur_input_embeds[graph_start_token_pos + 1:]),
                                                     dim=0)
                    cur_edge_idx += 1
                if cur_new_input_embeds!=None:
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_input += 1
                # assert cur_edge_idx == len(edge_embedding)
            # for i in inputs_embeds:
            #     print(i)
            inputs_embeds = torch.stack(new_input_embeds, dim=0).to(torch.float16)
            


        # for i in inputs_embeds:
        #     print(i)
        # with open(r'shit.pkl', 'wb') as f:
        #     pickle.dump(inputs_embeds, f)
        # with open(r'shit.pkl', 'rb') as file:
        #     your_data = pickle.load(file)
        # print(len(your_data))
        # print(len(your_data[0]))
        # comparison = torch.eq(your_data, inputs_embeds)
        # # 在第二个维度（332维度）上找到不相等的元素
        # # comparison 的形状为 (1, 332, 4096)，使用 all() 方法在最后一个维度上进行比较
        # # 如果所有 4096 个元素都相等，则返回 True，否则返回 False
        # # 这将给我们一个形状为 (1, 332) 的张量
        # not_equal = ~comparison.all(dim=2)
        # # 获取不相等元素的索引
        # index_of_not_equal = not_equal.nonzero(as_tuple=True)[1]
        # print(index_of_not_equal)

        return super(DyGLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class DyGLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlamaConfig

    def __init__(self, config, edge_embedding: Optional[List[torch.Tensor]] = None, add_example: Optional[bool] = None):
        super(LlamaForCausalLM, self).__init__(config)
        self.edge_emb = edge_embedding
        self.add_example = add_example
        self.model = DyGLlamaModel(config, edge_embedding, add_example)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

# AutoConfig.register("DyGLlama", DyGLlamaConfig)
# AutoModelForCausalLM.register(DyGLlamaConfig, DyGLlamaForCausalLM)
