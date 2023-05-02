import wandb

wandb.init(project="bert", entity="owenjin")

# # 학습시작위치


import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

import json
import glob
import tqdm
import random


# check special tokens
from transformers import BertTokenizerFast, BertTokenizer

hf_model_path='tokenizer_model'

tokenizer_check = BertTokenizer.from_pretrained(hf_model_path+'_special')


print(tokenizer_check("")['input_ids'])
print('check special tokens : %s'%tokenizer_check.all_special_tokens[:20])

print('vocab size : %d' % tokenizer_check.vocab_size)
# tokenized_input_for_pytorch = tokenizer_check("나는 오늘 아침밥을 먹었다.","맛나다", return_tensors="pt")
tokenized_input_for_pytorch = tokenizer_check("나는 오늘 아침밥을 먹었다.","맛나다")

print("Tokens (str)      : {}".format([tokenizer_check.convert_ids_to_tokens(s) for s in tokenized_input_for_pytorch['input_ids']]))
print("Tokens (int)      : {}".format(tokenized_input_for_pytorch['input_ids']))
print("Tokens (int)      : {}".format(tokenized_input_for_pytorch['input_ids'][1:-1]))
print("Tokens (attn_mask): {}\n".format(tokenized_input_for_pytorch['attention_mask']))




berttokenizer= BertTokenizerFast.from_pretrained(hf_model_path+'_special')



target_length = 512
random.seed(42)



def truncate_ab(seq_a, seq_b, target_seq_length):
    while True:

        if (len(seq_a) + len(seq_b)) <= target_seq_length:
            break

        trunc = seq_a if len(seq_a) > len(seq_b) else seq_b
        assert len(trunc) >= 1, "def truncate_ab, trunc >= 1"

        if random.random() < 0.5:
            del trunc[0]
        else:
            trunc.pop()

    return seq_a, seq_b


def mask_ab(seq_a, seq_b,mask_token ):
    
    index_list = range(len(seq_a)+ len(seq_b))

    mask_15 = random.sample(index_list, int(len(index_list)*0.15))
    len_mask_15 = len(mask_15)
    mask_15_80 = random.sample(mask_15, int(len_mask_15*0.8))
    mask_ = list(set(mask_15) - set(mask_15_80))
    mask_15_10r, mask_15_10s = [], []
    if len_mask_15 * 0.2 > 2:
        mask_15_10r = random.sample(mask_, int(len_mask_15*0.1))
        mask_ = list(set(mask_) - set(mask_15_10r))
        mask_15_10s = random.sample(mask_, int(len_mask_15*0.1))

    masked_idx = []
    original_token = []
    
    for i in mask_15_80:
        if i < len(seq_a):
            original_token.extend([seq_a[i]])
            seq_a[i] = mask_token[0]
            masked_idx.extend([i+1])
        else:
            original_token.extend([seq_b[i-len(seq_a)]])
            seq_b[i-len(seq_a)] = mask_token[0]
            masked_idx.extend([i+2])
    for i in mask_15_10r:
        if i < len(seq_a):
            original_token.extend([seq_a[i]])
            seq_a[i] = random.randint(217,52000-1)
            masked_idx.extend([i+1])
        else:
            original_token.extend([seq_b[i-len(seq_a)]])
            seq_b[i-len(seq_a)] = random.randint(217,52000-1)
            masked_idx.extend([i+2])

    for i in mask_15_10s:
        if i < len(seq_a):
            original_token.extend([seq_a[i]])
            masked_idx.extend([i+1])
        else:
            original_token.extend([seq_b[i-len(seq_a)]])
            masked_idx.extend([i+2])
    # mask idx, original_token

    return seq_a, seq_b, masked_idx, original_token 

def mkBertTrainset(doc_token, target_length, special_cls_sep, mask_token):

    ab_list = []
    ab_RN = []
    ab_segment = []
    ab_mask = []
    ab_orgiToken = []
    total_origin_token = []
    for a_index in tqdm.tqdm(range(len(doc_token))):

        doc_element = doc_token[a_index]
        

        target_seq_length = target_length - 3 # [cls],[sep],[sep]

        if random.random() < 0.1 :
            target_seq_length = random.randint(2, target_seq_length)
        # print("target_seq_length : ",target_seq_length)

        # print("len(doc_element) : ",len(doc_element))

        result_ab = []
        cur_sentence = []
        cur_length = 0
        i = 0
        a_bp = 0
        b_index = 0
        is_RN = False

        #랜덤시작점
        if len(doc_element) > 3:
            sosentence = random.randint(0, len(doc_element)//2)
            doc_element = doc_element[sosentence:]
            # print("sosentence : ",sosentence)

        while i < len(doc_element):
            line = doc_element[i][1:-1]
            
            cur_sentence.append(line)
            cur_length += len(line)

            if i == len(doc_element)-1 or cur_length >= target_seq_length:
                break
            else:
                i += 1

        if len(cur_sentence) >= 2:
            a_bp = random.randint(1, len(cur_sentence) - 1)
        else:
            a_bp = 1

        seq_a = []
        for j in range(a_bp):
            seq_a.extend(cur_sentence[j])

        seq_b = []
        # 0.5 확률로 랜덤
        if (len(cur_sentence) == 1 or random.random() < 0.5) and len(doc_token) > 1:
            # print(True)
            is_RN = True
            target_b_length = target_seq_length - len(seq_a)

            while True:
                b_index = random.randint(0, len(doc_token)-1)
                if b_index != a_index:
                    break

            b_element = doc_token[b_index]

            # print("len(b_element) : ",len(b_element))

            #랜덤시작점
        
            if len(b_element) > 2:
                sosentence = random.randint(0, len(b_element)-1)
                b_element = b_element[sosentence:]
                # print("sosentence : ",sosentence)
            btoken_length = 0
            for b_sentence in b_element:
                seq_b.extend(b_sentence[1:-1])
                btoken_length += len(b_sentence)
                if btoken_length >= target_b_length:
                    break
        else:
            # print(False)
            for j in range(a_bp, len(cur_sentence)):
                seq_b.extend(cur_sentence[j])

        seq_a, seq_b = truncate_ab(seq_a, seq_b, target_seq_length)
        seq_a, seq_b, masked_idx, original_token   = mask_ab(seq_a, seq_b,mask_token)
        # mask

        seg_a = [0 for _ in range(len(seq_a)+2)]
        seg_b = [1 for _ in range(len(seq_b)+1)]
        pad_seg = [0] * ( target_length - (len(seg_a) +len(seg_b)) )
        tmp_seg = []
        for seq_ele in [[special_cls_sep[0]], seq_a,[ special_cls_sep[1]], seq_b, [special_cls_sep[1]]]:
            result_ab.extend(seq_ele)

        
        for seg_ele in [seg_a, seg_b]:
            tmp_seg.extend(seg_ele)
        # print(result_ab)
        # print(np.pad(result_ab, (0, target_length-len(result_ab)), 'constant', constant_values = 0 ).shape)
    
        tmp_mask_list = np.zeros((512))
        tmp_masktoken_list = np.zeros((512))
        for maskidx_elementidx in range(len(masked_idx)):
            tmp_mask_list[masked_idx[maskidx_elementidx]] = 1
            tmp_masktoken_list[masked_idx[maskidx_elementidx]] = original_token[maskidx_elementidx]
        ab_mask.append(tmp_mask_list)
        ab_orgiToken.append(tmp_masktoken_list)
        
        ab_list.append(np.pad(result_ab, (0, target_length-len(result_ab)), 'constant', constant_values = 0 ))
        ab_RN.append(int(is_RN))

        ab_segment.append(np.pad(tmp_seg, (0, target_length-len(tmp_seg)), 'constant', constant_values = 0 ))
        
    for list_idx in range(len(ab_list)):
        tmp_total_origin_token = []
        for token_idx in range(len(ab_list[list_idx])):
            if ab_orgiToken[list_idx][token_idx] != 0:

                tmp_total_origin_token.append(ab_orgiToken[list_idx][token_idx])
            else:
                tmp_total_origin_token.append(ab_list[list_idx][token_idx])
        total_origin_token.append(tmp_total_origin_token)

    ab_list = np.concatenate(ab_list, axis=0).reshape(-1, target_length)
    ab_RN = np.array(ab_RN).reshape(-1,1)
    ab_segment = np.concatenate(ab_segment, axis=0).reshape(-1, target_length)
    ab_mask = np.array(ab_mask,dtype=np.int32).reshape(-1, target_length)

    
    ab_orgiToken = np.array(ab_orgiToken, dtype=np.int32).reshape(-1, target_length)
    total_origin_token = np.array(total_origin_token, dtype=np.int32).reshape(-1, target_length)
    print(total_origin_token.shape)

    return ab_list, ab_RN, ab_segment, ab_mask, ab_orgiToken, total_origin_token
            




def mkDoc2Token(file_path):
    fileLine = []
    special_cls_sep= berttokenizer("")['input_ids'] 
    mask_token = berttokenizer("[MASK]")['input_ids'][1:-1]
    
    for i in range(len(file_path)):
        with open(file_path[i], 'r', encoding="utf8") as file:
            lines = file.readlines()
            for line in tqdm.tqdm(lines):
                # print(line)
                doclines = line.split("\n")[0].split("[DOC_SEP]")[-1].split("[SEN_SEP]")[:-1]
                # print(doclines)
                try:
                    linetoken = berttokenizer(doclines)['input_ids']
                    fileLine.append(linetoken)
                except:
                    print(doclines, line)
                
                
    
    return fileLine, special_cls_sep, mask_token



class CustomDataset(Dataset):
    def __init__(self, file_dir_list, target_length):
        self.file_dir_list = file_dir_list
        self.select_list = random.sample(self.file_dir_list, 2)
        self.target_length = target_length

        # self.doc_list, self.special_cls_sep, self.mask_token = mkDoc2Token(self.select_list)
        # self.ab_array, self.ab_RN_array, self.ab_segment_array, self.ab_mask_array, self.ab_orgiToken_array= mkBertTrainset(self.doc_list, \
        #                                                                         self.target_length, self.special_cls_sep, self.mask_token)
        self.select()



    def __len__(self):
        return len(self.doc_list)


    def __getitem__(self, index):
        # print(self.select_list)
        return self.ab_array[index],self.ab_RN_array[index],self.ab_segment_array[index], self.ab_mask_array[index], self.ab_orgiToken_array[index], self.total_origin_token[index]
        
        # {"label_rn":self.ab_RN_array[index] ,"label_seg":self.ab_segment_array[index] , \
        #                     "label_abmask":self.ab_mask_array[index] , "label_oritoken":self.ab_orgiToken_array[index]}

    def select(self):
        
        self.select_list = random.sample(self.file_dir_list, 2)
        print(self.select_list)

        self.doc_list, self.special_cls_sep, self.mask_token = mkDoc2Token(self.select_list)
        self.ab_array, self.ab_RN_array, self.ab_segment_array, self.ab_mask_array, self.ab_orgiToken_array, self.total_origin_token = mkBertTrainset(self.doc_list, \
                                                                                self.target_length, self.special_cls_sep, self.mask_token)
        



file_dir_list = glob.glob('./pretrain_data/shuffle/*.txt')






import torch.nn as nn
def create_padding_mask(x):
    mask = (x == 0).int()
    # print(mask)
    # (batch_size, 1, 1, key의 문장 길이)

    mask=mask[:, None, None, :]
    attention_mask = mask | mask.permute(0,1,3,2)
    return attention_mask


config = {

    "vocab_size" : 52000,
    "hidden_size" : 768,
    "seq_size" : 128,
    "hidden_dropout_prob" : 0.15,
    "multi_attention_head" : 8,
    "ff_size" : 3072,
    "batch_size" : 24,
    "epoch" : 10000000,
    "check_point":100

}
train_dataset = CustomDataset(file_dir_list=file_dir_list,target_length = config["seq_size"])



class BertEmbModel(nn.Module):

    def __init__(self, config) -> None:
        super(BertEmbModel, self).__init__()
        self.word_emb = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.position_emb = nn.Embedding(config["seq_size"], config["hidden_size"])
        self.seg_emb = nn.Embedding(config["seq_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])  # 0.1

    def forward(self, input_ids, segment_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long , device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # print("input_ids",input_ids.size())
        word_embedding = self.word_emb(input_ids)
        position_embedding = self.position_emb(position_ids)
        segment_embedding = self.seg_emb(segment_ids)

        bert_embedding = word_embedding + position_embedding + segment_embedding
        bert_embedding = self.LayerNorm(bert_embedding)
        bert_embedding = self.dropout(bert_embedding)
        return bert_embedding #(batch, seq_length, hiddensize) ex) 8,512,768



# # train_dataset.select()
# train_dataloader = DataLoader(train_dataset,
#                                     batch_size=64,
#                                     shuffle=True)
# batch_iterator = iter(train_dataloader)
# bert_traindata = next(batch_iterator)
# print(bert_traindata[0].shape)
# # print(images[1])
# for i in bert_traindata[1].keys():
#     print(i, bert_traindata[1][i].shape)
#     print(bert_traindata[1][i][:20])


# emb_output = emb_bert(bert_traindata[0], bert_traindata[1]["label_seg"] )
# emb_output.shape


class BertSelfAttentionModel(nn.Module):

    def __init__(self, config) -> None:
        super(BertSelfAttentionModel, self).__init__()
        #hiddensize = 768, attention head = 8
        if config["hidden_size"] % config["multi_attention_head"] != 0:
            raise ValueError(
                "hidden size (%d) is not multiple of the number of, \
                    multi attention head (%d)" %(config["hidden_size"],config["multi_attention_head"] )
            )

        self.Num_multi_attention_head = config["multi_attention_head"] # 8
        
        self.d_qkv_size = torch.tensor(int(config["hidden_size"] / self.Num_multi_attention_head))
        

        self.d_model_size = self.Num_multi_attention_head * self.d_qkv_size # 768
        
        self.query = nn.Linear(config["hidden_size"], self.d_model_size) #768 768
        self.key = nn.Linear(config["hidden_size"], self.d_model_size) #768 768
        self.value = nn.Linear(config["hidden_size"], self.d_model_size) #768 768

        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.attnfeedf = nn.Linear(config["hidden_size"], config["hidden_size"]) #768 768


    def attention_transpose(self, x, tom = False):
        #(batch, seq_length, d_model_size) -> (batch,Num_multi_attention_head, seq_length, d_qkv_size )
        attention_shape = x.size()[:-1] + (self.Num_multi_attention_head,self.d_qkv_size )
        x = torch.reshape(x, attention_shape) # batch, seq_length, Num_multi_attention_head, d_qkv_size
        if not tom:
            return x.permute(0,2,1,3) #(batch,Num_multi_attention_head, seq_length, d_qkv_size )
        else:
            return x.permute(0,2,3,1) #(batch,Num_multi_attention_head, d_qkv_size, seq_length )




    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states) #batch, seq_len, 768
        mixed_key_layer = self.key(hidden_states) #batch, seq_len, 768
        mixed_value_layer = self.value(hidden_states) #batch, seq_len, 768


        query_layer = self.attention_transpose(mixed_query_layer) # batch , 8, 512, 96
        key_layer = self.attention_transpose(mixed_key_layer, tom = True) # batch , 8, 96, 512
        value_layer = self.attention_transpose(mixed_value_layer) # batch , 8, 512, 96
        
        #dot product
        attention_score = torch.matmul(query_layer, key_layer)
        attention_score = attention_score / torch.sqrt(self.d_qkv_size)
        attention_mask = torch.gt(attention_mask, 0)
        attention_score = attention_score.masked_fill(attention_mask, -1e9)

        attention_probs = nn.functional.softmax(attention_score, dim = -1)

        attention_probs = self.dropout(attention_probs)

        attn_out = torch.matmul(attention_probs,value_layer)  # batch , 8, 512, 96
        attn_out = attn_out.permute(0,2,1,3).contiguous() # batch, 512, 8, 96
        
        attn_out_size = attn_out.size()[:-2] + (self.d_model_size,)
        
        attn_out = torch.reshape(attn_out, attn_out_size)
        
        attn_add = self.attnfeedf(attn_out) # batch, 512, 768
        attn_add = self.dropout(attn_add) # batch, 512, 768
        attn_add = self.LayerNorm(attn_add + hidden_states)

        return attn_add


        


class BertFeedForward(nn.Module):
    def __init__(self, config) -> None:
        super(BertFeedForward, self).__init__()

        self.ffdense = nn.Linear(config["hidden_size"], config["ff_size"])
        self.ffactivate = nn.GELU()
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.encoderdense = nn.Linear(config["ff_size"],config["hidden_size"] )
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=1e-12)

    def forward(self, hidden_state):
        encoder_layer_out = self.ffdense(hidden_state) #hidden -> ff_size
        encoder_layer_out = self.ffactivate(encoder_layer_out)
        encoder_layer_out = self.encoderdense(encoder_layer_out)#ff_size -> hidden
        encoder_layer_out = self.dropout(encoder_layer_out)
        encoder_layer_out = self.LayerNorm(encoder_layer_out + hidden_state)

        return encoder_layer_out


        


class BertOutLayer(nn.Module):
    def __init__(self, config) -> None:
        super(BertOutLayer, self).__init__()
        self.out_linear = nn.Linear(config["hidden_size"], config["hidden_size"]) # 768 -> 768
        self.word_emb = nn.Linear(config["hidden_size"], config["vocab_size"]) # 768 -> 52000(vocab size)
        self.cls_linear = nn.Linear(config["hidden_size"], 1) # 768 -> 768

        self.tanh = nn.Tanh()
        self.relu = nn.GELU()
        self.softmax = nn.Softmax(dim = 2)

    def forward(self, hidden_state):
        bert_out = self.out_linear(hidden_state)

        #embedding token
        bert_out_emb = self.word_emb(bert_out) # 768 -> 52000
        bert_out_emb = self.relu(bert_out_emb) # 768 -> 52000
        
        

        #cls activate tanh
        bert_cls =  self.cls_linear(bert_out[:,0])
        bert_cls = self.tanh(bert_cls)


        return bert_out_emb, bert_cls




def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp      


# # train_dataset.select()
# train_dataloader = DataLoader(train_dataset,
#                                     batch_size=64,
#                                     shuffle=True, collate_fn=lambda x: tuple(x_.cuda() for x_ in default_collate(x)))
# batch_iterator = iter(train_dataloader)
# ab_array, ab_RN_array, ab_segment_array, ab_mask_array, ab_orgiToken_array= next(batch_iterator)
# attn_mask = create_padding_mask(ab_array)
# print("attn_mask : ", attn_mask.size())
# print(attn_mask[0])
# emb_bert = BertEmbModel(config).cuda()
# emb_output = emb_bert(ab_array, ab_segment_array )
# print(emb_output.shape)

# print("bert_traindata", bert_traindata[0].size())
# print("label_seg", bert_traindata[1]["label_seg"].size())
# print("attn_mask", attn_mask.size())
# attn_bert = BertSelfAttentionModel(config).cuda()
# attn_out = attn_bert(emb_output, attn_mask)
# print(attn_out.size())


# bertff = BertFeedForward(config).cuda()
# bertff_out = bertff(attn_out)
# print(bertff_out.size())

# bert_out_layer = BertOutLayer(config).cuda()
# bert_out_emb, bert_out_cls = bert_out_layer(bertff_out)
# print(bert_out_emb.size(),bert_out_cls.size())


class BertConnectModel(nn.Module):
    def __init__(self, config, Nencoder = 3) -> None:
        super(BertConnectModel, self).__init__()
        self.Nencoder = Nencoder
        self.emb_bert = BertEmbModel(config)
        self.attn_bert = BertSelfAttentionModel(config)
        self.bertff = BertFeedForward(config)

        self.attn_layers = []
            
        self.bert_out_layer = BertOutLayer(config)
        

    def forward(self, input_token, input_seg, attn_mask):
        # print(input_token.size(),input_seg.size(), attn_mask.size() )
        bert_hidden = self.emb_bert(input_token, input_seg )

        for i in range(self.Nencoder):
            bert_hidden = self.bertff(self.attn_bert(bert_hidden, attn_mask))


        bert_soft, bert_log = self.bert_out_layer(bert_hidden)

        return bert_soft, bert_log



from torchsummary import summary as summary_

train_dataloader = DataLoader(train_dataset,
                                    batch_size=64,
                                    shuffle=True, collate_fn=lambda x: tuple(x_.cuda() for x_ in default_collate(x)))
batch_iterator = iter(train_dataloader)
ab_array, ab_RN_array, ab_segment_array, ab_mask_array, ab_orgiToken_array, total_origin_token_array= next(batch_iterator)

print(ab_array.size())

# attn_mask = create_padding_mask(ab_array)
# BERT = BertConnectModel(config, Nencoder=2)
# BERT.cuda()
# BERT(ab_array, ab_segment_array , attn_mask)

# print(BERT)




# BERT2 = BertConnectModel(config, Nencoder=3).cpu()
# BERT3 = BertConnectModel(config, Nencoder=6).cpu()
# # summary_(BERT2, [(512,), (512,), (1,512,512)]).cpu()


# BERT2(torch.randint(10,(2,512)),torch.randint(10, (2,512)) ,torch.randint(1,(2,1,512,512)) )


# # summary_(BERT2, [(512,),(512,),(1,512,512)], device="cpu")


# import torch
# import torch.nn as nn
# from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary_model(model, input_size, batch_size=-1, device="cpu"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        # dtype = torch.FloatTensor
        dtype = torch.IntTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))
    # print(x[0].size())
    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    # print(summary)
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        # print(summary[layer]["nb_params"])
        total_params += summary[layer]["nb_params"]
        # print(summary[layer]["output_shape"])
        total_output += np.prod(summary[layer]["output_shape"][0])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    # total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    # total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    # total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    # total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    # print("Input size (MB): %0.2f" % total_input_size)
    # print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    # print("Params size (MB): %0.2f" % total_params_size)
    # print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary



# summary_model(BERT2, [(512,),(512,),(1,512,512)], device="cpu")
# summary_model(BERT3, [(512,),(512,),(1,512,512)], device="cpu")


import tqdm

def train_bert(config, train_dataset):
    def train_data_shuffle(config, train_dataset):
        train_dataloader = DataLoader(train_dataset,
                                        batch_size=config["batch_size"],
                                        shuffle=True, collate_fn=lambda x: tuple(x_.cuda() for x_ in default_collate(x)))
        # batch_iterator = iter(train_dataloader)
        # ab_array, ab_RN_array, ab_segment_array, ab_mask_array, ab_orgiToken_array= next(batch_iterator)
        return train_dataloader

    f = open("history.txt", "w", encoding='utf-8')
    f.close()
    
    BERT2 = BertConnectModel(config, Nencoder=4)
    summary_model(BERT2, [(config["seq_size"],),(config["seq_size"],),(1,config["seq_size"],config["seq_size"])], device="cpu")
    BERT2.cuda()
    optimizer = optim.Adam(BERT2.parameters())
    BERT2.train()
    wandb.watch(BERT2, log="all")
    wandbtable = wandb.Table(columns=["origin", "model_out"])
    for i in range(config["epoch"]):
        train_dataset.select()
        print("EPOCH : {%d} / {%d}"%(i+1, config["epoch"]) )
        if random.random() < 0.2:
            print("전체 학습")
            train_loader = train_data_shuffle(config, train_dataset)
            running_train_loss = 0
            running_loss_cls = 0
            running_loss_token = 0
            running_cnt = 0
            running_acc_cnt = 0
            running_train_acc = 0
            csv_save_output = []
            csv_save_token = []

            pbar = tqdm.tqdm(train_loader, leave=True)
            for ab_array, ab_RN_array, ab_segment_array, ab_mask_array, ab_orgiToken_array, total_origin_token_array in pbar:
            
                # print(ab_RN_array)
                attn_mask = create_padding_mask(ab_array)
                optimizer.zero_grad()
                BERT_out_token, BERT_out_cls = BERT2(ab_array, ab_segment_array, attn_mask)
                

                loss_cls = nn.functional.binary_cross_entropy_with_logits(BERT_out_cls, ab_RN_array.float())*5

                
                BERT_out_token = nn.functional.log_softmax(BERT_out_token, dim=2)
                BERT_out_token = BERT_out_token.view(BERT_out_token.size(0)*BERT_out_token.size(1), BERT_out_token.size(2))
                
                
                token_target = total_origin_token_array.flatten().contiguous()
                
                # print("ab_orgiToken_array",ab_orgiToken_array.flatten().contiguous())
                # print(torch.where(torch.ones_like(ab_mask_array).mul(ab_mask_array).flatten().contiguous() == 0)[0])
                # print(token_target)
                # print(BERT_out_token.size())
                # print(token_target.size())



                loss_toekn = nn.functional.nll_loss(BERT_out_token, token_target.type(torch.LongTensor).cuda())
                # loss_toekn = nn.functional.cross_entropy(BERT_out_token, token_target.type(torch.LongTensor).cuda())
                # print(BERT_out_token.size())
                
                # print(torch.argmax(BERT_out_token, dim= 1)[:10])
                # print(token_target.type(torch.LongTensor)[:10])

                csv_save_output = token_target.type(torch.LongTensor)[:30]
                csv_save_token =  torch.argmax(BERT_out_token, dim= 1)[:30]
                if i % 1 == 0 and running_cnt == 10:
                    wandbtable.add_data( token_target.type(torch.LongTensor)[:10], torch.argmax(BERT_out_token, dim= 1)[:10])
            


                # torch.gt(torch.ones_like(ab_mask_array).mul(ab_mask_array).flatten().contiguous()

                loss = loss_cls + loss_toekn

                loss.backward()
                running_train_loss += loss.item()
                running_loss_cls += loss_cls
                running_loss_token += loss_toekn
                running_cnt += 1
                pbar.set_description(f"loss {running_train_loss / running_cnt:.3f},loss_cls {running_loss_cls / running_cnt:.3f},loss_toekn {running_loss_token / running_cnt:.3f}")
                
                optimizer.step()
            wandb.log({'loss rn': running_loss_cls / running_cnt, 'loss mk': running_loss_token / running_cnt, "total_loss":running_train_loss / running_cnt})
            # wandb.log({"table": wandbtable})
            
            if i % config["check_point"] == 0:
                PATH = "./torch_check/"
                torch.save({
                    "epoch" : i+1,
                    "model_state_dict" : BERT2.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "total_loss": running_train_loss / running_cnt,
                    "loss_mk": running_loss_token / running_cnt,
                    "loss_rn": running_loss_cls / running_cnt,

                }, PATH+"e%d.pt"%(i+1))

            f = open('history.txt', 'a',encoding='utf-8')

            f.write("epoch "+str(i+1) +" / "+str(csv_save_output) + ' / ' + str(csv_save_token) + '\n')
            f.close()

        else:
            for _ in range(2):
                train_loader = train_data_shuffle(config, train_dataset)
                running_train_loss = 0
                running_loss_cls = 0
                running_loss_token = 0
                running_cnt = 0
                running_acc_cnt = 0
                running_train_acc = 0
                csv_save_output = []
                csv_save_token = []

                pbar = tqdm.tqdm(train_loader, leave=True)
                for ab_array, ab_RN_array, ab_segment_array, ab_mask_array, ab_orgiToken_array,_ in pbar:
                
                    # print(ab_RN_array)
                    attn_mask = create_padding_mask(ab_array)
                    optimizer.zero_grad()
                    BERT_out_token, BERT_out_cls = BERT2(ab_array, ab_segment_array, attn_mask)
                    

                    loss_cls = nn.functional.binary_cross_entropy_with_logits(BERT_out_cls, ab_RN_array.float())*5

                    
                    BERT_out_token = nn.functional.log_softmax(BERT_out_token, dim=2)
                    BERT_out_token = BERT_out_token.view(BERT_out_token.size(0)*BERT_out_token.size(1), BERT_out_token.size(2))
                    
                    BERT_out_token_acc = BERT_out_token[torch.where(torch.ones_like(ab_mask_array).mul(ab_mask_array).flatten().contiguous() == 1)[0]]
                    BERT_out_token_acc = torch.argmax(BERT_out_token_acc, dim= 1)
                    token_target_acc = torch.masked_select(ab_orgiToken_array.flatten().contiguous() , torch.gt(torch.ones_like(ab_mask_array).mul(ab_mask_array).flatten().contiguous(), 0))
                    
                    running_train_acc += torch.sum(BERT_out_token_acc == token_target_acc)
                    running_acc_cnt += len(token_target_acc)
                    accuracy = running_train_acc / running_acc_cnt
                    # BERT_out_token = nn.functional.softmax(BERT_out_token, dim=2)
                    # BERT_out_token = BERT_out_token.view(BERT_out_token.size(0)*BERT_out_token.size(1), BERT_out_token.size(2))
                    ab_orgiToken_array.flatten().contiguous()[torch.where(torch.ones_like(ab_mask_array).mul(ab_mask_array).flatten().contiguous() == 0)[0]] = -100
                    token_target = ab_orgiToken_array.flatten().contiguous()
                    # print("ab_orgiToken_array",ab_orgiToken_array.flatten().contiguous())
                    # print(torch.where(torch.ones_like(ab_mask_array).mul(ab_mask_array).flatten().contiguous() == 0)[0])
                    # print(token_target)
                    # print(BERT_out_token.size())
                    # print(token_target.size())



                    loss_toekn = nn.functional.nll_loss(BERT_out_token, token_target.type(torch.LongTensor).cuda())
                    # loss_toekn = nn.functional.cross_entropy(BERT_out_token, token_target.type(torch.LongTensor).cuda())
                    # print(BERT_out_token.size())
                    
                    # print(torch.argmax(BERT_out_token, dim= 1)[:10])
                    # print(token_target.type(torch.LongTensor)[:10])

                    csv_save_output = token_target.type(torch.LongTensor)[:30]
                    csv_save_token =  torch.argmax(BERT_out_token, dim= 1)[:30]
                    if i % 1 == 0 and running_cnt == 10:
                        wandbtable.add_data( token_target.type(torch.LongTensor)[:10], torch.argmax(BERT_out_token, dim= 1)[:10])
                


                    # torch.gt(torch.ones_like(ab_mask_array).mul(ab_mask_array).flatten().contiguous()

                    loss = loss_cls + loss_toekn

                    loss.backward()
                    running_train_loss += loss.item()
                    running_loss_cls += loss_cls
                    running_loss_token += loss_toekn
                    running_cnt += 1
                    pbar.set_description(f"loss {running_train_loss / running_cnt:.3f},loss_cls {running_loss_cls / running_cnt:.3f},loss_toekn {running_loss_token / running_cnt:.3f}, acc {accuracy:.3f}")
                    
                    optimizer.step()
                print("RN_loss : ", (running_train_loss/len(train_loader)))
                wandb.log({'loss rn': running_loss_cls / running_cnt, 'loss mk': running_loss_token / running_cnt, "total_loss":running_train_loss / running_cnt, "accuracy":accuracy})
                # wandb.log({"table": wandbtable})
                
                if i % config["check_point"] == 0:
                    PATH = "./torch_check/"
                    torch.save({
                        "epoch" : i+1,
                        "model_state_dict" : BERT2.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "total_loss": running_train_loss / running_cnt,
                        "loss_mk": running_loss_token / running_cnt,
                        "loss_rn": running_loss_cls / running_cnt,

                    }, PATH+"e%d.pt"%(i+1))

                f = open('history.txt', 'a',encoding='utf-8')

                f.write("epoch "+str(i+1) +" / "+str(csv_save_output) + ' / ' + str(csv_save_token) + '\n')
                f.close()

                            



train_bert(config, train_dataset)
