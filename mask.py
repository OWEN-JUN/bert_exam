
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
        

