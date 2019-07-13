import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertModel, BertTokenizer,BertForSequenceClassification,BertAdam
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tensorboardX import SummaryWriter

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = inputs

        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

DEVICE = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained(r'D:\bert_weight_Chinese\chinese_L-12_H-768_A-12')
train_df = pd.read_csv(r'E:\2019ATEC\MyCode\JupyterPro\data\train_balance.csv')
vld_df = pd.read_csv(r'E:\2019ATEC\MyCode\JupyterPro\data\dev_balance.csv')

import re

qlen = 35


class BulldozerDataset(Dataset):  # 集成Dataset，要重写3个函数 init，len，getitem
    def __init__(self, loadin_data):
        self.df = loadin_data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''根据IDX返回数据 '''
        text1 = self.df.iloc[idx, 1]
        text1 = re.sub("[\s+\.\!\/_,$%^*()+-?\"\']+|[+——！，。；？、~@#￥%……&*（）]+", "", text1)
        text2 = self.df.iloc[idx, 2]
        text2 = re.sub("[\s+\.\!\/_,$%^*()+-?\"\']+|[+——！，。；？、~@#￥%……&*（）]+", "", text2)
        label = torch.tensor(self.df.iloc[idx, 3])

        tokens_1 = tokenizer.tokenize(text1)
        tokens_2 = tokenizer.tokenize(text2)

        if len(tokens_1) > qlen - 2:
            tokens_1 = tokens_1[:qlen - 2]
        seq_word_1 = ["[CLS]"] + tokens_1 + ["[SEP]"]
        real_len_1 = len(seq_word_1)
        ids_1 = tokenizer.convert_tokens_to_ids(seq_word_1)
        ids_tensor_1 = torch.tensor(ids_1)
        pad0 = torch.zeros(qlen - real_len_1).long()
        ids_tensor_1 = torch.cat([ids_tensor_1, pad0])
        token_type_ids_1 = torch.tensor([0] * qlen).long()
        attention_mask_1 = torch.tensor([1] * (2 + len(tokens_1)) + [0] * (qlen - 2 - len(tokens_1))).long()

        if len(tokens_2) > qlen - 2:
            tokens_2 = tokens_2[:qlen - 2]
        seq_word_2 = ["[CLS]"] + tokens_2 + ["[SEP]"]
        real_len_2 = len(seq_word_2)
        ids_2 = tokenizer.convert_tokens_to_ids(seq_word_2)
        ids_tensor_2 = torch.tensor(ids_2)
        pad0 = torch.zeros(qlen - real_len_2).long()
        ids_tensor_2 = torch.cat([ids_tensor_2, pad0])
        token_type_ids_2 = torch.tensor([0] * qlen).long()
        attention_mask_2 = torch.tensor([1] * (2 + len(tokens_2)) + [0] * (qlen - 2 - len(tokens_2))).long()

        res1 = [ids_tensor_1, token_type_ids_1, attention_mask_1]
        res2 = [ids_tensor_2, token_type_ids_2, attention_mask_2]

        res = [res1, res2, label]
        return res  # __getitem__的返回值要是tensor或者list或者数值类型


train_dataset = BulldozerDataset(train_df)
vld_dataset = BulldozerDataset(vld_df)

EPOCH = 3
BATCH_SIZE = 64

train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
vld_iter = torch.utils.data.DataLoader(vld_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def observe_res(module, epoch, step, observe_set='train'):
    loss_funtion = FocalLoss(2, size_average=False)
    dataset_len = len(train_dataset) if observe_set == 'train' else len(vld_dataset)
    data_iter = train_iter if observe_set == 'train' else vld_iter

    check_point = step / iter_num if step else 1

    module.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for i, batchgroup in enumerate(data_iter):
            torch.cuda.empty_cache()  # 清除gpu缓存
            text1, text2, label = batchgroup
            label = label.to(DEVICE)
            ids_tensor1, token_type_ids1, attention_mask1 = text1[0].to(DEVICE), text1[1].to(DEVICE), text1[2].to(
                DEVICE)
            ids_tensor2, token_type_ids2, attention_mask2 = text2[0].to(DEVICE), text2[1].to(DEVICE), text2[2].to(
                DEVICE)
            output = module(ids_tensor1, token_type_ids1, attention_mask1, ids_tensor2, token_type_ids2,
                            attention_mask2)
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            loss += loss_funtion(output, label).item()
            correct += pred.eq(label.view_as(pred)).sum().item()
        loss_avg = loss / dataset_len
        acc = correct / dataset_len
        print(
        'epoch:', epoch, '\t', 'check_point:', check_point, '\t', observe_set + '_acc:', acc, '\t', 'loss:', loss_avg)
        return loss_avg, acc


class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        #         bert_model_path = os.path.join(BERT_PATH, 'bert-base-chinese.tar')
        #         self.bert = BertModel.from_pretrained(bert_model_path)
        self.bert = BertModel.from_pretrained(r'D:\bert_weight_Chinese\chinese_L-12_H-768_A-12\bert-base-chinese.tar')
        for param in self.bert.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(1, 180, kernel_size=(2, 16 * 768), stride=1,
                               dilation=(3, 1))  # params: 输入通道数，输出通道数（filter个数），核视野（H,W）,步长
        self.conv2 = nn.Conv2d(1, 180, kernel_size=(2, 16 * 768), stride=1)
        self.conv3 = nn.Conv2d(1, 180, kernel_size=(3, 16 * 768), stride=1)
        self.conv4 = nn.Conv2d(1, 180, kernel_size=(4, 16 * 768), stride=1)
        self.conv5 = nn.Conv2d(1, 180, kernel_size=(5, 16 * 768), stride=1)

        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.2)
        self.dp3 = nn.Dropout(0.3)

        self.dense1 = nn.Linear(900 * 6, 900 * 3)  # 第一个参数计算 filters*5*6

        self.dense2 = nn.Linear(900 * 3, 900)

        self.dense3 = nn.Linear(900, 400)

        self.dense4 = nn.Linear(400, 150)

        self.dense5 = nn.Linear(150, 2)

    def mask_expand_for_BertOut(self, bert_out, attention_mask):
        '''
        attention_mask: b*qlen——>b*1*qlen——>（转置）b*qlen*1——>(expand)b*qlen*4dim得到mask_expand_for_bertout
        输出mask_expand_for_bertout，与bert_out同维度，但在pad的行全为0，非pad的行全为1。 将bert_out与mask_expand_for_bertout
        逐元素相乘，可将bert_out变换为但在pad的行全为0，非pad的行为原元素的样子。
        :param attention_mask: b*qlen
        :return: mask_expand_for_bertout
        '''
        attention_mask_ = attention_mask.unsqueeze(1)
        attention_mask_ = torch.transpose(attention_mask_, 1, 2)
        mask_expand_for_bertout = attention_mask_.expand_as(bert_out)
        mask_expand_for_bertout = mask_expand_for_bertout.cuda()
        return mask_expand_for_bertout.float()

    def bert_encoder(self, ids_tensor, token_type_ids, attention_mask):
        _, pooled = self.bert(ids_tensor, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              output_all_encoded_layers=True)

        L4 = _[-1]
        L3 = _[-2]
        L2 = _[-3]
        L1 = _[-4]
        bert_out = torch.cat([L1, L2, L3, L4], dim=2)
        mask_expand_for_bertout = self.mask_expand_for_BertOut(bert_out, attention_mask)
        bert_out = bert_out * mask_expand_for_bertout
        return bert_out

    def mask_expand_for_Attention(self, attention, attention_mask):  # 在soft_align_attention中调用
        '''
        attention_mask: b*qlen——>*1w-1w——>b*1*qlen——>（expand）b*qlen*qlen
        mask_expand_for_attention，与attention同维度，但在attention_mask为0的列值是1w，为1的列值是0，作用是：以后将attention与mask_expand_for_attention
        逐元素相加，再对每一行做softmax
        :param attention:
        :param attention_mask:
        :return: mask_expand_for_attention
        '''
        attention_mask_ = attention_mask * 10000 - 10000
        attention_mask_.unsqueeze_(1)
        mask_expand_for_attention = attention_mask_.expand_as(attention)
        return mask_expand_for_attention.float()

    def soft_align_attention(self, vec1, vec2, attention_mask1, attention_mask2):  # 输入vec1，vec2  局部推理
        attention1_2 = torch.bmm(vec1, torch.transpose(vec2, 1, 2))  # attention1_2  b*sq*sq
        attention2_1 = torch.bmm(vec2, torch.transpose(vec1, 1, 2))  # attention2_1  b*sq*sq

        mask_expand_for_attention1_2 = self.mask_expand_for_Attention(attention1_2,
                                                                      attention_mask2)  # b*sq*sq  ； 注意传参attention_mask2
        masked_attention1_2 = attention1_2 + mask_expand_for_attention1_2
        softmax_attention1_2 = F.softmax(masked_attention1_2, dim=2)
        # 借用函数mask_expand_for_BertOut处理注意attention_mask1  传参attention_mask1，
        mask_expand_for_SMA1_2 = self.mask_expand_for_BertOut(softmax_attention1_2, attention_mask1)
        pad_softmax_attention1_2 = softmax_attention1_2 * mask_expand_for_SMA1_2
        vec1_align = torch.bmm(pad_softmax_attention1_2, vec2)

        mask_expand_for_attention2_1 = self.mask_expand_for_Attention(attention2_1,
                                                                      attention_mask1)  # b*sq*sq  ； 注意传参attention_mask1
        masked_attention2_1 = attention2_1 + mask_expand_for_attention2_1
        softmax_attention2_1 = F.softmax(masked_attention2_1, dim=2)
        mask_expand_for_SMA2_1 = self.mask_expand_for_BertOut(softmax_attention2_1, attention_mask2)
        pad_softmax_attention2_1 = softmax_attention2_1 * mask_expand_for_SMA2_1
        vec2_align = torch.bmm(pad_softmax_attention2_1, vec1)
        return vec1_align, vec2_align  # vec2_align  b*sq*(4*768)

    def fix_encoder(self, vec1, vec2, vec1_align, vec2_align):  # 形成新的字嵌入
        x1_mul = vec1 * vec1_align
        x1_sub = vec1 - vec1_align
        x2_mul = vec2 * vec2_align
        x2_sub = vec2 - vec2_align
        x1_combined = torch.cat([vec1, vec1_align, x1_sub, x1_mul], dim=2)  # b*sq*（768*16）
        x2_combined = torch.cat([vec2, vec2_align, x2_sub, x2_mul], dim=2)  #
        return x1_combined, x2_combined

    def cnn_maxpool_encoder(self, vec):
        vec.unsqueeze_(1)  # :batch*1*qlength*4dim  nn.Cov2d()的输入参数是4-D：batch，输入通道数，H，W
        conv1_out = F.relu(self.conv1(vec))  # conv1_out:batch*filters*H*1 , H=qlength-kernel(H)+1
        conv1_out.squeeze_(3)  # conv1_out:batch*filters*W  , W=H
        conv1_encoder_vector = nn.MaxPool2d((1, conv1_out.size(2)), stride=(1, 1))(
            conv1_out)  # conv1_encoder_vector：batch*100*1
        conv1_encoder_vector.squeeze_(2)  # conv1_encoder_vector：batch*100

        conv2_out = F.relu(self.conv2(vec))
        conv2_out.squeeze_(3)
        conv2_encoder_vector = nn.MaxPool2d((1, conv2_out.size(2)), stride=(1, 1))(conv2_out)
        conv2_encoder_vector.squeeze_(2)  # conv2_encoder_vector：batch*100

        conv3_out = F.relu(self.conv3(vec))
        conv3_out.squeeze_(3)
        conv3_encoder_vector = nn.MaxPool2d((1, conv3_out.size(2)), stride=(1, 1))(conv3_out)
        conv3_encoder_vector.squeeze_(2)  # conv3_encoder_vector：batch*100

        conv4_out = F.relu(self.conv4(vec))
        conv4_out.squeeze_(3)
        conv4_encoder_vector = nn.MaxPool2d((1, conv4_out.size(2)), stride=(1, 1))(conv4_out)
        conv4_encoder_vector.squeeze_(2)  # conv3_encoder_vector：batch*100

        conv5_out = F.relu(self.conv5(vec))
        conv5_out.squeeze_(3)
        conv5_encoder_vector = nn.MaxPool2d((1, conv5_out.size(2)), stride=(1, 1))(conv5_out)
        conv5_encoder_vector.squeeze_(2)  # conv3_encoder_vector：batch*100

        cnn_maxpool_vector = torch.cat(
            [conv1_encoder_vector, conv2_encoder_vector, conv3_encoder_vector, conv4_encoder_vector,
             conv5_encoder_vector], 1)  # cnn_encoder_vector: batch*300
        return cnn_maxpool_vector, [conv1_out, conv2_out, conv3_out, conv4_out, conv5_out]

    def cnn_avgpool_encoder(self, cnn_out_list1, cnn_out_list2):
        avgpool_list1 = []
        avgpool_list2 = []
        for convout1, convout2 in zip(cnn_out_list1, cnn_out_list2):  # convout1  b*filter*H
            # score_matrix矩阵元素表示，句1每个短语与句2每个短语的点积相似度
            score_matrix = torch.bmm(torch.transpose(convout1, 1, 2), convout2)
            weight_convout1 = torch.sum(score_matrix, dim=2)  # convout1的池化权重 b*convout1.size(2) sum函数会降为
            sum_weight1 = torch.sum(weight_convout1, dim=1, keepdim=True).expand_as(weight_convout1)
            weight_convout1 = weight_convout1 / sum_weight1
            #             print('====================')
            #             print(weight_convout1)

            weight_convout2 = torch.sum(score_matrix, dim=1)
            sum_weight2 = torch.sum(weight_convout2, dim=1, keepdim=True).expand_as(weight_convout2)
            weight_convout2 = weight_convout2 / sum_weight2
            avgpool1 = torch.bmm(weight_convout1.unsqueeze(1), torch.transpose(convout1, 1, 2))  # b*1*filters
            avgpool2 = torch.bmm(weight_convout2.unsqueeze(1), torch.transpose(convout2, 1, 2))  # b*1*filters
            avgpool_list1.append(avgpool1.squeeze(1))
            avgpool_list2.append(avgpool2.squeeze(1))
        cnn_avgpool_vector1 = torch.cat(avgpool_list1, dim=1)  # b*5filters
        cnn_avgpool_vector2 = torch.cat(avgpool_list2, dim=1)
        return cnn_avgpool_vector1, cnn_avgpool_vector2

    def forward(self, ids_tensor1, token_type_ids1, attention_mask1, ids_tensor2, token_type_ids2, attention_mask2):
        vec1 = self.bert_encoder(ids_tensor1, token_type_ids1, attention_mask1)  # 两个都是b*q*(4*768)
        vec2 = self.bert_encoder(ids_tensor2, token_type_ids2, attention_mask2)

        vec1_align, vec2_align = self.soft_align_attention(vec1, vec2, attention_mask1, attention_mask2)
        x1_combined, x2_combined = self.fix_encoder(vec1, vec2, vec1_align, vec2_align)

        cnn_maxpool_vector1, cnn_out_list1 = self.cnn_maxpool_encoder(x1_combined)
        cnn_maxpool_vector2, cnn_out_list2 = self.cnn_maxpool_encoder(x2_combined)

        cnn_avgpool_vector1, cnn_avgpool_vector2 = self.cnn_avgpool_encoder(cnn_out_list1, cnn_out_list2)

        vec = torch.cat([cnn_maxpool_vector1, cnn_maxpool_vector2, cnn_maxpool_vector1 - cnn_maxpool_vector2,
                         cnn_maxpool_vector1 * cnn_maxpool_vector2, cnn_avgpool_vector1, cnn_avgpool_vector2], dim=1)


        out = self.dp2(vec)
        out = F.relu(self.dense1(out))
        out = F.relu(self.dense2(out))
        out = F.relu(self.dense3(out))
        out = F.relu(self.dense4(out))
        out = self.dense5(out)
        #         print(out)

        out = F.softmax(out, dim=1)
        return out

model4 = Model4().to(DEVICE)
optimizer = BertAdam(filter(lambda p: p.requires_grad, model4.parameters()),lr=0.00002)#(filter(lambda p: p.requires_grad, model4.parameters()),lr=0.001)
loss_funtion = FocalLoss(2)
iter_num = int(len(train_dataset) / BATCH_SIZE)
observe_point = int(iter_num / 2)
step = 0

for epoch in range(20):
    model4.train()
    for i, batchgroup in enumerate(train_iter):
        step += 1
        torch.cuda.empty_cache()  # 清除gpu缓存
        text1, text2, label = batchgroup
        label = label.to(DEVICE)

        ids_tensor1, token_type_ids1, attention_mask1 = text1[0].to(DEVICE), text1[1].to(DEVICE), text1[2].to(DEVICE)
        ids_tensor2, token_type_ids2, attention_mask2 = text2[0].to(DEVICE), text2[1].to(DEVICE), text2[2].to(DEVICE)

        predicted = model4(ids_tensor1, token_type_ids1, attention_mask1, ids_tensor2, token_type_ids2, attention_mask2)

        optimizer.zero_grad()
        loss = loss_funtion(predicted, label)

        optimizer.step()
        if (i + 1) % 300 == 0:
            print(loss)

    train_loss, train_correct = observe_res(model4, epoch, step=None, observe_set='train')
    vld_loss, vld_correct = observe_res(model4, epoch, step=None, observe_set='vld')
