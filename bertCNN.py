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
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')#r'D:\bert_weight_Chinese\chinese_L-12_H-768_A-12'
tokenizer = BertTokenizer.from_pretrained(r'D:\bert_weight_Chinese\chinese_L-12_H-768_A-12')
train_df = pd.read_csv(r'E:\2019ATEC\MyCode\JupyterPro\data\train_balance.csv')#.iloc[:1000,:]
vld_df = pd.read_csv(r'E:\2019ATEC\MyCode\JupyterPro\data\dev_balance.csv')#.iloc[:100,:]

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
        #         self.bert = BertModel.from_pretrained('bert-base-chinese')#r'D:\bert_weight_Chinese\chinese_L-12_H-768_A-12\bert-base-chinese.tar')
        self.bert = BertModel.from_pretrained(r'D:\bert_weight_Chinese\chinese_L-12_H-768_A-12\bert-base-chinese.tar')
        for param in self.bert.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(1, 100, kernel_size=(1, 4 * 768), stride=1)  # params: 输入通道数，输出通道数（filter个数），核视野（H,W）,步长
        self.conv2 = nn.Conv2d(1, 100, kernel_size=(2, 4 * 768), stride=1)
        self.conv3 = nn.Conv2d(1, 100, kernel_size=(3, 4 * 768), stride=1)
        self.conv4 = nn.Conv2d(1, 100, kernel_size=(4, 4 * 768), stride=1)
        self.conv5 = nn.Conv2d(1, 100, kernel_size=(5, 4 * 768), stride=1)

        self.dp1 = nn.Dropout(0.1)

        self.dense1 = nn.Linear(500 * 4, 500 * 2)

        self.dense2 = nn.Linear(500 * 2, 200)

        self.dense3 = nn.Linear(200, 2)

    def bert_encoder(self, ids_tensor, token_type_ids, attention_mask):
        _, pooled = self.bert(ids_tensor, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              output_all_encoded_layers=True)
        L4 = _[-1]
        L3 = _[-2]
        L2 = _[-3]
        L1 = _[-4]
        bert_out = torch.cat([L1, L2, L3, L4], dim=2)
        #         bert_out = bert_out[:,1:,:]
        return bert_out

    def cnn_encoder(self, vec):
        vec.unsqueeze_(1)  # new_embedding:batch*1*qlength*328  nn.Cov2d()的输入参数是4-D：batch，输入通道数，H，W
        conv1_out = self.conv1(vec)  # conv1_out:batch*100*H*1 , H=qlength-kernel(H)+1
        conv1_out.squeeze_()  # conv1_out:batch*100*W  , W=H
        conv1_encoder_vector = nn.MaxPool2d((1, conv1_out.size(2)), stride=(1, 1))(
            conv1_out)  # conv1_encoder_vector：batch*100*1
        conv1_encoder_vector.squeeze_()  # conv1_encoder_vector：batch*100

        conv2_out = self.conv2(vec)
        conv2_out.squeeze_()
        conv2_encoder_vector = nn.MaxPool2d((1, conv2_out.size(2)), stride=(1, 1))(conv2_out)
        conv2_encoder_vector.squeeze_()  # conv2_encoder_vector：batch*100

        conv3_out = self.conv3(vec)
        conv3_out.squeeze_()
        conv3_encoder_vector = nn.MaxPool2d((1, conv3_out.size(2)), stride=(1, 1))(conv3_out)
        conv3_encoder_vector.squeeze_()  # conv3_encoder_vector：batch*100

        conv4_out = self.conv4(vec)
        conv4_out.squeeze_()
        conv4_encoder_vector = nn.MaxPool2d((1, conv4_out.size(2)), stride=(1, 1))(conv4_out)
        conv4_encoder_vector.squeeze_()  # conv3_encoder_vector：batch*100

        conv5_out = self.conv5(vec)
        conv5_out.squeeze_()
        conv5_encoder_vector = nn.MaxPool2d((1, conv5_out.size(2)), stride=(1, 1))(conv5_out)
        conv5_encoder_vector.squeeze_()  # conv3_encoder_vector：batch*100

        cnn_encoder_vector = torch.cat(
            [conv1_encoder_vector, conv2_encoder_vector, conv3_encoder_vector, conv4_encoder_vector,
             conv5_encoder_vector], 1)  # cnn_encoder_vector: batch*300
        return cnn_encoder_vector

    def forward(self, ids_tensor1, token_type_ids1, attention_mask1, ids_tensor2, token_type_ids2, attention_mask2):
        vec1 = self.bert_encoder(ids_tensor1, token_type_ids1, attention_mask1)
        vec2 = self.bert_encoder(ids_tensor2, token_type_ids2, attention_mask2)

        cnn_encoder_vector1 = self.cnn_encoder(vec1)
        cnn_encoder_vector2 = self.cnn_encoder(vec2)

        vec = torch.cat([cnn_encoder_vector1, cnn_encoder_vector2, cnn_encoder_vector1 - cnn_encoder_vector2,
                         cnn_encoder_vector1 * cnn_encoder_vector2], dim=1)

        out = self.dp1(vec)
        out = F.relu(self.dense1(out))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)

        out = F.softmax(out, dim=1)
        return out


model4 = Model4().to(DEVICE)
optimizer = BertAdam(filter(lambda p: p.requires_grad, model4.parameters()),lr=0.00005)#(filter(lambda p: p.requires_grad, model4.parameters()),lr=0.001)
loss_funtion = FocalLoss(2)

writer = SummaryWriter()
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
        loss.backward()
        optimizer.step()
    #         if i+1 == observe_point:
    #             train_loss,train_correct = observe_res(model4,epoch,step,observe_set='train')
    #             vld_loss,vld_correct = observe_res(model4,epoch,step,observe_set='vld')
    #             writer.add_scalars('loss',{'train_loss':train_loss,'vld_loss':vld_loss},step)
    #             writer.add_scalars('acc',{'train_acc':train_correct,'vld_acc':vld_correct},step)

    train_loss, train_correct = observe_res(model4, epoch, step=None, observe_set='train')
    vld_loss, vld_correct = observe_res(model4, epoch, step=None, observe_set='vld')
    print('\n')
    writer.add_scalars('loss_for_epoch', {'train_loss': train_loss, 'vld_loss': vld_loss}, epoch)
    writer.add_scalars('acc_for_epoch', {'train_acc': train_correct, 'vld_acc': vld_correct}, epoch)
