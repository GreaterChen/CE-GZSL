import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import os
import random

# 假设您已经有以下导入
# from your_module import util, model, losses, classifier_embed_contras
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ZDFY', help='FLO')
parser.add_argument('--dataroot', default='/home/LAB/chenlb24/ZhengDaFuyi', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='sent', help='att or sent')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=768, help='size of semantic features')
parser.add_argument('--nz', type=int, default=1024, help='noise for generation')
parser.add_argument('--embedSize', type=int, default=2048, help='size of embedding h')
parser.add_argument('--outzSize', type=int, default=512, help='size of non-liner projection z')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator G')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator D')
parser.add_argument('--nhF', type=int, default=2048, help='size of the hidden units comparator network F')
parser.add_argument('--ins_weight', type=float, default=0.001, help='weight of the classification loss when learning G')
parser.add_argument('--cls_weight', type=float, default=0.001, help='weight of the score function when learning G')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to training')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=3483, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=3, help='number of all classes')
parser.add_argument('--nclass_seen', type=int, default=2, help='number of all classes')
parser.add_argument('--gpus', default='0', help='the number of the GPU to use')
parser.add_argument('--load_model', type=bool, default=True, help='load model from disk')
parser.add_argument('--model_epoch', type=int, default=2000, help='the epoch of the model to load')
parser.add_argument('--model_kind', type=str, default='seen', help='the kind of the model to load: seen, unseen, or H')

opt = parser.parse_args()
print(opt)

def evaluate_model(netG, netMap, test_feature, test_label, batch_size=32):
    netG.eval()
    netMap.eval()

    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(0, test_feature.size(0), batch_size):
            inputs = test_feature[i:i + batch_size]
            targets = test_label[i:i + batch_size]

            if opt.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            noise_gen = torch.randn(inputs.size(0), opt.nz).cuda()
            fake = netG(noise_gen, inputs)
            _, outputs = netMap(fake)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = correct / total
    return accuracy, all_preds, all_targets

def main():
    

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    data = DataLoader(opt)

    netG = model.MLP_G(opt)
    netMap = model.Embedding_Net(opt)
    netD = model.MLP_CRITIC(opt)
    F_ha = model.Dis_Embed_Att(opt)

    model_path = './models/' + opt.dataset

    optimizerD = optim.Adam(itertools.chain(netD.parameters(), netMap.parameters(), F_ha.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    epoch, opt = load_models(opt.model_epoch, netG, netD, netMap, F_ha, optimizerG, optimizerD, model_path, opt.model_kind)
    print(f"Loaded model from epoch {epoch} with kind {opt.model_kind}")

    acc_seen, _, _ = evaluate_model(netG, netMap, data.test_seen_feature, data.test_seen_label, opt.batch_size)
    acc_unseen, _, _ = evaluate_model(netG, netMap, data.test_unseen_feature, data.test_unseen_label, opt.batch_size)

    h_score = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen + 1e-12)

    print(f'Seen Accuracy: {acc_seen:.4f}, Unseen Accuracy: {acc_unseen:.4f}, H-score: {h_score:.4f}')

if __name__ == '__main__':
    main()
