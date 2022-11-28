from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import clip
import faiss


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values

def build_knn_datastore(cfg, cache_keys, cache_values):

    if cfg['knn_mode'] == True:
        cache_keys = cache_keys.permute(1, 0)
        n, d = cache_keys.shape[0], cache_keys.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(cache_keys.cpu().numpy())

    else:
        index = None

    return index

def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels


def search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, adapter=None, datastore=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        if cfg['knn_mode'] == True:
            gamma_list = [i * (cfg['search_scale'][2] - 0.1) / cfg['search_step'][2] + 0.1 for i in range(cfg['search_step'][2])]

        best_acc = 0
        best_beta, best_alpha, best_gamma = 0, 0, 0

        clip_logits = 100. * test_features @ clip_weights

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(test_features)
                else:
                    affinity = test_features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

                if cfg['knn_mode'] == True:
                    topk = cfg['topk']
                    test_embeddings = np.array(test_features.cpu().detach(), dtype=np.float32)
                    D, I = datastore.search(test_embeddings, topk)
                    D = torch.from_numpy(D).to(test_features.device)

                    knn_logits = torch.full((clip_logits.shape[0], clip_logits.shape[1]), 0.).to(test_features.device)

                    for i in range(clip_logits.shape[0]):
                        soft_knn_i = torch.softmax(D[i], dim=-1)
                        for j in range(topk):
                            knn_logits[i][I[i][j] // cfg['shots']] += soft_knn_i[j]

                    for gamma in gamma_list:
                        tip_knn_logits = clip_logits + cache_logits * alpha + knn_logits * gamma
                        acc = cls_acc(tip_knn_logits, test_labels)

                        if acc > best_acc:
                            print("New best setting, beta: {:.2f}, alpha: {:.2f}, gamma: {:.2f}; accuracy: {:.2f}".format(beta, alpha, gamma, acc))
                            best_acc = acc
                            best_beta = beta
                            best_alpha = alpha
                            best_gamma = gamma
                        
                        tip_knn_logits = tip_knn_logits.cpu()
                else:
                    tip_logits = clip_logits + cache_logits * alpha
                    acc = cls_acc(tip_logits, test_labels)
            
                    if acc > best_acc:
                        print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                        best_acc = acc
                        best_beta = beta
                        best_alpha = alpha
                    
                    tip_logits = tip_logits.cpu()

                cache_logits = cache_logits.cpu()

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))
        print("After searching, the best setting, beta: {:.2f}, alpha: {:.2f}, gamma: {:.2f}.\n".format(best_beta, best_alpha, best_gamma))

        return best_beta, best_alpha, best_gamma

    else:
        return cfg['init_beta'], cfg['init_alpha'], cfg['init_gamma']
