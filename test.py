# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import os
import cv2
from models import *
import torch
import numpy as np
import time
from config import Config
from torch.nn import DataParallel


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image

def get_featurs(model, test_list, batch_size=10):
    images = []
    features = []
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))
            continue

        images.append(image)

        if len(images) % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(np.stack(images))
            data = data.to(torch.device("cuda"))

            # Reshape the input tensor to 4 dimensions (batch_size, channels, height, width)
            data = data.reshape(-1, *data.shape[2:])

            output = model(data)
            output = output.data.cpu().numpy()

            # Separate the feature vectors for each image in the batch
            features_batch = np.split(output, len(images))

            if not features:
                features = features_batch
            else:
                assert len(features_batch[0]) == len(features[0]), "Feature sizes do not match"
                features = [np.concatenate((f1, f2), axis=1) for f1, f2 in zip(features, features_batch)]

            images = []

    if features:
        features = np.concatenate(features, axis=0)
    else:
        features = np.empty((0,))  # Empty array when no features are extracted

    return features[:len(test_list)], cnt





def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(identity_list, features):
    fe_dict = {}
    for i, each in enumerate(identity_list):
        if i < len(features):
            fe_dict[each] = features[i]
        else:
            break
    return fe_dict



def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)

def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        if len(splits) == 3:
            fe_1 = fe_dict.get(splits[0])
            fe_2 = fe_dict.get(splits[1])
            if fe_1 is not None and fe_2 is not None:
                label = int(splits[2])
                sim = cosin_metric(fe_1, fe_2)

                sims.append(sim)
                labels.append(label)
        else:
            continue

    acc, th = cal_accuracy(sims, labels)
    return acc, th

def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc


if __name__ == '__main__':

    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    model = DataParallel(model)
    # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(opt.test_model_path))
    model.to(torch.device("cuda"))

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    model.eval()
    lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)



