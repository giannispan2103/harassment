from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch
from torch.autograd import Variable
from time import time
import random
import numpy as np
import pandas as pd
import os


from globals import SAMPLE_SUBMISSION, SUBMISSION_PATH
SEED = 1985


def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, train_batches, test_batches, optimizer,  criterion,
                           epochs, init_patience, cuda=True, metric='f1'):
    patience = init_patience
    best_sc = 0.0
    best_thr = 0.0
    for i in range(1, epochs + 1):
        start = time()
        eval_dict = run_epoch(model, train_batches, test_batches, optimizer,  criterion,
                                         cuda, metric)
        end = time()
        print('epoch %d, score: %2.3f, thr: %2.3f  Time: %d minutes, %d seconds'
              % (i, 100 * eval_dict['sc'], eval_dict['thr'], (end - start) / 60, (end - start) % 60))
        if best_sc < eval_dict['sc']:
            best_sc = eval_dict['sc']
            best_thr = eval_dict['thr']
            patience = init_patience
            save_model(model)
            if i > 1:
                print('best epoch so far')
        else:
            patience -= 1
        if patience == 0:
            break
    return {'sc': best_sc, 'thr': best_thr}


def run_epoch(model, train_batches, test_batches, optimizer, criterion, cuda, metric):
    model.train(True)
    perm = np.random.permutation(len(train_batches))
    for i in perm:
        batch = train_batches[i]
        inner_perm = np.random.permutation(len(batch['label']))
        data = []
        for inp in model.input_list:
            if cuda:
                data.append(Variable(torch.from_numpy(batch[inp][inner_perm]).long().cuda()))
            else:
                data.append(Variable(torch.from_numpy(batch[inp][inner_perm]).long()))
        if cuda:
            aux = Variable(torch.from_numpy(batch['aux'][inner_perm]).cuda())
        else:
            aux = Variable(torch.from_numpy(batch['aux'][inner_perm]))
        outputs = model(*data)
        loss = criterion(outputs, aux)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return evaluate(model, test_batches, cuda, metric)


def get_scores_and_labels(model, test_batches, cuda):
    scores_list = []
    labels_list = []
    for batch in test_batches:
        data = []
        for inp in model.input_list:
            if cuda:
                data.append(Variable(torch.from_numpy(batch[inp]).long().cuda()))
            else:
                data.append(Variable(torch.from_numpy(batch[inp]).long()))
        outputs = model(*data)[:, 0]
        outputs = torch.sigmoid(outputs)
        labels_list.extend(batch['target'].tolist())
        scores_list.extend(outputs.data.view(-1).tolist())
    return labels_list, scores_list


def evaluate(model, test_batches, cuda, metric='auc'):
    model.train(False)
    labels_list, scores_list = get_scores_and_labels(model, test_batches, cuda)
    if metric == 'auc':
        return {'sc': roc_auc_score(np.asarray(labels_list, dtype='float32'),
                                np.asarray(scores_list, dtype='float32')),'thr':None}
    else:
        return tune_threshold(np.asarray(labels_list, dtype='float32'),
                                np.asarray(scores_list, dtype='float32'),metric)


def tune_threshold(labels, scores, metric='accuracy'):
    lower = 0.2
    upper = 0.70
    step = 0.01
    best_sc = 0.0
    best_thr = lower
    for x in np.arange(lower, upper, step):
        predictions = predict(scores, x)
        if metric == 'accuracy':
            sc = accuracy_score(labels, predictions)
        else:
            sc = f1_score(labels, predictions)
        if sc > best_sc:
            best_sc = sc
            best_thr = x
    return {'sc': best_sc, 'thr': best_thr}


def get_scores_and_ids(model, test_batches, cuda):
    scores_list = []
    ids = []
    for batch in test_batches:
        data = []
        for inp in model.input_list:
            if cuda:
                data.append(Variable(torch.from_numpy(batch[inp]).long().cuda()))
            else:
                data.append(Variable(torch.from_numpy(batch[inp]).long()))
        outputs = model(*data)[:, 0]
        outputs = torch.sigmoid(outputs)
        scores_list.extend(outputs.data.view(-1).tolist())
        ids.extend(batch['post_id'])
    return scores_list, ids


def predict(scores, thr):
    return [x > thr for x in scores]


def save_model(model):
    torch.save(model.state_dict(), model.name + '.pkl')


def load_model(model):
    model.load_state_dict(torch.load(model.name + '.pkl'))
    return model


def generate_results_predictions(model, test_batches, cuda, threshold):
    sc, ids = get_scores_and_ids(model, test_batches, cuda)
    predictions = predict(sc, threshold)
    predictions = [int(x) for x in predictions]
    d = {i: s for i, s in zip(ids, predictions)}
    return d


def generate_results(model, test_batches, cuda):
    sc, ids = get_scores_and_ids(model, test_batches, cuda)
    d = {i: s for i, s in zip(ids, sc)}
    return d


def generate_csv(dicts):
    def final_score(idx):
        c = 0
        for d in dicts:
            c += d[idx]
        return c / float(len(dicts))

    df = pd.read_csv(SAMPLE_SUBMISSION)
    df['prediction'] = df['id'].map(lambda x: final_score(x))
    df.to_csv(SUBMISSION_PATH, index=False)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def generate_wheigthed_csv(dicts, scores, low):
    s = sum(scores) - low * len(scores)
    weights = list(map(lambda x: (x - low) / s, scores))

    def final_score(idx):
        c = 0
        for w, d in zip(weights, dicts):
            c += w * d[idx]
        return int(c > 0.5)

    df = pd.read_csv(SAMPLE_SUBMISSION)
    df['prediction'] = df['id'].map(lambda x: final_score(x))
    df.to_csv(SUBMISSION_PATH, index=False)






