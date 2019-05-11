from tqdm import tqdm
import numpy as np
import pandas as pd
import gc
import regex as re
from sklearn.metrics import roc_auc_score
import torch
from torch.autograd import Variable
from time import time
import random
import os
from torch.nn import Module, Embedding, GRU, LSTM, Linear, ModuleList, Dropout, \
    Dropout2d, Softmax, BCEWithLogitsLoss
from torch.nn import Module
from torch.nn.functional import relu
from torch.optim import Adam, Adagrad

FLAGS = re.MULTILINE | re.DOTALL
# global variables
PAD_TOKEN = "*$*PAD*$*"
UNK_TOKEN = "*$*UNK*$*"
MAXLEN = 300
BATCH_SIZE = 512
SEED = 1985
GLOVE_EMBEDDINGS_PATH = '../input/embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt'
TRAIN_DATA_PATH = '../input/harrasment/Train_data_compeition.csv'
TEST_DATA_PATH = '../input/harrasment/Validation_data_competition.csv'
SUBMISSION_PATH = 'submission.csv'
SAMPLE_SUBMISSION = "../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv"

GLOVE_MEAN = -0.005753148614825279
GLOVE_STD = 0.4111380286218564



def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join([""] + [re.sub(r"([A-Z])", r" \1", hashtag_body, flags=FLAGS)])
    return result


def allcaps(text):
    text = text.group()
    return text.lower() + " "


class Tweet(object):

    def __init__(self, post_id, tweet_content,
                 harrasment=None,
                 indirectH=None,
                 physicalH=None,
                 sexualH=None ):
        self.post_id = post_id
        self.target = harrasment
        self.indirectH = indirectH
        self.physicalH = physicalH
        self.sexualH = sexualH

        self.label = self.target
        self.tokens = [self.clean_token(x) for x in self.tokenize(tweet_content, True).split() if len(x) > 0]
        self.text_size = self.get_text_size()

    def get_text_size(self):
        return len(self.tokens)

    @staticmethod
    def clean(comment_text, lower=True):
        """
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        :param comment_text: The string to be cleaned
        :param lower: If True text is converted to lower case
        :return: The clean string
        """
        text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", comment_text)
        text = re.sub(r"[1-9][\d]+", " 5 ", text)
        text = re.sub(r"\'s", " \'s", text)
        text = re.sub(r"\'ve", " \'ve", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " \'re", text)
        text = re.sub(r"\'d", " \'d", text)
        text = re.sub(r"\'ll", " \'ll", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ) ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip().lower() if lower else text.strip()

    @staticmethod
    def clean_token(tkn):
        if tkn not in ["\'s", "\'ve", "n\'t", "\'d", "\'ll"]:
            tkn = tkn.replace("'", "")
        return tkn

    @staticmethod
    def tokenize(text, lower=True):
        # Different regex parts for smiley faces
        eyes = r"[8:=;]"
        nose = r"['`-]?"

        # function so code less repetitive
        def re_sub(pattern, repl):
            return re.sub(pattern, repl, text, flags=FLAGS)

        text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
        text = re_sub(r"@\w+", "<user>")
        text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
        text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
        text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
        text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
        text = re_sub(r"/", " / ")
        text = re_sub(r"<3", "<heart>")
        text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
        text = re_sub(r"#\S+", hashtag)
        text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
        text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

        ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
        # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
        text = re_sub(r"([A-Z]){2,}", allcaps)
        return text.lower() if lower else text

def get_embeddings(tokens, path=GLOVE_EMBEDDINGS_PATH):
    embeddings_dict = {}
    with open(path,'r', encoding='utf8') as f:
        for line in tqdm(f):
                values = line.strip().split(" ")
                try:
                    tokens[values[0]]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_dict[values[0]] = coefs
                except KeyError:
                    pass
    return embeddings_dict


def normalize(embeddings, m, s):
    return (embeddings - m)/s


def create_freq_vocabulary(tokenized_texts):
    token_dict = {}
    for text in tokenized_texts:
        for token in text:
            try:
                token_dict[token] += 1
            except KeyError:
                token_dict[token] = 1
    return token_dict


def get_frequent_words(token_dict, min_freq):
    return {x:None for x in token_dict if token_dict[x] >= min_freq}


def get_unique_tokens(posts, min_freq):
    tokenized_texts = [x.tokens for x in posts]
    voc = create_freq_vocabulary(tokenized_texts)
    print("tokens found in training data set:", len(voc))
    freq_words = get_frequent_words(voc, min_freq)
    print("tokens with frequency >= %d: %d" % (min_freq, len(freq_words)))
    return freq_words


def create_final_dictionary(freq_words, embeddings_dict, unk_token, pad_token):
    words = list(set(freq_words).intersection(embeddings_dict.keys()))
    print("embedded tokens: %d" % (len(words)))
    words = [pad_token, unk_token] + words
    print("ok final dict")
    return {w: i for i, w in enumerate(words)}


def get_embeddings_matrix(word_dict, embeddings_dict, size):
    embs = np.zeros(shape=(len(word_dict), size))
    for word in word_dict:
        try:
            embs[word_dict[word]] = embeddings_dict[word]
        except KeyError:
            print('no embedding for: ', word)
    embs[1] = np.mean(embs[2:])
    print("ok emb matrix")
    return embs


def get_indexed_value(w2i, word, unk_token):
    try:
        return w2i[word]
    except KeyError:
        return w2i[unk_token]

def get_indexed_text(w2i, words, unk_token):
    return [get_indexed_value(w2i, word, unk_token) for word in words]


def pad_text(tokenized_text, maxlen, pad_tkn):
    if len(tokenized_text) < maxlen:
        return [pad_tkn] * (maxlen - len(tokenized_text)) + tokenized_text
    else:
        return tokenized_text[len(tokenized_text) - maxlen:]


def load_data(csv_file):
    data = pd.read_csv(csv_file)
    data = data.fillna({'tweet_content': "nan"})
    return data


def get_comments(df):
    comments = []
    for i, row in df.iterrows():

        post = Tweet(post_id=row['post_id'],
                           tweet_content=row['tweet_content'],
                           harrasment=row['harassment'],
                           indirectH=row['IndirectH'],
                           physicalH=row['PhysicalH'],
                           sexualH=row['SexualH'])

        comments.append(post)
    return comments

def create_batches(comments, w2i, pad_tnk=PAD_TOKEN, unk_tkn=UNK_TOKEN, batch_size=128,
                   max_len=100, sort_data=True):
    if sort_data:
        comments.sort(key=lambda x: -len(x.tokens))
    offset = 0
    batches = []
    while offset < len(comments):
        batch_texts = []
        batch_targets = []
        batch_ids = []
        batch_aux = []
        start = offset
        end = min(offset + batch_size, len(comments))
        batch_max_size = comments[start].text_size if sort_data else max(list(map(lambda x: x.text_size, comments[start:end])))
        for i in range(start, end):
            batch_texts.append(get_indexed_text(w2i, pad_text(comments[i].tokens, max(min(max_len, batch_max_size), 1), pad_tkn=pad_tnk), unk_token=unk_tkn))
            batch_targets.append(comments[i].target)
            batch_aux.append([comments[i].target, comments[i].sexualH,
                              comments[i].physicalH, comments[i].indirectH])
            batch_ids.append(comments[i].post_id)
        batches.append({'text': np.array(batch_texts), 'aux': np.array(batch_aux, dtype='float32'),
                        'target': np.array(batch_targets, dtype='float32'),
                        'post_id': batch_ids})
        offset += batch_size
    return batches


def generate_data(embs_path,
                  batch_size=128,
                  min_freq=1,
                  maxlen=1000):
    df = load_data(TRAIN_DATA_PATH)
    df_test = load_data(TEST_DATA_PATH)
    train_posts = get_comments(df)
    test_posts = get_comments(df_test)
    posts = train_posts + test_posts
    print('posts for training:', len(train_posts))
    print('posts for validation:', len(test_posts))
    tokens = get_unique_tokens(posts, min_freq)
    embeddings_dict = get_embeddings(tokens, embs_path)

    w2i = create_final_dictionary(tokens, embeddings_dict, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN)
    emb_matrix = get_embeddings_matrix(w2i, embeddings_dict, 200)
    del embeddings_dict
    gc.collect()

    train_batches = create_batches(train_posts, w2i, max_len=maxlen, batch_size=batch_size)
    test_batches = create_batches(test_posts, w2i, max_len=maxlen, batch_size=batch_size)


    return {
            'emb_matrix': emb_matrix,
            'train_batches': train_batches,
            'test_batches': test_batches}


def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, train_batches, test_batches, optimizer,  criterion,
                          epochs, init_patience, cuda=True):
    patience = init_patience
    best_auc = 0.0
    for i in range(1, epochs + 1):
        start = time()
        auc = run_epoch(model, train_batches, test_batches, optimizer,  criterion,
                                         cuda)
        end = time()
        print('epoch %d, auc: %2.3f  Time: %d minutes, %d seconds'
              % (i, 100 * auc, (end - start) / 60, (end - start) % 60))
        if best_auc < auc:
            best_auc = auc
            patience = init_patience
            save_model(model)
            if i > 1:
                print('best epoch so far')
        else:
            patience -= 1
        if patience == 0:
            break
    return best_auc


def run_epoch(model, train_batches, test_batches, optimizer, criterion, cuda):
    model.train(True)
    perm = np.random.permutation(len(train_batches))
    for i in perm:
        batch = train_batches[i]
        inner_perm = np.random.permutation(len(batch['target']))
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
    return evaluate(model, test_batches, cuda)


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


def evaluate(model, test_batches, cuda):
    model.train(False)
    labels_list, scores_list = get_scores_and_labels(model, test_batches, cuda)
    return roc_auc_score(np.asarray(labels_list, dtype='float32'), np.asarray(scores_list, dtype='float32'))


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


class SpatialDropout(Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class Layer(Module):
    def __init__(self):
        super(Layer, self).__init__()

    def get_output_size(self):
        raise NotImplementedError

    def get_input_size(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class PretrainedEmbeddingLayer(Layer):
    def __init__(self, embeddings, dropout=0.0, trainable=True):
        """
        :param embeddings: a numpy array with the embeddings
        :param trainable: if false the embeddings will be frozen
        """
        super(PretrainedEmbeddingLayer, self).__init__()
        self.__input_size = embeddings.shape[0]
        self.__output_size = embeddings.shape[1]
        self.dropout = SpatialDropout(dropout)
        self.embed = Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embed.weight.data.copy_(torch.from_numpy(embeddings))
        if not trainable:
            self.embed.weight.requires_grad = False

    def forward(self, x):
        return self.dropout(self.embed(x))

    def get_output_size(self):
        return self.__output_size

    def get_input_size(self):
        return self.__input_size


class EmbeddingLayer(Layer):
    def __init__(self, input_size, output_size, dropout=0.0):
        super(EmbeddingLayer, self).__init__()
        self.__input_size = input_size
        self.__output_size = output_size
        self.embed = Embedding(input_size, output_size)
        self.dropout = SpatialDropout(dropout)

    def forward(self, x):
        return self.dropout(self.embed(x))

    def get_output_size(self):
        return self.__output_size

    def get_input_size(self):
        return self.__input_size


class CellLayer(Layer):
    def __init__(self, is_gru, input_size,  hidden_size, bidirectional, stacked_layers):
        """
        :param is_gru: GRU cell type if true, otherwise LSTM
        :param input_size: the size of the tensors that will be used as input (embeddings or projected embeddings)
        :param hidden_size: the size of the cell
        :param bidirectional: boolean
        :param stacked_layers: the number of stacked layers
        """
        super(CellLayer, self).__init__()
        if is_gru:
            self.cell = GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True,
                            bidirectional=bidirectional, num_layers=stacked_layers)

        else:
            self.cell = LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True,
                             bidirectional=bidirectional, num_layers=stacked_layers)

        self.__output_size = hidden_size * 2 if bidirectional else hidden_size
        self.__input_size = input_size

    def forward(self, x):
        return self.cell(x)[0]

    def get_output_size(self):
        return self.__output_size

    def get_input_size(self):
        return self.__input_size


class MultiLayerPerceptron(Layer):
    def __init__(self, num_of_layers, init_size, out_size, dropout=0.0, inner_activation=None, outer_activation=None):
        """
        :param num_of_layers: the total number of layers
        :param init_size: unit size of hidden layers
        :param out_size: output size
        :param inner_activation: the activation function for the inner layers
        :param outer_activation: the activation function for the outer layers
        """
        super(MultiLayerPerceptron, self).__init__()
        self.num_of_layers = num_of_layers
        self.__input_size = init_size
        self.__output_size = out_size
        self.dropout = Dropout(dropout)
        if self.num_of_layers > 0:
            self.layers = ModuleList([Linear(init_size, init_size) for _ in range(num_of_layers-1)] + [Linear(init_size, out_size)])
            self.activation_list = [inner_activation for _ in range(num_of_layers - 1)] + [outer_activation]

    def forward(self, x):
        if self.num_of_layers > 0:
            for layer, activation in zip(self.layers, self.activation_list):
                if activation is None:
                    x = self.dropout(layer(x))
                else:
                    x = self.dropout(activation(layer(x)))
        return x

    def get_output_size(self):
        return self.__output_size

    def get_input_size(self):
        return self.__input_size


class LastState(Layer):
    def __init__(self, input_size, output_size):
        super(LastState, self).__init__()
        self.__input_size = input_size
        self.__output_size = output_size

    def forward(self, x):
        return x[:, -1, :]

    def get_input_size(self):
        return self.__input_size

    def get_output_size(self):
        return self.__output_size


class AvgPoolingState(Layer):
    def __init__(self, input_size, output_size):
        super(AvgPoolingState, self).__init__()
        self.__input_size = input_size
        self.__output_size = output_size

    def forward(self, x):
        return torch.mean(x, 1)

    def get_input_size(self):
        return self.__input_size

    def get_output_size(self):
        return self.__output_size


class MaxPoolingState(Layer):
    def __init__(self, input_size, output_size):
        super(MaxPoolingState, self).__init__()
        self.__input_size = input_size
        self.__output_size = output_size

    def forward(self, x):
        m, _ = torch.max(x, 1)
        return m

    def get_input_size(self):
        return self.__input_size

    def get_output_size(self):
        return self.__output_size


class AttendedState(Layer):
    def __init__(self, num_of_layers, hidden_size, dropout=0.0, inner_activation=None):
        super(AttendedState, self).__init__()
        self.__input_size = hidden_size
        self.__output_size = hidden_size
        self.mlp = MultiLayerPerceptron(num_of_layers=num_of_layers,
                                        init_size=hidden_size, out_size=hidden_size,
                                        dropout=dropout,
                                        inner_activation=inner_activation,
                                        outer_activation=inner_activation)

        self.attention = Linear(hidden_size, 1)
        self.at_softmax = Softmax()

    def forward(self, x):
        states_mlp = self.mlp(x)
        att_sc_dist = self.attention(states_mlp).squeeze(-1)
        att_weights = self.at_softmax(att_sc_dist).unsqueeze(2)
        out_attended = torch.sum(torch.mul(att_weights, x), dim=1)
        return out_attended

    def get_input_size(self):
        return self.__input_size

    def get_output_size(self):
        return self.__output_size


class ConcatenationLayer(Layer):
    def __init__(self, layer1, layer2):
        super(ConcatenationLayer, self).__init__()
        self.__input_size = layer1.get_input_size() + layer2.get_input_size()
        self.__output_size = self.__input_size

    def forward(self, x, y):
        return torch.cat((x, y), 1)

    def get_input_size(self):
        return self.__input_size

    def get_output_size(self):
        return self.__output_size


class SequentialModel(Layer):
    def __init__(self, layers):
        super(Layer, self).__init__()
        for i in range(len(layers)-1):
            assert (layers[i].get_output_size() == layers[i+1].get_input_size())
        self.layers = ModuleList(layers)
        self.__output_size = self.layers[-1].get_output_size()
        self.__input_size = self.layers[0].get_input_size()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_input_size(self):
        return self.__input_size

    def get_output_size(self):
        return self.__output_size

    def add_layer(self, layer):
        assert (layer.get_input_size() == self.__input_size)
        self.layers.append(layer)
        self.__output_size = layer.get_output_size()



class AttentionRNN(Module):
    def __init__(self, embeddings,
                 trainable_embeddings=True,
                 embeddings_dropout=0.0,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 att_mlp_layers=1,
                 att_mlp_dropout=0.5,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param trainable_embeddings: boolean indicating if the embedding will be trainable or frozen
        :param embeddings_dropout: dropout of the embeddings layer
        :param is_gru: GRU cell type if true, otherwise LSTM
        :param cell_hidden_size: the cell size of the RNN
        :param stacked_layers: the number of stacked layers of the RNN
        :param bidirectional: boolean indicating if the cell is bidirectional
        :param att_mlp_layers: number of layers of the attention mlp
        :param att_mlp_dropout: dropout of the attention mlp
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(AttentionRNN, self).__init__()
        self.input_list = ['text']
        self.name = "AttentionTextRNN"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)

        self.cell = CellLayer(is_gru, self.word_embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size,
                                                   out_size=4,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.last_state = AttendedState(att_mlp_layers, large_size, att_mlp_dropout, relu)
        self.seq = SequentialModel([self.word_embedding_layer, self.cell, self.last_state, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        out = self.seq(x)
        return out

def run(cuda):
    data = generate_data(embs_path=GLOVE_EMBEDDINGS_PATH, maxlen=100, batch_size=64)
    emb_matrix = data['emb_matrix']
    train_batches = data['train_batches']
    test_batches = data['test_batches']
    #     val_label_dict = data['val_labels']
    set_seeds(SEED + 11192108)
    del data
    gc.collect()
    models = 7
    dicts = []
    for i in range(models):
        print("iteration:", i)
        if cuda:
            model = AttentionRNN(emb_matrix, embeddings_dropout=0.3).cuda()
        else:
            model = AttentionRNN(emb_matrix, embeddings_dropout=0.3)
        optimizer = Adam(model.params, 0.001)
        criterion = BCEWithLogitsLoss()
        print("adam training...")

        sc = train(model=model, train_batches=train_batches, test_batches=test_batches,
              optimizer=optimizer, criterion=criterion, epochs=4, init_patience=2, cuda=cuda)
        print(sc)
        d = generate_results(model, test_batches, cuda)
        dicts.append(d)
        print("adagrad training...")
        optimizer = Adagrad(model.params)
        sc = train(model=model, train_batches=train_batches, test_batches=test_batches,
                   optimizer=optimizer, criterion=criterion, epochs=4, init_patience=2, cuda=cuda)
        print(sc)
        d = generate_results(model, test_batches, cuda)
        dicts.append(d)

    print("added: ", len(dicts))
    if cuda:
        generate_csv(dicts)


run(torch.cuda.is_available())



