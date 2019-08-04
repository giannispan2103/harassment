from preprocess import generate_data
from modeling.multiheaded_attention_rnn import MultiHeadedAttentionRNN
from modeling.projected_multiheaded_attention_rnn import ProjectedMultiHeadedAttentionRNN
from modeling.projected_mutliheaded_vanilla_rnn import ProjectedMultiHeadedVanillaRNN
from modeling.vanilla_rnn import VanillaRnn
from globals import GLOVE_EMBEDDINGS_PATH, SEED
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from utils import train, load_model, generate_csv, generate_results, set_seeds
import gc
import pandas as pd
import torch


def run_projected_attention(cuda):
    data = generate_data(embs_path=GLOVE_EMBEDDINGS_PATH, maxlen=100, batch_size=32)
    emb_matrix = data['emb_matrix']
    train_batches = data['train_batches']
    val_batches = data['val_batches']
    test_batches = data['test_batches']
    set_seeds(SEED)
    del data
    gc.collect()
    models = 1
    dicts_h = []
    dicts_i = []
    dicts_p = []
    dicts_s = []

    for i in range(1, models+1):
        print("iteration:", i)
        model = ProjectedMultiHeadedAttentionRNN(emb_matrix, embeddings_dropout=0.3)
        optimizer = Adam(model.params, 0.001)
        criterion = BCEWithLogitsLoss()
        print("adam training...")

        train(model=model, train_batches=train_batches, test_batches=val_batches,
                   optimizer=optimizer, criterion=criterion, epochs=20, init_patience=10, cuda=cuda)
        model = load_model(model)
        d = generate_results(model, test_batches, cuda)
        dicts_h.append(d['harassment'])
        dicts_s.append(d['sexual'])
        dicts_p.append(d['physical'])
        dicts_i.append(d['indirect'])

    generate_csv(dicts_h=dicts_h, dicts_i=dicts_i, dicts_p=dicts_p, dicts_s=dicts_s)
    data = pd.read_csv("submission_inbalanced.csv")

    print("Indirect comments:", sum(data.IndirectH))
    print("Sexual comments:", sum(data.SexualH))

    print("Physical comments:", sum(data.PhysicalH))
    print("Harassment comments:", sum(data.Harassment))


def run_attention(cuda):
    data = generate_data(embs_path=GLOVE_EMBEDDINGS_PATH, maxlen=100, batch_size=32)
    emb_matrix = data['emb_matrix']
    train_batches = data['train_batches']
    val_batches = data['val_batches']
    test_batches = data['test_batches']
    set_seeds(SEED)
    del data
    gc.collect()
    models = 1
    dicts_h = []
    dicts_i = []
    dicts_p = []
    dicts_s = []

    for i in range(1, models+1):
        print("iteration:", i)
        model = MultiHeadedAttentionRNN(emb_matrix, embeddings_dropout=0.3, trainable_embeddings=False)
        # model = ProjectedMultiHeadedAttentionRNN(emb_matrix, embeddings_dropout=0.3)
        # model = VanillaRnn(emb_matrix, embeddings_dropout=0.3, trainable_embeddings=False)
        optimizer = Adam(model.params, 0.001)
        criterion = BCEWithLogitsLoss()
        print("adam training...")

        train(model=model, train_batches=train_batches, test_batches=val_batches,
                   optimizer=optimizer, criterion=criterion, epochs=20, init_patience=10, cuda=cuda)
        model = load_model(model)
        d = generate_results(model, test_batches, cuda)
        dicts_h.append(d['harassment'])
        dicts_s.append(d['sexual'])
        dicts_p.append(d['physical'])
        dicts_i.append(d['indirect'])

    generate_csv(dicts_h=dicts_h, dicts_i=dicts_i, dicts_p=dicts_p, dicts_s=dicts_s)
    data = pd.read_csv("submission_inbalanced.csv")

    print("Indirect comments:", sum(data.IndirectH))
    print("Sexual comments:", sum(data.SexualH))

    print("Physical comments:", sum(data.PhysicalH))
    print("Harassment comments:", sum(data.Harassment))

def run_vanilla(cuda):
    data = generate_data(embs_path=GLOVE_EMBEDDINGS_PATH, maxlen=100, batch_size=32)
    emb_matrix = data['emb_matrix']
    train_batches = data['train_batches']
    val_batches = data['val_batches']
    test_batches = data['test_batches']
    set_seeds(SEED)
    del data
    gc.collect()
    models = 1
    dicts_h = []
    dicts_i = []
    dicts_p = []
    dicts_s = []

    for i in range(1, models+1):
        print("iteration:", i)
        model = VanillaRnn(emb_matrix, embeddings_dropout=0.3, trainable_embeddings=False)
        optimizer = Adam(model.params, 0.001)
        criterion = BCEWithLogitsLoss()
        print("adam training...")

        train(model=model, train_batches=train_batches, test_batches=val_batches,
                   optimizer=optimizer, criterion=criterion, epochs=20, init_patience=10, cuda=cuda)
        model = load_model(model)
        d = generate_results(model, test_batches, cuda)
        dicts_h.append(d['harassment'])
        dicts_s.append(d['sexual'])
        dicts_p.append(d['physical'])
        dicts_i.append(d['indirect'])

    generate_csv(dicts_h=dicts_h, dicts_i=dicts_i, dicts_p=dicts_p, dicts_s=dicts_s)
    data = pd.read_csv("submission_inbalanced.csv")

    print("Indirect comments:", sum(data.IndirectH))
    print("Sexual comments:", sum(data.SexualH))

    print("Physical comments:", sum(data.PhysicalH))
    print("Harassment comments:", sum(data.Harassment))


def run_projected_vanilla(cuda):
    data = generate_data(embs_path=GLOVE_EMBEDDINGS_PATH, maxlen=100, batch_size=32)
    emb_matrix = data['emb_matrix']
    train_batches = data['train_batches']
    val_batches = data['val_batches']
    test_batches = data['test_batches']
    set_seeds(SEED)
    del data
    gc.collect()
    models = 1
    dicts_h = []
    dicts_i = []
    dicts_p = []
    dicts_s = []

    for i in range(1, models+1):
        print("iteration:", i)
        model = ProjectedMultiHeadedVanillaRNN(emb_matrix, embeddings_dropout=0.3)
        optimizer = Adam(model.params, 0.001)
        criterion = BCEWithLogitsLoss()
        print("adam training...")

        train(model=model, train_batches=train_batches, test_batches=val_batches,
                   optimizer=optimizer, criterion=criterion, epochs=20, init_patience=10, cuda=cuda)
        model = load_model(model)
        d = generate_results(model, test_batches, cuda)
        dicts_h.append(d['harassment'])
        dicts_s.append(d['sexual'])
        dicts_p.append(d['physical'])
        dicts_i.append(d['indirect'])

    generate_csv(dicts_h=dicts_h, dicts_i=dicts_i, dicts_p=dicts_p, dicts_s=dicts_s)
    data = pd.read_csv("submission_inbalanced.csv")

    print("Indirect comments:", sum(data.IndirectH))
    print("Sexual comments:", sum(data.SexualH))

    print("Physical comments:", sum(data.PhysicalH))
    print("Harassment comments:", sum(data.Harassment))


if __name__ == "__main__":
    run_projected_attention(torch.cuda.is_available())