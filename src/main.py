from preprocess import generate_data
from modeling.multiheaded_attention_rnn import MultiHeadedAttentionRNN
from modeling.projected_multiheaded_attention_rnn import ProjectedMultiHeadedAttentionRNN
from modeling.projected_mutliheaded_vanilla_rnn import ProjectedMultiHeadedVanillaRNN
from modeling.vanilla_rnn import VanillaRnn
from globals import GLOVE_EMBEDDINGS_PATH, SEED, MODELS_DIR, TEST_RESULTS, SUBMISSION_PATH
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from utils import train, load_model, generate_test_submission_values, generate_results, set_seeds
import gc
import pandas as pd
import torch
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def run_projected_attention(cuda):
    data = generate_data(embs_path=GLOVE_EMBEDDINGS_PATH, maxlen=100, batch_size=32)
    emb_matrix = data['emb_matrix']
    train_batches = data['train_batches']
    val_batches = data['val_batches']
    test_batches = data['test_batches']
    set_seeds(SEED)
    model = ProjectedMultiHeadedAttentionRNN(emb_matrix, embeddings_dropout=0.3)
    optimizer = Adam(model.params, 0.001)
    criterion = BCEWithLogitsLoss()
    print("adam training...")

    train(model=model, train_batches=train_batches, test_batches=val_batches,
                   optimizer=optimizer, criterion=criterion, epochs=20, init_patience=10, cuda=cuda)
    model = load_model(model)
    d = generate_results(model, test_batches, cuda)
    df = generate_test_submission_values(d['harassment'], d['indirect'], d['physical'], d['sexual'])
    df_results = pd.read_csv(TEST_RESULTS)
    print("HARASSMENT - ACCURACY: ", accuracy_score(df_results.harassment.values, df.Harassment.values))
    print("HARASSMENT  - F1: ", f1_score(df_results.harassment.values, df.Harassment.values))
    print("HARASSMENT - PRECISION:", precision_score(df_results.harassment.values, df.Harassment.values))
    print("HARASSMENT - RECALL:", recall_score(df_results.harassment.values, df.Harassment.values))

    print("INDIRECT - ACCURACY: ", accuracy_score(df_results.IndirectH.values, df.IndirectH.values))
    print("INDIRECT  - F1: ", f1_score(df_results.IndirectH.values, df.IndirectH.values))
    print("INDIRECT - PRECISION:", precision_score(df_results.IndirectH.values, df.IndirectH.values))
    print("INDIRECT - RECALL:", recall_score(df_results.IndirectH.values, df.IndirectH.values))

    print("SEXUAL NEW - ACCURACY: ", accuracy_score(df_results.SexualH.values, df.SexualH.values))
    print("SEXUAL NEW - F1: ", f1_score(df_results.SexualH.values, df.SexualH.values))
    print("SEXUAL - PRECISION:", precision_score(df_results.SexualH.values, df.SexualH.values))
    print("SEXUAL - RECALL:", recall_score(df_results.SexualH.values, df.SexualH.values))

    print("PHYSICAL- ACCURACY: ", accuracy_score(df_results.PhysicalH.values, df.PhysicalH.values))
    print("PHYSICAL - F1: ", f1_score(df_results.PhysicalH.values, df.PhysicalH.values))
    print("PHYSICAL - PRECISION:", precision_score(df_results.PhysicalH.values, df.PhysicalH.values))
    print("PHYSICAL - RECALL:", recall_score(df_results.PhysicalH.values, df.PhysicalH.values))


def run_attention(cuda):
    data = generate_data(embs_path=GLOVE_EMBEDDINGS_PATH, maxlen=100, batch_size=32)
    emb_matrix = data['emb_matrix']
    train_batches = data['train_batches']
    val_batches = data['val_batches']
    test_batches = data['test_batches']
    set_seeds(SEED)
    del data
    gc.collect()

    model = MultiHeadedAttentionRNN(emb_matrix, embeddings_dropout=0.3, trainable_embeddings=False)
    optimizer = Adam(model.params, 0.001)
    criterion = BCEWithLogitsLoss()

    train(model=model, train_batches=train_batches, test_batches=val_batches,
                   optimizer=optimizer, criterion=criterion, epochs=20, init_patience=10, cuda=cuda)
    model = load_model(model)
    d = generate_results(model, test_batches, cuda)

    df = generate_test_submission_values(d['harassment'], d['indirect'], d['physical'], d['sexual'])
    df_results = pd.read_csv(TEST_RESULTS)
    print("HARASSMENT - ACCURACY: ", accuracy_score(df_results.harassment.values, df.Harassment.values))
    print("HARASSMENT  - F1: ", f1_score(df_results.harassment.values, df.Harassment.values))
    print("HARASSMENT - PRECISION:", precision_score(df_results.harassment.values, df.Harassment.values))
    print("HARASSMENT - RECALL:", recall_score(df_results.harassment.values, df.Harassment.values))

    print("INDIRECT - ACCURACY: ", accuracy_score(df_results.IndirectH.values, df.IndirectH.values))
    print("INDIRECT  - F1: ", f1_score(df_results.IndirectH.values, df.IndirectH.values))
    print("INDIRECT - PRECISION:", precision_score(df_results.IndirectH.values, df.IndirectH.values))
    print("INDIRECT - RECALL:", recall_score(df_results.IndirectH.values, df.IndirectH.values))

    print("SEXUAL NEW - ACCURACY: ", accuracy_score(df_results.SexualH.values, df.SexualH.values))
    print("SEXUAL NEW - F1: ", f1_score(df_results.SexualH.values, df.SexualH.values))
    print("SEXUAL - PRECISION:", precision_score(df_results.SexualH.values, df.SexualH.values))
    print("SEXUAL - RECALL:", recall_score(df_results.SexualH.values, df.SexualH.values))

    print("PHYSICAL- ACCURACY: ", accuracy_score(df_results.PhysicalH.values, df.PhysicalH.values))
    print("PHYSICAL - F1: ", f1_score(df_results.PhysicalH.values, df.PhysicalH.values))
    print("PHYSICAL - PRECISION:", precision_score(df_results.PhysicalH.values, df.PhysicalH.values))
    print("PHYSICAL - RECALL:", recall_score(df_results.PhysicalH.values, df.PhysicalH.values))


def run_vanilla(cuda):
    data = generate_data(embs_path=GLOVE_EMBEDDINGS_PATH, maxlen=100, batch_size=32)
    emb_matrix = data['emb_matrix']
    train_batches = data['train_batches']
    val_batches = data['val_batches']
    test_batches = data['test_batches']
    set_seeds(SEED)

    model = VanillaRnn(emb_matrix, embeddings_dropout=0.3, trainable_embeddings=False)
    optimizer = Adam(model.params, 0.001)
    criterion = BCEWithLogitsLoss()
    print("adam training...")

    train(model=model, train_batches=train_batches, test_batches=val_batches,
                   optimizer=optimizer, criterion=criterion, epochs=20, init_patience=10, cuda=cuda)
    model = load_model(model)
    d = generate_results(model, test_batches, cuda)
    df = generate_test_submission_values(d['harassment'], d['indirect'], d['physical'], d['sexual'])
    df_results = pd.read_csv(TEST_RESULTS)
    print("HARASSMENT - ACCURACY: ", accuracy_score(df_results.harassment.values, df.Harassment.values))
    print("HARASSMENT  - F1: ", f1_score(df_results.harassment.values, df.Harassment.values))
    print("HARASSMENT - PRECISION:", precision_score(df_results.harassment.values, df.Harassment.values))
    print("HARASSMENT - RECALL:", recall_score(df_results.harassment.values, df.Harassment.values))

    print("INDIRECT - ACCURACY: ", accuracy_score(df_results.IndirectH.values, df.IndirectH.values))
    print("INDIRECT  - F1: ", f1_score(df_results.IndirectH.values, df.IndirectH.values))
    print("INDIRECT - PRECISION:", precision_score(df_results.IndirectH.values, df.IndirectH.values))
    print("INDIRECT - RECALL:", recall_score(df_results.IndirectH.values, df.IndirectH.values))

    print("SEXUAL NEW - ACCURACY: ", accuracy_score(df_results.SexualH.values, df.SexualH.values))
    print("SEXUAL NEW - F1: ", f1_score(df_results.SexualH.values, df.SexualH.values))
    print("SEXUAL - PRECISION:", precision_score(df_results.SexualH.values, df.SexualH.values))
    print("SEXUAL - RECALL:", recall_score(df_results.SexualH.values, df.SexualH.values))

    print("PHYSICAL- ACCURACY: ", accuracy_score(df_results.PhysicalH.values, df.PhysicalH.values))
    print("PHYSICAL - F1: ", f1_score(df_results.PhysicalH.values, df.PhysicalH.values))
    print("PHYSICAL - PRECISION:", precision_score(df_results.PhysicalH.values, df.PhysicalH.values))
    print("PHYSICAL - RECALL:", recall_score(df_results.PhysicalH.values, df.PhysicalH.values))



def run_projected_vanilla(cuda):
    data = generate_data(embs_path=GLOVE_EMBEDDINGS_PATH, maxlen=100, batch_size=32)
    emb_matrix = data['emb_matrix']
    train_batches = data['train_batches']
    val_batches = data['val_batches']
    test_batches = data['test_batches']
    set_seeds(SEED)

    model = ProjectedMultiHeadedVanillaRNN(emb_matrix, embeddings_dropout=0.3)
    optimizer = Adam(model.params, 0.001)
    criterion = BCEWithLogitsLoss()

    train(model=model, train_batches=train_batches, test_batches=val_batches,
                   optimizer=optimizer, criterion=criterion, epochs=20, init_patience=10, cuda=cuda)
    model = load_model(model)
    d = generate_results(model, test_batches, cuda)
    df = generate_test_submission_values(d['harassment'], d['indirect'], d['physical'], d['sexual'])
    df_results = pd.read_csv(TEST_RESULTS)
    print("HARASSMENT - ACCURACY: ", accuracy_score(df_results.harassment.values, df.Harassment.values))
    print("HARASSMENT  - F1: ", f1_score(df_results.harassment.values, df.Harassment.values))
    print("HARASSMENT - PRECISION:", precision_score(df_results.harassment.values, df.Harassment.values))
    print("HARASSMENT - RECALL:", recall_score(df_results.harassment.values, df.Harassment.values))

    print("INDIRECT - ACCURACY: ", accuracy_score(df_results.IndirectH.values, df.IndirectH.values))
    print("INDIRECT  - F1: ", f1_score(df_results.IndirectH.values, df.IndirectH.values))
    print("INDIRECT - PRECISION:", precision_score(df_results.IndirectH.values, df.IndirectH.values))
    print("INDIRECT - RECALL:", recall_score(df_results.IndirectH.values, df.IndirectH.values))

    print("SEXUAL NEW - ACCURACY: ", accuracy_score(df_results.SexualH.values, df.SexualH.values))
    print("SEXUAL NEW - F1: ", f1_score(df_results.SexualH.values, df.SexualH.values))
    print("SEXUAL - PRECISION:", precision_score(df_results.SexualH.values, df.SexualH.values))
    print("SEXUAL - RECALL:", recall_score(df_results.SexualH.values, df.SexualH.values))

    print("PHYSICAL- ACCURACY: ", accuracy_score(df_results.PhysicalH.values, df.PhysicalH.values))
    print("PHYSICAL - F1: ", f1_score(df_results.PhysicalH.values, df.PhysicalH.values))
    print("PHYSICAL - PRECISION:", precision_score(df_results.PhysicalH.values, df.PhysicalH.values))
    print("PHYSICAL - RECALL:", recall_score(df_results.PhysicalH.values, df.PhysicalH.values))

def run_model(cuda, mdl):
    data = generate_data(embs_path=GLOVE_EMBEDDINGS_PATH, maxlen=100, batch_size=32)
    emb_matrix = data['emb_matrix']
    train_batches = data['train_batches']
    val_batches = data['val_batches']
    test_batches = data['test_batches']
    set_seeds(SEED)
    model = ProjectedMultiHeadedVanillaRNN(emb_matrix, embeddings_dropout=0.3)
    optimizer = Adam(model.params, 0.001)
    criterion = BCEWithLogitsLoss()
    print("adam training...")

    harassment_f1_scores = []
    harassment_recall_scores = []
    harassment_precision_scores = []

    indirect_f1_scores = []
    indirect_recall_scores = []
    indirect_precision_scores = []

    sexual_f1_scores = []
    sexual_recall_scores = []
    sexual_precision_scores = []

    physical_f1_scores = []
    physical_recall_scores = []
    physical_precision_scores = []
    runs = 10
    for i in range(1, runs+1):
        train(model=model, train_batches=train_batches, test_batches=val_batches,
              optimizer=optimizer, criterion=criterion, epochs=20, init_patience=10, cuda=cuda)
        model = load_model(model)
        d = generate_results(model, test_batches, cuda)
        df = generate_test_submission_values(d['harassment'], d['sexual'], d['physical'], d['indirect'])
        df_results = pd.read_csv(TEST_RESULTS)
        harassment_f1_scores.append(f1_score(df_results.harassment.values, df.Harassment.values))
        harassment_precision_scores.append(precision_score(df_results.harassment.values, df.Harassment.values))
        harassment_recall_scores.append(recall_score(df_results.harassment.values, df.Harassment.values))

        indirect_f1_scores.append(f1_score(df_results.IndirectH.values, df.IndirectH.values))
        print("INDIRECT - PRECISION:", precision_score(df_results.IndirectH.values, df.IndirectH.values))
        print("INDIRECT - RECALL:", recall_score(df_results.IndirectH.values, df.IndirectH.values))

        print("SEXUAL NEW - ACCURACY: ", accuracy_score(df_results.SexualH.values, df.SexualH.values))
        print("SEXUAL NEW - F1: ", f1_score(df_results.SexualH.values, df.SexualH.values))
        print("SEXUAL - PRECISION:", precision_score(df_results.SexualH.values, df.SexualH.values))
        print("SEXUAL - RECALL:", recall_score(df_results.SexualH.values, df.SexualH.values))

        print("PHYSICAL- ACCURACY: ", accuracy_score(df_results.PhysicalH.values, df.PhysicalH.values))
        print("PHYSICAL - F1: ", f1_score(df_results.PhysicalH.values, df.PhysicalH.values))
        print("PHYSICAL - PRECISION:", precision_score(df_results.PhysicalH.values, df.PhysicalH.values))
        print("PHYSICAL - RECALL:", recall_score(df_results.PhysicalH.values, df.PhysicalH.values))
if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    run_vanilla(torch.cuda.is_available())
    # run_projected_vanilla(torch.cuda.is_available())
    # run_attention(torch.cuda.is_available())
    # run_projected_attention(torch.cuda.is_available())
