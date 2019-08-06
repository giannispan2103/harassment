from globals import GLOVE_EMBEDDINGS_PATH, SEED, MODELS_DIR, TEST_RESULTS, DATA_DIR, CONFIG
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from utils import train, load_model, generate_test_submission_values, generate_results, set_seeds
import pandas as pd
import os
from sklearn.metrics import f1_score, precision_score, recall_score
from preprocess import generate_data
from modeling.multiheaded_attention_rnn import MultiHeadedAttentionRNN
from modeling.projected_multiheaded_attention_rnn import ProjectedMultiHeadedAttentionRNN
from modeling.projected_mutliheaded_vanilla_rnn import ProjectedMultiHeadedVanillaRNN
from modeling.vanilla_rnn import VanillaRnn


def run_model(mdl, cuda):
    if mdl not in ['vanilla', 'vanilla_projected', 'attention', 'projected_attention']:
        NotImplementedError("You must choose one of these: ['vanilla', 'vanilla_projected', 'attention', 'projected_attention']")
    else:
        data = generate_data(embs_path=GLOVE_EMBEDDINGS_PATH, maxlen=CONFIG['maxlen'], batch_size=CONFIG['batch_size'])
        emb_matrix = data['emb_matrix']
        train_batches = data['train_batches']
        val_batches = data['val_batches']
        test_batches = data['test_batches']
        set_seeds(SEED)

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
        runs = CONFIG['iterations']
        for i in range(1, runs+1):
            if mdl == "vanilla_projected":
                model = ProjectedMultiHeadedVanillaRNN(emb_matrix, embeddings_dropout=CONFIG['dropout'])
            elif mdl == "attention":
                model = MultiHeadedAttentionRNN(emb_matrix, embeddings_dropout=CONFIG['dropout'], trainable_embeddings=False)
            elif mdl == "projected_attention":
                model = ProjectedMultiHeadedAttentionRNN(emb_matrix, embeddings_dropout=CONFIG['dropout'])
            else:
                model = VanillaRnn(emb_matrix, embeddings_dropout=CONFIG['dropout'], trainable_embeddings=False)
            optimizer = Adam(model.params, CONFIG['lr'])
            criterion = BCEWithLogitsLoss()
            train(model=model, train_batches=train_batches, test_batches=val_batches,
                  optimizer=optimizer, criterion=criterion, epochs=CONFIG['epochs'], init_patience=CONFIG['patience'], cuda=cuda)
            model = load_model(model)
            d = generate_results(model, test_batches, cuda)
            df = generate_test_submission_values(d['harassment'], d['sexual'], d['physical'], d['indirect'])
            df_results = pd.read_csv(TEST_RESULTS)
            harassment_f1_scores.append(f1_score(df_results.harassment.values, df.Harassment.values))
            harassment_precision_scores.append(precision_score(df_results.harassment.values, df.Harassment.values))
            harassment_recall_scores.append(recall_score(df_results.harassment.values, df.Harassment.values))

            indirect_f1_scores.append(f1_score(df_results.IndirectH.values, df.IndirectH.values))
            indirect_precision_scores.append(precision_score(df_results.IndirectH.values, df.IndirectH.values))
            indirect_recall_scores.append(recall_score(df_results.IndirectH.values, df.IndirectH.values))

            sexual_f1_scores.append(f1_score(df_results.SexualH.values, df.SexualH.values))
            sexual_precision_scores.append(precision_score(df_results.SexualH.values, df.SexualH.values))
            sexual_recall_scores.append(recall_score(df_results.SexualH.values, df.SexualH.values))

            physical_f1_scores.append(f1_score(df_results.PhysicalH.values, df.PhysicalH.values))
            physical_precision_scores.append(precision_score(df_results.PhysicalH.values, df.PhysicalH.values))
            physical_recall_scores.append(recall_score(df_results.PhysicalH.values, df.PhysicalH.values))
        results_dict = {'harassment_f1_score': harassment_f1_scores,
                        'harassment_recall': harassment_recall_scores,
                        'harassment_precision': harassment_precision_scores,
                        'indirect_f1_score': indirect_f1_scores,
                        'indirect_recall': indirect_recall_scores,
                        'indirect_precision': indirect_precision_scores,
                        'sexual_f1_score': sexual_f1_scores,
                        'sexual_recall': sexual_recall_scores,
                        'sexual_precision': sexual_precision_scores,
                        'physical_f1_score': physical_f1_scores,
                        'physical_recall':physical_recall_scores,
                        'physical_precision': physical_precision_scores
                        }
        print("df...")
        df = pd.DataFrame.from_dict(results_dict)
        print("save...")
        df.to_csv(DATA_DIR+mdl+".csv", index=False)

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    run_model("vanilla", False)
    run_model("vanilla_projected", False)
    run_model("attention", False)
    run_model("projected_attention", False)

