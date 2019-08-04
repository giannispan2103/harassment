# global variables
PAD_TOKEN = "*$*PAD*$*"
UNK_TOKEN = "*$*UNK*$*"
MAXLEN = 300
BATCH_SIZE = 512
SEED = 1985
GLOVE_EMBEDDINGS_PATH = '../input/embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt'
TRAIN_DATA_PATH = '../input/harrasment/Train_data_compeition.csv'
VALID_DATA_PATH = '../input/harrasment/Validation_data_competition.csv'
TEST_DATA_PATH = "../input/harrasment/testset-competition.csv"
TRANSLATED_DE_PATH ="translations-de.csv"
TRANSLATED_FR_PATH = "translations-fr.csv"
TRANSLATED_GR_PATH = "translations-el.csv"
SUBMISSION_PATH = 'submission_inbalanced.csv'
SAMPLE_SUBMISSION = "../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv"