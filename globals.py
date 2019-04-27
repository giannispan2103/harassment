# global variables
PAD_TOKEN = "*$*PAD*$*"
UNK_TOKEN = "*$*UNK*$*"
MAXLEN = 300
BATCH_SIZE = 512
SPLIT_POINT = 1700000
SEED = 1985
GLOVE_EMBEDDINGS_PATH = '../input/embeddings/glove.twitter.27B/glove.twitter.27B.200.txt'
TRAIN_DATA_PATH = '../input/harrasment/Train_data_compeition.csv'
TEST_DATA_PATH = '../input/harrasment/Validation_data_competition.csv'
SUBMISSION_PATH = 'submission.csv'
SAMPLE_SUBMISSION = "../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv"

GLOVE_MEAN = -0.005753148614825279
GLOVE_STD = 0.4111380286218564
