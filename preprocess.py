import numpy as np
import gc

from preprocess_utils import get_indexed_text, pad_text, load_data, get_comments, \
    get_embeddings, get_embeddings_matrix, \
    get_unique_tokens, create_final_dictionary
from globals import TRAIN_DATA_PATH, TEST_DATA_PATH, PAD_TOKEN, UNK_TOKEN


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
    emb_matrix = get_embeddings_matrix(w2i, embeddings_dict, 300)
    del embeddings_dict
    gc.collect()

    train_batches = create_batches(train_posts, w2i, max_len=maxlen, batch_size=batch_size)
    test_batches = create_batches(test_posts, w2i, max_len=maxlen, batch_size=batch_size)


    return {
            'emb_matrix': emb_matrix,
            'train_batches': train_batches,
            'test_batches': test_batches}


