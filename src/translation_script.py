from googletrans import Translator
import time
import pandas as pd
from encapsulations import Tweet
from globals import TRAIN_DATA_PATH, DATA_DIR


def get_posts(csv_path):
    df = pd.read_csv(csv_path)
    df = df[(df['IndirectH'] == 1) | (df['PhysicalH'] == 1)]
    posts = []
    for i, row in df.iterrows():
        post = Tweet(post_id=row['post_id'],
                     tweet_content=row['tweet_content'],
                     harrasment=row['harassment'],
                     indirectH=row['IndirectH'],
                     physicalH=row['PhysicalH'],
                     sexualH=row['SexualH'])
        posts.append(post)
    return posts


def translate(translator, text, src="en", dest="de"):
    # translation = Translator()
    try:
        translation = translator.translate(text, src=src, dest=dest)
        return translation.text
    except:
        time.sleep(10)
        print('first catch:::', src, "->", dest)
        try:
            translation = translator.translate(text, src=src, dest=dest)
            return translation.text
        except:
            time.sleep(10)
            print("second catch:::",src, "->",  dest)
            try:
                print("third catch:::", src, "->", dest)
                translation = translator.translate(text, src=src, dest=dest)
                return translation.text
            except:
                print("return")
                return "$$"


def translate_back(translator, text, lan="de"):
    translation = translate(translator, text, "en", lan)
    if translation == "$$":
        return "$$"
    else:
        return translate(translator, translation, lan, "en")


def translate_reviews(tweets, lan="de"):
    translator = Translator()
    texts, harassment, sexual, indirect, physical = [], [], [], [], []
    for i, d in enumerate(tweets, start=1):
        print(i)
        text = translate_back(translator, d.text, lan)
        if text != "$$":
            texts.append(text)
            harassment.append(d.target)
            sexual.append(d.sexualH)
            physical.append(d.physicalH)
            indirect.append(d.indirectH)

    df = pd.DataFrame({'tweet_content': texts,  'harassment': harassment,
                       'IndirectH': indirect, 'PhysicalH': physical, 'SexualH':sexual})
    df.to_csv(DATA_DIR+"translations-{}.csv".format(lan), index=False, encoding="utf8")


if __name__ == "__main__":
    data = get_posts(TRAIN_DATA_PATH)
    translate_reviews(data, lan="de")
    translate_reviews(data, lan="el")
    translate_reviews(data, lan="fr")
