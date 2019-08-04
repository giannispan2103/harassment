from googletrans import Translator
import time
import pandas as pd
from main_script2 import Tweet, TRAIN_DATA_PATH


def get_posts(csv_path):
    df = pd.read_csv(csv_path)
    df = df[(df['IndirectH']==1)| (df['PhysicalH']==1)]
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


def translate_reviews(data, lan="de"):
    translator = Translator()
    texts, harassment, sexual, indirect, physical = [], [], [], [], []
    for i, d in enumerate(data, start=1):
        print(i)
        """
        self.target = harrasment
        self.indirectH = indirectH
        self.physicalH = physicalH
        self.sexualH = sexualH
        """
        text = translate_back(translator, d.text, lan)
        if text != "$$":
            texts.append(text)
            harassment.append(d.target)
            sexual.append(d.sexualH)
            physical.append(d.physicalH)
            indirect.append(d.indirectH)

    df = pd.DataFrame({'tweet_content': texts,  'harassment': harassment,
                       'IndirectH': indirect, 'PhysicalH': physical, 'SexualH':sexual})
    df.to_csv("translations-{}.csv".format(lan), index=False, encoding="utf8")


if __name__ == "__main__":
    """
    LANGUAGES = {
    'af': 'afrikaans',
    'sq': 'albanian',
    'am': 'amharic',
    'ar': 'arabic',
    'hy': 'armenian',
    'az': 'azerbaijani',
    'eu': 'basque',
    'be': 'belarusian',
    'bn': 'bengali',
    'bs': 'bosnian',
    'bg': 'bulgarian',
    'ca': 'catalan',
    'ceb': 'cebuano',
    'ny': 'chichewa',
    'zh-cn': 'chinese (simplified)',
    'zh-tw': 'chinese (traditional)',
    'co': 'corsican',
    'hr': 'croatian',
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'eo': 'esperanto',
    'et': 'estonian',
    'tl': 'filipino',
    'fi': 'finnish',
    'fr': 'french',
    'fy': 'frisian',
    'gl': 'galician',
    'ka': 'georgian',
    'de': 'german',
    'el': 'greek',
    'gu': 'gujarati',
    'ht': 'haitian creole',
    'ha': 'hausa',
    'haw': 'hawaiian',
    'iw': 'hebrew',
    'hi': 'hindi',
    'hmn': 'hmong',
    'hu': 'hungarian',
    'is': 'icelandic',
    'ig': 'igbo',
    'id': 'indonesian',
    'ga': 'irish',
    'it': 'italian',
    'ja': 'japanese',
    'jw': 'javanese',
    'kn': 'kannada',
    'kk': 'kazakh',
    'km': 'khmer',
    'ko': 'korean',
    'ku': 'kurdish (kurmanji)',
    'ky': 'kyrgyz',
    'lo': 'lao',
    'la': 'latin',
    'lv': 'latvian',
    'lt': 'lithuanian',
    'lb': 'luxembourgish',
    'mk': 'macedonian',
    'mg': 'malagasy',
    'ms': 'malay',
    'ml': 'malayalam',
    'mt': 'maltese',
    'mi': 'maori',
    'mr': 'marathi',
    'mn': 'mongolian',
    'my': 'myanmar (burmese)',
    'ne': 'nepali',
    'no': 'norwegian',
    'ps': 'pashto',
    'fa': 'persian',
    'pl': 'polish',
    'pt': 'portuguese',
    'pa': 'punjabi',
    'ro': 'romanian',
    'ru': 'russian',
    'sm': 'samoan',
    'gd': 'scots gaelic',
    'sr': 'serbian',
    'st': 'sesotho',
    'sn': 'shona',
    'sd': 'sindhi',
    'si': 'sinhala',
    'sk': 'slovak',
    'sl': 'slovenian',
    'so': 'somali',
    'es': 'spanish',
    'su': 'sundanese',
    'sw': 'swahili',
    'sv': 'swedish',
    'tg': 'tajik',
    'ta': 'tamil',
    'te': 'telugu',
    'th': 'thai',
    'tr': 'turkish',
    'uk': 'ukrainian',
    'ur': 'urdu',
    'uz': 'uzbek',
    'vi': 'vietnamese',
    'cy': 'welsh',
    'xh': 'xhosa',
    'yi': 'yiddish',
    'yo': 'yoruba',
    'zu': 'zulu',
    'fil': 'Filipino',
    'he': 'Hebrew'
}
    """
    data = get_posts(TRAIN_DATA_PATH)
    translate_reviews(data, lan="jw")
