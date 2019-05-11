import regex as re

FLAGS = re.MULTILINE | re.DOTALL


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

    def __init__(self, post_id, tweet_content, harassment=None, indirectH=None,
                 physicalH=None, sexualH=None ):
        self.post_id = post_id
        self.target = harassment
        self.indirectH = indirectH
        self.physicalH = physicalH
        self.sexualH = sexualH

        self.label = self.target
        self.tokens = [self.clean_token(x) for x in self.tokenize(tweet_content, False).split() if len(x) > 0]
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
        return text.lower().split() if lower else text.lower()



