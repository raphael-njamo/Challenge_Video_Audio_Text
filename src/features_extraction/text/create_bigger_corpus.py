import wikipedia
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class CreateCorpus:

    def __init__(self, corpus_len: int, lang: str):
        """Class constructor

        Arguments:
            corpus_len {int} -- lenght of the corpus
            lang {str} -- Language

        Raises:
            ValueError -- lenght of the corpus need to be greater than 0 

        Returns:
            [type] -- [description]
        """

        if corpus_len <= 0:
            raise ValueError(
                f'The lenght of the corpus need to be greater than 0')

        self.corpus_len = corpus_len
        self.lang = lang
        self.corpus = {}
        self.stopwords = set(stopwords.words('french'))
        self.stopwords.update(
            ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '-', '…'])

    def random_page(self, random_state=42, clean=True) -> str:
        """Returns random wikipedia page.
        
        Keyword Arguments:
            random_state {int} -- Fix the random seed (default: {42})
            clean {bool} -- Delete stopwords (default: {True})
        
        Returns:
            {str} -- random wikipedia page.
        """
        wikipedia.set_lang(self.lang)
        random = wikipedia.random(random_state)
        try:
            result = wikipedia.page(random).content
        except:
            result = self.random_page()

        result = result.lower()
        if clean:
            result = ' '.join([i for i in word_tokenize(
                result) if i not in self.stopwords])

        return result

    def generate_corpus(self):

        for idx in tqdm(range(self.corpus_len)):
            self.corpus.update({idx: self.random_page()})
        return self.corpus


if __name__ == '__main__':

    Corpus = CreateCorpus(500, 'fr')
    Corpus = Corpus.generate_corpus()
    Corpus = pd.DataFrame.from_dict(Corpus, orient='index', columns=['Text'])

    print(Corpus.head())

    Corpus = pd.concat([Corpus, pd.read_csv(
        'data/external/corpus_wiki_300.csv', sep="§",error_bad_lines=False)], axis=0)
    print(f'the shape of the corpus is : {len(Corpus)}')
    Corpus.to_csv('data/external/corpus_wiki_300.csv', sep='§', index=False)
