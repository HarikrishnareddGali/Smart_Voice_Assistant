import logging
import pandas as pd
import tiktoken
import openai
import matplotlib.pyplot as plt


class OpenAIEmbeddings:
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.df = None
        self.embedding_model = None
        self.embeddings = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def openai_embedd(self, csv_path, api_key, embedding_model, embedd_path):
        self.df = pd.read_csv(csv_path, index_col=0)
        self.df.columns = ['title', 'text']
        openai.api_key = api_key
        self.embedding_model = embedding_model
        self.embeddings = embedd_path

        self.tokenize_text()
        self.shorten_texts()
        self.generate_embeddings()
        print(self.display_head())

    def tokenize_text(self):
        try:
            self.df['n_tokens'] = self.df.text.apply(lambda x: len(self.tokenizer.encode(x)))
        except Exception as e:
            self.logger.error(f"Error while Tokenisation: {str(e)}")
            raise e
        self.df.n_tokens.hist()
        plt.show()

    def split_into_many(self, text, max_tokens=500, stride=250):  # Note: added stride parameter
        try:
            tokens = self.tokenizer.encode(text)
            chunks = []

            # Use a sliding window approach to split text with overlap
            start_token = 0
            while start_token < len(tokens):
                end_token = start_token + max_tokens
                chunk_tokens = tokens[start_token:end_token]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)

                start_token += stride

            return chunks
        except Exception as e:
            self.logger.error(f"Error while splitting with overlap: {str(e)}")
            raise e

    def shorten_texts(self, max_tokens=500):
        try:
            shortened = []
            for row in self.df.iterrows():
                if row[1]['text'] is None:
                    continue
                if row[1]['n_tokens'] > max_tokens:
                    shortened += self.split_into_many(row[1]['text'])
                else:
                    shortened.append(row[1]['text'])

            self.df = pd.DataFrame(shortened, columns=['text'])
            self.df['n_tokens'] = self.df.text.apply(lambda x: len(self.tokenizer.encode(x)))
            self.df.n_tokens.hist()
            plt.show()
        except Exception as e:
            self.logger.error(f"Error while shortening the texts: {str(e)}")
            raise e

    def generate_embeddings(self):
        try:
            self.df['embeddings'] = self.df.text.apply(
                lambda x: openai.Embedding.create(input=x, engine=self.embedding_model)['data'][0]['embedding'])
            self.df.to_csv(self.embeddings)
        except Exception as e:
            self.logger.error(f"Error while generating embeddings: {str(e)}")
            raise e

    def display_head(self):
        return self.df.head()
