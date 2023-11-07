import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import distances_from_embeddings
import logging


class OpenAIMain:
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.maxtokens = None
        self.sourcechunks = None
        self.model = None

    def create_context(self, question, embeddings_path, max_len=1800, size="ada"):

        try:
            df = pd.read_csv(
                embeddings_path,
                index_col=0)
            df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

            df.head()

            q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0][
                'embedding']
            df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values,
                                                        distance_metric='cosine')
            returns = []
            cur_len = 0
            for i, row in df.sort_values('distances').iterrows():
                if cur_len + row['n_tokens'] < max_len:
                    returns.append(i)
                    cur_len += row['n_tokens']
                if cur_len > max_len - 300:
                    break
            return df.loc[returns].text.str.cat(sep=". "), returns
        except Exception as e:
            self.logger.error(f"Failed to create context: {str(e)}")
            raise e

    def openai_chat(self, api_key, modelname, maximumtokens, targetchunks, embeddings_path, question):
        try:
            openai.api_key = api_key
            self.model = modelname
            self.maxtokens = maximumtokens
            self.sourcechunks = targetchunks

            past_conversations = []
            context, _ = self.create_context(question, embeddings_path)
            past_context = "\n\n###\n\n".join(past_conversations[-10:])  # get the last 10 conversations

            response = openai.Completion.create(engine=self.model, prompt=f"Answer the question based on the "
                                                                          f"context below, and if the question "
                                                                          f"can't be answered based on"
                                                                          f"the context, say \"I don't "
                                                                          f"know\"\n\nprevious conversation: "
                                                                          f"{past_context}\n\nCo"
                                                                          f"ntext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
                                                max_tokens=self.maxtokens, stop=None, n=self.sourcechunks,
                                                temperature=0.5)

            answer = response.choices[0].text.strip()
            past_conversations.append(f"Question: {question}\nAnswer: {answer}")
            return {
                'answer': answer,
                'max_tokens': self.maxtokens,
                'source_chunks': self.sourcechunks
            }
        except Exception as e:
            self.logger.error(f"Failed to generate response: {str(e)}")
            raise e
