"""Implement HuggingfaceModel models."""
import copy
import logging
from collections import Counter


from openai import OpenAI


from uncertainty.models.base_model import BaseModel
from uncertainty.models.base_model import STOP_SEQUENCES

import os
import hashlib
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type,stop_after_attempt

from openai import OpenAI
import openai
import time
from uncertainty.utils.openai import predict,CLIENT


class KeyError(Exception):
    """OpenAIKey not provided in environment variable."""
    pass


import tiktoken

class APIModel(BaseModel):
    """Hugging Face Model."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        if max_new_tokens is None:
            raise
        self.max_new_tokens = max_new_tokens

        # TODO 处理不同的分词器

        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        self.tokenizer = tiktoken.get_encoding("o200k_base") 

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES


        self.model_name = model_name
        self.stop_sequences = stop_sequences


        self.token_limit =  10000
        

        
        self.client = CLIENT

        if not self.client.api_key:
            raise KeyError('Need to provide OpenAI API key in environment variable `OPENAI_API_KEY`.')
        


        

    @retry(wait=wait_random_exponential(min=1, max=10),stop=stop_after_attempt(4))
    def get_p_true(self, input_data):
        """
        计算 GPT-4 在给定 input_data 后，紧跟一个 " A" 的对数似然（log probability）。
        返回值是 log p(" A"|input_data)，
        如果要把它转成概率可以用 math.exp()。
        """

        messages = [
                {'role': 'user', 'content': input_data + " A"},
            ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_new_tokens,
                temperature=0,
                logprobs =True,
            top_logprobs=2
        )
        except openai.RateLimitError as e:
            logging.error(f"RateLimitError: {e}")
            raise e
        # 只取第一个 choice
        choice = response.choices[0]
        logprobs_info = choice.logprobs.content

        last_token_logprob = logprobs_info[-1].logprob
        
        last_token = logprobs_info[-1].token
        
        return last_token_logprob
    
    @retry(wait=wait_random_exponential(min=1, max=10),stop=stop_after_attempt(4))
    def predict(self,prompt, temperature=1.0,return_full=False):
        """Predict with GPT models."""



        if isinstance(prompt, str):
            messages = [
                {'role': 'user', 'content': prompt},
            ]
        else:
            messages = prompt


        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_new_tokens,
                temperature=temperature,
                logprobs =True,
            top_logprobs=2
        )
        except openai.RateLimitError as e:
            logging.error(f"RateLimitError: {e}")
            raise e
        except openai.APITimeoutError as e:
            logging.error(f"APITimeoutError: {e}")
            raise e
        token_log_likelihoods = []
        output = response.choices[0].message.content

        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs is not None:
            for token_info in response.choices[0].logprobs.content:
                if hasattr(token_info, 'logprob'):
                    token_log_likelihoods.append(token_info.logprob)
        else:
            logging.error(f"没有找到logprobs")
            token_log_likelihoods = None
        
        # TODO 获取 last token embedding 以便于计算 p(ik) // p(I dont know), 确保向量维度一样
        last_token_embedding = None
    
        # TODO 检查
        return output, token_log_likelihoods, last_token_embedding





