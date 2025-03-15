from typing import Any
import os
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from config import *

from zhipuai import ZhipuAI, APIReachLimitError


class ZhipuModel(CustomLLM):
    context_window: int = CONTEXT_WINDOW
    max_tokens: int = MAX_OUTPUT_TOKENS
    model_name: str = ZHIPU_MODEL

    temperature: float = TEMPERATURE
    is_call: bool = True
    client: Any

    input_token = 0

    def __init__(self, model_name: str = None, api_key: str = None,
                 is_call: bool = True, temperature: float = None,
                 max_token: int = None,
                 **kwargs):
        super().__init__(**kwargs)  # 调用父类构造函数
        api_key = ZHIPU_API_KEY if not api_key else api_key
        self.client = ZhipuAI(api_key=api_key)
        self.model_name = self.model_name if not model_name else model_name
        self.is_call = is_call  # is_call 为真时调用 llm 并返回交互结果，is_call 为假时仅返回调用提示词
        self.temperature = self.temperature if not temperature else temperature
        self.max_tokens = self.max_tokens if not max_token else max_token

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model_name,
        )

    def set_api_key(self, api_key: str):
        self.client.api_key = api_key

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # print(prompt)
        # print("----------------------------------------------")
        if self.is_call:
            response = self.client.chat.completions.create(
                model=self.model_name,  # 填写需要调用的模型编码
                messages=[
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE
            )
            completion_response = response.choices[0].message.content
            self.input_token += response.usage.prompt_tokens
        else:
            completion_response = prompt

        return CompletionResponse(text=completion_response)

    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)


# #
if __name__ == "__main__":
    # from baselines.LinkAlign.llms.ApiPool import ZhipuApiPool

    llm = ZhipuModel()
    answer = llm.complete("桌子上有4个苹果，小红吃了1个，小刚拿走了2个，还剩下几个苹果？").text
    print(llm.input_token)
