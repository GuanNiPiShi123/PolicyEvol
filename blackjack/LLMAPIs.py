import os
import openai
import dashscope
import replicate
from http import HTTPStatus
from openai import OpenAI


class GPT35API:

    def __init__(self) -> None:
        openai.api_key = "YOUR KEY"

    def response(self, mes):
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=mes,
            top_p=0.95,
            temperature=1,
        )
        return response.get("choices")[0]["message"]["content"]


class GPT4API:

    def __init__(self) -> None:
        openai.api_key = "YOUR KEY"

    def response(self, mes):
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=mes,
            top_p=0.95,
            temperature=1,
        )
        return response.get("choices")[0]["message"]["content"]


class llama2_70b_chatAPI:

    def response(self, mes):
        os.environ["REPLICATE_API_TOKEN"] = "YOUR KEY"
        system_prompt = ""
        prompt = ""
        for item in mes:
            if item.get('role') == 'system':
                system_prompt = item.get('content')
            if item.get('role') == 'user':
                prompt = item.get('content')
        try:
            iterator = replicate.run(
                "meta/llama-2-70b-chat",
                input={
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "temperature": 1,
                    "top_p": 0.95,
                    "max_new_tokens": 4000,
                },
            )
            result_string = ''.join(text for text in iterator)
        except replicate.exceptions.ModelError as e:
            with open("/replicate_modelerror_times.txt", "a") as f:
                f.write("llama2_70b_chat:replicate_modelerror_times\n")
            print(e)
        except Exception as e:
            with open("Exception.txt", "a") as f:
                f.write("llama2_70b_chat:Exception_times\n")
            print(e)
        return result_string


class QwenAPI:

    def __init__(self) -> None:
        dashscope.api_key = ''

    def response(self, mes):
        for i in range(5):
            try:
                response = dashscope.Generation.call(
                    'qwen-max',
                    messages=mes,
                    temperature=1,
                    top_p=0.95,
                    result_format='message',
                    )
                #if response.status_code == HTTPStatus.OK:
                data_res = response['output']['choices'][0]['message']['content']
                return data_res
            except:
                continue
                #print(
                    #'Request id: %s, Status code: %s, error code: %s, error message: %s'
                    #% (response.request_id, response.status_code, response.code,
                       #response.message))
            
class DeepSeekAPI:

    def __init__(self) -> None:
        self.api_key = ''

    def response(self, mes):
        client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=mes,
            stream=False
            )
        return response.choices[0].message.content

