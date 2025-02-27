import copy
import os
import re
import time
import json
import math
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
from datasets import load_dataset
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import asyncio
import numpy as np


# from retrievers import calculate_retrieval_metrics
import functools
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def extract_program(a_string,lan='python',first_block_only=False):
    indices_object = re.finditer(pattern="```", string=a_string)
    indices = [index.start() for index in indices_object]
    contents = ''
    if len(indices) == 0:
        contents = a_string
    elif len(indices) % 2 == 0:
        for i in range(0, len(indices), 2):
            cur_str = a_string[indices[i]:indices[i + 1]]
            if cur_str.startswith(f"```{lan}"):
                cur_str = cur_str[len(f"```{lan}"):]
            elif cur_str.startswith(f"```\n{lan}"):
                cur_str = cur_str[len(f"```\n{lan}"):]
            elif cur_str.startswith("```"):
                cur_str = cur_str[len("```"):]
            contents += cur_str
            if first_block_only:
                break
    else:
        contents = a_string.replace(f"```{lan}", '').replace("```", '').replace(f"{lan}\n", '')
    lines = contents.strip().split('\n')
    if lines[-1].isidentifier():
        contents = '\n'.join(lines[:-1])
    return contents.replace(f"{lan}\n", '')


def call_api(func):
    count = 0
    while True:
        try:
            count += 1
            output = func()
            break
        except Exception as e:
            logger.info(f"Exception while using api: {e}")
            if "rate limit" in str(e).lower() or "rate_limit" in str(e).lower():
                logger.info("Rate limit exceeded, waiting 10 secs and retrying...")
                time.sleep(10)
            elif count < 5:
                logger.info("Encountered error, retrying...")
                time.sleep(5)
            else:
                logger.info("Skipping generation due to unknown error after 5 retries.")
                output = None
                break
    return output


def format_chat(message, include_system=True, system_message="You are a helpful assistant."):
    if include_system:
        chat = [{"role": "system", "content": system_message}, {"role": "user", "content": message}]
    else:
        chat = [{"role": "user", "content": message}]
    return chat


class ClaudeModel:

    def __init__(self, version):
        from anthropic import AnthropicVertex
        PROJECT_ID = "xxx"  # @param
        LOCATION = "xxx"  # @param
        self.model = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)
        self.version = version

    def rerank(self, docs, query, topk):
        doc_string = ''
        indices_map = {}
        for doc_idx,doc in enumerate(docs):
            assert isinstance(doc,list)
            doc_string += "[{}]. {}\n\n".format(doc_idx + 1, re.sub('\n+', ' ', doc[1]))
            indices_map[doc_idx + 1] = doc[0]
        cur_query = query.replace('\n','  ')
        prompt = (f'The following passages are related to query: {cur_query}\n\n'
                  f'{doc_string}'
                  f'First identify the essential problem in the query.\n'
                  f'Think step by step to reason about why each document is relevant or irrelevant.\n'
                  f'Rank these passages based on their relevance to the query.\n'
                  f'Please output the ranking result of passages as a list, where the first element is the id of the most relevant '
                  f'passage, the second element is the id of the second most element, etc.\n'
                  f'Please strictly follow the format to output a list of {topk} ids corresponding to the most relevant {topk} passages:\n'
                  f'```json\n'
                  f'[...]\n'
                  f'```')
        func = functools.partial(
            self.model.messages.create,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.version,
            temperature=0.8,
            top_p=0.8
        )
        message = call_api(func)
        response = json.loads(message.model_dump_json(indent=2))
        ranks = extract_program(response['content'][0]['text'],lan='json')
        return [indices_map[r] for r in ranks]


class OpenAIModel:
    def __init__(self, model_name, temperature=0.8, top_p=0.8):
        import openai
        if "azure" in model_name:
            # env var: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and OPENAI_API_VERSION
            self.model = openai.AzureOpenAI()
            model_name = model_name[model_name.index("/")+1:]
        else:
            # make sure to set the OPENAI_API_KEY environment variable
            self.model = openai.OpenAI()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = 2048

    def rerank(self, docs, query, topk):
        doc_string = ''
        indices_map = {}
        for doc_idx,doc in enumerate(docs):
            assert isinstance(doc,list)
            doc_string += "[{}]. {}\n\n".format(doc_idx + 1, re.sub('\n+', ' ', doc[1]))
            indices_map[doc_idx + 1] = doc[0]
        cur_query = query.replace('\n','  ')
        prompt = (f'The following passages are related to query: {cur_query}\n\n'
                  f'{doc_string}'
                  f'First identify the essential problem in the query.\n'
                  f'Think step by step to reason about why each document is relevant or irrelevant.\n'
                  f'Rank these passages based on their relevance to the query.\n'
                  f'Please output the ranking result of passages as a list, where the first element is the id of the most relevant '
                  f'passage, the second element is the id of the second most element, etc.\n'
                  f'Please strictly follow the format to output a list of {topk} ids corresponding to the most relevant {topk} passages, sorted from the most to least relevant passage. First think step by step and write the reasoning process, then output the ranking results as a list of ids in a json format.'
                  )
        inputs = format_chat(prompt, system_message="You are a helpful assistant")
        func = functools.partial(
            self.model.chat.completions.create,
            model=self.model_name,
            messages=inputs,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        def parse_json(text):
            matches = re.findall(r"(?:```json\s*)(.+)(?:```)", text, re.DOTALL)
            if len(matches) > 0:
                try:
                    return json.loads(matches[-1].strip())
                except:
                    return None
            return None

        output = call_api(func)
        if output is not None:
            response = parse_json(output.choices[0].message.content)
            if response is None:
                return None
            return [indices_map[r] for r in response if r in indices_map]
            # return output.choices[0].message.content
        return None

class JudgeRank:
    def __init__(self):
        # model_name = "meta-llama/Llama-3.1-70B-Instruct"
        model_name = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
        self.sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=2048)
        self.sampling_params_final = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=32, logprobs=10)
        self.model = LLM(model=model_name, dtype="bfloat16", tensor_parallel_size=torch.cuda.device_count(), max_model_len=16384)

        self.yes_token = self.model.get_tokenizer().encode(" Yes", add_special_tokens=False)[0]
        self.no_token = self.model.get_tokenizer().encode(" No", add_special_tokens=False)[0]
        # set random seed
        torch.manual_seed(42)

    def rerank(self, docs, query, topk):
        scores = []
        query_prompt = (f'You will be presented with a query.\n\n'
                        f'Your task consists of the following step:\n\n'
                        f'1. Analyze the query:\n'
                        f'- Carefully read each sentence of the query.\n'
                        f'- Identify the core problem or question being asked.\n\n'
                        f'Here is the query:\n'
                        f'{query}\n\n'
                        )

        output = self.model.generate([query_prompt], self.sampling_params)
        query_prompt_output = output[0].outputs[0].text

        batch_size = 35
        all_prompts = []
        # for doc in docs:
        for i in range(0, len(docs), batch_size):
            list_docs = docs[i:i+batch_size]
            doc_prompt = (f'You will be presented with a query, an analysis of the query, and a passage.\n\n'
                            f'Your task consists of the following steps:\n\n'
                            f'1. Analyze the passage:\n'
                            f'- Thoroughly examine each sentence of the passage.\n'
                            f'- List all sentences from the passage that are relevant the query.\n'
                            f'- Briefly explain how each sentence listed is related the query.\n\n'
                            f'2. Assess overall relevance:\n'
                            f'- If the passage, particularly the relevant sentences (if applicable), are related to the query, briefly explain why.\n'
                            f'- Otherwise, briefly explain why not.\n\n'
                            f'Here is the query:\n'
                            f'{query}\n\n'
                            f'Here is the analysis of the query:\n'
                            f'{query_prompt_output}\n\n'
                            f'Here is the passage:\n'
                            )
            doc_prompts = [doc_prompt+'{}\n\n'.format(doc["text"]) for doc in list_docs]
            
            output = self.model.generate(doc_prompts, self.sampling_params)
            doc_prompt_outputs = [o.outputs[0].text for o in output]
            # print(doc_prompt_output)

            judgement_prompt = (f'You will be presented with a query, an analysis of the query, a passage, and an analysis of the passage.\n\n'
                            f'Your task is to assess if the passage is relevant to the query in one word:\n'
                            f'- Yes: If the passage is relevant to the query.\n'
                            f'- No: Otherwise.\n\n'
                            f'Important: Respond using exactly one of the following two words without quotation marks: Yes or No.\n\n'
                            f'Here is the query:\n'
                            f'{query}\n\n'
                            f'Here is the analysis of the query:\n'
                            f'{query_prompt_output}\n\n'
                            f'Here is the passage:\n'
                            )
            judgement_prompt1 = ('{}\n\n'
                            f'Here is the analysis of the passage:\n'
                            )
            judgement_prompt2 = ('{}\n\n'
                            'The final answer is: '
                            )
            judgement_prompts = [judgement_prompt+judgement_prompt1.format(doc["text"])+judgement_prompt2.format(doc_prompt_output) for doc, doc_prompt_output in zip(list_docs, doc_prompt_outputs)]
            all_prompts.extend(judgement_prompts)
            judgement_prompt_output = self.model.generate(judgement_prompts, self.sampling_params_final)

            for j in judgement_prompt_output:
                logprobs = j.outputs[0].logprobs[0]
                if self.yes_token not in logprobs or self.no_token not in logprobs:
                    if self.yes_token not in logprobs and self.no_token not in logprobs:
                        scores.append(float('-inf'))
                        continue
                    elif self.yes_token not in logprobs and (float("-inf") in [item.logprob for item in logprobs.values()]):
                        prob_yes = 0.0
                        prob_no = math.exp(logprobs[self.no_token].logprob)
                    elif self.no_token not in logprobs and (float("-inf") in [item.logprob for item in logprobs.values()]):
                        prob_no = 0.0
                        prob_yes = math.exp(logprobs[self.yes_token].logprob)
                    else:
                        import pdb; pdb.set_trace()
                else:
                    prob_yes = math.exp(logprobs[self.yes_token].logprob)
                    prob_no = math.exp(logprobs[self.no_token].logprob)

                score = prob_yes / (prob_yes + prob_no)
                scores.append(score)

        
        ranking = {doc["id"]: score for doc, score in zip(docs, scores)}
        ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:topk])
        write_dict = {"query": query, "judgement_prompts": all_prompts, "scores": scores, "ranking": ranking}
        # write dict to json file
        with open("judge_rank_r_results.jsonl", "a") as f:
            f.write(json.dumps(write_dict) + "\n")
        return ranking
    

class R1:
    def __init__(self):
        # model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        self.sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=3072, logprobs=10)
        self.model = LLM(model=model_name, dtype="bfloat16", tensor_parallel_size=torch.cuda.device_count(), max_model_len=24576)

        self.yes_token = self.model.get_tokenizer().encode(" Yes", add_special_tokens=False)[0]
        self.no_token = self.model.get_tokenizer().encode(" No", add_special_tokens=False)[0]
        # set random seed
        torch.manual_seed(42)

    def rerank(self, docs, query, topk):
        scores = []

        batch_size = 50
        all_outs = []
        # for doc in docs:
        for i in range(0, len(docs), batch_size):
            list_docs = docs[i:i+batch_size]
            # doc_prompt = ("Given a query and a document, tell me whether the document answers the query.\n"
            #               "The answer should be either 'Relevance judgement: Yes.' or 'Relevance judgement: No.'\n\n" 
            #                 "Here is the query:\n"
            #                 "<start_query>\n"
            #                 f"{query}\n"
            #                 "<end_query>\n\n"
            #                 "Here is the document:\n"
            #                 "<start_document>\n"
            #                 )
            # doc_prompt = ("A document is relevant if it contains information that helps answer or address the query.\n"
            #               "A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.\n\n"
            #               "Is the document below relevant to answering the query below?\n"
            #               "The answer should be either 'Relevance judgement: Yes.' or 'Relevance judgement: No.'\n\n" 
            #                 "Here is the query:\n"
            #                 "<start_query>\n"
            #                 f"{query}\n"
            #                 "<end_query>\n\n"
            #                 "Here is the document:\n"
            #                 "<start_document>\n"
            #                 )
            doc_prompt = ("A document is relevant if it contains information that helps answer or address the query.\n"
              "A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.\n\n"
              "Is the document below relevant to answering the query below?\n"
              "The answer should be 'Relevance score: X.' where X is a number from 0-5.\n"
              "0 means completely irrelevant, 5 means highly relevant and completely addresses the query.\n\n"
              "Here is the query:\n"
              "<start_query>\n"
              f"{query}\n"
              "<end_query>\n\n"
              "Here is the document:\n"
              "<start_document>\n"
                )
            doc_prompts = [doc_prompt+'{}\n<end_document>\n\n'.format(doc["text"]) for doc in list_docs]
            
            output = self.model.generate(doc_prompts, self.sampling_params)
            doc_prompt_outputs = [o.outputs[0].text for o in output]
            # print(doc_prompt_output)

            
            all_outs.extend(doc_prompt_outputs)

            for i in range(len(doc_prompt_outputs)):

                # if len(j.outputs[0].logprobs) < 3:
                #     scores.append(float('-inf'))
                #     continue
                # logprobs = j.outputs[0].logprobs[-3]
                # if self.yes_token not in logprobs or self.no_token not in logprobs:
                #     if self.yes_token not in logprobs and self.no_token not in logprobs:
                #         scores.append(float('-inf'))
                #         continue
                #     elif self.yes_token not in logprobs and (float("-inf") in [item.logprob for item in logprobs.values()]):
                #         prob_yes = 0.0
                #         prob_no = math.exp(logprobs[self.no_token].logprob)
                #     elif self.no_token not in logprobs and (float("-inf") in [item.logprob for item in logprobs.values()]):
                #         prob_no = 0.0
                #         prob_yes = math.exp(logprobs[self.yes_token].logprob)
                #     else:
                #         scores.append(float('-inf'))
                # else:
                #     prob_yes = math.exp(logprobs[self.yes_token].logprob)
                #     prob_no = math.exp(logprobs[self.no_token].logprob)

                # score = prob_yes / (prob_yes + prob_no)
                # scores.append(score)
                pos_score = doc_prompt_outputs[i].rfind("Relevance score:")
                if pos_score != -1:
                    try:
                        score = float(doc_prompt_outputs[i][pos_score+16:pos_score+18])/5
                        scores.append(score)
                    except:
                        print("In exception!!")
                        scores.append(-2)
                else:
                    scores.append(-1)

        
        ranking = {doc["id"]: score for doc, score in zip(docs, scores)}
        ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:topk])
        write_dict = {"query": query, "outputs": all_outs, "scores": scores, "ranking": ranking}
        # write dict to json file
        with open(f"r1_8b_score_rank_results.jsonl", "a") as f:
            f.write(json.dumps(write_dict) + "\n")
        return ranking

class R1_api:
    def __init__(self):
        from together import Together
        from together import AsyncTogether
        self.client = Together(api_key="bbbc9744fae2e45985861c6eb7a30e9b5f81f7f5d11f4b79b86c810798067634")
        self.async_client = AsyncTogether(api_key="bbbc9744fae2e45985861c6eb7a30e9b5f81f7f5d11f4b79b86c810798067634")

    def rerank(self, docs, query, topk):
        scores = []
        model_scores = []

        all_outs = []
        doc_prompt = ("A document is relevant if it contains information that helps answer or address the query.\n"
              "A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.\n\n"
              "Is the document below relevant to answering the query below?\n"
              "The answer should end with either 'Relevance judgement: Yes.' or 'Relevance judgement: No.' followed by\n"
              "'Relevance score: X' where X is a number from 0-5.\n"
              "0 means completely irrelevant, 5 means highly relevant and completely addresses the query.\n\n"
              "Here is the query:\n"
              "<start_query>\n"
              f"{query}\n"
              "<end_query>\n\n"
              "Here is the document:\n"
              "<start_document>\n"
                )
        doc_prompts = [doc_prompt+'{}\n<end_document>\n\n'.format(doc["text"]) for doc in docs]

        async def async_chat_completion(prompts):
            tasks = [
                self.async_client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-R1",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=3072,
                )
                for prompt in prompts
            ]
            responses = await asyncio.gather(*tasks)

            return [response.choices[0].message.content for response in responses]

        import pdb; pdb.set_trace()
        doc_prompt_output = asyncio.run(async_chat_completion(doc_prompts[:2]))

        
        for i in range(len(docs)):
            # old_doc_prompt = ("A document is relevant if it contains information that helps answer or address the query.\n"
            #               "A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.\n\n"
            #               "Is the document below relevant to answering the query below?\n"
            #               "The answer should be either 'Relevance judgement: Yes.' or 'Relevance judgement: No.'\n\n" 
            #                 "Here is the query:\n"
            #                 "<start_query>\n"
            #                 f"{query}\n"
            #                 "<end_query>\n\n"
            #                 "Here is the document:\n"
            #                 "<start_document>\n"
            #                 )
            # doc_prompts = doc_prompt+'{}\n<end_document>\n\n'.format(docs[i]["text"])

            # response = self.client.chat.completions.create(
            #         model="deepseek-ai/DeepSeek-R1",
            #         messages=[{"role": "user", "content": doc_prompts}],
            #         temperature=0.6,
            #         top_p=0.95,
            #         max_tokens=3072,
            #     )
            # print(response.usage)
            
            # doc_prompt_output = response.choices[0].message.content
            
            all_outs.extend(doc_prompt_output)

            pos_yes = doc_prompt_output[i].rfind("Relevance judgement: Yes.")
            pos_no = doc_prompt_output[i].rfind("Relevance judgement: No.")
            if pos_yes > pos_no:
                scores.append(1.0)
            elif pos_yes < pos_no:
                scores.append(0.0)
            else:
                scores.append(float('-inf'))

            pos_score = doc_prompt_output[i].rfind("Relevance score:")
            if pos_score != -1:
                score = float(doc_prompt_output[i][pos_score+16:])/5
                model_scores.append(score)
            else:
                model_scores.append(-1)

        
        ranking = {doc["id"]: score for doc, score in zip(docs, scores)}
        ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:topk])
        write_dict = {"query": query, "outputs": all_outs, "scores": scores, "ranking": ranking, "model_scores": model_scores, "doc_ids": [doc["id"] for doc in docs]}
        # write dict to json file
        with open("r1api_rank_r_results.jsonl", "a") as f:
            f.write(json.dumps(write_dict) + "\n")
        return ranking

class llama:
    def __init__(self):
        # model_name = "meta-llama/Llama-3.1-70B-Instruct"
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=32, logprobs=10)
        self.model = LLM(model=model_name, dtype="bfloat16", tensor_parallel_size=torch.cuda.device_count(), max_model_len=16384)

        self.yes_token = self.model.get_tokenizer().encode(" Yes", add_special_tokens=False)[0]
        self.no_token = self.model.get_tokenizer().encode(" No", add_special_tokens=False)[0]
        # set random seed
        torch.manual_seed(42)

    def rerank(self, docs, query, topk):
        scores = []

        batch_size = 50
        all_outs = []
        # for doc in docs:
        for i in range(0, len(docs), batch_size):
            list_docs = docs[i:i+batch_size]
            # doc_prompt = ("A document is relevant if it contains information that helps answer or address the query.\n"
            #               "A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.\n\n"
            #               "Is the document below relevant to answering the query below?\n"
            #               "The answer should be either 'Relevance judgement: Yes.' or 'Relevance judgement: No.' Don't output anything else. \n\n" 
            #                 "Here is the query:\n"
            #                 "<start_query>\n"
            #                 f"{query}\n"
            #                 "<end_query>\n\n"
            #                 "Here is the document:\n"
            #                 "<start_document>\n"
            #                 )
            # doc_prompt = ("A document is relevant if it contains information that helps answer or address the query.\n"
            #   "A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.\n\n"
            #   "Is the document below relevant to answering the query below?\n"
            #   "The answer should be 'Relevance score: X.' where X is a number from 0-5.\n"
            #   "0 means completely irrelevant, 5 means highly relevant and completely addresses the query.\n"
            #   "Don't output anything else.\n\n"
            #   "Here is the query:\n"
            #   "<start_query>\n"
            #   f"{query}\n"
            #   "<end_query>\n\n"
            #   "Here is the document:\n"
            #   "<start_document>\n"
            #     )
            doc_prompt = ("A document is relevant if it contains information that helps answer or address the query.\n"
              "A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.\n\n"
              "Is the document below relevant to answering the query below?\n"
              "The answer should be just a number from 0-5.\n"
              "0 means completely irrelevant, 5 means highly relevant and completely addresses the query.\n"
              "Don't output anything else.\n\n"
              "Here is the query:\n"
              "<start_query>\n"
              f"{query}\n"
              "<end_query>\n\n"
              "Here is the document:\n"
              "<start_document>\n"
                )
            # doc_prompt = ("A document is relevant if it contains information that helps answer or address the query.\n"
            #   "A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.\n\n"
            #   "Is the document below relevant to answering the query below? Reason about what's in the query and document and produce a judgement.\n"
            #   "Think step by step to reason about what is in the query and document, and then assess whether the document is relevant.\n"
            #   "The answer should be in the following format:\n"
            #   "'Reasoning: [text]' where [text] is the reasoning.\n"
            #   "'Relevance score: X' where X is a number from 0-5.\n"
            #   "0 means completely irrelevant, 5 means highly relevant and completely addresses the query.\n\n"
            #   "Here is the query:\n"
            #   "<start_query>\n"
            #   f"{query}\n"
            #   "<end_query>\n\n"
            #   "Here is the document:\n"
            #   "<start_document>\n"
            #     )
            doc_prompts = [doc_prompt+'{}\n<end_document>\n\nRelevance Score: '.format(doc["text"]) for doc in list_docs]
            
            output = self.model.generate(doc_prompts, self.sampling_params)
            doc_prompt_outputs = [o.outputs[0].text for o in output]
            # print(doc_prompt_output)

            
            all_outs.extend(doc_prompt_outputs)

            # for j in output:
            #     if len(j.outputs[0].logprobs) < 3:
            #         scores.append(float('-inf'))
            #         continue
            #     logprobs = j.outputs[0].logprobs[-3]
            #     if self.yes_token not in logprobs or self.no_token not in logprobs:
            #         if self.yes_token not in logprobs and self.no_token not in logprobs:
            #             scores.append(float('-inf'))
            #             continue
            #         elif self.yes_token not in logprobs and (float("-inf") in [item.logprob for item in logprobs.values()]):
            #             prob_yes = 0.0
            #             prob_no = math.exp(logprobs[self.no_token].logprob)
            #         elif self.no_token not in logprobs and (float("-inf") in [item.logprob for item in logprobs.values()]):
            #             prob_no = 0.0
            #             prob_yes = math.exp(logprobs[self.yes_token].logprob)
            #         else:
            #             scores.append(float('-inf'))
            #     else:
            #         prob_yes = math.exp(logprobs[self.yes_token].logprob)
            #         prob_no = math.exp(logprobs[self.no_token].logprob)

            #     score = prob_yes / (prob_yes + prob_no)
            #     scores.append(score)
            #og
            # for i in range(len(doc_prompt_outputs)):
            #     pos_score = doc_prompt_outputs[i].rfind("Relevance score:")
            #     if pos_score != -1:
            #         try:
            #             score = float(doc_prompt_outputs[i][pos_score+16:pos_score+18])/5
            #             scores.append(score)
            #         except:
            #             print("In exception!! {}".format(doc_prompt_outputs[i][pos_score:pos_score+25]))
            #             scores.append(-2)
            #     else:
            #         scores.append(-1)
            #new
            for i in range(len(doc_prompt_outputs)):
                # pos_score = doc_prompt_outputs[i].rfind("Relevance score:")
                # if pos_score != -1:
                try:
                    score = float(doc_prompt_outputs[i].strip()[0])/5
                    scores.append(score)
                except:
                    print("In exception!! {}".format(doc_prompt_outputs[i]))
                    scores.append(-1)

        
        ranking = {doc["id"]: score for doc, score in zip(docs, scores)}
        ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:topk])
        write_dict = {"query": query, "outputs": all_outs, "scores": scores, "ranking": ranking}
        # write dict to json file
        with open(f"rank1_theoremqa_theorems_llama_score_rank_results1.jsonl", "a") as f:
            f.write(json.dumps(write_dict) + "\n")
        return ranking

class STReranker:
    def __init__(self, model_name, batch_size=8):
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    @torch.no_grad()
    def rerank(self, docs, query, topk):
        inputs = [(query, doc["text"]) for doc in docs]
        scores = self.model.predict(inputs, batch_size=self.batch_size)
        ranking = {doc["id"]: score.item() for doc, score in zip(docs, scores)}
        ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:topk])
        return ranking

class BGEReranker:
    def __init__(self, model_name, batch_size=8):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def rerank(self, docs, query, topk):
        pairs = [[query, doc["text"]] for doc in docs]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float().tolist()
        ranking = {doc["id"]: score for doc, score in zip(docs, scores)}
        ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:topk])
        return ranking

class MxbaiReranker:
    def __init__(self):
        self.model = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1")

    @torch.no_grad()
    def rerank(self, docs, query, topk):
        doc_texts = [doc["text"] for doc in docs]
        results = self.model.rank(query, doc_texts, top_k=100)
        scores = [0]*100
        for i, r in enumerate(results):
            scores[r["corpus_id"]] = 1.*r["score"]
        
        ranking = {doc["id"]: score for doc, score in zip(docs, scores)}
        ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:topk])
        write_dict = {"query": query, "scores": scores, "ranking": ranking}
        # write dict to json file
        with open(f"mxbai_score_rank_r_results.jsonl", "a") as f:
            f.write(json.dumps(write_dict) + "\n")
        return ranking
    
class GritLM:
    def __init__(self):
        from gritlm import GritLM
        self.model = GritLM("GritLM/GritLM-7B", torch_dtype="auto", mode="embedding")
        self.instruction = "Given a query, retrieve documents that contain information that helps answer or address the query."

    def gritlm_instruction(self, instruction):
        return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

    @torch.no_grad()
    def rerank(self, docs, query, topk):
        doc_texts = [doc["text"] for doc in docs]
        d_reps = self.model.encode(doc_texts, instruction=self.gritlm_instruction(""))
        q_rep = self.model.encode(query, instruction=self.gritlm_instruction(self.instruction))

        scores = np.einsum("ij,j->i", d_reps, q_rep).tolist()
        
        ranking = {doc["id"]: score for doc, score in zip(docs, scores)}
        ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:topk])
        write_dict = {"query": query, "scores": scores, "ranking": ranking}
        # write dict to json file
        with open(f"gritlm_emb_score_rank_r_results.jsonl", "a") as f:
            f.write(json.dumps(write_dict) + "\n")
        return ranking


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology','earth_science','economics','pony','psychology','robotics','theoremqa_questions', "theoremqa_theorems",
                                 'stackoverflow','sustainable_living','aops','leetcode'])
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--llm', type=str, default=None)
    parser.add_argument('--score_file', type=str, default=None)
    parser.add_argument('--rerank_score_file', type=str, default=None)
    parser.add_argument('--input_k', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--reasoning', type=str, default=None)
    args = parser.parse_args()

    # if os.path.exists(args.rerank_score_file):
    #     print(f"Rerank score file {args.rerank_score_file} already exists.")
    #     exit()

    if args.reasoning is not None:
        raw_examples = load_dataset('xlangai/bright', f"{args.reasoning}_reason", cache_dir=args.cache_dir)[args.task]
    else:
        raw_examples = load_dataset('xlangai/bright', 'examples',cache_dir=args.cache_dir)[args.task]

    # raw_examples = load_dataset('xlangai/bright', 'examples', cache_dir=args.cache_dir)[args.task]
    examples = {}
    for e in raw_examples:
        examples[e['id']] = e
    if args.long_context:
        doc_pairs = load_dataset('xlangai/bright', 'long_documents', cache_dir=args.cache_dir)[args.task]
    else:
        doc_pairs = load_dataset('xlangai/bright', 'documents', cache_dir=args.cache_dir)[args.task]
    documents = {}
    for d in doc_pairs:
        documents[d['id']] = d['content']
    with open(args.score_file) as f:
        all_scores = json.load(f)
    new_scores = copy.deepcopy(all_scores)

    if 'claude' in args.llm:
        model = ClaudeModel(version=args.llm)
    elif "gpt" in args.llm:
        model = OpenAIModel(model_name=args.llm)
    elif "bge" in args.llm:
        model = BGEReranker(model_name=args.llm)
    elif "mxbai" in args.llm:
        model = MxbaiReranker()
    elif "gritlm" in args.llm:
        model = GritLM()
    elif "judge" in args.llm:
        model = JudgeRank()
    elif "r1_api" in args.llm:
        model = R1_api()
    elif "r1" in args.llm:
        model = R1()
    elif "llama" in args.llm:
        model = llama()
    else:
        model = STReranker(model_name=args.llm)

    for qid,scores in tqdm(all_scores.items()):
        print(qid)
        # skip the first 17 examples
        # if int(qid) < 27:
        #     continue
        docs = []
        sorted_scores = sorted(scores.items(),key=lambda x:x[1],reverse=True)[:args.input_k]
        for did, _ in sorted_scores:
            docs.append([did, documents[did]])

        if 'claude' in args.llm or "gpt" in args.llm:
            new_rank = model.rerank(docs=docs, query=examples[qid]['query'], topk=args.k)
            cur_score = {}
            if new_rank is None:
                # use the original ranks if fail
                for rank_id, (did, _) in enumerate(sorted_scores):
                    cur_score[did] = args.k - rank_id
            else:
                for rank_id, r in enumerate(new_rank):
                    cur_score[r] = args.k - rank_id
            new_scores[qid] = cur_score
        else:
            ctxs = [{'id': did, 'text': documents[did]} for did, _ in sorted_scores]
            # for iterx in range(5):
            cur_score = model.rerank(query=examples[qid]['query'], docs=ctxs, topk=args.k)
            new_scores[qid] = cur_score

    import sys
    sys.exit()

    os.makedirs(os.path.dirname(args.rerank_score_file), exist_ok=True)
    with open(args.rerank_score_file, 'w') as f:
        json.dump(new_scores, f, indent=2)

    if args.long_context:
        key = 'gold_ids_long'
    else:
        key = 'gold_ids'
    ground_truth = {}
    for e in raw_examples:
        ground_truth[e['id']] = {}
        for gid in e[key]:
            ground_truth[e['id']][gid] = 1
        for i in e["excluded_ids"]:
            if i in documents:
                ground_truth[e['id']][i] = 0

    results = calculate_retrieval_metrics(results=new_scores, qrels=ground_truth)
    with open(args.rerank_score_file.replace(".json", "_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
