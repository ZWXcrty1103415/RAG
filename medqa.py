
"""
Built with retrievers and memory sessions from langchain.
"""

import os

os.environ['OPENAI_API_KEY'] = 'sk-e0774de50faa4a4fb0437e74307e39bb'
os.environ['DASHSCOPE_API_KEY'] = 'sk-e0774de50faa4a4fb0437e74307e39bb'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from custom_bm25_retriever import BM25Retriever2
from use_dashscope import LLMClient
from use_bge_rerank import bge_reranker
import jieba
import pickle
import sys
import logging
import json


embedding_retrieve = 6
bm25_retrieve = 6

def tokenize(text):
    return list(jieba.cut_for_search(text))

def load_bm25retriever(filedir, k=1):
    with open(filedir, "rb") as f:
        token_n_text = pickle.load(f)
    tokenized_texts = token_n_text["token"]
    texts = token_n_text["text"]
    metadatas = token_n_text["metadata"]
    print(sys.getsizeof(tokenized_texts), sys.getsizeof(texts))
    bm25retriever = BM25Retriever2.from_tokenized_text(texts, tokenized_texts, metadatas=metadatas, preprocess_func=tokenize, k=k)
    return bm25retriever

def multi_retrieve(questions):
    retrieves_docs, retrieves_metadatas = [], []
    for q in questions:
        vector_retrieve_list= vector_store.similarity_search(q, k=2)
        keyword_retrieve_list = bm25retriever.invoke(q)
        for retrieve in vector_retrieve_list:
            retrieves_docs.append(retrieve.page_content)
            retrieves_metadatas.append(retrieve.metadata)
        logging.info("vector retrieve finished")
        for retrieve in keyword_retrieve_list:
            retrieves_docs.append(retrieve.page_content)
            retrieves_metadatas.append(retrieve.metadata)
        logging.info("bm25 retrieve finished")
    return retrieves_docs, retrieves_metadatas

def filter_context(context, rerank_score, metadata, threshold, k = 2):
    filtered_context = ""
    doc_count = 0
    exits_ids = []
    if rerank_score:
        rerank, score = rerank_score
        for res in rerank:
            doc_id = metadata[res]["doc_id"]
            if score[res] < threshold:
                break
            elif doc_id not in exits_ids:
                doc_count += 1
                filtered_context += context[res] + "\n"
                exits_ids.append(doc_id)
            if doc_count >= k:
                break
    else:
        for doc, md in zip(context, metadata):
            doc_id = md["doc_id"]
            if doc_id not in exits_ids:
                doc_count += 1
                filtered_context += doc + "\n"
                exits_ids.append(doc_id)
            if doc_count >= k:
                break
    return filtered_context


model_name = "BAAI/bge-small-zh"
model_kwargs = {
    "device": "cuda"
}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceEmbeddings(
    cache_folder="./hf_models",
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    multi_process=False,
)

vector_store = FAISS.load_local(
    "faiss_index", hf, allow_dangerous_deserialization=True
)

bm25retriever = load_bm25retriever("dataset/huatuo_encyclopedia_qa/train_datasets_tokenized.pickle", k=2)

reranker = bge_reranker()

prompt_cm = [
    ('system', "任务：给出下面的历史对话和一个后续问题，用原来的对话将后续问题改写为独立的问题。\n---\n规则：\n - 判断相关性：你需要分析后续问题和历史对话是否有关；如果有关：生成一个问题，该问题必须包含所有必要细节，确保任何人（无需历史对话背景）都能直接回答它。；如果不相关：直接输出新信息本身（不作修改或添加）。\n - 判断准确性：如果无法生成明确的问题，需要更多信息则回复：“请提供更多信息。”并给出可能的选项。\n---\n输出约束：始终输出一个以“？”为结尾的字符串\n---\n对话历史：{history}"),
    ('user', "{question}")
]
prompt_qg = [
    ('system', "角色：你是一个检索内容生成机器人。你所生成的内容会被用于知识库检索并由问答机器人整理成最终的答案\n---\n任务：根据当前问题生成2-3个检索词/句\n---\n规则：- 包含两类内容：\n**宽泛型**：覆盖核心主题的大范围概念\n**聚焦型**：锁定具体细节或特殊角度\n - 确保内容来自 **不同维度**（如：对象/行为/属性/场景）\n---\n输出约束：只包含2-3个词或句子。词或句子中间用“，”隔开。\n---\n输出示例：{example}"),
    ('user', "{question}")
]
example = "感冒药，感冒治疗药物有哪些，感冒如何治疗"
prompt_qa = [
    ('system', "角色：你是一个医药问答机器人，核心使命是基于循证医学提供准确、安全、可靠的健康信息。\n---\n任务：根据以下用户问题以及知识库中的相关内容，生成简短的回答。\n---\n规则：\n - 如果相关内容中有足够的信息来回答，则据此生成回答；如果没有，直接回复：“我没有关于此问题的完整信息”\n - 优先使用知识中的信息回复\n---\n约束：\n - 所有信息和建议都需基于科学；如用户询问中存在不科学的内容，尝试纠正观念\n - 用户的生命安全有最高优先级；如表述中体现出用户遭遇紧急情况，建议其立即就医。\n - 所有回复必须合法合规；如用户询问中有潜在的犯罪倾向（如尸体处理，毒品，毒药等），不要进行回答。\n---\n知识：{context}"),
    ('user', "{question}")
]
prompt_router = [
    ('system', '角色：你是一个医药问答系统的前台分流机器人，以历史对话为依据，负责回答**基础问题**；并向问答机器人问询**医药问题**。你的核心使命是基于循证医学提供准确、安全、可靠的健康信息。\n---\n规则：\n - 判断问题专业性：对**医药问题**（任何关于疾病、症状、病因、诊断、治疗（包括非药物疗法）、药物、保健品、养生功效的咨询），只要问题涉及“如何做来改善或影响健康”，“直接”向医药问答机器人提问；对**基础问题**（不需要医药知识的问题，包括和医学无关的问题、违法的问题、伪科学的问题等），直接进行回答。\n - 结合历史对话信息：你需要分析后续问题和历史对话是否有关。\n对于**医药问题**，如果和历史对话有关：生成一个问题，该问题必须包含所有必要细节，确保任何人（无需历史对话背景）都能直接回答它。；如果和历史对话不相关：**直接输出用户原本的问题**（不作修改或添加）。\n对于**基础问题**结合历史对话进行回答\n---\n格式约束：输出一个json格式的字符串，包含两个key：“action”和“resp”：\n**action**：“rag”（如果认为是医药问题）；“answer”（如果认为是基础问题）\n**resp**：\n如果认为是医药问题，按规则生成问题用于向问答机器人提问，问题是一个以“？”结尾的字符串，不需要额外的问询。\n如果认为是基础问题；生成回答\n---\n约束：\n - 所有信息和建议都需基于科学；如用户询问中存在不科学的内容，尝试纠正观念\n - 用户的生命安全有最高优先级；如表述中体现出用户遭遇紧急情况，建议其立即就医；如表述中出现有危害健康的内容（如使用强效药，用药不适量等），进行劝阻。\n - 所有回复必须合法合规；如用户询问中有潜在的犯罪倾向（如尸体处理，毒品，毒药等），不要进行回答。\n---\n示例：\n{{"action": "rag", "resp":"什么药对感冒有用？"}}\n{{"action":"answer", "resp":"我不能提供任何有关假装疾病症状以获取处方药物的建议。这种行为不仅不道德，而且可能违法。"}}\n---\n历史对话：{history}'),
    ('user', '{question}')
]

cm = LLMClient("qwen3-32b", prompt_cm)

qg = LLMClient("qwen3-32b", prompt_qg)

qa = LLMClient("qwen3-32b", prompt_qa)

router = LLMClient("qwen3-32b", prompt_router)

memory_dic = {}
current_memory = ConversationBufferWindowMemory(k=3)

def retrieve_memory(thread: int, user: str):
    user_thread = user + str(thread)
    return memory_dic[user_thread]


def find_memory(thread: int, user: str):
    global current_memory
    user_thread = user + str(thread)
    if user_thread not in memory_dic.keys():
        current_memory = ConversationBufferWindowMemory(k=3)
        memory_dic[user_thread] = current_memory
    else:
        current_memory = memory_dic[user_thread]
    history = current_memory.buffer_as_str
    return history

def generate_retrieve(question: str):
    retrieve_q = qg.invoke([{"example":example}, {"question": question}],**{"extra_body":{"enable_thinking":False}})
    logging.info("pg finished")
    print(f"retrieve_q: \n {retrieve_q}")
    context, metadata = multi_retrieve(retrieve_q.split("，"))
    rerank_score = reranker.rerank(question, context, top_n=8)
    # rerank_score = None
    logging.info("rerank finished")
    context = filter_context(context, rerank_score, metadata, -8, k=3)
    logging.info("filter finished")
    print(f"retrieved: \n {context}")
    return context

def run(question: str, thread = 1, user = "temp_user"):
    logging.info("process start")
    history = find_memory(thread, user)
    logging.info("memory retrieved")
    n_question = cm.invoke([{"history": history}, {"question": question}],**{"extra_body":{"enable_thinking":False}})
    logging.info("cm finished")
    print(f"question: \n {n_question}")
    context = generate_retrieve(n_question)
    logging.info("retrieve finished")
    msg = qa.invoke([{"context": context}, {"question": question}],**{"extra_body":{"enable_thinking":False}})
    logging.info("anwser finished")
    current_memory.save_context({"input": question}, {"output": msg})
    return msg

def raw(question: str, thread = 1, user = "temp_user"):
    msg = qa.invoke([{"context": ""}, {"question": question}], **{"extra_body": {"enable_thinking": False}})
    return msg

def stream(question: str, thread = 1, user = "temp_user"):
    logging.info("process start")
    history = find_memory(thread, user)
    logging.info("memory retrieved")
    n_question = cm.invoke([{"history": history}, {"question": question}], **{"extra_body": {"enable_thinking": False}})
    logging.info("cm finished")
    print(f"question: \n {n_question}")
    context = generate_retrieve(n_question)
    logging.info("retrieve finished")
    return qa.stream([{"context": context}, {"question": question}])
    # current_memory.save_context({"input": question}, {"output": message})

def routing(question: str, thread = 1, user = "temp_user", retry = 3, streaming = False):
    history = find_memory(thread, user)
    logging.info("memory retrieved")
    routing_resp = True
    trys = 0
    while routing_resp and trys < retry:
        trys += 1
        routing_resp = router.invoke([{"history": history}, {"question": question}],
                                   **{"response_format": {"type": "json_object"}, "extra_body": {"enable_thinking": False}})
        try:
            routing_resp = json.loads(routing_resp)
            logging.info("routing finished")
            print(f"question: \n {routing_resp}")
            if 'action' in routing_resp.keys() and 'resp' in routing_resp.keys():
                if routing_resp['action'] == 'rag':
                    context = generate_retrieve(routing_resp['resp'])
                    logging.info("retrieve finished")
                    msg = qa.invoke([{"context": context}, {"question": routing_resp['resp']}],
                                    **{"extra_body": {"enable_thinking": False}})
                    save_message(question, msg)
                    return msg
                elif routing_resp['action'] == 'answer':
                    save_message(question, routing_resp['resp'])
                    return routing_resp['resp']
                else:
                    raise json.decoder.JSONDecodeError
        except json.decoder.JSONDecodeError:
            routing_resp = True
    return "LLM error occurred"

def save_message(question, message, thread = 1, user = "temp_user"):
    logging.info("save memory process start")
    history = retrieve_memory(thread, user)
    history.save_context({"input": question}, {"output": message})

if __name__ == "__main__":
    while True:
        question = input("You: ")
        if question.lower() in ['quit', 'exit', 'bye'] or not question:
            print("再见")
            break
        print(f'PsyQA: {run(question, 1, "john")}')
