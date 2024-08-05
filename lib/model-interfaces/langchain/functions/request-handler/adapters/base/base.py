import os
from enum import Enum
from aws_lambda_powertools import Logger
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversational_retrieval.prompts import (
    QA_PROMPT,
    CONDENSE_QUESTION_PROMPT,
)
from typing import Dict, List, Any

from genai_core.langchain import WorkspaceRetriever, DynamoDBChatMessageHistory
from genai_core.types import ChatbotMode

logger = Logger()


import json
import boto3

PROCESSING_BUCKET_NAME = os.environ.get("PROCESSING_BUCKET_NAME", "")

def save_lipid_data_to_s3(bucket_name, key, json_obj):
    s3 = boto3.resource("s3")
    s3_client = boto3.client("s3")
    s3_client.put_object(
        Body=json.dumps(json_obj),
        Bucket=bucket_name,
        Key=key,
        ContentType="application/json",
    )


from functools import reduce
from langchain_core.outputs.llm_result import LLMResult


def process_lipid_plot(json_obj, label_name):

    def all_series_names(json_obj, label):
        series = []
        if len(json_obj) > 0:
            series = reduce(lambda a, x: a + list(x.keys()), json_obj, [])
            series = list(set(series))
            series = filter(lambda x: x != label, series)
        return series

    def main_data_series(data_points, color):
        series = dict()
        series["fill"] = False
        series["tension"] = 0.1
        series["borderColor"] = color
        series["data"] = data_points
        return series

    def reference_data_series(reference_value, size, color):
        series = dict()
        series["fill"] = False
        series["tension"] = 0.1
        series["borderColor"] = color
        series["data"] = [reference_value] * size
        return series

    series_names = list(all_series_names(json_obj, label_name))

    charts = []
    for s_name in series_names:
        sub_list = list(filter(lambda x: (s_name in x) and (label_name in x), json_obj))
        list_len = len(sub_list)
        if list_len < 2:
            continue
        labels = list(map(lambda x: x[label_name], sub_list))
        s_list = list(map(lambda x: x[s_name], sub_list))

        main_series = main_data_series(s_list, "rgb(75, 192, 192)")
        # reference_series_1 = reference_data_series(125, list_len, "rgb(192, 192, 192)")
        # reference_series_2 = reference_data_series(200, list_len, "rgb(192, 192, 192)")
        datasets = [main_series]

        chart_data = dict()
        chart_data["labels"] = labels
        chart_data["datasets"] = datasets

        chart = dict()
        chart["chartTitle"] = f"Historical {s_name} readings"
        chart["chartData"] = chart_data

        charts = charts + [chart]

    return charts


def plot_chart_for_lipid_panel_trend(inputs):
    return process_lipid_plot(inputs, "Report Date")


tools = [
    {
        "type": "function",
        "function": {
            "name": "plot_chart_for_lipid_panel_trend",
            "description": """
            this function is called to plot lipid panel data for different dates
            to show the trend. as much data points should be included as possible.
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "inputs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "date": {
                                    "type": "string",
                                    "description": "date when this data point is reported",
                                },
                                "total_cholesterol": {
                                    "type": "string",
                                    "description": "cholesterol or total cholesterol level",
                                },
                                "triglycerides": {
                                    "type": "string",
                                    "description": "Triglycerides level",
                                },
                                "hdl_cholesterol": {
                                    "type": "string",
                                    "description": "hdl cholesterol level",
                                },
                                "ldl_cholesterol": {
                                    "type": "string",
                                    "description": "ldl cholesterol or calculated ldl cholesterol level",
                                },
                                # "Non-HDL Cholesterol": {
                                #     "type": "string",
                                #     "description": "non-hdl cholesterol level",
                                # },
                                # "Cholesterol/HDL": {
                                #     "type": "string",
                                #     "description": "cholesterol to hdl ratio",
                                # },
                                # "LDL/HDL": {
                                #     "type": "string",
                                #     "description": "ldl/hdl ratio",
                                # },
                            },
                            "required": ["Report Date"],
                        },
                    }
                },
            },
        },
    }
]


additional_requirement = """
Do not plot charts unless you are explicitly instructed to do so.
If you do need to invoke plot tool function call, invoke at most once.
"""

prompt_template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
""" + f"{additional_requirement}" + """

{context}

Question: {question}
Helpful Answer:
"""

MODIFIED_QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


condense_question_template = """
Given the following conversation and a follow up input, 
rephrase the follow up question to be a standalone question in its original language. 
If the user requests plot, explicitly say plot a chart in the standalone question

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""

MODIFIED_CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)


class Mode(Enum):
    CHAIN = "chain"


class LLMStartHandler(BaseCallbackHandler):
    prompts = []
    llm_responses = []

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        logger.info(prompts)
        self.prompts.append(prompts)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.llm_responses.append(response)


class ModelAdapter:
    def __init__(
        self, session_id, user_id, mode=ChatbotMode.CHAIN.value, model_kwargs={}
    ):
        self.session_id = session_id
        self.user_id = user_id
        self._mode = mode
        self.model_kwargs = model_kwargs

        self.callback_handler = LLMStartHandler()
        self.__bind_callbacks()

        self.chat_history = self.get_chat_history()
        self.llm = self.get_llm(model_kwargs)

    def __bind_callbacks(self):
        callback_methods = [method for method in dir(self) if method.startswith("on_")]
        valid_callback_names = [
            attr for attr in dir(self.callback_handler) if attr.startswith("on_")
        ]

        for method in callback_methods:
            if method in valid_callback_names:
                setattr(self.callback_handler, method, getattr(self, method))

    def get_llm(self, model_kwargs={}):
        raise ValueError("llm must be implemented")

    def get_embeddings_model(self, embeddings):
        raise ValueError("embeddings must be implemented")

    def get_chat_history(self):
        return DynamoDBChatMessageHistory(
            table_name=os.environ["SESSIONS_TABLE_NAME"],
            session_id=self.session_id,
            user_id=self.user_id,
        )

    def get_memory(self, output_key=None, return_messages=False):
        return ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=self.chat_history,
            return_messages=return_messages,
            output_key=output_key,
        )

    def get_prompt(self):
        template = """The following is a friendly conversation between a human and an AI. If the AI does not know the answer to a question, it truthfully says it does not know.

        Current conversation:
        {chat_history}

        Question: {input}"""

        return PromptTemplate.from_template(template)

    def get_condense_question_prompt(self):
        return CONDENSE_QUESTION_PROMPT

    def get_qa_prompt(self):
        return QA_PROMPT

    def run_with_chain(self, user_prompt, workspace_id=None):
        if not self.llm:
            raise ValueError("llm must be set")

        self.callback_handler.prompts = []
        self.callback_handler.llm_responses = []

        if workspace_id:
            llm_w_tools = self.llm.bind_tools(tools=tools, tool_choice="auto")
            conversation = ConversationalRetrievalChain.from_llm(
                llm_w_tools,
                WorkspaceRetriever(workspace_id=workspace_id),
                condense_question_llm=self.get_llm({"streaming": False}),
                condense_question_prompt=MODIFIED_CONDENSE_QUESTION_PROMPT,
                combine_docs_chain_kwargs={"prompt": MODIFIED_QA_PROMPT},
                return_source_documents=True,
                memory=self.get_memory(output_key="answer", return_messages=True),
                verbose=True,
                callbacks=[self.callback_handler],
            )
            result = conversation({"question": user_prompt})
            logger.info(result["source_documents"])
            documents = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in result["source_documents"]
            ]

            tool_calls = []
            handlers = self.callback_handler
            if handlers.llm_responses and len(handlers.llm_responses) > 0:
                last_response = handlers.llm_responses[-1]
                if last_response.generations and len(last_response.generations) > 0:
                    response_message = last_response.generations[0][0].message
                    if response_message.tool_calls and len(response_message.tool_calls) > 0:
                        tool_calls = response_message.tool_calls

            charts = []
            for tool_call in tool_calls:
                if tool_call["name"] == "plot_chart_for_lipid_panel_trend":
                    save_lipid_data_to_s3(PROCESSING_BUCKET_NAME, "test_file", tool_call["args"]["inputs"])
                    if (len(charts) <= 0):
                        charts = tool_call["args"]["inputs"]

            answer = result["answer"]
            if (len(charts) > 0):
                if ((not answer) or (len(answer) < 1)):
                    pass
                    # llm = self.get_llm({"streaming": True})
                    # current_prompt = handlers.prompts[-1]
                    # new_prompt = list(map(lambda x: x.replace(additional_requirement, ""), current_prompt))
                    # answer = llm.invoke(new_prompt)
                    # self.callback_handler.prompts.pop()
                answer = answer + "\n**Following chart(s) are generated for your reference.**\n"

            medias = [
                #    "https://youtu.be/AVaFEanJAUs?si=cWXjA7MeVzOZnWbX",
                #    "https://www.youtube.com/watch?v=AVaFEanJAUs&t=400s",
            ]

            metadata = {
                "modelId": self.model_id,
                "modelKwargs": self.model_kwargs,
                "mode": self._mode,
                "sessionId": self.session_id,
                "userId": self.user_id,
                "workspaceId": workspace_id,
                "documents": documents,
                "prompts": self.callback_handler.prompts,
                "charts": charts,
                "medias": medias,
            }

            self.chat_history.add_metadata(metadata)

            return {
                "sessionId": self.session_id,
                "type": "text",
                "content": answer,
                "metadata": metadata,
            }

        conversation = ConversationChain(
            llm=self.llm,
            prompt=self.get_prompt(),
            memory=self.get_memory(),
            verbose=True,
        )
        answer = conversation.predict(
            input=user_prompt, callbacks=[self.callback_handler]
        )

        charts = []

        medias = [
        #    "https://youtu.be/AVaFEanJAUs?si=cWXjA7MeVzOZnWbX",
        #    "https://www.youtube.com/watch?v=AVaFEanJAUs&t=400s",
        ]

        metadata = {
            "modelId": self.model_id,
            "modelKwargs": self.model_kwargs,
            "mode": self._mode,
            "sessionId": self.session_id,
            "userId": self.user_id,
            "documents": [],
            "prompts": self.callback_handler.prompts,
            "charts": charts,
            "medias": medias,
        }

        self.chat_history.add_metadata(metadata)

        return {
            "sessionId": self.session_id,
            "type": "text",
            "content": answer,
            "metadata": metadata,
        }

    def run(self, prompt, workspace_id=None, *args, **kwargs):
        logger.debug(f"run with {kwargs}")
        logger.debug(f"workspace_id {workspace_id}")
        logger.debug(f"mode: {self._mode}")

        if self._mode == ChatbotMode.CHAIN.value:
            return self.run_with_chain(prompt, workspace_id)

        raise ValueError(f"unknown mode {self._mode}")
