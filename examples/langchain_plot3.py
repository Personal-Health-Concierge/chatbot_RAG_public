import os
from enum import Enum

# from aws_lambda_powertools import Logger
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversational_retrieval.prompts import (
    QA_PROMPT,
    CONDENSE_QUESTION_PROMPT,
)
from typing import Dict, List, Any


from functools import reduce
from langchain_core.outputs.llm_result import LLMResult
from langchain_openai import ChatOpenAI

# from langchain_community.chat_models import ChatOpenAI

# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever, Document

# from typing import Dict, List, Any

# from typing import Optional, Type
# from langchain.callbacks.manager import (
#     AsyncCallbackManagerForToolRun,
#     CallbackManagerForToolRun,
# )

# from langchain.pydantic_v1 import BaseModel, Field

# from langchain.tools import BaseTool, StructuredTool, tool

os.environ["OPENAI_API_KEY"] = (
)


import re
import json
from functools import reduce


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


class WorkspaceRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # result = genai_core.semantic_search.semantic_search(
        #     self.workspace_id, query, limit=20, full_response=True
        # )
        result = {
            "items": [
                {
                    "content": """
                    ### Summary of Health-Related Information (Lipid Panel) **Date of Report:** December 23, 2020 #### Results: - **Total Cholesterol:** 279 mg/dL (High; Reference Range: <200 mg/dL) - **Triglycerides:** 53 mg/dL (Normal; Reference Range: <150 mg/dL) - **HDL Cholesterol:** 91 mg/dL (Normal; Reference Range: >39 mg/dL) - **Calculated LDL Cholesterol:** 173 mg/dL (High; Reference Range: <100 mg/dL) - **Risk Ratio (LDL/HDL):** 1.90 (Normal; Reference Range: <3.22) #### Notes: - The calculated LDL is based on the Martin-Hopkins method, which adjusts for triglyceride levels and VLDL cholesterol ratio. - Elevated total cholesterol and LDL levels may indicate a higher risk for cardiovascular issues. For further details, consult the provided link or a healthcare professional.
                    """
                },
                {
                    "content": """
                    health_report: date: "04/24/2024" lipid_panel: cholesterol: value: 211 reference_range: "<200 mg/dL" status: "High" triglycerides: value: 33 reference_range: "<150 mg/dL" HDL_cholesterol: value: 90 reference_range: ">39 mg/dL" cholesterol_hdl_ratio: value: 2.34 reference_range: "0.0-4.4 (Ratio)" non_hdl_cholesterol: value: 121 reference_range: "<130 mg/dL" LDL_cholesterol: value: 114 reference_range: "<130 mg/dL" interpretation: - "Optimal: Less than 100 mg/dL" - "Near Optimal: 100 to 129 mg/dL" - "Above Optimal: 130 to 159 mg/dL" - "Borderline High: 160 to 189 mg/dL" - "High: 190 mg/dL and above" - "Very High: 190 mg/dL and above" LDL_history: previous_tests: - date: "04/19/2023" LDL_result: 112 change: "0%" - date: "04/23/2024" LDL_result: 114
                    """
                },
                {
                    "content": """
                    health_report: date: "04/20/2023" lipid_panel: cholesterol: value: 207 reference_range: "<200 mg/dL" status: "High" triglycerides: value: 56 reference_range: "<150 mg/dL" status: "Normal" hdl_cholesterol: value: 84 reference_range: ">39 mg/dL" status: "Normal" cholesterol_hdl_ratio: value: 2.46 reference_range: "0.00-0.44" status: "High" non_hdl_cholesterol: value: 123 reference_range: "<130 mg/dL" status: "Normal" ldl_cholesterol_calculation: value: 112 reference_range: "<130 mg/dL" status: "Normal" ldl_hdl_ratio: value: 1.3 reference_range: "<3.3" status: "Normal" patient_history: test_date: "04/13/2022" ldl_results: value: 113 units: "mg/dL" change_percentage: "-" 
                    """
                },
                {
                    "content": """
                    ### Summary of Health-Related Information from Lipid Panel **Date of Report:** July 2, 2021 #### Results: - **Total Cholesterol:** 244 mg/dL (High; Reference Range: <200 mg/dL) - **Triglycerides:** 94 mg/dL (Normal; Reference Range: <150 mg/dL) - **HDL Cholesterol:** 74 mg/dL (Normal; Reference Range: >39 mg/dL) - **Calculated LDL Cholesterol:** 150 mg/dL (High; Reference Range: <100 mg/dL) - **Risk Ratio (LDL/HDL):** 2.03 (Normal; Reference Range: <3.22) #### Notes: - The calculated LDL is based on the Martin-Hopkins method, which adjusts for triglyceride levels and VLDL cholesterol ratio. - Elevated total cholesterol and LDL levels may indicate a higher risk for cardiovascular issues. For further details, consult a healthcare provider.
                    """
                },
                {
                    "content": """
                    health_report: date: "04/14/2022" lipid_panel: cholesterol: value: 201 reference_range: "<200 mg/dL" status: "High" triglycerides: value: 42 reference_range: "<150 mg/dL" HDL_cholesterol: value: 80 reference_range: ">39 mg/dL" cholesterol_hdl_ratio: value: 2.51 reference_range: "0.00-0.44 (Ratio)" non_hdl_cholesterol: value: 121 reference_range: "<130 mg/dL" LDL_cholesterol: calculated_value: 113 reference_range: "<130 mg/dL" status: "Optimal" LDL_hdl_ratio: value: 1.4 reference_range: "<3.3 (Ratio)" test_date: "04/13/2022"
                    """
                },
                {
                    "content": """
                    ### Summary of Health Information from Lipid Panel (Date: December 28, 2016) - **Total Cholesterol**: 234 mg/dL (High; Reference Range: 126-200 mg/dL) - **HDL Cholesterol**: 80 mg/dL (Normal; Reference Range: >46 mg/dL) - **Triglycerides**: 75 mg/dL (Normal; Reference Range: <150 mg/dL) - **LDL Cholesterol**: 139 mg/dL (High; Reference Range: <130 mg/dL) - **Cholesterol Ratio**: 2.9 (Normal; Reference Range: <5.0) - **Non-HDL Cholesterol**: 154 mg/dL ### Notes: - Desirable LDL cholesterol levels are <100 mg/dL for patients with coronary heart disease (CHD) or diabetes. - The target for non-HDL cholesterol is 30 mg/dL higher than the LDL cholesterol target.
                    """
                },
                {
                    "content": """
                    ### Summary of Health-Related Information from Lipid Panel **Date of Report:** January 5, 2016 #### Lipid Panel Results: - **Total Cholesterol:** 191 mg/dL (Reference Range: 125-200 mg/dL) - **HDL Cholesterol:** 42 mg/dL (Low; Reference Range: >46 mg/dL) - **Triglycerides:** 99 mg/dL (Normal; Reference Range: <150 mg/dL) - **LDL Cholesterol:** 129 mg/dL (Normal; Reference Range: <130 mg/dL) - **Cholesterol Ratio:** 4.5 (Normal; Reference Range: <5.0) - **Non-HDL Cholesterol:** 149 mg/dL (Target: 30 mg/dL higher than LDL target) #### Additional Notes: - Desirable range for LDL cholesterol is <100 mg/dL for patients with coronary heart disease (CHD) or diabetes, and <70 mg/dL for diabetic patients with known heart disease. This summary provides an overview of the lipid panel results, indicating areas of concern, particularly with HDL cholesterol levels.
                    """
                },
                {
                    "content": """
                    Here’s a summary of the health-related information from the lipid panel report dated January 4, 2020: ### Lipid Panel Results: - **Total Cholesterol**: 222 mg/dL (High; Reference range: <200 mg/dL) - **Triglycerides**: 58 mg/dL (Normal; Reference range: <150 mg/dL) - **HDL Cholesterol**: 73 mg/dL (Normal; Reference range: >39 mg/dL) - **Calculated LDL Cholesterol**: 137 mg/dL (High; Reference range: <100 mg/dL) - **Risk Ratio (LDL/HDL)**: 1.88 (Normal; Reference range: <3.22) ### Summary: - Total cholesterol and calculated LDL cholesterol levels are elevated. - Triglycerides and HDL cholesterol levels are within normal ranges. - The risk ratio indicates a low risk based on the LDL and HDL levels. It may be advisable to consult a healthcare provider for further interpretation and recommendations based on these results.
                    """
                },
                {
                    "content": """
                    Here’s a summary of the health-related information from the lipid panel report dated January 3, 2019: ### Lipid Panel Results: - **Total Cholesterol**: 229 mg/dL (High; Reference <200 mg/dL) - **HDL Cholesterol**: 83 mg/dL (Normal; Reference >50 mg/dL) - **Triglycerides**: 57 mg/dL (Normal; Reference <150 mg/dL) - **LDL Cholesterol**: 132 mg/dL (High; Reference <100 mg/dL) - **Chol/HDL Ratio**: 2.8 (Normal; Reference <5.0) - **Non-HDL Cholesterol**: 146 mg/dL (High; Reference <130 mg/dL) ### Interpretation: - **Total Cholesterol** and **LDL Cholesterol** levels are elevated, which may indicate an increased risk for cardiovascular issues. - **HDL Cholesterol** is within a healthy range, which is beneficial. - **Triglycerides** are normal. - The **Non-HDL Cholesterol** is also high, suggesting a need for further evaluation, especially for patients with diabetes or other risk factors.
                    """
                },
                {
                    "content": """
                    The lipid panel report from December 13, 2021, includes the following health-related information: - **Total Cholesterol**: 281 mg/dL (High; reference range <200 mg/dL) - **Triglycerides**: 85 mg/dL (Normal; reference range <150 mg/dL) - **HDL Cholesterol**: 80 mg/dL (Normal; reference range >39 mg/dL) - **Calculated LDL Cholesterol**: 162 mg/dL (High; reference range <100 mg/dL) - **Risk Ratio (LDL/HDL)**: 1.90 (Normal; reference range <3.22) **Notes**: The calculated LDL is based on the Martin-Hopkins method, which considers the triglyceride:VLDL cholesterol ratio. Elevated LDL levels may be influenced by higher triglycerides or lower non-HDL cholesterol levels. **Fasting Status**: The test was performed while fasting.
                    """
                },
            ]
        }
        return [self._get_document(item) for item in result.get("items", [])]

    def _get_document(self, item):
        content = item["content"]
        content_complement = item.get("content_complement")
        page_content = content
        if content_complement:
            page_content = content_complement
        metadata = {
            #    "chunk_id": item["chunk_id"],
            #    "workspace_id": item["workspace_id"],
            #    "document_id": item["document_id"],
            #    "document_sub_id": item["document_sub_id"],
            #    "document_type": item["document_type"],
            #    "document_sub_type": item["document_sub_type"],
            #    "path": item["path"],
            #    "title": item["title"],
            #    "score": item["score"],
        }
        return Document(page_content=page_content, metadata=metadata)


# chat_memory = BaseChatMessageHistory()


def get_memory(output_key=None, return_messages=False):
    return ConversationBufferMemory(
        memory_key="chat_history",
        # chat_memory=[],
        return_messages=return_messages,
        output_key=output_key,
    )


class LLMStartHandler(BaseCallbackHandler):
    prompts = []
    llm_responses = []

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        print("\non_llm_start 1")
        # print(prompts)
        print("on_llm_start 2\n")
        self.prompts.append(prompts)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print("\non_llm_end 1")
        # print(response)
        print("on_llm_end 2\n")
        self.llm_responses.append(response)


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
                                "Report Date": {
                                    "type": "string",
                                    "description": "date when this data point is reported",
                                },
                                "Cholesterol": {
                                    "type": "string",
                                    "description": "cholesterol or total cholesterol level",
                                },
                                "Triglycerides": {
                                    "type": "string",
                                    "description": "Triglycerides level",
                                },
                                "HDL Cholesterol": {
                                    "type": "string",
                                    "description": "hdl cholesterol level",
                                },
                                "LDL Cholesterol": {
                                    "type": "string",
                                    "description": "ldl cholesterol or calculated ldl cholesterol level",
                                },
                                "Non-HDL Cholesterol": {
                                    "type": "string",
                                    "description": "non-hdl cholesterol level",
                                },
                                "Cholesterol/HDL": {
                                    "type": "string",
                                    "description": "cholesterol to hdl ratio",
                                },
                                "LDL/HDL": {
                                    "type": "string",
                                    "description": "ldl/hdl ratio",
                                },
                            },
                            "required": ["report_date"],
                        },
                    }
                },
            },
        },
    }
]

handlers = LLMStartHandler()

params = {}
params["streaming"] = False
params["temperature"] = 0.0
params["max_tokens"] = 4095

# llm_base = ChatOpenAI(model_name="gpt-3.5-turbo-1106", callbacks=[handlers], **params)
llm_base = ChatOpenAI(model_name="gpt-4o", callbacks=[handlers], **params)
llm = llm_base.bind_tools(tools=tools, tool_choice="auto")
# llm = llm_base

conversation = ConversationalRetrievalChain.from_llm(
    llm,
    WorkspaceRetriever(),
    condense_question_llm=llm,
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True,
    memory=get_memory(output_key="answer", return_messages=True),
    verbose=True,
    callbacks=[handlers],
)

# user_prompt = "List LIPID data based on provided data include all dates"
# user_prompt = "plot a chart based on provided data"
user_prompt = (
    "What are your recommendations based on the LIPID values in the documents?"
)

result = conversation({"question": user_prompt})
# print(result)
documents = [
    {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
    }
    for doc in result["source_documents"]
]
print("******")
# print(result["answer"])

# print(handlers.prompts)
# print(handlers.llm_responses)

# print("******")

tool_calls = []
if handlers.llm_responses and len(handlers.llm_responses) > 0:
    last_response = handlers.llm_responses[-1]
    if last_response.generations and len(last_response.generations) > 0:
        response_message = last_response.generations[0][0].message
        if response_message.tool_calls and len(response_message.tool_calls) > 0:
            tool_calls = response_message.tool_calls

charts = []
for tool_call in tool_calls:
    if tool_call["name"] == "plot_chart_for_lipid_panel_trend":
        charts = plot_chart_for_lipid_panel_trend(tool_call["args"]["inputs"])
print(charts)
