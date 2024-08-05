import os
from openai import OpenAI, OpenAIError


os.environ["OPENAI_API_KEY"] = (
)

content_arr = [
    {
        "type": "text",
        "text": "use the following information as context to perform action or answer question",
    },
    {
        "type": "text",
        "text": """

### Summary of Health-Related Information (Lipid Panel) **Date of Report:** December 23, 2020 #### Results: - **Total Cholesterol:** 279 mg/dL (High; Reference Range: <200 mg/dL) - **Triglycerides:** 53 mg/dL (Normal; Reference Range: <150 mg/dL) - **HDL Cholesterol:** 91 mg/dL (Normal; Reference Range: >39 mg/dL) - **Calculated LDL Cholesterol:** 173 mg/dL (High; Reference Range: <100 mg/dL) - **Risk Ratio (LDL/HDL):** 1.90 (Normal; Reference Range: <3.22) #### Notes: - The calculated LDL is based on the Martin-Hopkins method, which adjusts for triglyceride levels and VLDL cholesterol ratio. - Elevated total cholesterol and LDL levels may indicate a higher risk for cardiovascular issues. For further details, consult the provided link or a healthcare professional.

health_report: date: "04/24/2024" lipid_panel: cholesterol: value: 211 reference_range: "<200 mg/dL" status: "High" triglycerides: value: 33 reference_range: "<150 mg/dL" HDL_cholesterol: value: 90 reference_range: ">39 mg/dL" cholesterol_hdl_ratio: value: 2.34 reference_range: "0.0-4.4 (Ratio)" non_hdl_cholesterol: value: 121 reference_range: "<130 mg/dL" LDL_cholesterol: value: 114 reference_range: "<130 mg/dL" interpretation: - "Optimal: Less than 100 mg/dL" - "Near Optimal: 100 to 129 mg/dL" - "Above Optimal: 130 to 159 mg/dL" - "Borderline High: 160 to 189 mg/dL" - "High: 190 mg/dL and above" - "Very High: 190 mg/dL and above" LDL_history: previous_tests: - date: "04/19/2023" LDL_result: 112 change: "0%" - date: "04/23/2024" LDL_result: 114

health_report: date: "04/20/2023" lipid_panel: cholesterol: value: 207 reference_range: "<200 mg/dL" status: "High" triglycerides: value: 56 reference_range: "<150 mg/dL" status: "Normal" hdl_cholesterol: value: 84 reference_range: ">39 mg/dL" status: "Normal" cholesterol_hdl_ratio: value: 2.46 reference_range: "0.00-0.44" status: "High" non_hdl_cholesterol: value: 123 reference_range: "<130 mg/dL" status: "Normal" ldl_cholesterol_calculation: value: 112 reference_range: "<130 mg/dL" status: "Normal" ldl_hdl_ratio: value: 1.3 reference_range: "<3.3" status: "Normal" patient_history: test_date: "04/13/2022" ldl_results: value: 113 units: "mg/dL" change_percentage: "-" 

### Summary of Health-Related Information from Lipid Panel **Date of Report:** July 2, 2021 #### Results: - **Total Cholesterol:** 244 mg/dL (High; Reference Range: <200 mg/dL) - **Triglycerides:** 94 mg/dL (Normal; Reference Range: <150 mg/dL) - **HDL Cholesterol:** 74 mg/dL (Normal; Reference Range: >39 mg/dL) - **Calculated LDL Cholesterol:** 150 mg/dL (High; Reference Range: <100 mg/dL) - **Risk Ratio (LDL/HDL):** 2.03 (Normal; Reference Range: <3.22) #### Notes: - The calculated LDL is based on the Martin-Hopkins method, which adjusts for triglyceride levels and VLDL cholesterol ratio. - Elevated total cholesterol and LDL levels may indicate a higher risk for cardiovascular issues. For further details, consult a healthcare provider.

health_report: date: "04/14/2022" lipid_panel: cholesterol: value: 201 reference_range: "<200 mg/dL" status: "High" triglycerides: value: 42 reference_range: "<150 mg/dL" HDL_cholesterol: value: 80 reference_range: ">39 mg/dL" cholesterol_hdl_ratio: value: 2.51 reference_range: "0.00-0.44 (Ratio)" non_hdl_cholesterol: value: 121 reference_range: "<130 mg/dL" LDL_cholesterol: calculated_value: 113 reference_range: "<130 mg/dL" status: "Optimal" LDL_hdl_ratio: value: 1.4 reference_range: "<3.3 (Ratio)" test_date: "04/13/2022"

### Summary of Health Information from Lipid Panel (Date: December 28, 2016) - **Total Cholesterol**: 234 mg/dL (High; Reference Range: 126-200 mg/dL) - **HDL Cholesterol**: 80 mg/dL (Normal; Reference Range: >46 mg/dL) - **Triglycerides**: 75 mg/dL (Normal; Reference Range: <150 mg/dL) - **LDL Cholesterol**: 139 mg/dL (High; Reference Range: <130 mg/dL) - **Cholesterol Ratio**: 2.9 (Normal; Reference Range: <5.0) - **Non-HDL Cholesterol**: 154 mg/dL ### Notes: - Desirable LDL cholesterol levels are <100 mg/dL for patients with coronary heart disease (CHD) or diabetes. - The target for non-HDL cholesterol is 30 mg/dL higher than the LDL cholesterol target.

### Summary of Health-Related Information from Lipid Panel **Date of Report:** January 5, 2016 #### Lipid Panel Results: - **Total Cholesterol:** 191 mg/dL (Reference Range: 125-200 mg/dL) - **HDL Cholesterol:** 42 mg/dL (Low; Reference Range: >46 mg/dL) - **Triglycerides:** 99 mg/dL (Normal; Reference Range: <150 mg/dL) - **LDL Cholesterol:** 129 mg/dL (Normal; Reference Range: <130 mg/dL) - **Cholesterol Ratio:** 4.5 (Normal; Reference Range: <5.0) - **Non-HDL Cholesterol:** 149 mg/dL (Target: 30 mg/dL higher than LDL target) #### Additional Notes: - Desirable range for LDL cholesterol is <100 mg/dL for patients with coronary heart disease (CHD) or diabetes, and <70 mg/dL for diabetic patients with known heart disease. This summary provides an overview of the lipid panel results, indicating areas of concern, particularly with HDL cholesterol levels.

Here’s a summary of the health-related information from the lipid panel report dated January 4, 2020: ### Lipid Panel Results: - **Total Cholesterol**: 222 mg/dL (High; Reference range: <200 mg/dL) - **Triglycerides**: 58 mg/dL (Normal; Reference range: <150 mg/dL) - **HDL Cholesterol**: 73 mg/dL (Normal; Reference range: >39 mg/dL) - **Calculated LDL Cholesterol**: 137 mg/dL (High; Reference range: <100 mg/dL) - **Risk Ratio (LDL/HDL)**: 1.88 (Normal; Reference range: <3.22) ### Summary: - Total cholesterol and calculated LDL cholesterol levels are elevated. - Triglycerides and HDL cholesterol levels are within normal ranges. - The risk ratio indicates a low risk based on the LDL and HDL levels. It may be advisable to consult a healthcare provider for further interpretation and recommendations based on these results.

Here’s a summary of the health-related information from the lipid panel report dated January 3, 2019: ### Lipid Panel Results: - **Total Cholesterol**: 229 mg/dL (High; Reference <200 mg/dL) - **HDL Cholesterol**: 83 mg/dL (Normal; Reference >50 mg/dL) - **Triglycerides**: 57 mg/dL (Normal; Reference <150 mg/dL) - **LDL Cholesterol**: 132 mg/dL (High; Reference <100 mg/dL) - **Chol/HDL Ratio**: 2.8 (Normal; Reference <5.0) - **Non-HDL Cholesterol**: 146 mg/dL (High; Reference <130 mg/dL) ### Interpretation: - **Total Cholesterol** and **LDL Cholesterol** levels are elevated, which may indicate an increased risk for cardiovascular issues. - **HDL Cholesterol** is within a healthy range, which is beneficial. - **Triglycerides** are normal. - The **Non-HDL Cholesterol** is also high, suggesting a need for further evaluation, especially for patients with diabetes or other risk factors.

The lipid panel report from December 13, 2021, includes the following health-related information: - **Total Cholesterol**: 281 mg/dL (High; reference range <200 mg/dL) - **Triglycerides**: 85 mg/dL (Normal; reference range <150 mg/dL) - **HDL Cholesterol**: 80 mg/dL (Normal; reference range >39 mg/dL) - **Calculated LDL Cholesterol**: 162 mg/dL (High; reference range <100 mg/dL) - **Risk Ratio (LDL/HDL)**: 1.90 (Normal; reference range <3.22) **Notes**: The calculated LDL is based on the Martin-Hopkins method, which considers the triglyceride:VLDL cholesterol ratio. Elevated LDL levels may be influenced by higher triglycerides or lower non-HDL cholesterol levels. **Fasting Status**: The test was performed while fasting.

        """,
    },
    {
        "type": "text",
        "text": "plot charts using lipid data",
    },
]

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
                                "report_date": {
                                    "type": "string",
                                    "description": "date when this data point is reported",
                                },
                                "cholesterol": {
                                    "type": "string",
                                    "description": "cholesterol or total cholesterol level",
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

try:
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "user",
                "content": content_arr,
            }
        ],
        temperature=0.0,
        max_tokens=4095,
        top_p=0.9,
        tools=tools,
        tool_choice="auto",
    )
    print(completion.choices[0].message.content)
except OpenAIError as e:
    # Handle all OpenAI API errors
    print(f"Error: {e}")


completion.choices[0].message.tool_calls[0].function.arguments
