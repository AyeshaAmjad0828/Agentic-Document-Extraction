import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.prompt_template import PromptTemplate
from src.utils.LLM_utils import get_completion_gpt4
import numpy as np
from IPython.display import display, HTML


CLASSES = ["Invoice", "Purchase Order", "Bill", "Receipt", "Financial Document", "Salary Slip"]

def classify_document_with_llm(document_text):
    CLASSIFICATION_PROMPT = PromptTemplate(
        template="""
I have a form-like document, and I want you to classify it into one of the following categories:
- Invoice
- Purchase Order
- Utility Bill
- Receipt
- Financial Document
- Salary Slip

Here is the content of the document:

{{text}}

Based on the content, which category does this document belong to? Please reply with only the category name.
    """,
    input_variables=["text"]
    )
    prompt = CLASSIFICATION_PROMPT.format(
        text = document_text)

    # Call the OpenAI API to classify the document
    try:
        response = get_completion_gpt4([{"role": "user", "content": prompt}],
                                       logprobs=True,
                                       top_logprobs=1,
                                       )
        top_two_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        html_content = ""
        for i, logprob in enumerate(top_two_logprobs, start=1):
            html_content += (
                f"<span style='color: cyan'>Output token {i}:</span> {logprob.token}, "
                f"<span style='color: darkorange'>logprobs:</span> {logprob.logprob}, "
                f"<span style='color: magenta'>linear probability:</span> {np.round(np.exp(logprob.logprob)*100,2)}%<br>"
            )
        display(HTML(html_content))
        print("\n")
        
        classification = response.choices[0].message.content
        linear_probability = np.round(np.exp(top_two_logprobs[0].logprob)*100,2)

        
        if classification in CLASSES:
            return classification, linear_probability
        else:
            return "Unknown"  # Return "Unknown" if response is unexpected

    except Exception as e:
        print(f"Error during LLM classification: {e}")
        return "Error"

