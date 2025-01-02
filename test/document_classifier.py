from utils.LLM_utils import get_completion_gpt4
import numpy as np
from IPython.display import display, HTML


CLASSES = ["Invoice", "Purchase Order", "Bill", "Receipt", "Financial Document"]

def classify_document_with_llm(text):
    CLASSIFICATION_PROMPT = f"""
    I have a form-like document, and I want you to classify it into one of the following categories:
    - Invoice
    - Purchase Order
    - Bill
    - Receipt
    - Financial Document

    Here is the content of the document:

    "{text}"

    Based on the content, which category does this document belong to? Please reply with only the category name.
    """

    # Call the OpenAI API to classify the document
    try:
        response = get_completion_gpt4([{"role": "user", "content": CLASSIFICATION_PROMPT.format(text)}],
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

        print(linear_probability) 

        
        if classification in CLASSES:
            return classification, linear_probability
        else:
            return "Unknown"  # Return "Unknown" if response is unexpected

    except Exception as e:
        print(f"Error during LLM classification: {e}")
        return "Error"

# Example usage
if __name__ == "__main__":
    # Example initial state and actions
    doc = f"""
---------------Distribution Notice-------------------

                FFFIV Investments LLC
        Underlying Fund: KKR Americas Fund XII LP

                Distribution Notice
                    July 26, 2021

FFFIV Investments LLC is making a distribution to its investors. Your pro rata share is detailed as follows:

            Investor: XXXX XXX
            Current Distribution Amount: $6,312.50

Your summary of cash flows inception to date including this distribution are as follows:

            Contributions to Date: $190,872.64
            Distributions to Date: $17,237.50
            Net Cash Flow Since Inception: $173,635.14
            --------------------------------------------
"""
   
    generated_output = {
    "invoice_number": "INV-12345",
    "invoice_date": "2024-06-01",
    "sub_total": 1060,
    "total_amount": 1560, 
    "Line_Items": [
        {
        "item": "item_1",
        "quantity": 5,
        "price": "$100",
        "total": 500
        },
        {
        "item": "item_2",
        "quantity": 10,
        "price": "$50",
        "total": 500
        }
    ]
    } 
    groundtruth = {
    "invoice_number": "INV-12345",
    "customer": "XYZ Corp",
    "invoice_date": "2024-06-01",
    "sub_total": 1060,
    "total_GST":500,
    "total_amount": 1560,
    "Line_Items": [
        {
        "item": "item_1",
        "quantity": 5,
        "price": 100,
        "total": 500
        },
        {
        "item": "item_2",
        "quantity": 10,
        "price": 50,
        "total": 500
        },
        {
        "item": "item_3",
        "quantity": 6,
        "price": 10,
        "total": 60
        }
    ]
    }
    prompt, predicted_class, prob = classify_document_with_llm(doc)
    print(prompt)
    print(f"Predicted Class: {predicted_class}")
    print(f"Linear Probability: {prob}")