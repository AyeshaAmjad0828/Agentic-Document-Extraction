from utils.LLM_utils import get_completion_gpt4
import numpy as np



def schema_building_with_llm(text):
    # Define the prompt template
    SCHEMA_BUILDER_PROMPT = f"""
    You are a json schema builder. You will recieve an unstructured form-like document and you will perform following tasks:
    1. Extract only the key phrases in key-value pairs, and header names in tables from the document. 
    2. Arrange all keys phrases with data types in the parent region of the json and table header names in the child region.
    3. Ensure the order of key names and table headers aligns with the original text.


    Here is the content of the form-like document:

    '''

    {text}

    '''

    Only include unique column names as they appear in the document, without duplicates or associated data. Respond only with the json schema.
    """

 
    try:
        response = get_completion_gpt4([{"role": "user", "content": SCHEMA_BUILDER_PROMPT.format(text)}],
                                       logprobs=True,
                                       )
        logprobs = [token.logprob for token in response.choices[0].logprobs.content]
        response_text = response.choices[0].message.content
        max_starter_length = max(len(s) for s in ["Schema \n:", "Perplexity:"])


        perplexity_score = np.exp(-np.mean(logprobs))
        print("Response:".ljust(max_starter_length), response_text, "\n")
        print("Perplexity:".ljust(max_starter_length), perplexity_score, "\n")

    except Exception as e:
        print(f"Error during Schema Building: {e}")
        return "Error"
    return response_text, perplexity_score

# Example usage
document_text = "Invoice #789, Amount Due: $1200, Due Date: 2024-11-01"
a, b = schema_building_with_llm(document_text)

if __name__ == "__main__":
    # Example initial state and actions
    invoice = f"""
-----------------Invoice------------------
                              Page 1 of 3

Invoice Number: INV-12345
Customer: XYZ Corp
Invoice Date: 2024-06-01


Item    Quantity    Price     Total
item_1     5         $100      500
item_2     10        $50       500
item_3     6         $10       60

					Subtotal: 1060
					Total GST: 500
					Total Amount: $1560
--------------------------------------------
"""
    schema = schema = {
    "invoice_number": "string",
    "customer": "string",
    "invoice_date": "yyyy-mm-dd",
    "sub_total": "number",
    "total_GST": "number",
    "total_amount": "number",
    "Line_Items": [
        {
            "item": "string",
            "quantity": "number",
            "price": "number",
            "total": "number"
        }
    ] 
    }
   
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
    a, b, c = schema_building_with_llm(invoice)
    print(a)
    print(b)
    print(c)
