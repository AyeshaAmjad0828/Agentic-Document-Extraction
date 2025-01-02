from utils.prompt_template import PromptTemplate
from utils.LLM_utils import get_completion_gpt4


complete_invoice_agent = PromptTemplate(
	template="""
	### Instructions:
	You are a data extraction tool capable of extracting data from each page of an invoice.

	1. Please extract the data in this invoice and format it to the given output JSON schema.
	
	2. Extract all key-value pairs from the invoice.
	
	3. If there are tables in the invoice, capture all of the rows and columns in the JSON object. 
	Even if a column is blank, include it as a key in the JSON object with a null value.
	
	4. If a row is blank denote missing fields with "null" values.
	
	5. If the page contains no charge data, please output an empty JSON object and don't make up any data.
	
	6. Don't interpolate or make up data.

	7. Please maintain the table structure of the charges, i.e. capture all of the rows and columns in the JSON object.
	
	8. Ensuring the order of key-value pairs and tabular data aligns with the original text.


	The language model must interpret and execute these extraction and formatting instructions accurately.
	
	Perform the task as per above instructions on the following invoice document:
	
	### INPUT INVOICE:
	
	{{DocumentText}}
	
	### OUTPUT JSON SCHEMA:
	
	{{JsonSchema}}

	""",
	input_variables=["DocumentText","JsonSchema"]
)

##test
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
schema = {
    "invoice_number": "string",
    "customer": "string",
    "invoice_date": "yyyy-mm-dd",
    "sub_total": "number",
    "total_GST": "number",
    "total_amount": "number",
    "Line_Items": [
        {
            "item": "string",
            "description": "string",
            "quantity": "number",
            "price": "number",
            "discount": "number",
            "total": "number"
        }
    ]
}

def generate_invoice_prompt(document_text, json_schema):
     prompt = complete_invoice_agent.format(
		DocumentText=document_text,
		JsonSchema=json_schema,
        )
     return prompt

prompt = generate_invoice_prompt(invoice, schema)
messages = [
    {"role": "user", "content": prompt}
]
response = get_completion_gpt4(messages)
print(response.choices[0].message.content)


def baseline_invoice_agent(DocumentText):
    Prompt = f"""
### Task Instructions:

Develop a language model capable of performing the following tasks on each page of a document such as invoices or purchase orders.

1. Read and process the document data from the given page.
2. Extract all key-value pairs from this page. If it is a first page, you may find unique identifier information e.g., 'invoice_number', 'date', 'supplier', and more. If it is last page, you may find summary totals such as 'sub_total', 'total_gst', 'invoice_total' and more. 
3. Extract all line items i.e extract all columns and rows from the invoice table from the given page
4. Present the extracted information, including header names, in a JSON format, ensuring the order of key-value pairs and line items data aligns with the original text.

As an illustrative example, consider an invoice with the following text data on the given page:

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

Json output:

-------Output-----------------------------
{{
"invoice_number": "INV-12345",
"customer": "XYZ Corp",
"invoice_date": "2024-06-01",
"sub_total": 1060,
"total_GST":500,
"total_amount": "1560",
"Line_Items": [[
	{{
	"item": "item_1",
	"quantity": 5,
	"price": "$100",
	"total": 500
	}},
	{{
	"item": "item_2",
	"quantity": 10,
	"price": "$50",
	"total": 500
	}},
	{{
	"item": "item_3",
	"quantity": 6,
	"price": "$10",
	"total": 60
	}}
]]
}}
--------------------------------------------

The language model must interpret and execute these extraction and formatting instructions accurately for each page of the document.

### Perform the task as per the above instructions on the following : 	

{DocumentText}

If you cannot find a value then leave it as empty like this "" in the output. 
Only get me the JSON output object without any irrelevant text or explanation.
Do not leave out any information, extract all line items and key-value pairs. 
 

            """
    return Prompt


def firstpage():
    Prompt = """
			### Instructions:
			Develop a language model capable of performing the following tasks on the first page of a document such as invoices or purchase orders.

			1. Read and process the document data from the first page.
			2. Extract all key-value pairs and line items or tabular data from the first page text to complete the user-defined output JSON schema.
			3. Present the extracted information, including header names, in a JSON format, ensuring the order of key-value pairs and line items data aligns with the original text.
	
			As an illustrative example, consider an invoice with the following text data on the first page:

			-----------------Invoice------------------
			                              Page 1 of 3

			Invoice Number: INV-12345
			Customer: XYZ Corp
			Invoice Date: 2024-06-01


			Item    Quantity    Price     Total
			item_1     5         $100      500
			item_2     10        $50       500
			item_3     6         $10       60

			-------------------------------------------
			
			This is the expected Json output:

			-------Output-----------------------------
			{
			"invoice_number": "INV-12345",
			"customer": "XYZ Corp",
			"invoice_date": "2024-06-01",
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
				},
				{
				"item": "item_3",
				"quantity": 6,
				"price": "$10",
				"total": 60
				}
			]
			}
			--------------------------------------------
	
			The language model must interpret and execute these extraction and formatting instructions accurately, but only for the first page
            
            """
    return Prompt


def middlepages():
    Prompt = """
			### Instructions:
			Develop a language model capable of performing the following tasks on all pages except the first and last page of a document.

			1. Read and process the document data from these middle pages of documents such as invoices or purchase orders.
            2. Extract all key-value pairs and line items or tabular data found on this page to complete the user-defined output JSON schema.
			3. Present the extracted information, including header names, in a JSON format, ensuring the order of key-value pairs and line items data aligns with the original text.

			As an illustrative example, consider an invoice with the following text data on the second page:

			----------------Invoice---------------------
                                            Page 2 of 3

			Item    Quantity    Price     Total
			item_4     5         $100      500
			item_5     10        $50       500
			item_6     6         $10       60
			item_7     5         $15       75
			item_8     10        $10       100
			item_9     2         $200      400

			---------------------------------------------
			
			This is the expected Json output:

			-------Output-----------------------------
			{
			"Line_Items": [
				{
				"item": "item_4",
				"quantity": 5,
				"price": "$100",
				"total": 500
				},
				{
				"item": "item_5",
				"quantity": 10,
				"price": "$50",
				"total": 500
				},
				{
				"item": "item_6",
				"quantity": 6,
				"price": "$10",
				"total": 60
				},
				{
				"item": "item_7",
				"quantity": 5,
				"price": "$15",
				"total": 75
				},
				{
				"item": "item_8",
				"quantity": 10,
				"price": "$10",
				"total": 100
				},
				{
				"item": "item_9",
				"quantity": 2,
				"price": "$200",
				"total": 400
				}
			]
			}
			--------------------------------------------

			The language model must interpret and execute these extraction and formatting instructions accurately, but only for the middle pages.
            
            """
    return Prompt


def lastpage():
    Prompt = """
			### Instructions:
			Develop a language model capable of performing the following tasks on the last page of a document.

			1. Read and process the document data from the last page, such as invoices and purchase orders.
            2. Extract all key-value pairs any relevant summary data or totals typically found on the last page to complete the user-defined output JSON schema.
			3. Extract all remaining line items or tabular data if exists to complete the user-defined output JSON schema.
			4. Present the extracted information in a JSON format, ensuring the order of key-value pairs and tabular data aligns with the original text.

			As an illustrative example, consider an invoice with the following text data on the last page:

			--------------Invoice-----------------
                                         Page 3 of 3

			item_10     8         $100      800
			item_11     10        $50       500

								
								Subtotal: 5000
								Total GST: 500
								Total Amount: $5500
			----------------------------------------

			This is the expected Json output:

			----------Output-----------------------
			{
			"sub_total": 5000,
			"total_GST":500,
			"total_amount": "$5500",
			"Line_Items": [
				{
				"item": "item_10",
				"quantity": 8,
				"price": "$100",
				"total": 800
				},
				{
				"item": "item_11",
				"quantity": 10,
				"price": "$50",
				"total": 500
				}
			]
			}
			------------------------------------------

			The language model must interpret and execute these extraction and formatting instructions accurately, but only for the last page.
            
            """
    return Prompt

#Example_usage:

# def load_prompt_from_file(filename):
#     """Load prompt template from a text file"""
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     prompt_path = os.path.join(current_dir, 'Prompts', filename)
#     with open(prompt_path, 'r', encoding='utf-8') as file:
#         return file.read()


# # Load the base prompt template
# base_prompt = load_prompt_from_file('Invoice_prompt_baseline.txt')
# if __name__ == "__main__":
#     # Example initial state and actions
#     invoice = f"""
# -----------------Invoice------------------
#                               Page 1 of 3

# Invoice Number: INV-12345
# Customer: XYZ Corp
# Invoice Date: 2024-06-01


# Item    Quantity    Price     Total
# item_1     5         $100      500
# item_2     10        $50       500
# item_3     6         $10       60

# 					Subtotal: 1060
# 					Total GST: 500
# 					Total Amount: $1560
# --------------------------------------------
# """
#     schema = schema = {
#     "invoice_number": "string",
#     "customer": "string",
#     "invoice_date": "yyyy-mm-dd",
#     "sub_total": "number",
#     "total_GST": "number",
#     "total_amount": "number",
#     "Line_Items": [
#         {
#             "item": "string",
#             "quantity": "number",
#             "price": "number",
#             "total": "number"
#         }
#     ] 
#     }
   
#     generated_output = {
#     "invoice_number": "INV-12345",
#     "invoice_date": "2024-06-01",
#     "sub_total": 1060,
#     "total_amount": 1560, 
#     "Line_Items": [
#         {
#         "item": "item_1",
#         "quantity": 5,
#         "price": "$100",
#         "total": 500
#         },
#         {
#         "item": "item_2",
#         "quantity": 10,
#         "price": "$50",
#         "total": 500
#         }
#     ]
#     } 
#     groundtruth = {
#     "invoice_number": "INV-12345",
#     "customer": "XYZ Corp",
#     "invoice_date": "2024-06-01",
#     "sub_total": 1060,
#     "total_GST":500,
#     "total_amount": 1560,
#     "Line_Items": [
#         {
#         "item": "item_1",
#         "quantity": 5,
#         "price": 100,
#         "total": 500
#         },
#         {
#         "item": "item_2",
#         "quantity": 10,
#         "price": 50,
#         "total": 500
#         },
#         {
#         "item": "item_3",
#         "quantity": 6,
#         "price": 10,
#         "total": 60
#         }
#     ]
#     }
#     print(baseline_extractor_agent(base_prompt,'invoice',invoice))






