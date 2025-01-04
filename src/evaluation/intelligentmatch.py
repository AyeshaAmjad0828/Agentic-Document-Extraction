
import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

#--------------------------------------------------------------------------------------------#



from src.utils.LLM_utils import get_completion_gpt4
from src.utils.prompt_template import PromptTemplate
import json

def semantic_match_score(generated_output, groundtruth):
    """Use LLM to semantically compare generated output with groundtruth"""
    
    semantic_match_prompt = PromptTemplate(
        template="""
You are an expert evaluator for document extraction systems. 
Compare the generated output with the ground truth and:
1. Identify semantically matching fields (even if they have different key names)
2. Count total fields in generated output
3. Count how many fields and respective values from generated output match with ground truth (considering semantic similarity)
4. Provide a score between 0 and 1

Generated Output:
{{generated_output}}

Ground Truth:
{{groundtruth}}

Respond in JSON format:
{
    "total_fields": number,
    "matched_fields": number,
    "score": number,
    "field_matches": [
        {
            "generated_field": "field name",
            "groundtruth_field": "matching field name",
            "reason": "explain why the field matches"
        }
    ]
}
    """,
    input_variables=["generated_output", "groundtruth"]
    )
    prompt=semantic_match_prompt.format(
        generated_output=generated_output,
        groundtruth=groundtruth
    )

    try:
        response = get_completion_gpt4(
            [{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        result = json.loads(response.choices[0].message.content)
        
        # Print detailed matching information
        print("\nSemantic Matching Details:")
        print(f"Total fields in generated output: {result['total_fields']}")
        print(f"Matched fields: {result['matched_fields']}")
        print("\nField Matches:")
        for match in result['field_matches']:
            print(f"- {match['generated_field']} → {match['groundtruth_field']}")
            print(f"  Reason: {match['reason']}")
        
        return result['score']
        
    except Exception as e:
        print(f"Error in semantic matching: {str(e)}")
        return 0.0
    
# Example usage

# generated_output= {
#   "Faktura nr": "F 1234K20/1234/12",
#   "Data wystawienia": "2021-01-01",
#   "Sprzedawca": "Polski Koncern Naftowy ORLEN S.A.",
#   "NIP (Sprzedawca)": "774-00-01-454",
#   "Adres (Sprzedawca)": [
#     "Chemików 7 09-411 Płock",
#     "Polski Koncern Naftowy ORLEN S.A.",
#     "Stacja Paliw Nr 4445",
#     "58-100 Słotwina",
#     "Słotwina 62x"
#   ],
#   "Nabywca": "GŁÓWNY URZĄD STATYSTYCZNY",
#   "NIP (Nabywca)": "5261040828",
#   "Adres (Nabywca)": [
#     "Aleja Niepodległości 208 00-925 Warszawa"
#   ],
#   "Towary": [
#     {
#       "Lp": 1,
#       "Nazwa towaru": "EFECTA 95 CN27101245",
#       "Jm": "l",
#       "Ilość [jm]": 54.910,
#       "Cena brutto": 5.79,
#       "Wartość po rabacie": 1.10,
#       "Wartość netto": 5.77,
#       "VAT [%]": 23,
#       "Kwota VAT": 59.24,
#       "Wartość brutto": 316.83
#     }
#   ],
#   "Razem": {
#     "Wartość": 257.59,
#     "Kwota VAT": 59.24,
#     "Wartość brutto": 316.83
#   },
#   "Należność ogółem": "316,83 PLN",
#   "Słownie": "trzysta szesnaście PLN, 83/100",
#   "Do dokumentów": {
#     "Dok. wydania": "1234567",
#     "Dok.fisk.nr": "12345",
#     "Data": "2021-01-01"
#   },
#   "Zapłacono": "Aplikacja mobilna ORLEN Pay",
#   "Faktura wygenerowana automatycznie": True
# }

# groundtruth = {
#     "issuer": "Polski Koncern Naftowy ORLEN spółka akcyjna",
#     "date": "2021-01-01",
#     "invoice_number": "F",
#     "amount": 316.83,
#     "vat": 7740001454,
#     "sums":[
#         {
#             "net": 257.59,
#             "rate": 23,
#             "vat": 59.24,
#             "gross": 316.83
#         }
#     ],
#     "currency": "PLN",
#     "desc": "Invoice from Polski Koncern Naftowy ORLEN spółka akcyjna"
# }
# a = semantic_match_score(generated_output, groundtruth)