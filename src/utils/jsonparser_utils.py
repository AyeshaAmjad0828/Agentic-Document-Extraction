import pandas as pd
import json



def clean_llm_output(text: str) -> str:
    """
    Clean LLM output by removing markdown code blocks and language identifiers.
    
    Args:
        text (str): Raw text from LLM output
        
    Returns:
        str: Cleaned JSON string
    """
    # Remove markdown code block markers and language identifiers
    text = text.replace('```json', '').replace('```', '').strip()
    
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    try:
        # Parse and re-serialize to ensure valid JSON
        parsed = json.loads(text)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse JSON: {e}")
        return text



def json_to_dataframe(json_input):
    """
    Convert JSON input to a Pandas DataFrame.

    Parameters:
    json_input (str): JSON input as a string.

    Returns:
    pd.DataFrame: Pandas DataFrame containing the data from the JSON input.
    """
    # Parse the JSON input
    data = json.loads(json_input)
    
    # Extract line items and add main fields to each line item
    line_items = data.pop('Line_Items')
    for item in line_items:
        item.update(data)
    
    # Convert line items to DataFrame
    df = pd.DataFrame(line_items)
    
    return df

# # Example usage
# json_input = '''
# {
#   "Vendor": "16317",
#   "PO_Number": "38688",
#   "Ship_To": "CIENA COMMUNICATIONS, INC.",
#   "Bill_To": "GREAT PLAINS COMMUNICATIONS LLC",
#   "Discount": "",
#   "Sales_Tax": "",
#   "Total_Cost": "",
#   "Line_Items": [
#     {
#       "LN": "1",
#       "Item": "TE2308",
#       "LOC": "ELKH",
#       "Quantity": "2.000",
#       "UOM": "EA",
#       "Description": "8110/8114,DC PLUGGABLE POWER SUPPLY 48V",
#       "Unit_cost": "878.5300",
#       "Discount": "0.00",
#       "Total": "1,757.06"
#     },
#     {
#       "LN": "2",
#       "Item": "TE2309",
#       "LOC": "ELKH",
#       "Quantity": "1.000",
#       "UOM": "EA",
#       "Description": "8114,(6)100G QSFP28,(48) 25G/10G/1G SFP28 (2)SLOTS 800G MOD,EXT. TEMP,(2)SLOTS AC OR DC PLUG PWR SUP",
#       "Unit_cost": "14,448.690",
#       "Discount": "0.00",
#       "Total": "14,448.69"
#     },
#     {
#       "LN": "3",
#       "Item": "MISC",
#       "LOC": "ELKH",
#       "Quantity": "1.000",
#       "UOM": "EA",
#       "Description": "EIA561 RJ45F TO USB-C CONSOLE ADAPTER 6",
#       "Unit_cost": "39.1800",
#       "Discount": "0.00",
#       "Total": "39.18"
#     },
#     {
#       "LN": "4",
#       "Item": "TE2310",
#       "LOC": "ELKH",
#       "Quantity": "1.000",
#       "UOM": "EA",
#       "Description": "SAOS BASE OS, ETHERNET & OAM SOFTWARE LICENSE FOR 8114, PERPETUAL",
#       "Unit_cost": "7,420.0000",
#       "Discount": "0.00",
#       "Total": "7,420.00"
#     },
#     {
#       "LN": "5",
#       "Item": "TE2311",
#       "LOC": "ELKH",
#       "Quantity": "1.000",
#       "UOM": "EA",
#       "Description": "SAOS EVPN SOFTWARE LICENSE FOR 8114, PERPETUAL",
#       "Unit_cost": "1,157.5200",
#       "Discount": "0.00",
#       "Total": "1,157.52"
#     },
#     {
#       "LN": "6",
#       "Item": "TE2312",
#       "LOC": "ELKH",
#       "Quantity": "1.000",
#       "UOM": "EA",
#       "Description": "SAOS ROUTING AND MPLS SOFTWARE LICENSE FOR 8114, PERPETUAL",
#       "Unit_cost": "6,120.2500",
#       "Discount": "0.00",
#       "Total": "6,120.25"
#     },
#     {
#       "LN": "7",
#       "Item": "TE2297",
#       "LOC": "ELKH",
#       "Quantity": "1.000",
#       "UOM": "EA",
#       "Description": "SAOS SECURITY SOFTWARE LICENSE FOR 8114, PERPETUAL",
#       "Unit_cost": "106.8500",
#       "Discount": "0.00",
#       "Total": "106.85"
#     }
#   ]
# }
# '''
# # Example usage:
# raw_output = '''```json
# {
#   "invoice_number": "INV-123456",
#   "customer": "John Doe"
# }
# '''
# cleaned_output = clean_llm_output(raw_output)
# print(cleaned_output)

# # Convert JSON to DataFrame
# df = json_to_dataframe(json_input)
# df
