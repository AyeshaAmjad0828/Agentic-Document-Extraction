import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

#--------------------------------------------------------------------------------------------#
import json
from typing import Dict, Any

# ... existing imports ...

def calculate_schema_complexity(schema_json: Dict[str, Any]) -> float:
    """
    Calculate complexity score for a schema based on multiple factors:
    - Number of fields
    - Depth of nesting
    - Type complexity (simple types vs arrays/objects)
    - Field name complexity
    
    Returns a normalized score between 0 and 1
    """
    def get_nesting_depth(obj, current_depth=1):
        if not isinstance(obj, (dict, list)):
            return current_depth
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(get_nesting_depth(v, current_depth + 1) for v in obj.values())
        if isinstance(obj, list):
            if not obj:
                return current_depth
            return max(get_nesting_depth(item, current_depth + 1) for item in obj)
    
    def get_type_complexity(value):
        if isinstance(value, (str, int, float, bool)):
            return 1
        elif isinstance(value, list):
            return 2 + (0.5 * get_type_complexity(value[0]) if value else 0)
        elif isinstance(value, dict):
            return 3
        return 1

    try:
        # Parse schema if it's a string
        if isinstance(schema_json, str):
            schema = json.loads(schema_json)
        else:
            schema = schema_json

        # Calculate various complexity metrics
        num_fields = len(schema.keys())
        max_depth = get_nesting_depth(schema)
        
        # Calculate average type complexity
        type_complexity = sum(get_type_complexity(v) for v in schema.values()) / num_fields
        
        # Calculate field name complexity (length and special characters)
        field_complexity = sum(len(k) + sum(1 for c in k if not c.isalnum()) 
                             for k in schema.keys()) / num_fields
        
        # Combine metrics with weights
        weights = {
            'num_fields': 0.3,
            'depth': 0.3,
            'type_complexity': 0.25,
            'field_complexity': 0.15
        }
        
        # Normalize each component
        normalized_fields = min(1.0, num_fields / 20)  # Assume 20 fields is maximum complexity
        normalized_depth = min(1.0, max_depth / 5)     # Assume 5 levels is maximum depth
        normalized_type = min(1.0, type_complexity / 3)
        normalized_field = min(1.0, field_complexity / 10)
        
        complexity_score = (
            weights['num_fields'] * normalized_fields +
            weights['depth'] * normalized_depth +
            weights['type_complexity'] * normalized_type +
            weights['field_complexity'] * normalized_field
        )
        
        return round(complexity_score, 3)
        
    except Exception as e:
        print(f"Error calculating schema complexity: {e}")
        return 0.0
    
# if __name__ == "__main__":
#     schema = {
#   "Charmlaw Foods Pty Ltd": "string",
#   "Invoice No": "string",
#   "Inv Date": "string",
#   "Your Ref": "string",
#   "Page": "string",
#   "Bill To": {
#     "Customer Name": "string",
#     "Address": "string",
#     "Customer ABN No": "string"
#   },
#   "Ship To": {
#     "Customer Name": "string",
#     "Address": "string"
#   },
#   "Transport": "string",
#   "Items": {
#     "Item": {
#       "Qty": "integer",
#       "Item No.EAN": "string",
#       "Description": "string",
#       "Weight (kg)": "number",
#       "Price": "number",
#       "Disc%": "number",
#       "GST%": "number",
#       "Ex GST": "number"
#     }
#   },
#   "Signature": "string",
#   "Print Name": "string",
#   "Date": "string",
#   "Total Weight (kg)": "number",
#   "Carton Count": "integer",
#   "GST Amt": "number",
#   "Sale Amt": "number",
#   "GST Applicable": "number",
#   "GST Exempt": "number",
#   "Discount": "number",
#   "Invoice Total": "number",
#   "Bank Details": {
#     "SME": "string",
#     "BSB": "string",
#     "Account": "string",
#     "Payment Terms": "string"
#   }
# }

#     print(calculate_schema_complexity(schema))
