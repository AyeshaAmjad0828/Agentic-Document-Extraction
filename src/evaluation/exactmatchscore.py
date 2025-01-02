import re
import json
from fuzzywuzzy import fuzz


#load data
# with open('..\Samples\invoice json.txt', 'r') as json_file:
#     invoice_json = json.load(json_file)

# with open('..\Samples\\unstructured invoice.txt', 'r') as unstructured_file:
#     invoice_text = unstructured_file.read()


def calculate_match_score(generated_output, groundtruth):
    """Calculate exact match score between generated output and groundtruth"""
    # Convert string to dict if needed
    if isinstance(generated_output, str):
        try:
            generated_output = json.loads(generated_output)
        except json.JSONDecodeError as e:
            print(f"Error parsing generated output: {e}")
            return 0.0

    total_fields = 0
    matched_fields = 0

    def normalize_value(value):
        """Normalize values for comparison"""
        if isinstance(value, str):
            # Remove '$' and whitespace, convert to lowercase
            return value.replace('$', '').strip().lower()
        return value

    def compare_values(val1, val2):
        """Compare two values, handling different types"""
        norm1 = normalize_value(val1)
        norm2 = normalize_value(val2)
        
        # Handle numeric comparisons
        try:
            if isinstance(norm1, str) and isinstance(norm2, (int, float)):
                norm1 = float(norm1)
            if isinstance(norm2, str) and isinstance(norm1, (int, float)):
                norm2 = float(norm2)
        except ValueError:
            pass

        if isinstance(norm1, (int, float)) and isinstance(norm2, (int, float)):
            return abs(float(norm1) - float(norm2)) < 0.01
        return str(norm1) == str(norm2)

    def check_nested_structure(gen_data, true_data):
        nonlocal total_fields, matched_fields
        
        if isinstance(gen_data, dict) and isinstance(true_data, dict):
            for key in true_data:
                total_fields += 1
                if key in gen_data:
                    if isinstance(true_data[key], (dict, list)):
                        check_nested_structure(gen_data[key], true_data[key])
                    else:
                        if compare_values(gen_data[key], true_data[key]):
                            matched_fields += 1
                        else:
                            print(f"Mismatch for key '{key}': Generated='{gen_data[key]}', Expected='{true_data[key]}'")
                else:
                    print(f"Missing key in generated output: {key}")

        elif isinstance(gen_data, list) and isinstance(true_data, list):
            # For lists (like Line_Items), compare items at same indices
            for i, (gen_item, true_item) in enumerate(zip(gen_data, true_data)):
                check_nested_structure(gen_item, true_item)

    # Start the recursive comparison
    check_nested_structure(generated_output, groundtruth)

    if total_fields == 0:
        print("Warning: No fields to compare")
        return 0.0

    score = matched_fields / total_fields
    print(f"\nExact Match Details:")
    print(f"Total fields: {total_fields}")
    print(f"Matched fields: {matched_fields}")
    
    return score

def debug_exact_match(generated_output, groundtruth):
    """Debug function to compare generated output with ground truth"""
    print("\n=== Debug Exact Match ===")
    
    # Convert string to dict if needed
    if isinstance(generated_output, str):
        try:
            generated_output = json.loads(generated_output)
        except json.JSONDecodeError as e:
            print(f"Error parsing generated output: {e}")
            return
    
    print("\nKeys Comparison:")
    print(f"Ground Truth Keys: {sorted(groundtruth.keys())}")
    print(f"Generated Keys: {sorted(generated_output.keys())}")
    
    print("\nValue Comparisons:")
    for key in groundtruth.keys():
        if key in generated_output:
            ground_val = groundtruth[key]
            gen_val = generated_output[key]
            match = ground_val == gen_val
            print(f"\nKey: {key}")
            print(f"Ground Truth: {ground_val} ({type(ground_val)})")
            print(f"Generated: {gen_val} ({type(gen_val)})")
            print(f"Match: {match}")
            
            if not match and isinstance(ground_val, (list, dict)):
                print("Detailed comparison for nested structure:")
                print(f"Ground Truth: {json.dumps(ground_val, indent=2)}")
                print(f"Generated: {json.dumps(gen_val, indent=2)}")
        else:
            print(f"\nKey missing in generated output: {key}")

    return


def check_values_in_text(generated, original):
    found_values = {}

    if isinstance(original, dict):
        original = json.dumps(original)

    for key, value in generated.items():
        if isinstance(value, list):
            # Handle the collection of child items (Line_Items)
            for idx, item in enumerate(value):
                for sub_key, sub_value in item.items():
                    sub_value_str = str(sub_value)
                    if sub_value_str in original:
                        found_values[f"Line_Item_{idx}_{sub_key}"] = True
                    else:
                        found_values[f"Line_Item_{idx}_{sub_key}"] = False
        else:
            # Handle parent-level values
            value_str = str(value)
            if value_str in original:
                found_values[key] = True
            else:
                found_values[key] = False

    return found_values




def fuzzy_match_in_text(generated, original, threshold=90):
    found_values = {}

    if isinstance(original, dict):
        original = json.dumps(original)

    for key, value in generated.items():
        if isinstance(value, list):
            # Handle the collection of child items (Line_Items)
            for idx, item in enumerate(value):
                for sub_key, sub_value in item.items():
                    sub_value_str = str(sub_value)
                    match_score = fuzz.partial_ratio(sub_value_str.lower(), original.lower())
                    found_values[f"Line_Item_{idx}_{sub_key}"] = match_score >= threshold
        else:
            # Handle parent-level values
            value_str = str(value)
            match_score = fuzz.partial_ratio(value_str.lower(), original.lower())
            found_values[key] = match_score >= threshold

    return found_values


