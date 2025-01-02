import sys
import os

# Print the module search path
print("Python search paths:")
for path in sys.path:
    print(path)
    


# Add the project root to sys.path
project_root = r"C:\Users\ayesha.amjad\OneDrive - Astera Software\Documents\GitHub\Unstructured-Data-Extraction"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Python search paths:")
for path in sys.path:
    print(path)

from src.utils.prompt_template import PromptTemplate
from src.utils.LLM_utils import get_completion_gpt4
    