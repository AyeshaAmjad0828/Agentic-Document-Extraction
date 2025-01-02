
#pip install openai
#pip install tiktoken

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

#--------------------------------------------------------------------------------------------#

import json
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.LLM_utils import openai_embedding





def compute_cosine_similarity(extractedtext, filetext):
    

    if isinstance(filetext, dict):
        filetext = json.dumps(filetext)
    if isinstance(extractedtext, dict):
        extractedtext = json.dumps(extractedtext)
           
    file_embed = openai_embedding(filetext)
    output_embed = openai_embedding(extractedtext)

    embedding1 = np.array(file_embed).reshape(1, -1)
    embedding2 = np.array(output_embed).reshape(1, -1)

    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    
    return similarity







