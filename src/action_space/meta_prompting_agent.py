import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

#--------------------------------------------------------------------------------------------#

from src.utils.LLM_utils import get_completion_gpt4
from src.utils.prompt_template import PromptTemplate
#from actor_agents.document_extractor import document_extractor_agent, baseline_extractor_agent 


def clarity_strategy(actor_prompt):
    clarity_prompt = PromptTemplate(
        template="""
Review the following instruction prompt for clarity and precision in its task instructions.
Improve the clarity by:
1. Break the process into simple and concise instructions.
2. Avoid ambiguous works or phrases.
3. Use active voice for directness.
4. Use bullet points of numbered lists for easy readability.
5. Clearly state the goal or purpose of the instructions at the beginning.

### Instruction Prompt:

{{prompt}}

Only return the improved instruction prompt.
    """,
    input_variables=["prompt"]
    )
    response = get_completion_gpt4([{"role": "user", "content": clarity_prompt.format(prompt=actor_prompt)}],
                                )

    response_text = response.choices[0].message.content
    return response_text

#test
#actorprompt = "Extract data from the given invoice document."

    
def best_practice_strategy(actor_prompt, task_type):
    best_practice_prompt = PromptTemplate(
        template="""
Improve the following instruction prompt to generate more accurate results.
Adhere to prompt engineering best practices.
Make sure your improvements cater to a wide variety of form-like documents. 
Make sure the structure is clear and intuitive and contains complete information to perform the given task of {{task_type}}.


### Instruction Prompt:
'''

{{prompt}}

    '''
Only return the improved instruction prompt.
    """,
    input_variables=["prompt"]
    )
    response = get_completion_gpt4([{"role": "user", "content": best_practice_prompt.format(prompt=actor_prompt, task_type=task_type)}],
                                )

    response_text = response.choices[0].message.content
    return response_text


def fewshot_strategy(actor_prompt, task_type):
    fewshot_prompt = PromptTemplate(
        template="""
Improve the following instruction prompt by using few-shot prompting technique. 
Include a maximum of 3 examples related to the task the LLM is instructed to perform.
Make sure the structure is clear and intuitive and contains complete information to perform the given task of {{task_type}}.


### Instruction Prompt:
'''

{{prompt}}

'''
Only return the improved instruction prompt.

    """,
    input_variables=["task_type", "prompt"]
    )
    response = get_completion_gpt4([{"role": "user", "content": fewshot_prompt.format(prompt=actor_prompt,task_type=task_type)}],
                                )

    response_text = response.choices[0].message.content
    return response_text


def compare_groundtruth_feedback(actor_prompt, generated_output, ground_truth):

    feedback_prompt = PromptTemplate(
        template="""
Evaluate the following instruction prompt to the LLM, the corresponding output it generated, and the expected ground truth.
Suggest feedback to improve the instruction prompt such that the LLM generated output becomes same as the expected ground truth. 

[BEGIN DATA]
************
[Instruction Prompt]: 
{{prompt}}
************
[Generated Output]: 
{{output}}
************
[Expected Ground Truth]: 
{{ground_truth}}
************
[END DATA]


Only return the feedback to improve the instruction prompt.
     """,
     input_variables=["prompt", "output", "ground_truth"]
    )

    response = get_completion_gpt4([{"role": "user", "content": feedback_prompt.format(prompt=actor_prompt,
    output=generated_output,ground_truth=ground_truth)}],
                                )

    response_text = response.choices[0].message.content
    return response_text




def LLm_feedback_strategy(actor_prompt, generated_output, ground_truth):
    feedback = compare_groundtruth_feedback(actor_prompt, generated_output, ground_truth)
    use_feedback_prompt = PromptTemplate(
        template="""
Based on the feedback provided, revise the following instruction prompt to improve its effectiveness. The feedback highlights the areas that needs enhancements to imrpove the accuracy of the output. 
Make sure to incorporate these suggestions into the revised instruction prompt for the task of classification, or schema building, or data extraction.

### Original Instruction Prompt:

{{prompt}}

### Feedback:

{{feedback}}


Please provide a new, optimized instruction prompt that addresses the feedback while retaining the original intent of the prompt.
Only return the improved instruction prompt.
    """,
    input_variables=["prompt", "feedback"]

    )
    response = get_completion_gpt4([{"role": "user", "content": use_feedback_prompt.format(prompt=actor_prompt,feedback=feedback)}],
                                )

    response_text = response.choices[0].message.content
    return response_text


def no_change_strategy(actor_prompt):
    no_change_prompt = PromptTemplate(
        template="""
DO NOT make any changes to the following instruction prompt. 

### Instruction Prompt:
'''

{{prompt}}

'''
Just return the instruction prompt as it is.
    """,
    input_variables=["prompt"]
    )
    response = get_completion_gpt4([{"role": "user", "content": no_change_prompt.format(prompt=actor_prompt)}],
                                )

    response_text = response.choices[0].message.content
    return response_text

def adjust_prompt(actor_prompt, task_type, state, action, generated_output, groundtruth):
    """
    Adjusts the prompt based on the selected action.
    
    Args:
        state (list): Current state of the environment, including exact match and similarity scores.
        action (int): The action ID representing a specific prompt adjustment.
    
    Returns:
        str: Updated prompt after applying the action.
    """
    
    if action == 0:
        updated_prompt = best_practice_strategy(actor_prompt=actor_prompt, task_type=task_type)
    elif action == 1:
        updated_prompt = clarity_strategy(actor_prompt=actor_prompt)
    elif action == 2:
        updated_prompt = fewshot_strategy(actor_prompt=actor_prompt, task_type=task_type)
    elif action == 3:
        updated_prompt = no_change_strategy(actor_prompt=actor_prompt)
    elif action == 4:
        updated_prompt = LLm_feedback_strategy(actor_prompt=actor_prompt, generated_output=generated_output, ground_truth=groundtruth)
    else:
        raise ValueError("Invalid action ID. Must be between 0 and 4.")

    return updated_prompt


