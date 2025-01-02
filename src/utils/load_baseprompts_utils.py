
import os



def load_prompt_from_file(filename):
    """Load prompt template from a text file"""
    # Get the absolute path to the project root directory
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    
    # Construct path to the prompts directory
    prompt_path = os.path.join(project_root, 'src', 'actor_agents', 'Prompts', filename)
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found at: {prompt_path}")