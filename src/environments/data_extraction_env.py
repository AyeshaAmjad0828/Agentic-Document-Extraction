import sys
import os


# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

#--------------------------------------------------------------------------------------------#

import gymnasium as gym
import json
import numpy as np

from src.utils.LLM_utils import get_llm_completion
from src.utils.jsonparser_utils import clean_llm_output
from src.actor_agents.document_extractor import document_extractor_agent 
from src.action_space.meta_prompting_agent import adjust_prompt
from src.evaluation.scoring import calculate_exact_match, calculate_similarity, calculate_semantic_match_score


class DataExtractionEnvBase(gym.Env):
    """
    Custom Gymnasium environment for the Agentic Data Extraction process.
    The agent interacts with the environment to improve data extraction accuracy.
    Observations include Exact Match Score and Similarity Score.
    Actions represent adjustments to prompt engineering strategies.
    """
    def __init__(self, baseprompt, document_type, document, schema, groundtruth, llm_choice):
        super(DataExtractionEnvBase, self).__init__()
        self.action_space = gym.spaces.Discrete(5)  # 5 possible prompt adjustments
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)  # [Exact Match, Semantic Match, Similarity]
        
        # Store document extraction related data
        self.baseprompt = baseprompt
        self.document_type = document_type
        self.document = document
        self.schema = schema
        self.groundtruth = groundtruth
        self.llm_choice = llm_choice
        self.task_type = "form-like document data extraction"
        self.state = None
        self.current_step = 0
        self.max_steps = 5  # Add reasonable limit




    def step(self, action):
        self.current_step += 1


        print(f"\n............................. DATA EXTRACTION - ITERATION {self.current_step} BEGINS.....................................")
        
        # Get updated prompt using meta-prompting agent
        updated_prompt = adjust_prompt(
            actor_prompt=self.baseprompt,
            task_type=self.task_type,
            state=self.state,
            action=action,
            generated_output=self.last_output if hasattr(self, 'last_output') else None,
            groundtruth=self.groundtruth
        )
        self.current_prompt = updated_prompt

        resolved_updated_prompt = document_extractor_agent(updated_prompt, self.document_type, self.document, self.schema)


        # Generate new output using the updated prompt
        self.last_output = clean_llm_output(get_llm_completion([{"role": "user", "content": resolved_updated_prompt}], llm_choice=self.llm_choice,
                                                                response_format={ "type": "json_object" },).choices[0].message.content)
        
        print(f"\nUpdated Prompt: {resolved_updated_prompt}")
        print("Updated Output:", self.last_output)


        # Calculate scores (you'll need to implement these scoring functions)
        exact_match_score = calculate_exact_match(self.last_output, self.groundtruth)
        semantic_match_score = calculate_semantic_match_score(self.last_output, self.groundtruth, self.llm_choice)
        similarity_score = calculate_similarity(self.last_output, self.groundtruth)

        # Update state
        self.state = np.array([exact_match_score, semantic_match_score, similarity_score], dtype=np.float32)

        # Calculate reward
        reward = exact_match_score * semantic_match_score * similarity_score - abs(action - 2) * 0.05

        # Check if task is complete
        done = bool(exact_match_score >= 0.90 and semantic_match_score >= 0.95 and similarity_score >= 0.95)

        # Create info dictionary with useful information
        info = {
            'exact_match': exact_match_score,
            'semantic_match': semantic_match_score,
            'similarity': similarity_score,
            'State' : self.state,
            'Updated Prompt': resolved_updated_prompt,
            'Updated Output': self.last_output
        }

        print(f"State: {self.state}")
        print(f"Done: {done}")

        # Add early success condition
        if hasattr(self, 'last_output') and all([
            exact_match_score >= 0.95,
            semantic_match_score >= 0.95,
            similarity_score >= 0.95
        ]):
            return self.state, reward, True, info

        return self.state, reward, done, info

    def reset(self):
        """Reset the environment to initial state"""
        # Reset prompt to initial state

        self.current_prompt = self.baseprompt
        self.resolved_current_prompt = document_extractor_agent(self.current_prompt, self.document_type, self.document, self.schema)

        self.last_output = clean_llm_output(get_llm_completion([{"role": "user", "content": self.resolved_current_prompt}], llm_choice=self.llm_choice,
                                                                response_format={ "type": "json_object" },).choices[0].message.content)
        print(f"Initial Prompt:\n {self.resolved_current_prompt}")
        print(f"Initial Output:\n {self.last_output}")
        
        # Initialize all metrics
        exact_match_score = calculate_exact_match(self.last_output, self.groundtruth)
        semantic_match_score = calculate_semantic_match_score(self.last_output, self.groundtruth, self.llm_choice)
        similarity_score = calculate_similarity(self.last_output, self.groundtruth)

        self.state = np.array([exact_match_score, semantic_match_score, similarity_score], dtype=np.float32)  # Initial values for all metrics
        return self.state, {}  # Return state and empty info dict for gymnasium compatibility
    


class DataExtractionEnvIterative(gym.Env):
    """
    Custom Gymnasium environment for the Agentic Data Extraction process.
    The agent interacts with the environment to improve data extraction accuracy by iteratively updating observation.
    Observations include Exact Match Score and Similarity Score.
    Actions represent adjustments to prompt engineering strategies.
    """

    def __init__(self, baseprompt, document_type, document, schema, groundtruth, llm_choice, max_steps=5):
        super(DataExtractionEnvIterative, self).__init__()
        self.action_space = gym.spaces.Discrete(5)  # 5 possible prompt adjustments
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=1, 
            shape=(3,),  # [Exact Match, Semantic Match, Similarity]
            dtype=np.float32
        )
        
        # Store document extraction related data
        self.baseprompt = baseprompt
        self.document_type = document_type
        self.document = document
        self.schema = schema
        self.groundtruth = groundtruth
        self.llm_choice = llm_choice
        self.current_prompt = baseprompt
        self.task_type = "form-like document data extraction"
        
        # Track best scores and results
        self.best_exact_match = 0.0
        self.best_semantic_match = 0.0
        self.best_similarity = 0.0
        self.best_output = None
        self.best_prompt = None
        
        # Track consecutive non-improvements
        self.non_improvement_count = 0
        self.max_non_improvements = 2  # Stop after 2 consecutive non-improvements
        self.current_step = 0
        
        self.state = None
        self.max_steps = max_steps


    def step(self, action):
        self.current_step += 1

        print(f"\n............................. DATA EXTRACTION - ITERATION {self.current_step} BEGINS.....................................")
        
        # Get updated prompt using meta-prompting agent
        updated_prompt = adjust_prompt(
            actor_prompt=self.current_prompt,
            task_type=self.task_type,
            state=self.state,
            action=action,
            generated_output=self.last_output if hasattr(self, 'last_output') else None,
            groundtruth=self.groundtruth
        )
        self.current_prompt = updated_prompt

        resolved_updated_prompt = document_extractor_agent(updated_prompt, 
                                                           self.document_type, 
                                                           self.document, self.schema)

        # Generate new output
        self.last_output = get_llm_completion([{"role": "user", "content": resolved_updated_prompt}], llm_choice=self.llm_choice,
                                                                response_format={ "type": "json_object" },).choices[0].message.content

        print(f"\nStep {self.current_step}")
        print(f"\nUpdated Prompt: {resolved_updated_prompt}")
        print("Updated Output:\n", self.last_output)


        # Calculate scores
        exact_match_score = calculate_exact_match(self.last_output, self.groundtruth)
        semantic_match_score = calculate_semantic_match_score(self.last_output, self.groundtruth, self.llm_choice)
        similarity_score = calculate_similarity(self.last_output, self.groundtruth)
        
        # Update state
        self.state = np.array([exact_match_score, semantic_match_score, similarity_score], dtype=np.float32)
        
        # Calculate combined score for comparison
        current_combined_score = exact_match_score + similarity_score + semantic_match_score
        best_combined_score = self.best_exact_match + self.best_similarity + self.best_semantic_match
        
        # Check if current scores are better than best scores
        if current_combined_score > best_combined_score:
            self.best_exact_match = exact_match_score
            self.best_semantic_match = semantic_match_score
            self.best_similarity = similarity_score
            self.best_output = self.last_output
            self.best_prompt = self.current_prompt
            self.non_improvement_count = 0
            improved = True
        else:
            self.non_improvement_count += 1
            improved = False

        # Calculate reward
        reward = current_combined_score - best_combined_score

        # Determine if we should terminate
        done = (self.non_improvement_count >= self.max_non_improvements) or (self.current_step >= self.max_steps)

        # Create info dictionary with useful information
        info = {
            'exact_match': exact_match_score,
            'similarity': similarity_score,
            'semantic_match': semantic_match_score,
            'best_exact_match': self.best_exact_match,
            'best_similarity': self.best_similarity,
            'best_semantic_match': self.best_semantic_match,
            'improved': improved,
            'non_improvement_count': self.non_improvement_count,
            'steps': self.current_step
        }


        print(f"Current Scores - Exact Match: {exact_match_score:.4f}, Semantic Match: {semantic_match_score:.4f}, Similarity: {similarity_score:.4f}")
        print(f"Best Scores    - Exact Match: {self.best_exact_match:.4f}, Semantic Match: {self.best_semantic_match:.4f}, Similarity: {self.best_similarity:.4f}")

        
        if done:
            print("\nTerminating due to:", end=" ")
            if self.current_step >= self.max_steps:
                print("maximum steps reached")
            else:
                print("no improvements in last two updates")
            print(f"Best Results Achieved:")
            print(f"Exact Match: {self.best_exact_match:.4f}")
            print(f"Semantic Match: {self.best_semantic_match:.4f}")
            print(f"Similarity: {self.best_similarity:.4f}")
            print(f"Best Prompt:\n {self.best_prompt}")
            print(f"\nBest Output:\n {self.best_output}")


        return self.state, reward, done, info

    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0
        self.non_improvement_count = 0
        self.current_prompt = self.baseprompt
        self.resolved_current_prompt = document_extractor_agent(self.current_prompt, self.document_type, self.document, self.schema)
        
        # Generate initial output
        self.last_output = get_llm_completion([{"role": "user", "content": self.resolved_current_prompt}], llm_choice=self.llm_choice,
                                                                response_format={ "type": "json_object" },).choices[0].message.content

        print(f"Start Prompt:\n {self.resolved_current_prompt}")
        print(f"Start Output:\n {self.last_output}")

        # Debug the state of outputs before similarity calculation
        print("Environment Reset - Groundtruth:", self.groundtruth)
        
            
        # Calculate initial scores
        exact_match_score = calculate_exact_match(self.last_output, self.groundtruth)
        semantic_match_score = calculate_semantic_match_score(self.last_output, self.groundtruth, self.llm_choice)
        similarity_score = calculate_similarity(self.last_output, self.groundtruth)


        # Initialize best scores with initial scores
        self.best_exact_match = exact_match_score
        self.best_semantic_match = semantic_match_score
        self.best_similarity = similarity_score
        self.best_output = self.last_output
        self.best_prompt = self.current_prompt
        
        self.state = np.array([exact_match_score, semantic_match_score, similarity_score], dtype=np.float32)
        return self.state, {}

    def get_best_results(self):
        """Return the best results achieved during the episode"""
        return {
            'best_exact_match': self.best_exact_match,
            'best_semantic_match': self.best_semantic_match,
            'best_similarity': self.best_similarity,
            'best_output': self.best_output,
            'best_prompt': self.best_prompt
        }
    

class DataExtractionEnvStepCount(gym.Env):
    """
    Custom Gymnasium environment for the Agentic Data Extraction process.
    The agent interacts with the environment to improve data extraction accuracy.
    Observations include Exact Match Score and Similarity Score.
    Actions represent adjustments to prompt engineering strategies.
    """
    def __init__(self, baseprompt, document_type, document, schema, groundtruth, llm_choice):
        super(DataExtractionEnvStepCount, self).__init__()
        self.action_space = gym.spaces.Discrete(5)  # 5 possible prompt adjustments
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)  # [Exact Match, Similarity]
        
        # Store document extraction related data
        self.baseprompt = baseprompt
        self.document_type = document_type
        self.document = document
        self.schema = schema
        self.groundtruth = groundtruth
        self.llm_choice = llm_choice
        self.current_prompt = baseprompt  # Initial prompt
        self.task_type = "form-like document data extraction"
        self.state = None


        # Add termination thresholds
        self.exact_match_threshold = 0.95
        self.similarity_threshold = 0.95
        self.max_steps = 5  # Add maximum steps limit
        self.current_step = 0



    def step(self, action):
        
        
        """Execute one step in the environment"""
        self.current_step += 1
        
        print(f"\n............................. DATA EXTRACTION - ITERATION {self.current_step} BEGINS.....................................")

        # Get updated prompt using meta-prompting agent
        updated_prompt = adjust_prompt(
            actor_prompt=self.baseprompt,
            task_type=self.task_type,
            state=self.state,
            action=action,
            generated_output=self.last_output if hasattr(self, 'last_output') else None,
            groundtruth=self.groundtruth
        )
        self.current_prompt = updated_prompt

        resolved_updated_prompt = document_extractor_agent(updated_prompt, self.document_type, self.document, self.schema)


        # Generate new output using the updated prompt
        self.last_output = clean_llm_output(get_llm_completion([{"role": "user", "content": resolved_updated_prompt}], llm_choice=self.llm_choice,
                                                                response_format={ "type": "json_object" },).choices[0].message.content)

        print(f"\nStep {self.current_step}/{self.max_steps}")
        print(f"\nUpdated Prompt: {resolved_updated_prompt}")
        print("Updated Output:", self.last_output)

        # Calculate scores (you'll need to implement these scoring functions)
        exact_match_score = calculate_exact_match(self.last_output, self.groundtruth)
        similarity_score = calculate_similarity(self.last_output, self.groundtruth)

        # Update state
        self.state = np.array([exact_match_score, similarity_score], dtype=np.float32)

        # Calculate reward
        step_penalty = -0.01 * self.current_step
        reward = (exact_match_score * similarity_score - abs(action - 2) * 0.05) + step_penalty

         # Check termination conditions
        done = bool(
            (exact_match_score >= self.exact_match_threshold and 
             similarity_score >= self.similarity_threshold) or
            self.current_step >= self.max_steps
        )

        # Add info dict with useful debugging information
        info = {
            'exact_match': exact_match_score,
            'similarity': similarity_score,
            'steps': self.current_step,
            'max_steps_reached': self.current_step >= self.max_steps,
            'success': exact_match_score >= self.exact_match_threshold and similarity_score >= self.similarity_threshold
        }


        print(f"State: {self.state}")
        print(f"Done: {done}")


        return self.state, reward, done, info



    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0  # Reset step counter


        self.current_prompt = self.baseprompt
        self.resolved_current_prompt = document_extractor_agent(self.current_prompt, self.document_type, self.document, self.schema)

        self.last_output = clean_llm_output(get_llm_completion([{"role": "user", "content": self.resolved_current_prompt}], llm_choice=self.llm_choice,
                                                                response_format={ "type": "json_object" },).choices[0].message.content)
        print(f"Start Prompt:\n {self.resolved_current_prompt}")
        print(f"Start Output:\n {self.last_output}")


        exact_match_score = calculate_exact_match(self.last_output, self.groundtruth)
        similarity_score = calculate_similarity(self.last_output, self.groundtruth)

        # Initialize all metrics
        self.state = np.array([exact_match_score, similarity_score], dtype=np.float32)  # Initial values for all metrics
        return self.state, {}  # Return state and empty info dict for gymnasium compatibility
    