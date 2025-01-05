import sys
import os


# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

#--------------------------------------------------------------------------------------------#



import gymnasium as gym
import numpy as np
from src.actor_agents.schema_builder import schema_building_with_llm
from src.action_space.meta_prompting_agent import adjust_prompt



class SchemaBuilderEnv(gym.Env):
    def __init__(self, baseprompt, document_text, groundtruth, max_steps=5):
        super(SchemaBuilderEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(5)  # 5 possible prompt adjustments
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=float('inf'),  # Perplexity can be any positive number
            shape=(1,),  # [Perplexity]
            dtype=np.float32
        )
        
        # Store schema building related data
        self.baseprompt = baseprompt
        self.document_text = document_text
        self.groundtruth = groundtruth
        self.current_prompt = baseprompt
        self.task_type = "json schema-building"
        
        # Track best scores and results
        self.best_perplexity = float('inf')
        self.best_schema = None
        self.best_prompt = None
        
        # Track consecutive non-improvements
        self.non_improvement_count = 0
        self.max_non_improvements = 2
        self.current_step = 0
        
        self.state = None

        self.max_steps = max_steps

    def step(self, action):
        self.current_step += 1

        print(f"\n............................. SCHEMA BUILDER - ITERATION {self.current_step} BEGINS.....................................")

        # Get updated prompt using meta-prompting agent
        updated_prompt = adjust_prompt(
            actor_prompt=self.current_prompt,
            task_type=self.task_type,
            state=self.state,
            action=action,
            groundtruth=self.groundtruth,
            generated_output=self.last_output if hasattr(self, 'last_output') else None
        )
        self.current_prompt = updated_prompt

        # Generate new schema and get scores
        _, self.last_output, perplexity_score = schema_building_with_llm(updated_prompt, self.document_text)

        
        print(f"\nStep {self.current_step}")
        print(f"\nUpdated Prompt: {_}")
        print("Updated Schema:\n", self.last_output)

        # Update state (lower perplexity is better)
        self.state = np.array([perplexity_score], dtype=np.float32)
        
        # Check if current scores are better
        # Check if current score is better than best score
        if perplexity_score < self.best_perplexity:
            self.best_perplexity = perplexity_score
            self.best_schema = self.last_output
            self.best_prompt = self.current_prompt
            self.non_improvement_count = 0
            improved = True
        else:
            self.non_improvement_count += 1
            improved = False

        # Calculate reward (negative because we want to minimize perplexity)
        reward = self.best_perplexity - perplexity_score

        # Determine if we should terminate
        done = (self.non_improvement_count >= self.max_non_improvements) or (self.current_step >= self.max_steps)

        # Create info dictionary
        info = {
            'perplexity': perplexity_score,
            'best_perplexity': self.best_perplexity,
            'improved': improved,
            'non_improvement_count': self.non_improvement_count,
            'steps': self.current_step
        }

        print(f"Current Perplexity: {perplexity_score:.4f}")
        print(f"Best Perplexity: {self.best_perplexity:.4f}")
        print(f"Improved: {improved}")
        print(f"Non-improvement count: {self.non_improvement_count}")
        
        if done:
            print("\nTerminating due to:", end=" ")
            if self.current_step >= self.max_steps:
                print("maximum steps reached")
            else:
                print("no improvements in last two updates")
            print(f"Best Results Achieved:")
            print(f"Perplexity: {self.best_perplexity:.4f}")
            print("\nBest Schema:")
            print(self.best_schema)

        return self.state, reward, done, info

    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0
        self.non_improvement_count = 0
        self.current_prompt = self.baseprompt
        
        # Generate initial schema and calculate scores
        _, self.last_output, perplexity_score = schema_building_with_llm(self.current_prompt, self.document_text)
        
        print(f"\nInitial Prompt: {_}")
        print("Initial Schema:\n", self.last_output)
        print(f"Initial Perplexity: {perplexity_score:.4f}")

        # Initialize best scores
        self.best_perplexity = perplexity_score
        self.best_schema = self.last_output
        self.best_prompt = self.current_prompt
        
        self.state = np.array([perplexity_score], dtype=np.float32)
        return self.state, {}
    
    def get_best_results(self):
        """Return the best results achieved during the episode"""
        return {
            'best_perplexity': self.best_perplexity,
            'best_schema': self.best_schema,
            'best_prompt': self.best_prompt
        }
    


class SchemaBuilderEnvSemantic(gym.Env):
    def __init__(self, baseprompt, document_text, groundtruth, max_steps=5):
        super(SchemaBuilderEnvSemantic, self).__init__()
        self.action_space = gym.spaces.Discrete(5)  # 5 possible prompt adjustments
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=1.0,  # Semantic match score between 0 and 1
            shape=(1,),
            dtype=np.float32
        )
        
        # Store schema building related data
        self.baseprompt = baseprompt
        self.document_text = document_text
        self.groundtruth = groundtruth
        self.current_prompt = baseprompt
        self.task_type = "json schema-building"
        
        # Track best scores and results
        self.best_semantic_match = 0.0  # Higher is better for semantic match
        self.best_schema = None
        self.best_prompt = None
        
        # Track consecutive non-improvements
        self.non_improvement_count = 0
        self.max_non_improvements = 2
        self.current_step = 0
        
        self.state = None
        self.max_steps = max_steps

    def step(self, action):
        self.current_step += 1

        print(f"\n............................. SCHEMA BUILDER - ITERATION {self.current_step} BEGINS.....................................")

        # Get updated prompt using meta-prompting agent
        updated_prompt = adjust_prompt(
            actor_prompt=self.current_prompt,
            task_type=self.task_type,
            state=self.state,
            action=action,
            groundtruth=self.groundtruth,
            generated_output=self.last_output if hasattr(self, 'last_output') else None
        )
        self.current_prompt = updated_prompt

        # Generate new schema and get scores
        _, self.last_output, semantic_score = schema_building_with_llm(
            updated_prompt, 
            self.document_text,
            evaluation_metric='semantic'  # Specify semantic matching
        )

        print(f"\nStep {self.current_step}")
        print(f"\nUpdated Prompt: {_}")
        print("Updated Schema:\n", self.last_output)

        # Update state
        self.state = np.array([semantic_score], dtype=np.float32)
        
        # Check if current score is better than best score
        # For semantic match, higher is better
        if semantic_score > self.best_semantic_match:
            self.best_semantic_match = semantic_score
            self.best_schema = self.last_output
            self.best_prompt = self.current_prompt
            self.non_improvement_count = 0
            improved = True
        else:
            self.non_improvement_count += 1
            improved = False

        # Calculate reward (positive because we want to maximize semantic match)
        reward = semantic_score - self.best_semantic_match

        # Determine if we should terminate
        done = (self.non_improvement_count >= self.max_non_improvements) or (self.current_step >= self.max_steps)

        # Create info dictionary
        info = {
            'semantic_match': semantic_score,
            'best_semantic_match': self.best_semantic_match,
            'improved': improved,
            'non_improvement_count': self.non_improvement_count,
            'steps': self.current_step
        }

        print(f"Current Semantic Match: {semantic_score:.4f}")
        print(f"Best Semantic Match: {self.best_semantic_match:.4f}")
        print(f"Improved: {improved}")
        print(f"Non-improvement count: {self.non_improvement_count}")
        
        if done:
            print("\nTerminating due to:", end=" ")
            if self.current_step >= self.max_steps:
                print("maximum steps reached")
            else:
                print("no improvements in last two updates")
            print(f"Best Results Achieved:")
            print(f"Semantic Match: {self.best_semantic_match:.4f}")
            print("\nBest Schema:")
            print(self.best_schema)

        return self.state, reward, done, info

    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0
        self.non_improvement_count = 0
        self.current_prompt = self.baseprompt
        
        # Generate initial schema and calculate scores
        _, self.last_output, semantic_score = schema_building_with_llm(
            self.current_prompt, 
            self.document_text,
            evaluation_metric='semantic'
        )
        
        print(f"\nInitial Prompt: {_}")
        print("Initial Schema:\n", self.last_output)
        print(f"Initial Semantic Match: {semantic_score:.4f}")

        # Initialize best scores
        self.best_semantic_match = semantic_score
        self.best_schema = self.last_output
        self.best_prompt = self.current_prompt
        
        self.state = np.array([semantic_score], dtype=np.float32)
        return self.state, {}
    
    def get_best_results(self):
        """Return the best results achieved during the episode"""
        return {
            'best_semantic_match': self.best_semantic_match,
            'best_schema': self.best_schema,
            'best_prompt': self.best_prompt
        }