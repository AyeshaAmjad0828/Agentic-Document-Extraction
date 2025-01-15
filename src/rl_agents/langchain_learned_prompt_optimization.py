import sys
import os
from typing import List, Dict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from langchain_experimental import rl_chain
from sentence_transformers import SentenceTransformer
from src.utils.LLM_utils import get_completion_gpt4
from src.action_space.meta_prompting_agent import (
    clarity_strategy,
    best_practice_strategy,
    fewshot_strategy,
    LLm_feedback_strategy
)

class LearnedPromptOptimizer:
    """
    Optimizes prompts by learning from different meta-prompting strategies using RL.
    Uses contextual bandits to select and combine the most effective prompt strategies.
    """
    
    def __init__(self, llm, model_save_dir: str = None):
        self.llm = llm
        self.model_save_dir = model_save_dir
        
        # Initialize embedding model
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        
        # Define the prompt template for optimization
        self.prompt_template = rl_chain.PromptTemplate(
            template="""Given the document type: {doc_type}
And the current extraction results: {current_output}
With groundtruth: {groundtruth}

Select and combine the most effective parts of these prompt strategies:

Strategy 1 (Clarity):
{clarity_prompt}

Strategy 2 (Best Practices):
{best_practice_prompt}

Strategy 3 (Few-Shot):
{fewshot_prompt}

Strategy 4 (LLM Feedback):
{feedback_prompt}

Generate an optimized prompt that combines the strengths of these strategies.
Focus on maximizing extraction accuracy and maintaining consistent formatting.
""",
            input_variables=[
                "doc_type", "current_output", "groundtruth",
                "clarity_prompt", "best_practice_prompt", 
                "fewshot_prompt", "feedback_prompt"
            ]
        )
        
        # Initialize RL chain
        self.chain = rl_chain.PickBest.from_llm(
            llm=self.llm,
            prompt=self.prompt_template,
            auto_embed=True,
            feature_embedder=rl_chain.PickBestFeatureEmbedder(
                auto_embed=True,
                model=self.embedder
            ),
            model_save_dir=model_save_dir
        )

    def generate_strategy_variations(self, base_prompt: str, doc_type: str, 
                                  current_output: str, groundtruth: str) -> List[str]:
        """Generate variations using different meta-prompting strategies"""
        
        variations = []
        
        # Generate prompt variations using each strategy
        variations.append(clarity_strategy(base_prompt))
        variations.append(best_practice_strategy(base_prompt, doc_type))
        variations.append(fewshot_strategy(base_prompt, doc_type))
        variations.append(LLm_feedback_strategy(base_prompt, current_output, groundtruth))
        
        return variations

    def optimize_prompt(self, base_prompt: str, doc_type: str, 
                       current_output: str = None, groundtruth: str = None) -> Dict:
        """
        Optimize the base prompt using learned combinations of meta-prompting strategies.
        
        Args:
            base_prompt: Initial extraction prompt
            doc_type: Type of document being processed
            current_output: Current extraction output (if available)
            groundtruth: Expected extraction results (if available)
            
        Returns:
            Dict containing optimized prompt and metadata
        """
        
        # Generate variations using different strategies
        variations = self.generate_strategy_variations(
            base_prompt, doc_type, current_output, groundtruth
        )
        
        # Run RL chain to select and combine best elements
        response = self.chain.run(
            doc_type=rl_chain.BasedOn(doc_type),
            current_output=rl_chain.BasedOn(current_output or ""),
            groundtruth=rl_chain.BasedOn(groundtruth or ""),
            clarity_prompt=rl_chain.ToSelectFrom([variations[0]]),
            best_practice_prompt=rl_chain.ToSelectFrom([variations[1]]),
            fewshot_prompt=rl_chain.ToSelectFrom([variations[2]]),
            feedback_prompt=rl_chain.ToSelectFrom([variations[3]])
        )
        
        # Save progress of learned policy
        if self.model_save_dir:
            self.chain.save_progress()
            
        return {
            'optimized_prompt': response['response'],
            'base_variations': variations,
            'selection_metadata': response.get('selection_metadata')
        }

    def update_with_results(self, optimization_response: Dict, 
                          extraction_success: float) -> None:
        """
        Update the learned policy with delayed feedback about extraction success
        
        Args:
            optimization_response: Response from optimize_prompt()
            extraction_success: Score indicating extraction quality (0-1)
        """
        self.chain.update_with_delayed_score(
            score=extraction_success,
            chain_response=optimization_response
        )