#pip install langchain_experimental vowpal_wabbit_next 

import sys
import os
from typing import List, Dict, Union, Any

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

#--------------------------------------------------------------------------------------------#

from langchain_experimental import rl_chain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from src.utils.LLM_utils import openai_embedding
from src.action_space.meta_prompting_agent import (
    clarity_strategy,
    best_practice_strategy,
    fewshot_strategy,
    no_change_strategy,    
    LLm_feedback_strategy
)
from dotenv import load_dotenv

load_dotenv('.env')

class OpenAIEmbedder:
    """Custom embedder class that uses OpenAI's embedding function"""
    
    def __init__(self):
        pass
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using OpenAI's embedding function"""
        embeddings = []
        for text in texts:
            embedding = openai_embedding(text)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                # Fallback to zero vector if embedding fails
                embeddings.append([0.0] * 1536)  # OpenAI ada-002 has 1536 dimensions
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using OpenAI's embedding function"""
        embedding = openai_embedding(text)
        if embedding is not None:
            return embedding
        # Fallback to zero vector if embedding fails
        return [0.0] * 1536  # OpenAI ada-002 has 1536 dimensions
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Alias for embed_documents to match the expected interface"""
        return self.embed_documents(texts)

class LearnedPromptOptimizer:
    """
    Optimizes prompts by learning from different meta-prompting strategies using RL.
    Uses contextual bandits to select and combine the most effective prompt strategies.
    """
    
    def __init__(self, llm=None, model_save_dir: str = None):
        # Initialize OpenAI LLM
        self.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if llm is None else llm
        self.model_save_dir = model_save_dir
        
        # Initialize OpenAI embedder
        self.embedder = OpenAIEmbedder()
        
        # Define the prompt template for optimization
        self.prompt_template = PromptTemplate(
            template="""Given the document type: {doc_type}
And the current extraction results: {current_output}
With groundtruth: {groundtruth}

Here is a proposed prompt strategy:
{prompt_strategy}

Evaluate this prompt strategy and generate an optimized prompt that maximizes extraction accuracy and maintains consistent formatting.""",
            input_variables=[
                "doc_type", "current_output", "groundtruth",
                "prompt_strategy"
            ]
        )

        # Define scoring criteria for the selection scorer
        scoring_criteria_template = (
            "Given the document type {doc_type} and groundtruth {groundtruth}, "
            "rank how effective this prompt strategy would be: {prompt_strategy}"
        )
        
        # Configure VowpalWabbit command-line arguments
        vw_cmd = [
            "--cb_explore_adf",  # Use contextual bandits with action-dependent features
            "--interactions", "all",  # Enable all feature interactions
            "--learning_rate", "0.1",  # Set learning rate
            "--cb_type", "mtr",  # Use Multiclass Tree Reduction
            "--epsilon", "0.2",  # Set exploration rate
            "--quiet"  # Suppress VW output
        ]
        
        # Initialize RL chain with custom selection scorer
        self.chain = rl_chain.PickBest.from_llm(
            llm=self.llm,
            prompt=self.prompt_template,
            selection_scorer=rl_chain.AutoSelectionScorer(
                llm=self.llm,
                scoring_criteria_template_str=scoring_criteria_template
            ),
            auto_embed=True,
            feature_embedder=rl_chain.PickBestFeatureEmbedder(
                auto_embed=True,
                model=self.embedder
            ),
            model_save_dir=model_save_dir,
            vw_cmd=vw_cmd  # Pass VW command-line arguments
        )

    def generate_strategy_variations(self, base_prompt: str, doc_type: str, 
                                  current_output: str, groundtruth: str) -> List[str]:
        """Generate variations using different meta-prompting strategies"""
        
        variations = []
        
        # Generate prompt variations using each strategy
        variations.append(clarity_strategy(base_prompt))
        variations.append(best_practice_strategy(base_prompt, doc_type))
        variations.append(fewshot_strategy(base_prompt, doc_type))
        variations.append(no_change_strategy(base_prompt))
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
        
        # Ensure we have at least one variation
        if not variations:
            variations = [base_prompt]  # Use the base prompt if no variations
        
        # Run RL chain to select and combine best elements
        try:
            # Create input for the chain
            chain_inputs = {
                "doc_type": doc_type,  # Remove rl_chain.BasedOn wrapper
                "current_output": current_output or "",
                "groundtruth": groundtruth or "",
                "prompt_strategy": variations  # Pass variations directly
            }
            
            # Run the chain using invoke instead of call
            chain_response = self.chain.invoke(chain_inputs)
            
            # Extract the selected prompt from variations
            if isinstance(chain_response, dict) and 'text' in chain_response:
                selected_prompt = chain_response['text']
            else:
                # If chain_response format is unexpected, use first variation
                selected_prompt = variations[0]
                print(f"Warning: Unexpected chain response format: {type(chain_response)}")
            
            # Find the closest matching variation
            selected_index = 0
            if selected_prompt in variations:
                selected_index = variations.index(selected_prompt)
            else:
                # If exact match not found, find most similar variation
                selected_index = 0  # Default to first variation
                
            # Create proper selection metadata
            selection_metadata = {
                "context": {
                    "doc_type": doc_type,
                    "current_output": current_output or "",
                    "groundtruth": groundtruth or ""
                },
                "to_select_from": {"prompt_strategy": variations},
                "selected": {
                    "index": selected_index,
                    "value": selected_prompt,
                    "score": None  # Score will be set later in update_with_results
                }
            }
            
            response = {
                'response': selected_prompt,
                'selection_metadata': selection_metadata
            }
            
        except Exception as e:
            print(f"Warning: RL chain failed with error: {e}")
            # Fallback to using the first variation if RL chain fails
            response = {
                'response': variations[0],
                'selection_metadata': {
                    "context": {
                        "doc_type": doc_type,
                        "current_output": current_output or "",
                        "groundtruth": groundtruth or ""
                    },
                    "to_select_from": {"prompt_strategy": variations},
                    "selected": {
                        "index": 0,
                        "value": variations[0],
                        "score": None
                    }
                }
            }
        
        # Save progress of learned policy
        if self.model_save_dir:
            try:
                self.chain.save_progress()
            except Exception as e:
                print(f"Warning: Failed to save progress: {e}")
            
        return {
            'optimized_prompt': response['response'],
            'base_variations': variations,
            'selection_metadata': response['selection_metadata']
        }

    def update_with_results(self, optimization_response: Dict, 
                       extraction_success: float) -> None:
        """
        Update the learned policy with delayed feedback about extraction success
        
        Args:
            optimization_response: Response from optimize_prompt()
            extraction_success: Score indicating extraction quality (0-1)
        """
        selection_metadata = optimization_response.get('selection_metadata')
        if selection_metadata is None:
            print("Warning: No selection metadata available, skipping policy update")
            return
            
        try:
            # Ensure score is within valid range
            score = max(0.0, min(1.0, float(extraction_success)))
            
            # Update the score in the selection metadata
            if 'selected' in selection_metadata:
                selection_metadata['selected']['score'] = score
            
            # Create the chain response format expected by update_with_delayed_score
            chain_response = {
                "selection_metadata": selection_metadata,
                "text": optimization_response.get('optimized_prompt', 
                                               selection_metadata['selected']['value'])
            }
            
            # Update the policy with proper error handling
            try:
                self.chain.update_with_delayed_score(
                    score=score,
                    chain_response=chain_response,
                    force_score=True  # Force the score update even with selection scorer
                )
            except AttributeError as e:
                print(f"Warning: Chain does not support update_with_delayed_score: {e}")
                # Try alternative update method if available
                if hasattr(self.chain, 'update_policy'):
                    self.chain.update_policy(chain_response, score)
            except Exception as e:
                print(f"Warning: Failed to update policy: {e}")
                
            # Save progress after successful update
            if self.model_save_dir:
                self.chain.save_progress()
                
        except Exception as e:
            print(f"Warning: Failed to process update: {e}")