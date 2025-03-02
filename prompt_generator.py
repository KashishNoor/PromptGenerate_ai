from pydantic import BaseModel, Field
from typing import Optional
from huggingface_hub import InferenceClient
import os
import json

class GeneratePromptFromScratchInput(BaseModel):
    initial_intent: str = Field(description="The input text to generate the prompt from scratch")
    examples: Optional[str] = None
    context: Optional[str] = None
    constraints: Optional[str] = None

class GeneratePromptFromScratchOutput(BaseModel):
    system_role: str = Field(description="The role/persona the AI should adopt")
    task_description: str = Field(description="Clear description of task")
    input_structure: dict = Field(description="Structure of expected input variables as a dictionary")
    step_by_step: list[str] = Field(description="Detailed step-by-step instructions as a list")
    output_format: dict = Field(description="Expected format of the output as a dictionary")
    guidelines: list[str] = Field(description="Guidelines for the model to follow as a list")
    constraints: list[str] = Field(description="Any constraints or restrictions to follow as a list")
    examples: list[str] = Field(description="Examples to guide the model")

# Use your provided API key
HF_API_KEY = "hf_API KEY"

def generate_prompt(input_data: GeneratePromptFromScratchInput, api_key: str = HF_API_KEY) -> GeneratePromptFromScratchOutput:
    client = InferenceClient(token=api_key)
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI prompt engineering assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
Create a detailed prompt structure for: {input_data.initial_intent}
Examples: {input_data.examples or 'None'}
Context: {input_data.context or 'None'}
Constraints: {input_data.constraints or 'None'}
Use JSON format with: system_role, task_description, input_structure, step_by_step, output_format, guidelines, constraints, examples<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    response = client.text_generation(prompt, max_new_tokens=2000)
    
    # Validate JSON response
    try:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
        return GeneratePromptFromScratchOutput.model_validate_json(json_str)
    except (IndexError, ValidationError, json.JSONDecodeError) as e:
        print(f"Validation error: {str(e)}")
        print(f"Model response: {response}")  # Log the raw response for debugging
        raise ValueError(f"Failed to validate model response: {str(e)}")

if __name__ == "__main__":
    test_input = GeneratePromptFromScratchInput(
        initial_intent="Write a comprehensive article about AI safety",
        examples="Example 1: ...\nExample 2: ...",
        context="The article should cover ethical considerations, technical challenges, and future implications.",
        constraints="The article should be no longer than 2000 words and should avoid technical jargon."
    )
    result = generate_prompt(test_input)
    print(result.model_dump_json(indent=2))