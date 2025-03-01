import streamlit as st
from typing import Optional
from pydantic import BaseModel, Field, ValidationError
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import time
import json

load_dotenv()

# Define unified input/output models
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
    examples: list[str] = Field(description="Examples to guide the model (can include relevant links if available)")

class FinalPrompt(BaseModel):
    prompt: str

# Hardcoded API keyad
HF_API_KEY = "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"

# List of high-performance models
MODELS = [
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "gpt-3.5-turbo",
    "bigscience/bloom",
    "google/flan-t5-xxl"
]

def generate_with_hf(prompt: str, api_key: str = HF_API_KEY, model: str = MODELS[0]) -> str:
    client = InferenceClient(token=api_key)
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI prompt engineering expert. Your task is to generate detailed and structured prompts for various tasks. Ensure that all fields, including Input Structure, are filled with relevant data. Include relevant links in the examples section if they are available and appropriate.<|eot_id|><|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    response = client.text_generation(
        formatted_prompt,
        max_new_tokens=2000,
        temperature=0.7,
        seed=42,
    )
    return response

def validate_json_response(text: str, model: type[BaseModel]) -> BaseModel:
    try:
        # Try to extract JSON from markdown code block
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
        else:
            # If no markdown block, assume the entire response is JSON
            json_str = text.strip()
        
        # Parse and validate JSON
        return model.model_validate_json(json_str)
    except (IndexError, ValidationError, json.JSONDecodeError) as e:
        st.error(f"Validation error: {str(e)}")
        st.error(f"Model response: {text}")  # Log the raw response for debugging
        raise ValueError(f"Failed to validate model response: {str(e)}")

def format_response(response: GeneratePromptFromScratchOutput) -> str:
    formatted_text = f"""
### System Role:
{response.system_role}

### Task Description:
{response.task_description}

### Input Structure:
"""
    # Format input structure in a clean, readable way (not JSON)
    for key, value in response.input_structure.items():
        formatted_text += f"- **{key}**:\n"
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                formatted_text += f"  - **{sub_key}**: {sub_value}\n"
        else:
            formatted_text += f"  - {value}\n"

    formatted_text += """
### Step-by-Step Instructions:
"""
    for step in response.step_by_step:
        formatted_text += f"- {step}\n"

    formatted_text += """
### Output Format:
"""
    # Improved formatting for nested dictionaries
    def format_dict(d, indent=0):
        formatted = ""
        for key, value in d.items():
            if isinstance(value, dict):
                formatted += " " * indent + f"- **{key}**:\n"
                formatted += format_dict(value, indent + 4)
            else:
                formatted += " " * indent + f"- **{key}**: {value}\n"
        return formatted

    formatted_text += format_dict(response.output_format)

    formatted_text += """
### Guidelines:
"""
    for guideline in response.guidelines:
        formatted_text += f"- {guideline}\n"

    formatted_text += """
### Constraints:
"""
    for constraint in response.constraints:
        formatted_text += f"- {constraint}\n"

    formatted_text += """
### Examples:
"""
    for example in response.examples:
        formatted_text += f"- {example}\n"

    return formatted_text        

def simulate_streaming(text: str, placeholder):
    displayed_text = ""
    progress_bar = st.progress(0)
    for i, char in enumerate(text):
        displayed_text += char
        placeholder.markdown(displayed_text)
        progress_bar.progress((i + 1) / len(text))
        time.sleep(0.01)

def main():
    st.title("Advanced Agentic Prompt Generator")

    # Model selection
    model = st.selectbox("Select Model", MODELS, index=0)

    # Input Fields
    initial_intent = st.text_input(
        "What is your initial prompt?",
        placeholder="e.g., write an article, analyze sentiment, generate a story"
    )
    
    include_examples = st.checkbox("Include examples?")
    examples = None
    if include_examples:
        examples = st.text_area(
            "Enter examples:",
            placeholder="Enter example inputs and outputs to guide the AI"
        )

    include_context = st.checkbox("Include context?")
    context = None
    if include_context:
        context = st.text_area(
            "Enter context:",
            placeholder="Enter any additional context to guide the AI"
        )

    include_constraints = st.checkbox("Include constraints?")
    constraints = None
    if include_constraints:
        constraints = st.text_area(
            "Enter constraints:",
            placeholder="Enter any constraints or restrictions"
        )

    # Generate Prompt Button
    if st.button("Generate Prompt") and initial_intent:
        try:
            with st.spinner("Generating prompt structure..."):
                input_data = GeneratePromptFromScratchInput(
                    initial_intent=initial_intent,
                    examples=examples,
                    context=context,
                    constraints=constraints
                )

                # First generation stage
                generation_prompt = f"""
                Create a detailed AI prompt structure based on:
                Intent: {input_data.initial_intent}
                Examples: {input_data.examples or 'None'}
                Context: {input_data.context or 'None'}
                Constraints: {input_data.constraints or 'None'}
                
                Output must include:
                - System role
                - Task description
                - Input structure (as a dictionary with detailed fields)
                - Step-by-step instructions (as a list)
                - Output format (as a dictionary)
                - Guidelines (as a list)
                - Constraints (as a list)
                - Examples (as a list, including relevant links if available)
                
                Format your response as a JSON object wrapped in a markdown code block:
                ```json
                {{
                    "system_role": "...",
                    "task_description": "...",
                    "input_structure": {{
                        "game_name": "Snake",
                        "game_rules": {{
                            "objective": "The objective of the game is to eat as many apples as possible without colliding with the game boundaries or the snake's own tail.",
                            "controls": {{
                                "up": "W",
                                "down": "S",
                                "left": "A",
                                "right": "D"
                            }},
                            "game_area_size": "20x20",
                            "snake_initial_length": 3
                        }}
                    }},
                    "step_by_step": ["step 1", "step 2", "..."],
                    "output_format": {{"key": "value"}},
                    "guidelines": ["guideline 1", "guideline 2", "..."],
                    "constraints": ["constraint 1", "constraint 2", "..."],
                    "examples": ["example 1", "example 2", "..."]
                }}
                ```
                """

                first_response = generate_with_hf(generation_prompt, model=model)
                validated_response = validate_json_response(first_response, GeneratePromptFromScratchOutput)

                # Format the response
                formatted_response = format_response(validated_response)
                
                output_placeholder = st.empty()
                st.success("Prompt generated successfully!")
                simulate_streaming(formatted_response, output_placeholder)
                
                st.download_button(
                    label="📥 Download Prompt",
                    data=formatted_response,
                    file_name="optimized_prompt.md",
                    mime="text/markdown"
                )
            
        except Exception as e:
            st.error("Error generating prompt. Please try again.")
            if st.button("Retry"):
                st.experimental_rerun()

    # Clear Inputs Button
    if st.button("Clear Inputs"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()