import yaml
import base64
from functools import partial
from pathlib import Path
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

import google.generativeai as genai
from openai import OpenAI
import anthropic


DIR = Path(__file__).parent
load_dotenv(DIR / ".." / ".env")

DEFAULT_TEMPERATURE = 1.0

MODELS = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
    "claude-3-5-sonnet-20240620",
    "gpt-4o",
    # + models supported by APIs
]


def llm_model(model_id, temperature=DEFAULT_TEMPERATURE):
    if "gemini" in model_id:
        return gemini_model(model_id, temperature)
    elif "claude" in model_id:
        return claude_model(model_id, temperature)
    elif "gpt" in model_id:
        return gpt_model(model_id, temperature)
    else:
        raise ValueError(f"Unsupported model: {model_id}")


def gemini_model(model_id, temperature=DEFAULT_TEMPERATURE):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_id, generation_config={"temperature": temperature})

    def _generate_content(messages):
        result = model.generate_content(messages)
        return result.text
    return _generate_content


def gpt_model(model_id, temperature=DEFAULT_TEMPERATURE):
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _generate_content(messages):
        """
        Args:
            msg: list of str or Image.Image
        """
        if not isinstance(messages, list):
            messages = [messages]

        messages = [
            {"role": "user", "content": [
                m for m in messages
            ]}
        ]
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

    return _generate_content



def claude_model(model_id, temperature=0):
    claude_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    def _format_msg(msg):
        if isinstance(msg, str):
            return {"type": "text", "text": msg}
        elif isinstance(msg, Image.Image):
            buffered = BytesIO()
            msg.save(buffered, format="JPEG")
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    # "data": base64.b64encode(msg.tobytes()).decode("utf-8")
                    "data": base64.b64encode(buffered.getvalue()).decode('utf-8')
                }
            }
        else:
            raise ValueError("Invalid message type")

    def _generate_content(messages):
        if not isinstance(messages, list):
            messages = [messages]

        messages = [
            {"role": "user", "content": [
                _format_msg(m) for m in messages
            ]}
        ]
        result = claude_client.messages.create(
            model=model_id,
            temperature=temperature,
            max_tokens=8192,
            messages=messages
        )
        return result.content[0].text
    
    return _generate_content


def parse_output(answer: str, output_type='yaml', output_name=None):
    """
    Parse the output block from the LLM answer.
    A block starts with '```{output_type}' and ends with '```'.

    Args:
        answer: str, the answer from the LLM
        output_type: str, the type of the output block
        output_name: str, the name of the output block
            if output_type is 'python', output_name is the name of the function to return
    """
    if output_type == 'yaml':
        # extract output block
        text_block = answer.split(f'```{output_type}')[1].split('```')[0]
        try:
            # test if the block is a valid yaml
            config = yaml.safe_load(text_block)
            return text_block, config
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML output: {e}")

    elif output_type == 'python':
        # extract output block
        text_block = answer.split(f'```{output_type}')[1].split('```')[0]
        try:
            # Create a new empty dictionary to serve as a local namespace
            local_namespace = {}
            # Execute the code block in the local namespace
            exec(text_block, globals(), local_namespace)

            if output_name:
                # If a function name is provided, return that specific function
                if output_name in local_namespace:
                    return text_block, local_namespace[output_name]
                else:
                    raise ValueError(f"Function '{output_name}' not found in the generated code")
            else:
                # If no function name is provided, return the entire local namespace
                return text_block, local_namespace
        except Exception as e:
            raise ValueError(f"Failed to parse Python output: {e}")

    else:
        raise ValueError(f"Unsupported output type: {output_type}")
