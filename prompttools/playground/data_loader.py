# Copyright (c) Hegel AI, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.


import os
import jinja2
import streamlit as st

from prompttools.selector.prompt_selector import PromptSelector
from prompttools.playground.constants import ENVIRONMENT_VARIABLE, EXPERIMENTS


def render_prompts(templates, vars):
    prompts = []
    for template in templates:
        for var_set in vars:
            environment = jinja2.Environment()
            jinja_template = environment.from_string(template)
            prompts.append(jinja_template.render(**var_set))
    return prompts


@st.cache_data
def load_data(
    model_type,
    model,
    instructions,
    user_inputs,
    temperature=0.0,
    top_p=1,
    max_tokens=None,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    api_key=None,
    base_url=None,
):
    if api_key:
        os.environ[ENVIRONMENT_VARIABLE.get(model_type, "OPENAI_API_KEY")] = api_key

    from prompttools.selector.prompt_selector import PromptSelector
    
    selectors = [PromptSelector(instruction, user_input) for instruction in instructions for user_input in user_inputs]

    experiment = None
    if model_type == "LlamaCpp Chat":
        call_params = dict(temperature=[temperature])
        experiment = EXPERIMENTS[model_type]([model], selectors, call_params=call_params)
    elif model_type in {"OpenAI Chat", "OpenAI Completion"}:
        experiment = EXPERIMENTS[model_type](
            [model],
            selectors,
            temperature=[temperature],
            top_p=[top_p],
            max_tokens=[max_tokens],
            frequency_penalty=[frequency_penalty],
            presence_penalty=[presence_penalty],
        )
    elif model_type == "Custom OpenAI Chat":
        # For Custom OpenAI Chat, we'll need to create a custom experiment
        # This will use the OpenAIChatExperiment with a custom base_url
        from prompttools.experiment import OpenAIChatExperiment
        from prompttools.selector.prompt_selector import PromptSelector
        
        # Create selectors from instructions and user inputs
        messages = []
        for instruction in instructions:
            for user_input in user_inputs:
                messages.append([
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_input}
                ])
        
        # Create experiment with custom base_url
        experiment = OpenAIChatExperiment(
            model=[model],
            messages=messages,
            temperature=[temperature],
            top_p=[top_p],
            max_tokens=[max_tokens],
            frequency_penalty=[frequency_penalty],
            presence_penalty=[presence_penalty],
            base_url=base_url,
            stream=[False],
        )
        experiment.run()
    elif model_type == "HuggingFace Hub":
        experiment = EXPERIMENTS[model_type]([model], selectors, temperature=[temperature])
    elif model_type == "Anthropic":
        experiment = EXPERIMENTS[model_type]([model], selectors, temperature=[temperature])
    elif model_type == "Google PaLM":
        experiment = EXPERIMENTS[model_type]([model], selectors, temperature=[temperature])
    elif model_type == "Replicate":
        input_kwargs = {"prompt": selectors,
                        "temperature": [temperature]}
        model_specific_kwargs = {model: {}}
        experiment = EXPERIMENTS[model_type]([model], input_kwargs, model_specific_kwargs)

    return experiment.to_pandas_df(True, True)


@st.cache_data
def run_multiple(
    model_types,
    models,
    instructions,
    prompts,
    openai_api_key=None,
    anthropic_api_key=None,
    google_api_key=None,
    hf_api_key=None,
    replicate_api_key=None,
    base_urls=None,
):
    import os

    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if google_api_key:
        os.environ["GOOGLE_PALM_API_KEY"] = google_api_key
    if hf_api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key
    if replicate_api_key:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    
    base_urls = base_urls or {}
    dfs = []
    
    for i in range(len(models)):
        # TODO Support temperature and other parameters
        selectors = []
        if i + 1 in instructions:
            selectors = [PromptSelector(instructions[i + 1], prompt) for prompt in prompts]
            
            if model_types[i] == "Replicate":
                input_kwargs = {"prompt": selectors}
                model_specific_kwargs = {models[i]: {}}
                experiment = EXPERIMENTS[model_types[i]]([models[i]], input_kwargs, model_specific_kwargs)
            elif model_types[i] == "Custom OpenAI Chat":
                # Handle Custom OpenAI Chat in run_multiple
                from prompttools.experiment.openai.openai_chat_experiment import OpenAIChatExperiment
                
                # Create messages from instructions and prompts
                messages = []
                for prompt in prompts:
                    messages.append([
                        {"role": "system", "content": instructions[i + 1]},
                        {"role": "user", "content": prompt}
                    ])
                
                # Create experiment with custom base_url
                experiment = OpenAIChatExperiment(
                    model_names=[models[i]],
                    messages=messages,
                    temperature=[0.0],
                    base_url=base_urls.get(i + 1)
                )
                experiment.run()
            else:
                experiment = EXPERIMENTS[model_types[i]]([models[i]], selectors)
        else:
            if model_types[i] == "Replicate":
                input_kwargs = {"prompt": prompts}
                model_specific_kwargs = {models[i]: {}}
                experiment = EXPERIMENTS[model_types[i]]([models[i]], input_kwargs, model_specific_kwargs)
            elif model_types[i] == "Custom OpenAI Chat":
                # For cases without instruction
                from prompttools.experiment.openai.openai_chat_experiment import OpenAIChatExperiment
                
                # Create messages with default system message
                messages = []
                for prompt in prompts:
                    messages.append([
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt}
                    ])
                
                # Create experiment with custom base_url
                experiment = OpenAIChatExperiment(
                    model_names=[models[i]],
                    messages=messages,
                    temperature=[0.0],
                    base_url=base_urls.get(i + 1)
                )
                experiment.run()
            else:
                experiment = EXPERIMENTS[model_types[i]]([models[i]], prompts)
                
        dfs.append(experiment.to_pandas_df(True, True))
    return dfs