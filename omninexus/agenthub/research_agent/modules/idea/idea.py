"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_IDEA_GENERATION_DESCRIPTION = """Based on the relevant research results, create a [research_domain]_ideas.md file containing innovative ideas about this domain.

* You MUST first identify the RESEARCH DOMAIN or research field you are addressing, such as Large Language Models (LLMs), Reinforcement Learning and Human Feedback (RLHF), or Multimodal Learning.
* You MUST provide a brief overview of the RESEARCH DIRECTIONS or topics within the domain, highlighting key areas of focus and recent advancements.
* You MUST identify the key CHALLENGES or limitations faced by the research field or the specific paper or topic you are addressing. You can refer to existing research papers, reviews, or discussions to identify these challenges.
* You MUST provide IMPROVEMENTS before generating novel ideas. These improvements should be innovative approaches or solutions to address the identified challenges.
* For each research direction, you MUST brainstorm at least 3 innovative and novel ideas or approaches to address the identified challenges. These ideas should be innovative, feasible, and have the potential to advance the research field.
* You MUST describe how to implement the novel ideas in the research project for each idea. You can provide a high-level ABSTRACT of the proposed methods, IMPLEMENTATION details, and ALGORITHM flow for each idea.

## For example,  the [research_domain]_ideas.md file may look like this:
(this is the start of the file)
** RESEARCH DOMAIN **
* Large Language Models (LLMs)

** RESEARCH DIRECTIONS **
* Efficiency and Resource Optimization: Research focuses on reducing computational costs using techniques like model quantization, pruning, and distillation.
* Alignment with Human Intent: Advances in RLHF and preference modeling aim to improve LLMs' ability to align with user intent while minimizing harmful outputs.
* Multimodal Capabilities: LLMs are increasingly integrated with other modalities, enabling models like GPT-4 Vision to process text, images, and audio seamlessly.
* Long-Context Understanding: Techniques like extended context windows and efficient attention mechanisms enhance LLMs' ability to process long documents and sequences.
* Trustworthiness and Explainability: Efforts focus on improving LLMs' interpretability, mitigating biases, and ensuring safe, reliable outputs for critical applications.
* Enhanced Reasoning: Models like GPT-o1 emphasize improving reasoning performance by enabling extended deliberation for complex tasks, such as scientific problem-solving and mathematical reasoning

** CHALLENGES **
* Computational Cost and Efficiency: Training and deploying LLMs require immense computational resources, leading to high costs and energy consumption, which hinder scalability and accessibility.
* Hallucinations and Reliability: LLMs often generate hallucinated outputs (factually incorrect or fabricated information), posing risks in critical applications like healthcare and finance.
* Alignment with Human Values: Ensuring that LLMs consistently align with human intent and ethical values remains challenging, especially in reducing biases, harmful content, and unintended behaviors.
* Long-Context Limitations: Despite progress, LLMs still struggle with effectively processing and reasoning over long sequences of text, leading to loss of context or degradation in performance.
* Data Privacy and Security: Training and fine-tuning LLMs on vast datasets raise concerns regarding the use of private or sensitive information, along with risks like model inversion and data leakage.

** IMPROVEMENTS **
* Efficient Computation: Develop lightweight LLM architectures, model distillation techniques, or dynamic computation strategies to reduce costs and improve efficiency.
* Factual Consistency: Integrate fact-checking modules, hallucination detection mechanisms, or knowledge verification systems to enhance LLMs' reliability and trustworthiness.
* Human-Centric Design: Implement user feedback loops, preference modeling, or interactive learning paradigms to align LLMs with human values and ethical considerations.

** NOVEL IDEAS **
* Idea 1: Adaptive Computation for Dynamic Reasoning
    - ABSTRACT: Develop a multi-stage adaptive computation mechanism where LLMs selectively apply deeper layers and longer reasoning paths only when task complexity demands it. This could involve early stopping, dynamic depth control, or routing tasks to lightweight sub-networks.
    - IMPLEMENTATION:
        * Task Complexity Estimation: Introduce a lightweight task complexity classifier that evaluates input queries based on entropy, token distribution, or confidence scores.
        * Dynamic Routing: Use a mixture-of-experts (MoE) approach or layer-wise gating mechanisms to activate specific parts of the model for complex tasks while skipping others for simpler ones.
        * Efficient Inference: Integrate adaptive early-exit techniques (e.g., depth-skip transformers) that allow stopping computation once confidence thresholds are met.
    - ALGORITHM:
        * Input Query -> Complexity Estimation -> Adaptive Computation -> Inference -> Output.
        * Pseudocode: Provide a high-level Pseudocode snippet.

* Idea 2: Hallucination-Aware Retrieval-Augmented Generation (RAG)
    - ABSTRACT: Design a feedback loop-augmented RAG system that integrates hallucination detection and correction during generation. The model validates generated content against a trusted external knowledge base, iteratively refining responses.
    - IMPLEMENTATION:
        * Knowledge-Enhanced RAG: Combine LLMs with real-time retrieval systems (e.g., semantic search) to fetch relevant knowledge dynamically.
        * Hallucination Detection Module: Add a discriminator or factual consistency checker trained to detect hallucinations by comparing generated content with retrieved data.
        * Refinement Loop: If inconsistencies are found, prompt the model to refine or regenerate specific parts of the response iteratively.
    - ALGORITHM:
        * Input Query -> Retrieve Context -> Generate Response -> Factual Consistency Check -> (If Failure) Regenerate -> Output.
        * Pseudocode: Provide a high-level Pseudocode snippet.
(this is the end of the file)

Note: Execute a bash command in the terminal to generate the [research_domain]_ideas.md. The command should be run in the root directory where the [research_domain]_ideas.md should be generated.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
"""

IdeaGenerationTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='idea_generation',
        description=_IDEA_GENERATION_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to execute to generate ideas file. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
