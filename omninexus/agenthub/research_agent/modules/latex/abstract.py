"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_LATEX_ABSTRACT_DESCRIPTION = r"""Write the ABSTRACT section of a research paper.

* You MUST write the research paper’s abstract in LaTeX format and place it in the file sections/abstract.tex. Below is a detailed guide and example to structure the section effectively.

** Purpose and Structure **
* Audience: The abstract is the first point of contact for potential readers (including reviewers), guiding them on whether to read the full paper.
* Style: Clearly state what you did, why it matters, what results you got, and how it impacts the field—using concise language and minimal jargon.
* Brevity: Abstract should be typically 250 words in English. Use this space to efficiently convey your research’s main idea.

** Common Structure **
You can split the abstract into four key parts or merge them into a single, concise paragraph.
* Motivation
    - Briefly introduce the background of your research area or problem.
    - Explain why it is important or worth investigating.
* Problem Statement
    - Clearly state the problem or research goal.
    - Highlight gaps or challenges in existing solutions to show why your work is innovative.
* Methods
    - Give a brief overview of your proposed method or framework.
    - Point out what’s new or improved compared to previous work.
* Results
    - Summarize your main experimental or theoretical findings.
    - Emphasize the value or potential impact of these results (academic or applied).
    - If experiments show significant improvements or theoretical proofs confirm feasibility, mention them briefly.

** Writing Tips **
* Be Concise： Keep it short and to the point. Avoid lengthy background info or unnecessary adjectives.
* Highlight Contributions: Clearly describe your paper’s key contributions (theory, methods, experiments). Use active statements like “We propose...,” “We demonstrate...,” etc.
* Avoid Overstatement: Don’t oversell with words like “the best” or “completely solve.” Reviewers tend to distrust exaggerated claims.
* Typically, no figures or lengthy references appear in the abstract.

** Example Abstract **
For demonstration only. Adapt it to match your specific research and results. Here is an example abstract for a research paper:
(this is the start of sections/abstract.tex)
\begin{abstract}
\label{sec:abstract}
Deep reinforcement learning (DRL) has achieved notable success in various fields, yet applying it to real-time strategy (RTS) games remains challenging due to large action spaces, vast states, and complex multi-agent dynamics. We propose a hierarchical DRL framework that splits decision-making into strategic and tactical layers, enabling efficient handling of macro-level tasks such as resource management and army composition alongside micro-level tasks like unit positioning and targeting. This decomposition reduces exploration complexity and stabilizes learning by assigning distinct roles to each layer. A shared state mechanism ensures both layers access vital global information, promoting coherent decisions across temporal and spatial scales. Moreover, we employ a curriculum training approach in which the agent progresses from simpler to more demanding scenarios, fostering robust policies capable of adapting to large-scale RTS environments. Experimental evaluations confirm that our method achieves a 15\%–25\% higher win rate compared to baselines, underscoring its potential to learn scalable and efficient strategies in non-stationary settings. Ablation studies further highlight the importance of both the hierarchical architecture and the curriculum, as removing either component significantly degrades performance and stability. Overall, our work demonstrates that structured, multi-level decision-making can effectively address the inherent complexity of RTS games, paving the way for more powerful AI systems capable of integrating strategic foresight and tactical precision in real time.
\end{abstract}
(this is the end of sections/abstract.tex)
"""

_PARAMETER_COMMAND_DESCRIPTION = (
    'The bash command to generate the abstract in the terminal.'
)

LatexAbstractTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='latex_abstract',
        description=_LATEX_ABSTRACT_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': _PARAMETER_COMMAND_DESCRIPTION,
                },
            },
            'required': ['command'],
        },
    ),
)
