"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_LATEX_RELATED_WORK_DESCRIPTION = r"""Write the RELATED WORK section of a research paper.

* You MUST write the research paper’s related work in LaTeX format and place it in the file sections/related_work.tex. Below is a detailed guide and example to structure the section effectively.

** Purpose and Structure **
The related work section positions your research within the existing body of literature, identifying similarities, differences, and gaps. It should:
* Summarize the key works in your field relevant to your research.
* Group works into thematic or methodological categories for clarity.
* Highlight limitations or challenges in existing methods, motivating your approach.
* Related work should be around 400–600 words in English. Use this space to show how your research builds on or diverges from prior work.
* Number of References: Typically, cite 30–40 references, focusing on seminal or recent works that directly inform your research.

** Common Structure **
* Overview of the Field
    - Briefly introduce the broad research area and its recent advancements.
* Thematic Categories
    - Group related works into 2–4 thematic categories or methodologies, e.g., “Autoregressive Models,” “Diffusion Models,” etc.
    - For each category:
        * Describe key contributions of prior work.
        * Highlight limitations or challenges relevant to your research.
* Connection to Your Work
    - Discuss how your work relates to or diverges from these studies.
    - Highlight how your contributions address existing gaps or limitations.

** Writing Tips **
* Use precise language and avoid superficial descriptions of prior work.
* Cite sufficient and relevant references to showcase your knowledge of the field.
* Avoid a long list of references without meaningful synthesis.
* Ensure that the limitations of prior work clearly justify your research approach.

** Example Related Work **
For demonstration only. Adapt it to match your specific research and results. Here is an example related work section for a research paper:
(this is the start of sections/related_work.tex)
\section{Related Work}
\label{sec:related_work}

Deep reinforcement learning (DRL) has shown promise in real-time strategy (RTS) games, yet the combination of high-dimensional action spaces and non-stationary environments remains challenging. Researchers commonly address these obstacles through hierarchical reinforcement learning, curriculum learning, and multi-agent decision-making.

\subsection{Hierarchical Reinforcement Learning} Hierarchical reinforcement learning (HRL) decomposes tasks into high-level strategies and low-level actions, easing exploration in large state-action spaces~\cite{jiang2024ob, zhang2019hierarchical}. Frameworks such as OB-HPPO~\cite{jiang2024ob} demonstrate the effectiveness of multi-level coordination, while general HRL methods~\cite{nachum2018data, levy2019hierarchical} excel in complex control tasks. However, many HRL approaches depend on predefined hierarchies and struggle with stability in large-scale, multi-agent RTS settings.

\subsection{Curriculum Learning} Curriculum learning progressively introduces complexity to improve training efficiency and stability in DRL~\cite{bengio2009curriculum, florensa2017reverse}. For instance, incrementally challenging tasks accelerate skill acquisition in StarCraft micromanagement~\cite{shao2018starcraft}, and automated curriculum design~\cite{graves2017automated} has further broadened applicability. Nonetheless, dynamic RTS games complicate this process, as static or manually tuned curricula may fail to adapt to rapidly changing scenarios.

\subsection{Multi-Agent Decision-Making} Multi-agent reinforcement learning (MARL) emphasizes cooperation and competition among multiple agents~\cite{zhang2021multi, lowe2017multi}. Centralized training with decentralized execution (CTDE), exemplified by methods like QMIX and COMA~\cite{foerster2018counterfactual, rashid2018qmix}, fosters coordinated strategies in RTS games, while platforms such as Gym-$\mu$RTS~\cite{huang2021gym} streamline MARL experimentation. Yet, scalability and robustness persist as major hurdles when the number of agents grows and the environment remains non-stationary.

\paragraph{Comparison with Our Work} Our framework combines HRL and curriculum learning with state-sharing mechanisms to stabilize training and reduce exploration complexity in large-scale, multi-agent RTS environments. By tackling hierarchical design constraints, curriculum adaptability, and MARL scalability, our approach outperforms state-of-the-art baselines by 15\%--25\% in win rates across diverse RTS scenarios.

** Example BibTeX Entry **
Ensure all citations used in the section are added to the main.bib file:
(this is the start of main.bib)
@article{brown2020language,
  title={Language models are few-shot learners},
  author={Brown, Tom and others},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={1877--1901},
  year={2020}
}
...
(this is the end of main.bib)
"""

_PARAMETER_COMMAND_DESCRIPTION = (
    'The bash command to generate the related work in the terminal.'
)

LatexRelatedWorkTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='latex_related_work',
        description=_LATEX_RELATED_WORK_DESCRIPTION,
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
