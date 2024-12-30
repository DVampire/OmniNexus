"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_LATEX_INTRODUCTION_DESCRIPTION = r"""Write the INTRODUCTION section of a research paper.

* You MUST write the research paper’s introduction in LaTeX format and place it in the file sections/introduction.tex. Below is a detailed guide and example to structure the section effectively.

** Purpose and Structure **
The introduction sets the stage for your research, providing context, challenges, and motivation while introducing your contributions. It should:
* Engage readers by highlighting the problem's significance.
* Provide enough background to understand the research gap.
* Concisely describe your methodology and contributions.
* Introduction should be around 600 words in English. Use this space to motivate your research and outline its key contributions.
* Number of References: Typically, cite 30-40 references to establish the research context and justify your approach.

** Common Structure **
* Background and Context
    - Introduce the research area and its significance.
    - Highlight major advancements or milestones in the field.
* Challenges and Motivation
    - Point out gaps or limitations in existing methods or literature.
    - Explain why these challenges matter, supported by examples or evidence.
* Research Objective and Approach
    - Clearly articulate the problem statement or research goal.
    - Briefly describe the methodology or framework you propose.
* Key Contributions
    - Summarize your main contributions and their significance to the field.
    - Optionally, highlight experimental results briefly to emphasize impact.

** Writing Tips **
* Use active voice and precise language.
* Avoid unnecessary technical details; focus on clarity and motivation.
* Cite relevant work to position your research within the field.
* Avoid overloading with results; save detailed findings for later sections.
* End with a brief contributions summary of your work.

NOTE: When cite a reference in LaTeX, you MUST add the BibTeX of it in the file main.bib. If the item is already in the main.bib, you can directly use the citation key in the LaTeX file.

** Example Introduction **
For demonstration only. Adapt it to match your specific research and results. Here is an example introduction for a research paper:
(this is the start of sections/introduction.tex)
\section{Introduction}
\label{sec:introduction}
Deep reinforcement learning (DRL) has achieved remarkable success across various domains, including robotics~\cite{kober2013reinforcement}, finance~\cite{deng2016deep}, and gaming~\cite{mnih2015human}. Despite these advancements, its application to real-time strategy (RTS) games presents significant challenges due to the high-dimensional action spaces, complex multi-agent dynamics, and non-stationary environments~\cite{vinyals2019grandmaster}. These challenges highlight the limitations of existing DRL methods, which often struggle with scalability, exploration, and stability in dynamic settings~\cite{silver2016mastering, lillicrap2015continuous}.

Addressing these issues is critical for advancing the state-of-the-art in DRL and enabling robust performance in complex environments. Current approaches typically focus on flat policy learning~\cite{schulman2017proximal}, which fails to capture the hierarchical nature of decision-making required in RTS games. Moreover, they lack effective mechanisms for handling non-stationary dynamics, resulting in suboptimal learning outcomes and poor generalization across tasks~\cite{haarnoja2018soft}.

To tackle these challenges, we propose a novel hierarchical DRL framework that decomposes decision-making into strategic and tactical layers. The strategic layer focuses on high-level planning, while the tactical layer handles fine-grained actions. By integrating state-sharing and curriculum training~\cite{bengio2009curriculum}, our approach improves exploration efficiency and stabilizes the learning process in dynamic environments.

Our key contributions are as follows:
\begin{itemize}
    \item We introduce a hierarchical DRL framework that decouples decision-making into strategic and tactical layers, enabling scalable and efficient learning.
    \item We propose state-sharing and curriculum training mechanisms to address the challenges of exploration and non-stationarity.
    \item We demonstrate the effectiveness of our framework through experiments on multiple large-scale RTS scenarios, achieving a 15\%–25\% higher win rate compared to existing methods.
    \item We conduct ablation studies to validate the importance of the hierarchical design and curriculum training in improving performance and robustness.
\end{itemize}
(this is the end of sections/introduction.tex)

** Example BibTeX Entry **
Ensure all citations used in the section are added to the main.bib file:
(this is the start of main.bib)
@article{kober2013reinforcement,
  title={Reinforcement learning in robotics: A survey},
  author={Kober, Jens and Bagnell, J Andrew and Peters, Jan},
  journal={The International Journal of Robotics Research},
  volume={32},
  number={11},
  pages={1238--1274},
  year={2013},
  publisher={SAGE Publications Sage UK: London, England}
}
...
(this is the end of main.bib)
"""

_PARAMETER_COMMAND_DESCRIPTION = (
    'The bash command to generate the introduction in the terminal.'
)

LatexIntroductionTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='latex_introduction',
        description=_LATEX_INTRODUCTION_DESCRIPTION,
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
