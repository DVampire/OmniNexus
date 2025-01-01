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

** Example Introduction **
For demonstration only. Adapt it to match your specific research and results. Here is an example introduction for a research paper:
(this is the start of sections/introduction.tex)
\section{Introduction}
\label{sec:introduction}

Deep reinforcement learning (DRL) has garnered significant attention in recent years due to its remarkable success in solving complex problems across diverse domains. In robotics, DRL has enabled the development of adaptive control policies for real-world tasks such as manipulation and locomotion~\cite{kober2013reinforcement}. In finance, DRL has shown potential for optimizing portfolio management and algorithmic trading strategies by capturing complex temporal patterns in financial data~\cite{deng2016deep}. Moreover, DRL has revolutionized the gaming industry, achieving superhuman performance in games ranging from Atari classics to the strategic board game Go~\cite{mnih2015human, silver2016mastering}.

Despite these successes, extending DRL to real-time strategy (RTS) games presents a set of unique and formidable challenges. RTS games, characterized by high-dimensional action spaces, complex multi-agent dynamics, and non-stationary environments, push the limits of current DRL methodologies~\cite{vinyals2019grandmaster}. Unlike games with discrete, sequential decision-making, RTS games demand continuous coordination across multiple units, requiring sophisticated strategies and real-time adaptability. These challenges expose the limitations of existing DRL methods, particularly in their ability to scale effectively, explore vast action spaces, and maintain stability in dynamic, adversarial settings~\cite{silver2016mastering, lillicrap2015continuous}.

Addressing these limitations is critical for advancing DRL's applicability to environments that demand both scalability and robust decision-making. Existing approaches, such as flat policy learning, focus on direct action mapping without adequately capturing the hierarchical structure inherent in many complex tasks~\cite{schulman2017proximal}. Flat policies struggle to manage the layered decision-making required in RTS games, where long-term strategic planning must coexist with fine-grained tactical execution. Furthermore, the non-stationary dynamics of RTS games exacerbate the difficulties, as agents must adapt to evolving opponent strategies and game states. Traditional DRL methods often fail to generalize effectively in such settings, resulting in suboptimal learning outcomes and performance degradation~\cite{haarnoja2018soft}.

To overcome these challenges, we propose a novel hierarchical DRL framework specifically tailored for RTS games. This framework introduces a two-tiered structure, decomposing decision-making into strategic and tactical layers. The strategic layer is responsible for high-level planning and resource allocation, while the tactical layer handles micro-level actions and unit control. By incorporating state-sharing mechanisms and curriculum training techniques~\cite{bengio2009curriculum}, our approach enhances exploration efficiency and stabilizes learning in highly dynamic environments. State-sharing ensures that the two layers remain aligned and mutually informed, while curriculum training provides a structured progression of learning objectives, reducing the complexity of exploration in early stages.

Our key contributions are as follows:
\begin{itemize}[left=0em]
    \item We introduce a hierarchical DRL framework that decouples decision-making into strategic and tactical layers, enabling scalable and efficient learning.
    \item We propose state-sharing and curriculum training mechanisms to address the challenges of exploration and non-stationarity.
    \item We demonstrate the effectiveness of our framework through experiments on multiple large-scale RTS scenarios, achieving a 15\%–25\% higher win rate compared to existing methods.
\end{itemize}

By addressing the core challenges of scalability, exploration, and stability in RTS games, this work represents a significant step forward in advancing DRL's applicability to complex, real-world environments.
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
