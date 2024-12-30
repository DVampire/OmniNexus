"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import os.path

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_LATEX_MAIN_DESCRIPTION = r"""Write the main.bib, main.sty and main.tex files for a research paper.

* You MUST write the research paper’s main.bib, main.sty and main.tex files in LaTeX format and place them in the files main.bib, main.sty and main.tex. Below is a detailed guide and example to structure the files effectively.

** Purpose and Structure **
* Audience: The main files are the core components of a research paper, containing the bibliography, style definitions and main content.
* Style: Use LaTeX commands and packages to structure the paper effectively, following the guidelines of the target conference or journal.
* Brevity: The main files should be concise and well-organized, with clear sections and references.

** Common Structure **
You can split the main files into three key parts: bibliography, style definitions and main content.
* main.bib: Contains the bibliography entries in BibTeX format, listing all references cited in the paper.
* main.sty: Contains the style definitions, including packages, formatting rules and custom commands.
* main.tex: Contains the main content of the paper, structured into sections, subsections and paragraphs.

** Writing Tips **
* Use BibTeX: Maintain the bibliography in a separate file (main.bib) to manage references efficiently.
* Define Style: Customize the paper’s appearance and layout by modifying the style definitions (main.sty).
* Structure Content: Organize the paper’s content logically.

** Example main.bib **
(this is the start of main.bib)
@article{goodfellow2014generative,
  title={Generative adversarial nets},
  author={Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua},
  journal={Advances in neural information processing systems},
  volume={27},
  pages={2672--2680},
  year={2014}
}
...
(this is the end of main.bib)

** Example main.sty **
Because we have a template for main.sty, we don't need to write it here. You can copy the template from the file `main.sty` in the `../../template/latex/NeurIPS_2024/`.

** Example main.tex **
(this is the start of main.tex)
\documentclass{article}

\usepackage{main}
\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc}    % use 8-bit T1 fonts
\usepackage{hyperref}       % hyperlinks
\usepackage{url}            % simple URL typesetting
\usepackage{booktabs}       % professional-quality tables
\usepackage{amsfonts}       % blackboard math symbols
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{xcolor}         % colors
\usepackage{appendix}

\title{Deep Reinforcement Learning in Real-Time Strategy Games}

\author{%
  David S.~Hippocampus\thanks{Use footnote for providing further information
    about author (webpage, alternative address)---\emph{not} for acknowledging
    funding agencies.} \\
  Department of Computer Science\\
  Cranberry-Lemon University\\
  Pittsburgh, PA 15213 \\
  \texttt{hippo@cs.cranberry-lemon.edu} \\
  % examples of more authors
  % \And
  % Coauthor \\
  % Affiliation \\
  % Address \\
  % \texttt{email} \\
}


\begin{document}


\maketitle

\input{sections/abstract}

\input{sections/introduction}

\input{sections/related_work}

\input{sections/method}

\input{sections/empirical_results}

\input{sections/limitations_and_future_work}

\input{sections/conclusion}

\newpage

\bibliography{main}
\bibliographystyle{plain}

\newpage

\newpage
\appendix
\appendixpage

\end{document}
(this is the end of main.tex)
"""

_PARAMETER_COMMAND_DESCRIPTION = 'The bash command to generate the main.bib, main.sty and main.tex files in the terminal.'

LatexMainTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='latex_main',
        description=_LATEX_MAIN_DESCRIPTION,
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
