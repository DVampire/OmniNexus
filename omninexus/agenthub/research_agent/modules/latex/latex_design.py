"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_LATEX_DESIGN_DESCRIPTION = """Design the organization structure for a LaTeX-based paper write-up project, along with a bash command to generate it directly in the terminal.
* The assistant should organize the project structure by ensuring modularity and maintainability.
* The assistant should name the project appropriately based on the task, and its directory structure is as follows:
[project_name]                   # Root directory of the project, [project_name] should be replaced with the actual project name.
├── sections                      # Directory containing all individual sections of the paper.
│   ├── abstract.tex              # Abstract section.
│   ├── introduction.tex          # Introduction section.
│   ├── related_work.tex          # Related work section.
│   ├── method.tex                # Methods section.
│   ├── empirical_results.tex     # Results section.
│   ├── limitations_and_future_work.tex # Limitations and future work section.
│   ├── conclusion.tex            # Conclusion section.
├── figures                       # Directory for all figures and graphics.
│   ├── overview.tex              # Overview in Introduction section. Use TikZ for creating diagrams.
│   └── architecture.tex          # Model architecture diagram file. Use TikZ for creating diagrams.
├── tables                        # Directory for all table files.
│   └── table1.tex                # Example table file.
├── main.bib            # Bibliography file containing references.
├── main.sty            # Custom LaTeX style file for the paper.
├── main.tex            # Main LaTeX file that compiles all sections.
├── scripts                       # Auxiliary scripts for automation.
│   ├── build.sh                  # Script to compile the LaTeX project.
│   ├── clean.sh                  # Script to clean up auxiliary files.
│   └── README.md                 # Documentation for the scripts.
├── LICENSE                       # License file for the project.
├── Makefile                      # Makefile for building or cleaning the project.
├── README.md                     # Main documentation for the project.
└── .gitignore                    # File for ignoring unnecessary files in Git.

* The assistant should ensure the structure is well-documented and follows best practices for LaTeX project organization.
"""

_PARAMETER_COMMAND_DESCRIPTION = (
    'The bash command to generate the LaTeX-based paper write-up in the terminal.'
)

LatexDesignTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='latex_design',
        description=_LATEX_DESIGN_DESCRIPTION,
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
