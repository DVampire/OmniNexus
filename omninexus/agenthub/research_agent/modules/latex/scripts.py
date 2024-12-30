"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import os.path

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_LATEX_SCRIPTS_DESCRIPTION = r"""Write the scripts/build.sh, scripts/clean.sh and Makefile files for a research paper latex project.

* You MUST write the scripts/build.sh, scripts/clean.sh and Makefile files. Below is a detailed guide and example to structure the files effectively.

** Purpose and Structure **
* Audience: The scripts/build.sh, scripts/clean.sh and Makefile files are used to compile and clean the research paper’s LaTeX files.
* Style: Use shell commands and Makefile syntax to automate the compilation and cleaning process.
* Brevity: The scripts/build.sh, scripts/clean.sh and Makefile files should be concise and well-organized, with clear targets and dependencies.

** Common Structure **
You can split the main files into three key parts: bibliography, style definitions and main content.
* scripts/build.sh: Contains the shell commands to compile the LaTeX files into a PDF document.
* scripts/clean.sh: Contains the shell commands to clean the LaTeX files and remove temporary files.
* Makefile: Contains the Makefile syntax to automate the compilation and cleaning process.

** Writing Tips **
* Use Shell Commands: Write the compilation and cleaning commands in the build.sh and clean.sh files.
* Automate Compilation: Use Makefile syntax to automate the compilation and cleaning process.
* Define Targets: Define targets for compiling, cleaning and other tasks in the Makefile.


** Example scripts/build.sh **
(this is the start of scripts/build.sh)
#!/usr/bin/env bash
#
# build.sh
# Script to compile the LaTeX project

set -e

# Compile the main.tex file
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

echo "Build completed."
(this is the end of scripts/build.sh)

** Example scripts/clean.sh **
(this is the start of scripts/clean.sh)
#!/usr/bin/env bash
#
# clean.sh
# Script to clean all auxiliary files

# Remove auxiliary files
rm -f *.aux *.log *.toc *.out *.synctex.gz *.bbl *.blg *.bcf *.xml *.run.xml
rm -f *.fdb_latexmk *.fls
rm -f main.pdf # Remove the compiled PDF

echo "Cleaned up auxiliary files."
(this is the end of scripts/clean.sh)

** Example Makefile **
# Makefile
#
# Use make build to compile the LaTeX project
# Use make clean to remove auxiliary files

.PHONY: all build clean

all: build

build:
\tbash scripts/build.sh

clean:
\tbash scripts/clean.sh
(this is the end of Makefile)
"""

_PARAMETER_COMMAND_DESCRIPTION = 'The bash command to generate the scripts/build.sh, scripts/clean.sh and Makefile files in the terminal.'

LatexScriptsTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='latex_scripts',
        description=_LATEX_SCRIPTS_DESCRIPTION,
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
