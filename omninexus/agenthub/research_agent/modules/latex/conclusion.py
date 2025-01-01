"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_LATEX_CONCLUSION_DESCRIPTION = r"""Write the CONCLUSION section of a research paper.

* You MUST write the “Conclusion” section in LaTeX format and place it in the file sections/conclusion.tex.
* Below is a detailed guide and example to structure this section effectively.

** Purpose and Structure **
* Audience: Readers (including reviewers) who want a concise summary of the main findings and significance of your research.
* Style: Summarize your key contributions, reinforce their importance, and conclude with a thoughtful final statement on the impact of your work. Be direct and concise.
* Importance: Conclusions often provide the lasting impression. A concise, impactful conclusion helps situate your findings in the broader research landscape and can encourage further interest or inquiry.
* Conclusion should be around 200 words in English. Use this space to highlight the significance of your results and their implications.

** Common Structure **
* Recap of Problem and Methods
   - Briefly restate the problem or question addressed.
   - Summarize the methods or approach in one or two sentences.
* Key Findings and Contributions
   - Highlight the main results or contributions that your research offers.
   - Emphasize how these findings compare to or advance beyond existing work.
* Final Reflection or Implications
   - Reflect on the broader implications of your results (academic, industrial, societal).
   - Avoid introducing entirely new claims or extensive material that wasn’t in the main text.

** Writing Tips **
* Keep It Concise: Aim for a few concise paragraphs.
* Be Specific: Mention your main results and achievements without re-describing all the details.
* Avoid Repetition: Do not duplicate entire sections of your paper; focus on the essence.
* Keep a Positive Yet Realistic Tone: Emphasize the strengths of your solution without overselling.

** Example Conclusion **
This is for demonstration—adapt to match your specific research context. Here is an example for sections/conclusion.tex
(this is the start of sections/conclusion.tex)
\section{Conclusion}
\label{sec:conclusion}

This work addresses the challenges of applying deep reinforcement learning (DRL) to real-time strategy (RTS) games, which feature large action spaces, vast state representations, and complex multi-agent dynamics. We proposed a hierarchical DRL framework that decomposes decision-making into strategic and tactical layers, simplifying exploration and stabilizing learning by separating macro-level tasks, such as resource management, from micro-level tasks, like unit positioning and targeting. A shared state mechanism ensures cohesive decisions across temporal and spatial scales, while curriculum training progressively builds robust policies adaptable to dynamic and large-scale environments. Experimental evaluations demonstrated a 15\%–25\% higher win rate compared to baseline methods, affirming the efficacy of the hierarchical structure and curriculum training. Ablation studies highlighted the critical roles of both components, with performance and stability significantly degraded when either was removed. These findings underscore the potential of structured, multi-level decision-making in addressing the inherent complexity of RTS games. By integrating strategic foresight with tactical precision, our framework offers a scalable and efficient approach for real-time decision-making in dynamic environments. Beyond gaming, the proposed method has broader implications for domains requiring hierarchical reasoning and adaptability, such as robotics and autonomous systems. Future research could extend this framework to even more complex, non-stationary scenarios, reinforcing its versatility and applicability. Ultimately, this study demonstrates that hierarchical frameworks are a promising direction for advancing DRL in environments where complexity and dynamism intersect.
(this is the end of sections/conclusion.tex)
"""

_PARAMETER_COMMAND_DESCRIPTION = (
    'The bash command to generate the conclusion section in the terminal.'
)

LatexConclusionTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='latex_conclusion',
        description=_LATEX_CONCLUSION_DESCRIPTION,
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
