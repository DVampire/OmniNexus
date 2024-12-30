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
In this paper, we presented a novel transformer-based encoder-decoder architecture for real-time anomaly detection in streaming data. Through extensive experiments on three benchmark datasets, we demonstrated that our approach not only outperforms traditional autoregressive models by up to 12\% in accuracy, but also maintains low latency suitable for high-throughput environments. Furthermore, our systematic ablation study confirms the positive impact of the multi-head self-attention mechanism and residual connections on model stability and scalability.

These findings underscore the potential for integrating attention-based methods into next-generation anomaly detection pipelines across various domains, such as cybersecurity, Internet-of-Things (IoT) monitoring, and health analytics. While our work focused primarily on the predictive performance of the proposed model, the demonstrated efficiency and interpretability form a foundation for future explorations into data governance, privacy-preserving techniques, and cross-domain transfer learning. Overall, the results suggest that our framework offers a promising direction for robust, real-time anomaly detection that can adapt to dynamic and evolving data streams.
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
