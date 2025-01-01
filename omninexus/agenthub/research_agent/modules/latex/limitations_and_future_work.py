"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_LATEX_LIMITATIONS_AND_FUTURE_WORK_DESCRIPTION = r"""Write the LIMITATIONS AND FUTURE WORK section of a research paper.

* You MUST write the Limitations and Future Work section in LaTeX format and place it in the file sections/limitations_and_future_work.tex. Below is a detailed guide and example to structure this section effectively.

** Purpose and Structure **
* Audience: Readers (including reviewers) who want to see a balanced view of your work—knowing the shortcomings helps validate your credibility and can inspire further research.
* Style: Demonstrate critical thinking about your method’s constraints, as well as vision for future exploration. Use direct, clear, and concise language.
* Importance: Properly acknowledging limitations often strengthens your argument by showing that you understand the boundaries of your work; it also naturally leads into potential improvements.
* Limitations and Future Work should be around 250 words in English. Use this space to reflect on the challenges and opportunities for your research.

** Common Structure **
You can organize this section into two main parts (Limitations, then Future Work) or blend them into a single narrative.
* Limitations
    - State the constraints of your dataset, methodology, or theoretical assumptions.
    - Discuss any issues related to scalability, complexity, or generalizability.
    - Identify potential biases or concerns about reproducibility.
* Future Work
    - Propose concrete next steps or research directions (e.g., exploring more diverse data sets, refining the algorithm, combining with other techniques).
    - Highlight opportunities for interdisciplinary collaboration.
    - Indicate how addressing these limitations could expand the impact and validity of the research.

** Writing Tips **
* Be Honest: Acknowledging real constraints helps reviewers trust the value of your findings.
* Be Constructive: Link each limitation to a potential solution or avenue of improvement if possible.
* Specificity: Provide enough detail so readers clearly understand the nature and degree of each limitation. Avoid vague language.
* Forward-Looking: Show enthusiasm for future innovation and solutions; don’t dwell overly on the negative.
* Ethics: If your work involves human subjects, sensitive data, or potential real-world harm, briefly mention how you handled ethical review, consent, and potential risks.

** Example Limitations and Future Work **
For demonstration only. Adapt it to match your specific research and results. Here is an example limitations and future work section for a research paper:
(this is the start of sections/limitations_and_future_work.tex)
\section{Limitations and Future Work}
\label{sec:limitations_and_future_work}
While our proposed framework demonstrates significant performance gains on benchmark datasets, it has several limitations. First, the reliance on a relatively small and homogeneous dataset limits its generalizability to diverse real-world conditions. Extending our approach to larger, more varied datasets could improve its robustness and reduce the risk of overfitting. Second, our method assumes i.i.d. samples, which poses challenges in non-stationary or adversarial environments where data distributions shift over time. Addressing these challenges may require continuous adaptation or novel training strategies tailored to dynamic settings. Third, the computational cost of the model remains high, potentially hindering its deployment in resource-constrained environments. Techniques such as model compression, pruning, or distillation could reduce inference overhead and enable real-time applications without compromising performance.

From an ethical standpoint, the biases inherent in the dataset may inadvertently propagate through the model. Although we performed a preliminary bias audit, a thorough fairness analysis is required to ensure equitable outcomes for diverse user groups. Furthermore, while we adhered to Institutional Review Board (IRB) guidelines for data privacy and consent, future work should explore advanced privacy-preserving methods, such as differential privacy or federated learning, to strengthen data security and user trust.

Looking forward, we plan to explore hybrid or hierarchical architectures that adapt dynamically to shifting input distributions. Collaborating with domain experts in areas like healthcare and finance will help tailor the framework to meet domain-specific requirements and regulatory standards. Additionally, integrating explainable AI methodologies will enhance interpretability, fostering responsible and transparent deployment in critical decision-making scenarios.
(this is the end of sections/limitations_and_future_work.tex)
"""

_PARAMETER_COMMAND_DESCRIPTION = 'The bash command to generate the limitations and future work section in the terminal.'

LatexLimitationsAndFutureWorkTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='latex_limitation_and_future_work',
        description=_LATEX_LIMITATIONS_AND_FUTURE_WORK_DESCRIPTION,
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
