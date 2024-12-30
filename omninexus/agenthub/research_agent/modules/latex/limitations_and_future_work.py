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
While our proposed framework demonstrates significant performance gains on benchmark datasets, it faces several limitations. First, we rely on a relatively small and homogeneous data source, which may not reflect the diversity of real-world conditions. Extending our approach to larger, more varied datasets could improve generalizability and reduce overfitting. Second, our method assumes i.i.d.\ samples and may struggle under non-stationary or adversarial environments, where continuous adaptation or novel training strategies may be needed. Third, the computational cost of the model remains high, potentially limiting its application to resource-constrained settings. Techniques such as model compression, pruning, or distillation could reduce inference overhead and enable real-time deployment.

From an ethical perspective, we acknowledge that biases present in the original dataset may inadvertently propagate through the model. We have taken steps to mitigate risks by performing a preliminary bias audit; however, a thorough fairness analysis is necessary to ensure equitable outcomes for diverse user groups. Additionally, data privacy and consent procedures were approved under local Institutional Review Board (IRB) guidelines, yet future work should explore privacy-preserving techniques (e.g., differential privacy or federated learning) to further safeguard sensitive information.

Looking forward, we plan to investigate hierarchical or hybrid architectures that can adapt to dynamic input distributions. We also aim to collaborate with domain experts in healthcare and finance to address domain-specific requirements and regulatory constraints. By combining our framework with emerging explainable AI methodologies, we hope to improve interpretability and encourage responsible deployment in critical decision-making scenarios. Addressing these challenges will further validate the robustness, fairness, and broad applicability of the proposed approach.
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
