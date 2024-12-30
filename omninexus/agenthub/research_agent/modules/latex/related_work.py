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

NOTE: When cite a reference in LaTeX, you MUST add the BibTeX of it in the file main.bib. If the item is already in the main.bib, you can directly use the citation key in the LaTeX file.

** Example Related Work **
For demonstration only. Adapt it to match your specific research and results. Here is an example related work section for a research paper:
(this is the start of sections/related_work.tex)
\section{Related Work}
\label{sec:related_work}

The field of generative models has witnessed significant advancements in recent years, driven by the development of autoregressive (AR) and diffusion-based methods. In this section, we review the related work across three key areas: autoregressive modeling, diffusion models, and multi-scale approaches.

\subsection{Autoregressive Modeling}
Autoregressive models have achieved remarkable success in various domains, particularly in natural language processing (NLP)~\cite{brown2020language, radford2019language} and audio generation~\cite{oord2016wavenet}. These models rely on a sequential token prediction strategy, where each token is conditioned on its predecessors, making them powerful for capturing sequential dependencies. In the domain of computer vision, pioneering works such as VQ-VAE~\cite{van2017neural} and DALL-E~\cite{ramesh2021zero} introduced the use of visual tokenization to adapt autoregressive techniques for image generation. However, these methods are limited by their reliance on raster-scan token ordering, which can result in inefficiencies and suboptimal scalability~\cite{esser2021taming}.

\subsection{Diffusion Models}
Diffusion models~\cite{ho2020denoising, song2020score} have emerged as a dominant paradigm for high-quality image synthesis, leveraging iterative denoising processes to model data distributions. Recent advancements, such as the Diffusion Transformer (DiT)~\cite{peebles2022scalable} and Stable Diffusion~\cite{rombach2022high}, have demonstrated state-of-the-art performance on various benchmarks, outperforming many autoregressive counterparts in terms of both quality and diversity. Despite their success, diffusion models are computationally intensive and require numerous forward passes during inference, which limits their applicability in real-time scenarios~\cite{nichol2021improved}.

\subsection{Comparison with Our Work}
Our proposed Visual AutoRegressive (VAR) modeling diverges from traditional raster-scan autoregressive techniques by redefining the prediction process as “next-scale prediction.” This approach draws inspiration from multi-scale designs~\cite{lin2017feature, chen2018encoder} and integrates them into an autoregressive framework. Unlike diffusion models~\cite{ho2020denoising, peebles2022scalable}, VAR achieves comparable or superior image quality with significantly faster inference, making it more suitable for practical applications. Furthermore, VAR models exhibit scaling laws and zero-shot generalization capabilities akin to large language models~\cite{brown2020language}, positioning them as a promising direction for unified visual generative learning.
(this is the end of sections/related_work.tex)

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
