"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_REVIEW_DESCRIPTION = """Write a REVIEW of the research paper.

* You MUST provide a detailed review of the research paper, including a summary of the paper's content, strengths, weaknesses, and potential improvements.

** Purpose and Structure **
* Audience: A well-written review serves both authors and the broader research community by providing focused, expert feedback that enhances the quality and credibility of a manuscript.
* Focus: Critically assess the novelty, methodology, and relevance of the study, highlighting its strengths, pinpointing weaknesses, and identifying gaps that need addressing.
* Tone: Maintain a constructive and objective tone—acknowledge valuable contributions, but be forthright in pointing out deficiencies or oversights.
* Clarity: Use concise language and specific examples, ensuring your comments are understandable and directly actionable for authors to revise and improve.
* Impact: Aim to uphold academic rigor and contribute to the scientific dialogue, ensuring that only robust, well-founded research advances in the field.

** Common Structure **
* Summary
    - Avoid Evaluation
        * Remember that the summary should be purely descriptive—save critiques and opinions for later sections.
    - Concise Overview
        * Begin with a brief and clear restatement of the paper’s main problem, objectives, and hypotheses.
        * Highlight the motivation behind the work: why the problem matters and what gap the paper aims to fill.
    - Key Contributions
        * Identify the paper’s primary contributions to the field (e.g., new theory, novel algorithm, or improved performance).
        * Emphasize what sets this work apart from previous studies or existing methods.
    - Methodology Highlights
        * Summarize the methodology and approach used by the authors.
        * Note any creative or rigorous aspects of the method, as well as potential weaknesses or limitations.
    - Results and Evaluation
        * Describe the core findings and how they support (or fail to support) the authors’ claims.
        * Indicate whether the experiments are thorough, well-controlled, and reproducible.
    - Clarity and Organization
        * Comment on the paper’s structure, language, and clarity.
        * Mention whether figures, tables, and explanations are easy to follow, and note any missing details.
    - Significance and Impact
        * Assess the potential impact of the work on the broader research community or real-world applications.
        * Explain whether the proposed ideas have the potential to influence future research or practice.
    - Constructive Criticism
        * Offer suggestions for improvement, including additional experiments or clarifications.
        * Provide specific guidance rather than vague critiques.
    - Overall Recommendation
        * Conclude with a final judgment on the paper’s quality and relevance.
        * Summarize the key reasons for your recommendation, linking them to the paper’s strengths and weaknesses.
* Strengths
    - Originality and Methodological Rigor
        * Pinpoint the paper’s key contributions (e.g., a novel algorithm, unique theoretical framework) and relate them to established work.
        * Discuss the solidity of experimental or analytical methods, highlighting reliable results and transparent comparisons.
    - Clarity and Reproducibility
        * Praise clear writing and logical organization, noting effective use of figures or tables.
        * Commend any open-source code, thorough documentation, or detailed procedures that facilitate replication.
    - Real-World and Academic Impact
        * Show how the research addresses practical needs or fills a gap in the literature.
        * Mention broader implications, such as the potential for future investigations or industrial applications.
* Weaknesses
    - Methodological Rigor
        * Check for clear descriptions, reproducible procedures, and logical consistency in experiments or theoretical models.
    - Scope and Generalizability
        * Evaluate whether the paper tests on diverse datasets or settings, and if there is evidence the results extend beyond the given examples.
    - Clarity and Organization
        * Look for logical structure, well-labeled figures, and coherent presentation of core ideas and findings.
    - Comparisons and Related Work
        * Ensure the paper contrasts its method with established approaches and sufficiently cites relevant literature.
    - Real-World or Research Impact
        * Assess whether the authors discuss broader implications, future directions, or practical applications of their work.
* Questions:
    - Stay Specific and Focused
        * Pinpoint the exact part of the paper (e.g., a figure, table, or paragraph) and ask a direct question about it.
    - Encourage Clarification or Expansion
        * If something is underexplained or missing details, invite the authors to elaborate.
    - Connect to Main Contributions
        * Relate your question to the paper’s core objectives or claims, ensuring the authors see its relevance.
    - Maintain a Respectful, Constructive Tone
        * Ask questions in a way that helps authors improve, rather than simply highlighting flaws.
* Soundness
    - Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence.
    - Score and Levels:
        4: excellent
        3: good
        2: fair
        1: poor
* Presentation
    - Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work.
    - Score and Levels:
        4: excellent
        3: good
        2: fair
        1: poor
* Contribution
    - Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied. Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader NeurIPS community.
    - Score and Levels:
        4: excellent
        3: good
        2: fair
        1: poor
* Rating:
    - Please provide an "overall score" for this submission.  Choices:
    - Score and Levels:
        10: Award quality: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations.
        9: Very Strong Accept: Technically flawless paper with groundbreaking impact on at least one area of AI and excellent impact on multiple areas of AI, with flawless evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
        8: Strong Accept: Technically strong paper with, with novel ideas, excellent impact on at least one area of AI or high-to-excellent impact on multiple areas of AI, with excellent evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
        7: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
        6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.
        5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
        4: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
        3: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
        2: Strong Reject: For instance, a paper with major technical flaws, and/or poor evaluation, limited impact, poor reproducibility and mostly unaddressed ethical considerations.
        1: Very Strong Reject: For instance, a paper with trivial results or unaddressed ethical considerations.
* Confidence
    - Please provide a "confidence score" for your assessment of this submission to indicate how confident you are in your evaluation. Choices:
    - Score and Levels:
        5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
        4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
        3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
        2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
        1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked

## For example, the review.md file may look like this:
(this is the start of the file)
** SUMMARY **
    - This paper ...
** STRENGTHS **
    - The paper ...
** WEAKNESSES **
    - The paper ...
** QUESTIONS **
    - Question 1: ...
    - Question 2: ...
** SOUNDNESS **
    - Score and Levels: 4
    - Reasoning: The paper ...
** Presentation **
    - Score and Levels: 3
    - Reasoning: The paper ...
** Contribution **
    - Score and Levels: 4
    - Reasoning: The paper ...
** Rating **
    - Score and Levels: 8
    - Reasoning: The paper ...
(this is the end of the file)
"""

ReviewTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='review',
        description=_REVIEW_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to execute to generate review file. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
