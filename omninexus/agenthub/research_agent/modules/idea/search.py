"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from browsergym.core.action.highlevel import HighLevelActionSet
from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

# from browsergym/core/action/highlevel.py
_browser_action_space = HighLevelActionSet(
    subsets=['bid', 'nav'],
    strict=False,  # less strict on the parsing of the actions
    multiaction=True,  # enable to agent to take multiple actions at once
)

_IDEA_RELEVANT_RESEARCH_RETRIEVAL_DESCRIPTION = """Interact with the browser using Python code to search for LATEST (e.g., 2025, 2024 or 2023) research papers that are
relevant to the given paper or topic.

# The python code should interact with the browser using the following functions:
* Multiple actions can be provided at once, but will be executed sequentially without any feedback from the page.
More than 2-3 actions usually leads to failure or unexpected behavior.

## For example, the following code snippet can be used to search for relevant research papers:
fill('a12', 'example with "quotes"')
click('a51')
click('48', button='middle', modifiers=['Shift'])

# Search results should be organized in the following format:
* You MUST first identify the RESEARCH DOMAIN you are addressing, such as Large Language Models (LLMs), Reinforcement Learning and Human Feedback (RLHF), or Multimodal Learning.
* You MUST identify at least 5 LATEST research directions or topics related to the paper or topic.
* For each research direction, you MUST search for at least 3 latest research papers that are relevant to that direction.
* You MUST provide the base information of the research papers, including the RESEARCH DIRECTION, TITLE, PAPER LINK, DATE, number of CITED BY, and ABSTRACT.
* You MUST read the abstract, introduction, and conclusion, limitations, and future work of the research papers. And then, you MUST summarize the CONTRIBUTIONS, LIMITATIONS and IMPROVEMENTS of the research papers.

## For example,  the [research_domain]_papers.md file may look like this:
(this is the start of the file)
** RESEARCH DOMAIN **
* Large Language Models (LLMs)

** RESEARCH DIRECTIONS **
* Transformer Models
* Pre-trained Language Models

** PAPER LIST **
* ITERATION 1:
- Paper 1:
    * RESEARCH DIRECTION: Transformer Models
    * TITLE: Attention is All You Need
    * PAPER LINK: https://arxiv.org/abs/1706.03762
    * DATE: June 2017
    * CITED BY: 145444
    * ABSTRACT: The dominant sequence transduction models are based on complex recurrent or
                convolutional neural networks that include an encoder and a decoder. The best
                performing models also connect the encoder and decoder through an attention
                mechanism. We propose a new simple network architecture, the Transformer,
                based solely on attention mechanisms, dispensing with recurrence and convolutions
                entirely. Experiments on two machine translation tasks show these models to
                be superior in quality while being more parallelizable and requiring significantly
                less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German
                translation task, improving over the existing best results, including
                ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
                our model establishes a new single-model state-of-the-art BLEU score of 41.8 after
                training for 3.5 days on eight GPUs, a small fraction of the training costs of the
                best models from the literature. We show that the Transformer generalizes well to
                other tasks by applying it successfully to English constituency parsing both with
                large and limited training data.
    * CONTRIBUTIONS:
        - The paper introduces the Transformer, a novel architecture that relies entirely on self-attention mechanisms, removing the need for recurrence or convolution.
        - It significantly improves computational efficiency by enabling better parallelization and faster training on long sequences.
        - The multi-head attention mechanism is proposed to allow the model to attend to multiple positions and subspaces simultaneously.
        - Positional encodings are introduced to represent sequence order in the absence of recurrence.
        - The Transformer achieves state-of-the-art results on machine translation tasks, including WMT 2014 English-to-German and English-to-French benchmarks.
        - The model demonstrates that attention-based architectures can outperform traditional recurrent models while simplifying overall design.
    * LIMITATIONS:
        - The self-attention mechanism has quadratic time and memory complexity, making it computationally expensive for long sequences.
        - The use of fixed positional encodings limits flexibility when handling sequences longer than those seen during training.
        - The Transformer requires large datasets and significant computational resources, which can be prohibitive for some users.
        - Unlike RNNs, the Transformer lacks an inherent sequential inductive bias, which may be a disadvantage for strict sequence modeling tasks.
        - The model struggles to capture long-range dependencies in extremely long sequences without additional modifications.
        - Its highly parameterized design can lead to overfitting and poor generalization on tasks with limited training data.
    * IMPROVEMENTS:
        - Use sparse or approximate attention to reduce the quadratic complexity of self-attention for long sequences.
        - Replace fixed positional encodings with learnable or dynamic positional embeddings to enhance flexibility.
        - Incorporate mechanisms like sliding windows or memory-augmented modules to better handle long-range dependencies and extended contexts.
- Paper 2:
    * RESEARCH DIRECTION: Pre-trained Language Models
    * TITLE: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    * PAPER LINK: https://arxiv.org/abs/1810.04805
    * DATE: October 2018
    * CITED BY: 121777
    * ABSTRACT: We introduce BERT (Bidirectional Encoder Representations from Transformers), a new
                method of pre-training language representations that achieves state-of-the-art results
                on a wide range of natural language processing tasks. Unlike previous works, BERT is
                deeply bidirectional, jointly conditioning on both left and right context in all layers,
                which improves performance over unidirectional language models. BERT is pre-trained using
                two unsupervised tasks: Masked Language Modeling (MLM), where random words are masked, and
                the model predicts them, and Next Sentence Prediction (NSP), which helps the model understand
                sentence relationships. BERT achieves new state-of-the-art results on 11 NLP tasks, including
                the GLUE benchmark, SQuAD v1.1, and SQuAD v2.0, demonstrating its versatility and effectiveness.
    * CONTRIBUTIONS:
        - Introduces BERT, a pre-trained bidirectional Transformer-based model that learns deep contextualized word representations.
        - Proposes the Masked Language Modeling (MLM) task to enable bidirectional pre-training of Transformers by randomly masking input tokens.
        - Develops the Next Sentence Prediction (NSP) task to help the model capture sentence-level relationships for downstream tasks.
        - Demonstrates the effectiveness of BERT on diverse NLP tasks, achieving state-of-the-art performance on GLUE, SQuAD v1.1, and SQuAD v2.0 benchmarks.
        - Shows that pre-training a deep Transformer on a large text corpus can be fine-tuned to achieve exceptional results on specific tasks with minimal modifications.
        - Establishes the pre-train, fine-tune paradigm, which has become a foundation for modern NLP research.
    * LIMITATIONS:
        - BERT’s quadratic complexity in self-attention limits its scalability to very long sequences.
        - The masked token prediction leads to a mismatch between pre-training and fine-tuning, as the model does not encounter masked tokens during downstream tasks.
        - Training BERT requires significant computational resources, making it less accessible to smaller research groups.
        - BERT does not handle long-term dependencies efficiently, as it has a fixed input length (e.g., 512 tokens).
        - The model’s fine-tuning process can be unstable for certain tasks, requiring careful hyperparameter tuning.
        - BERT’s bidirectional nature may lead to information leakage when applied to autoregressive tasks like language generation.
    * IMPROVEMENTS:
        - Optimize self-attention mechanisms using sparse attention or sliding windows to handle longer sequences efficiently.
        - Address pre-training and fine-tuning mismatch by introducing more realistic masking strategies or dynamic masking during fine-tuning.
        - Develop lightweight or distilled versions of BERT (e.g., DistilBERT) to reduce computational costs while maintaining performance.

* ITERATION 2:
...
* ITERATION 3:
...
* ITERATION 4:
...
* ITERATION 5:
...

** SUMMARY **
* RESEARCH DIRECTIONS:
    - Transformer Models: Focus on architectures based on self-attention mechanisms like the Transformer, enabling parallelization and efficient sequence processing.
    - Pre-trained Language Models: Explore methods for pre-training deep contextualized representations like BERT, leveraging large text corpora for transfer learning.
* CHALLENGES:
    - Computational Complexity: Models like the Transformer and BERT face challenges with quadratic self-attention complexity, limiting scalability to long sequences.
    - Pre-training and Fine-tuning Mismatch: Mismatches between pre-training objectives (e.g., MLM) and downstream tasks can hinder model performance and generalization.
* IMPROVEMENTS:
    - Exploring more efficient and scalable pre-training methods for large language models.
    - Investigating ways to improve the interpretability and explainability of deep learning models.
(this is the end of the file)
"""

_BROWSER_TOOL_DESCRIPTION = """
The following 15 functions are available. Nothing else is supported.

goto(url: str)
    Description: Navigate to a url.
    Examples:
        goto('http://www.example.com')

go_back()
    Description: Navigate to the previous page in history.
    Examples:
        go_back()

go_forward()
    Description: Navigate to the next page in history.
    Examples:
        go_forward()

noop(wait_ms: float = 1000)
    Description: Do nothing, and optionally wait for the given time (in milliseconds).
    You can use this to get the current page content and/or wait for the page to load.
    Examples:
        noop()

        noop(500)

scroll(delta_x: float, delta_y: float)
    Description: Scroll horizontally and vertically. Amounts in pixels, positive for right or down scrolling, negative for left or up scrolling. Dispatches a wheel event.
    Examples:
        scroll(0, 200)

        scroll(-50.2, -100.5)

fill(bid: str, value: str)
    Description: Fill out a form field. It focuses the element and triggers an input event with the entered text. It works for <input>, <textarea> and [contenteditable] elements.
    Examples:
        fill('237', 'example value')

        fill('45', 'multi-line\nexample')

        fill('a12', 'example with "quotes"')

select_option(bid: str, options: str | list[str])
    Description: Select one or multiple options in a <select> element. You can specify option value or label to select. Multiple options can be selected.
    Examples:
        select_option('a48', 'blue')

        select_option('c48', ['red', 'green', 'blue'])

click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
    Description: Click an element.
    Examples:
        click('a51')

        click('b22', button='right')

        click('48', button='middle', modifiers=['Shift'])

dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
    Description: Double click an element.
    Examples:
        dblclick('12')

        dblclick('ca42', button='right')

        dblclick('178', button='middle', modifiers=['Shift'])

hover(bid: str)
    Description: Hover over an element.
    Examples:
        hover('b8')

press(bid: str, key_comb: str)
    Description: Focus the matching element and press a combination of keys. It accepts the logical key names that are emitted in the keyboardEvent.key property of the keyboard events: Backquote, Minus, Equal, Backslash, Backspace, Tab, Delete, Escape, ArrowDown, End, Enter, Home, Insert, PageDown, PageUp, ArrowRight, ArrowUp, F1 - F12, Digit0 - Digit9, KeyA - KeyZ, etc. You can alternatively specify a single character you'd like to produce such as "a" or "#". Following modification shortcuts are also supported: Shift, Control, Alt, Meta, ShiftLeft, ControlOrMeta. ControlOrMeta resolves to Control on Windows and Linux and to Meta on macOS.
    Examples:
        press('88', 'Backspace')

        press('a26', 'ControlOrMeta+a')

        press('a61', 'Meta+Shift+t')

focus(bid: str)
    Description: Focus the matching element.
    Examples:
        focus('b455')

clear(bid: str)
    Description: Clear the input field.
    Examples:
        clear('996')

drag_and_drop(from_bid: str, to_bid: str)
    Description: Perform a drag & drop. Hover the element that will be dragged. Press left mouse button. Move mouse to the element that will receive the drop. Release left mouse button.
    Examples:
        drag_and_drop('56', '498')

upload_file(bid: str, file: str | list[str])
    Description: Click an element and wait for a "filechooser" event, then select one or multiple input files for upload. Relative file paths are resolved relative to the current working directory. An empty list clears the selected files.
    Examples:
        upload_file('572', '/home/user/my_receipt.pdf')

        upload_file('63', ['/home/bob/Documents/image.jpg', '/home/bob/Documents/file.zip'])
"""


for _, action in _browser_action_space.action_set.items():
    assert (
        action.signature in _BROWSER_TOOL_DESCRIPTION
    ), f'Browser description mismatch. Please double check if the BrowserGym updated their action space.\n\nAction: {action.signature}'
    assert (
        action.description in _BROWSER_TOOL_DESCRIPTION
    ), f'Browser description mismatch. Please double check if the BrowserGym updated their action space.\n\nAction: {action.description}'

RelevantResearchRetrievalTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='relevant_research_retrieval',
        description=_IDEA_RELEVANT_RESEARCH_RETRIEVAL_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'code': {
                    'type': 'string',
                    'description': (
                        'The Python code that interacts with the browser to search for relevant research papers.\n'
                        + _BROWSER_TOOL_DESCRIPTION
                    ),
                }
            },
            'required': ['code'],
        },
    ),
)
