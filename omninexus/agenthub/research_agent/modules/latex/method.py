"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_LATEX_METHOD_DESCRIPTION = r"""Write the METHOD section of a research paper.

* You MUST first design an appropriate structure for the method section based on the paper's idea, such as dividing it into several subsections and determining the content of each subsection.
* You MUST write the research paper’s METHOD in LaTeX format and place it in the file sections/method.tex. Below is a detailed guide and example to structure the section effectively.

** Purpose and Structure **
The METHOD section is a core part of your paper. It needs to maintain academic rigor while presenting your research in a clear and comprehensible manner. Without delving into specific implementation details or tedious data processing steps, it should:
* Explain the Approach: Provide a detailed explanation of the proposed approach, including its underlying principles and innovative aspects, to ensure readers understand how it addresses the research problem.
* Establish Theoretical Rigor: Demonstrate the scientific basis of the method by linking it to existing theories, mathematical formulations, or relevant literature, reinforcing its validity.
* Clarify Implementation: Outline the workflow, model architecture, or algorithmic steps in a clear and structured manner, enabling reproducibility and facilitating comprehension of the methodology.
* Method should be around 3000 words in English. Use this space to explain your method's core ideas and theoretical underpinnings.
* References: Include references as needed to substantiate the theoretical foundation and contributions of your method.

** Common Structure **
Common structure is merely a guide to the main content. You SHOULD not directly divide the subsections strictly following it.
* Define Research Objectives and Problem Statement
    - Research Background and Motivation
        * Begin by briefly reviewing the background and motivation for your research problem. Highlight the limitations of existing methods or the challenges that need addressing.
        * Ensure readers grasp the problem you are tackling before diving into the technical details.
    - Problem Formulation
        * Use mathematical notation or standardized definitions to precisely define your task or model inputs and outputs.
        * Clearly introduce the notation you will use throughout the section.
* Core Ideas and Theoretical Foundation
    - Key Assumptions and Main Ideas
        * Present the core ideas of your proposed method, supported by prior literature or theories.
        * Clearly state any critical assumptions, such as data distribution or model structure assumptions.
    - Theoretical Derivations and Formulas
        * If your method involves algorithmic derivations or theoretical proofs, introduce the key formulas and explain each step logically.
        * Ensure the derivation is complete, but omit overly detailed intermediate steps in the main text. Lengthy proofs can be included in the Appendix and cited in the main text.
* Model Architecture and Algorithm Workflow
    - Module or Step Details
        * Generate figures/architecture.tex and use TikZ to create a comprehensive diagram that visually represents the model's overall structure and the relationships between its components.
        * Ensure that the diagram includes labels for the main modules, indicating their names and functions to facilitate understanding.
        * Describe each module or step in the sequence of input → processing → output. Focus on the function and motivation of each module, avoiding implementation specifics.
    - Algorithm Pseudo-Code
        * If introducing a new algorithm, provide pseudo-code to concisely illustrate the key steps. Use an "Algorithm 1" format, specifying inputs, outputs, and major steps with brief annotations.
* Training Strategy and Key Hyperparameters
    - Objective Function and Optimization Method
        * Building on earlier theoretical explanations, describe the loss function, optimization algorithm (e.g., SGD, Adam), and reasons for selecting them.
        * Highlight any special regularization terms or custom loss components.
    - Key Hyperparameters
        * Summarize hyperparameters that significantly impact results (e.g., learning rate, batch size, network depth). Provide their approximate settings or range in the main text or a table.
        * Leave exhaustive hyperparameter tuning details for the Experiments section or Appendix.
    - Training Process and Convergence
        * Briefly mention special training procedures (e.g., staged training, multitask learning) if applicable.
        * If relevant, discuss the convergence and stability of training, supported by theoretical or empirical findings.
* Scope and Limitations
    - Applicability
        * Indicate the types of data, tasks, or application scenarios your method targets.
        * Emphasize generalizability or scalability (e.g., "This method is not limited to image classification but can also be applied to video frame prediction").
    - Potential Limitations
        * If your method relies on specific assumptions (e.g., independent and identically distributed data, large-scale training data), acknowledge potential constraints in practical applications.
        * Proactively discussing limitations demonstrates a thoughtful and rigorous approach.
* Clear Organization and Concise Language
    - Structured Layout
        * Use logical subsections and headings to organize content, making it easy for readers to locate specific parts of interest.
        * Follow a "Overview → Details → Summary" structure or introduce theoretical foundations first, followed by algorithmic workflows.
    - Precise Language
        * Keep descriptions concise and avoid redundancy. Use established academic terminology and define any new terms or abbreviations.
    - Citations and Comparisons
        * Where necessary, compare your method to existing approaches and cite relevant work. While Related Work and Experiments sections often cover this, brief comparisons in the Method section can highlight your contributions.
* Avoid Unnecessary Implementation or Data Processing Details
    - Implementation details or data preprocessing steps can be included in supplementary materials or an open-source repository.
    - Focus on the logic and theory behind your method rather than implementation specifics.
    - Similarly, detailed data handling (e.g., cleaning, formatting, splitting) is better suited for the Empirical Result or Appendix sections.
* Smooth Transition to Experiments
    - Conclude the Method section with a brief transition to the Experiments section. For example, outline the core questions or evaluation metrics to be validated in the experiments, creating anticipation for the results.

** Writing Tips **
* Use structured headings and subsections to organize content logically.
* Provide clear and concise explanations of your approach, avoiding jargon.
* Use diagrams, pseudo-code, or mathematical formulations to enhance clarity where appropriate.
* While describing the design of a module, the MOTIVATION and BACKGROUND of the design needs to be explained in detail.
* Focus on the innovation and rationale of your method, linking it to the problem statement.
* Include references to support the theoretical foundation and highlight connections to prior work.
* Minimize the use of itemized lists unless essential; consider using i), ii), iii), etc., within cohesive paragraphs for smoother readability.
* Avoid implementation-specific details; keep the focus on the conceptual framework.
* End with a transition to the next section, setting up the context for experimental validation.

NOTE: When cite a reference in LaTeX, you MUST add the BibTeX of it in the file main.bib. If the item is already in the main.bib, you can directly use the citation key in the LaTeX file.

** Example Method **
For demonstration only. Adapt it to match your specific research and results. Here is an example method for a research paper:
(this is the start of sections/method.tex)
\section{Method}
\label{sec:method}

\input{figures/architecture}

\subsection{Problem Formulation and Overview}
We propose a hierarchical deep reinforcement learning framework that decomposes complex decision-making processes into two distinct layers: a strategic layer for high-level planning and a tactical layer for concrete action execution. This decomposition allows for more effective handling of tasks that require both long-term planning and precise immediate actions.

Let $\mathcal{S}$ denote the state space of the environment, $\mathcal{A}$ the primitive action space, and $\mathcal{G}$ the space of strategic goals. The state $s_t \in \mathcal{S}$ is represented as a vector of observable features, including environmental conditions and agent status. Actions $a_t \in \mathcal{A}$ are discrete or continuous vectors that directly modify the environment. Rewards $r_t$ are scalar signals from the environment, encapsulating feedback on performance. Formally, the reward function can be expressed as $r_t = R(s_t, a_t, s_{t+1})$, where $R: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}$ quantifies the outcome of an action.

The strategic layer operates at a lower temporal resolution, making decisions every $k$ time steps. Its output, a goal $g_t \in \mathcal{G}$, defines a target condition or subgoal. The tactical layer refines these goals into atomic actions executed at every time step. Together, the layers form a Semi-Markov Decision Process (SMDP) \cite{sutton1999between}.

\subsection{Hierarchical Framework Architecture}

\subsubsection{Strategic Layer}
The strategic layer policy $\pi_{\theta_s}$ maps the state $s_t$ to a goal $g_t$:
\begin{equation}
    g_t = \pi_{\theta_s}(s_t), \quad g_t \in \mathcal{G}
\end{equation}

Goals $g_t$ encapsulate high-level objectives and are defined in a continuous goal space $\mathcal{G} \subseteq \mathbb{R}^d$, where $d$ determines the dimensionality of the goal representation. For instance, in a navigation task, $g_t$ may represent a waypoint $(x, y)$.

The strategic policy maximizes the cumulative discounted reward over a longer time horizon:
\begin{equation}
    J_s(\theta_s) = \mathbb{E}_{\pi_{\theta_s}}\left[\sum_{t=0}^{T/k} \gamma^{kt} R_t\right]
\end{equation}
where $R_t$ aggregates rewards over the interval $[t, t+k)$.

\subsubsection{Tactical Layer}
The tactical layer policy $\pi_{\theta_t}$ translates the strategic goal $g_t$ into primitive actions:
\begin{equation}
    a_t = \pi_{\theta_t}(s_t, g_t), \quad a_t \in \mathcal{A}
\end{equation}

The motivation for the tactical layer design stems from the need to handle fine-grained decision-making in dynamic environments. While the strategic layer provides long-term direction, real-world scenarios often require rapid adjustments to changing conditions. For example, in a robotics context, even if a strategic layer defines a waypoint, obstacles and environmental changes necessitate immediate, precise actions to ensure safe and efficient navigation.

The tactical layer optimizes a combined objective that incorporates both environmental feedback and goal alignment:
\begin{equation}
    J_t(\theta_t) = \mathbb{E}_{\pi_{\theta_t}}\left[\sum_{t=0}^{k} \left(\gamma^t r_t + \lambda \cdot r^g_t\right)\right]
\end{equation}
where $r^g_t = -\|s_t - g_t\|_2^2$ measures the proximity to the goal, and $\lambda$ is a weighting coefficient. This design ensures that the tactical layer can effectively translate abstract goals into actionable steps while maintaining alignment with the overall strategy.

To enhance adaptability, the tactical layer leverages contextual features derived from the current state and recent observations. This allows the policy to dynamically adjust its behavior, mitigating potential discrepancies between high-level goals and immediate environmental demands.

\subsection{Learning Algorithm}
The learning process alternates between optimizing the strategic and tactical layers, ensuring that both layers are trained to fulfill their respective roles while maintaining coherence across temporal scales.

The hierarchical framework necessitates careful coordination between the strategic and tactical layers. To achieve this, the framework employs separate replay buffers for each layer, enabling transitions relevant to long-term goals and short-term actions to be sampled independently. Prioritized experience replay further ensures that the strategic layer focuses on transitions with significant temporal dependencies, which are particularly critical for achieving long-term objectives. Alternating update schedules are adopted, striking a balance between computational efficiency and training stability.

Strategic policies are trained using Soft Actor-Critic (SAC) \cite{haarnoja2018soft}, a method particularly suited for continuous control tasks. The SAC algorithm incorporates entropy maximization, encouraging exploratory behavior. The value function is defined as:
\begin{equation}
    V^\pi(s) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^{kt} \left(R_t + \alpha \mathcal{H}(\pi(\cdot|s_t))\right)\right]
\end{equation}
where $\mathcal{H}$ denotes policy entropy, and $\alpha$ controls the trade-off between exploration and exploitation. Policy updates are computed using the gradient:
\begin{equation}
    \nabla_{\theta_s} J_s = \mathbb{E}\left[\nabla_{\theta_s} \log \pi_{\theta_s}(g|s) \left(Q(s,g) - V(s)\right)\right]
\end{equation}

The tactical layer employs Proximal Policy Optimization (PPO) \cite{schulman2017proximal}, a robust and scalable method for updating policies. The PPO clipped objective ensures training stability:
\begin{equation}
    L^{CLIP}(\theta_t) = \mathbb{E}_t\left[\min\left(r_t(\theta_t)A_t, \text{clip}(r_t(\theta_t), 1-\epsilon, 1+\epsilon)A_t\right)\right]
\end{equation}
To align tactical actions with strategic goals, the reward function incorporates a goal achievement term $r^g_t$:
\begin{equation}
    r^g_t = -\|s_t - g_t\|_2^2
\end{equation}
This term incentivizes the tactical layer to adhere to high-level directives while remaining adaptable to local variations.

Temporal coordination between layers is achieved through decoupled training cycles. Strategic updates occur every $k$ steps to reflect the longer temporal horizon, while tactical updates are performed at each time step to adapt to the dynamic environment. Additionally, gradients from the tactical layer are propagated to the strategic layer, providing feedback on the feasibility and effectiveness of high-level goals. Curriculum learning is also incorporated, gradually increasing the complexity of strategic goals as training progresses. This staged approach enables both layers to incrementally adapt to more challenging scenarios.

\subsection{Goal Space Design}
The goal space $\mathcal{G}$ is represented as a continuous embedding learned by an encoder $f_\phi$:
\begin{equation}
    g = f_\phi(s, z), \quad z \in \mathcal{Z}
\end{equation}
where $z$ captures contextual features derived from historical observations. This design ensures that goals remain interpretable and expressive while aligning with the environment's structure.

\subsection{Intrinsic Motivation and Goal Evaluation}
To facilitate efficient exploration, we incorporate intrinsic motivation through a learned distance function $D$:
\begin{equation}
    r^g_t = -D(s_{t+k}, g_t), \quad D(s, g) = \|f(s) - g\|_2^2
\end{equation}
where $f(s)$ projects states into the same space as $g_t$.

\subsection{Training Process}
Training alternates between updating the strategic and tactical policies using prioritized replay and gradient-based optimization:

\begin{algorithm}[H]
\caption{Hierarchical DRL Training}
\begin{algorithmic}[1]
\State Initialize policies $\pi_{\theta_s}$, $\pi_{\theta_t}$
\For{each episode}
    \State Sample initial state $s_0 \sim p(s)$
    \For{$t=0$ to $T$}
        \If{$t \bmod k = 0$}
            \State Generate goal $g_t = \pi_{\theta_s}(s_t)$
        \EndIf
        \State Execute action $a_t = \pi_{\theta_t}(s_t, g_t)$
        \State Observe next state $s_{t+1}$ and reward $r_t$
        \State Store transition $(s_t, g_t, a_t, r_t, s_{t+1})$
        \If{buffer size sufficient}
            \State Update $\theta_t$ with PPO
            \If{$t \bmod k = 0$}
                \State Update $\theta_s$ with SAC
            \EndIf
        \EndIf
    \EndFor
\EndFor
\end{algorithmic}
\end{algorithm}

\subsection{Implementation Considerations}
The framework employs transformer-based architectures for the strategic layer to capture temporal dependencies and multi-layer perceptrons (MLPs) for the tactical layer to ensure efficient computation. Techniques like gradient clipping and replay prioritization stabilize training and improve sample efficiency.

Compared to monolithic reinforcement learning methods, this approach effectively abstracts long-term dependencies, enhances exploration via goal-driven behavior, and facilitates transferability across tasks.

\noindent The next section presents empirical results demonstrating the effectiveness of the proposed framework.
(this is the end of sections/method.tex)

** Example BibTeX Entry **
(this is the start of main.bib)
@article{sutton1999between,
  title={Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning},
  author={Sutton, Richard S and Precup, Doina and Singh, Satinder},
  journal={Artificial intelligence},
  volume={112},
  number={1-2},
  pages={181--211},
  year={1999},
  publisher={Elsevier}
}

@article{haarnoja2018soft,
  title={Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor},
  author={Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
  journal={arXiv preprint arXiv:1801.01290},
  year={2018}
}

@inproceedings{schulman2017proximal,
  title={Proximal policy optimization algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2017},
  pages={1--14},
  publisher={PMLR}
}
...
(this is the end of main.bib)

** Example figures/architecture.tex **
(this is the start of figures/architecture.tex)
\begin{figure*}[t]
\centering
\begin{tikzpicture}[
    % Modern style definitions
    box/.style={
        rectangle,
        draw=gray!50,
        fill=white,
        minimum width=2.8cm,
        minimum height=1.2cm,
        text centered,
        rounded corners=3pt,
        font=\sffamily
    },
    layer/.style={
        rectangle,
        draw=none,
        fill=blue!10,
        minimum width=12cm,
        minimum height=3cm,
        rounded corners=5pt,
        font=\sffamily\bfseries
    },
    env/.style={
        rectangle,
        draw=gray!50,
        fill=green!5,
        minimum width=12cm,
        minimum height=2cm,
        rounded corners=5pt,
        font=\sffamily
    },
    arrow/.style={
        -stealth,
        thick,
        draw=gray!70
    },
    darrow/.style={
        -stealth,
        thick,
        draw=gray!70,
        dashed
    },
    label/.style={
        font=\sffamily\small,
        text=gray!70
    }
]

    % Background layers
    \node[layer] (strategic_layer) at (0,2) {Strategic Layer};
    \node[layer] (tactical_layer) at (0,-2) {Tactical Layer};
    \node[env] (env_layer) at (0,-5) {Environment};

    % Strategic components
    \node[box] (state) at (-4,2) {State $s_t$};
    \node[box] (strategic) at (0,2) {Strategic Policy $\pi_{\theta_s}$};
    \node[box] (goal) at (4,2) {Goal $g_t$};

    % Tactical components
    \node[box] (tactical) at (0,-2) {Tactical Policy $\pi_{\theta_t}$};
    \node[box] (action) at (4,-2) {Action $a_t$};

    % Environment components
    \node[box] (next_state) at (-4,-5) {Next State $s_{t+1}$};
    \node[box] (reward) at (4,-5) {Reward $r_t$};

    % Information flow arrows
    \draw[arrow] (state) -- (strategic);
    \draw[arrow] (strategic) -- (goal);
    \draw[arrow] (goal) -- (tactical);
    \draw[arrow] (state) -- (-4,0) -- (-4,-2) -- (tactical);
    \draw[arrow] (tactical) -- (action);
    \draw[arrow] (action) -- (4,-3.5);
    \draw[arrow] (next_state) -- (-4,-3.5);
    \draw[arrow] (reward) -- (4,-3.5);

    % Feedback arrows
    \draw[darrow] (reward) -- (4,0) -- (strategic);
    \draw[darrow] (reward) -- (tactical);
    \draw[arrow] (next_state) to[out=90,in=180] (-4,0) -- (state);

    % Time scale annotations
    \node[label, above=0.1cm of strategic_layer] {Updates every $k$ steps};
    \node[label, above=0.1cm of tactical_layer] {Updates every step};

    % Add legend in bottom right corner
    \node[draw=gray!50,fill=white,rounded corners=3pt,font=\sffamily\small,anchor=south east]
    at (6,-6.5)
    {\begin{tabular}{ll}
        \textcolor{gray!70}{$\longrightarrow$} & Direct flow\\
        \textcolor{gray!70}{$\dashrightarrow$} & Reward signal
    \end{tabular}};

\end{tikzpicture}
\caption{Enhanced architecture of the hierarchical DRL framework. The system operates at two timescales: the strategic layer makes high-level decisions every $k$ steps, while the tactical layer produces actions at every timestep. The framework integrates environmental feedback through both direct state transitions and reward signals.}
\label{fig:architecture}
\end{figure*}
(this is the end of figures/architecture.tex)
"""

_PARAMETER_COMMAND_DESCRIPTION = (
    'The bash command to generate the method in the terminal.'
)

LatexMethodTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='latex_method',
        description=_LATEX_METHOD_DESCRIPTION,
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
