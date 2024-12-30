"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_LATEX_EMPIRICAL_RESULTS_DESCRIPTION = r"""Write the EMPIRICAL RESULTS section of a research paper.

* You MUST first design an appropriate structure for the empirical results section based on the paper's idea, such as dividing it into several subsections and determining the content of each subsection.
* You MUST write the research paper’s empirical results in LaTeX format and place it in the file sections/empirical_results.tex. Below is a detailed guide and example to structure the section effectively.

** Purpose and Structure **
The Empirical Results section is a core component of your paper, demonstrating how well your proposed method addresses the central research questions and hypotheses. Specifically, it aims to:
* Show the effectiveness of your method on relevant tasks or datasets.
* Compare and contrast your method with existing baselines in a fair manner.
* Provide both quantitative and qualitative evidence supporting your contributions.
* Offer insights into why and how the method succeeds (or fails) under various scenarios.
* Minimize the use of itemized lists unless essential; consider using i), ii), iii), etc., within cohesive paragraphs for smoother readability.
* Present ablation studies and deeper analyses that reveal the impact of each component.
* Empirical Results should be around 3000 words in English. Use this space to describe your experiments, results, and analyses in detail.
* Number of References: If need baselines or datasets, you can cite them. Otherwise, you can skip the references.

** Common Structure **
* Experimental Design and Research Questions
    - Clearly restate (or reference) the central research questions you plan to answer.
    - Align each experiment with one or more of these questions, ensuring the rationale for each experiment is explicit.
* Experiment Setup and Reproducibility
    - Provide all essential details: hyperparameters (learning rate, batch size, etc.), optimization algorithms (e.g., SGD, Adam, RMSProp), training epochs, and hardware used (e.g., GPU, TPU).
    - Group shared settings in a single subsection to avoid repetitive descriptions across multiple experiments.
    - Introduce baseline methods, explaining their relevance, official/standard implementations, and any modifications needed for a fair comparison.
* Evaluation Metrics
    - Justify the chosen metrics (e.g., Accuracy, MAE, F1, mAP) and clarify what aspect of performance each metric captures.
    - If you use multiple metrics, briefly explain how they complement each other (e.g., accuracy vs. precision/recall trade-offs).
* Main Results and Comparisons
    - Present key findings using well-designed tables and charts.
    - For instance, use `\input{tables/baselines_comparison.tex}` to integrate a table (see Table~\ref{tab:baselines_comparison}) comparing your method with various baselines.
    - Highlight best results in \textbf{bold} or \underline{underlined}, and mark statistical significance (e.g., p-value < 0.05) with symbols like `*` or `†`.
    - Go beyond listing numbers: interpret the results in terms of model effectiveness, robustness, and alignment with your hypotheses.
    - You may also include a bar chart (e.g., `figures/efficiency.tex`) to illustrate efficiency or throughput comparisons (see Figure~\ref{fig:efficiency}).
* Ablation Studies and Detailed Exploration
    - Demonstrate the contribution of individual modules or hyperparameters by removing or altering them in controlled experiments.
    - Refer to a separate table (e.g., `\input{tables/ablation_comparison.tex}`, Table~\ref{tab:ablation_comparison}) showing performance variations with/without each module.
    - Examine sensitivity to data quantity or distribution shifts, where applicable.
    - Offer both quantitative outcomes (e.g., performance changes) and qualitative observations (stability, interpretability).
* Deeper Analysis and Visualization
    - Include error analyses to uncover common failure patterns or challenging examples for your model.
    - Visualize intermediate features, attention maps, or other internal representations, if relevant, to provide interpretability.
    - Discuss practical aspects like inference speed or memory usage if they are critical to your application domain.
* Organization, Style, and Completeness
   - Follow a logical progression, typically: “Experiment Setup” → “Main Results” → “Ablation Analysis” → “Further Analyses (visualizations, error cases).”
   - Refer back to your Introduction or Methods sections to show how empirical findings support (or refine) your theoretical claims.
   - Maintain clarity in tables/figures: label axes, provide a legend, and ensure each is referenced in the text.
   - If space is constrained, place extensive tables, detailed hyperparameter searches, or additional figures in the appendix.

** Writing Tips **
* Keep it concise and insightful: Aim to highlight the \emph{most significant} results and findings. Avoid overloading this section with extraneous data.
* Fair Baselines: Ensure the versions, parameters, and implementations of baseline methods are up-to-date and optimized to avoid biased comparisons.
* Statistical Significance: For small performance gains or close results, employ appropriate tests (e.g., t-tests) and report p-values or confidence intervals.
* Negative or Unexpected Findings: Address them candidly. Providing well-reasoned explanations or hypotheses enhances the paper’s credibility.
* Adhere to Formatting: Follow any page limit or style constraints required by the conference/journal. Move extensive details to appendices if needed.

NOTE: When cite a reference in LaTeX, you MUST add the BibTeX of it in the file main.bib. If the item is already in the main.bib, you can directly use the citation key in the LaTeX file.

** Example Empirical Results **
For demonstration only. Adapt it to match your specific research and results. Here is an example empirical results section:
(this is the start of sections/empirical_results.tex)
\section{Empirical Results}
\label{sec:empirical_results}

In this section, we present a comprehensive empirical evaluation of our hierarchical deep reinforcement learning (DRL) framework. We target three key research questions:

\begin{itemize}[left=0em]
    \item \textbf{RQ1:} Does the proposed hierarchical DRL method outperform flat (single-layer) DRL approaches in different real-time strategy (RTS) scenarios?
    \item \textbf{RQ2:} How do the strategic and tactical layers individually and collectively improve exploration efficiency, learning stability, and final win rates?
    \item \textbf{RQ3:} What are the main limitations of the hierarchical framework, and what potential improvements can be derived from deeper analyses?
\end{itemize}

\subsection{Experiment Setup}
\label{sec:experiment_setup}

\paragraph{Dataset (Environments).}
We follow established RTS-style evaluation protocols~\cite{silver2016mastering,mnih2015human,lillicrap2015continuous} and employ three environments of escalating complexity. \emph{Scenario A (Small-Scale Skirmish)} is a compact map with a single resource type and low-dimensional state space, focusing on unit micromanagement. \emph{Scenario B (Medium-Scale Expansion)} introduces additional resources, base-building, and moderate-scale combat, requiring mid-term strategic coordination. \emph{Scenario C (Large-Scale Campaign)} involves multiple resource types, adversarial expansions, and complex event scheduling, thus testing high-level planning and fine-grained tactics at scale.

\paragraph{Baselines.}
We compare our hierarchical DRL (\textit{HierDRL}) with two widely used approaches. \emph{FlatDRL}~\cite{schulman2017proximal} applies a single policy network directly on raw states without hierarchical decomposition. \emph{Multi-Agent RL (MARL)}~\cite{silver2016mastering} assigns a separate agent to each controllable unit and shares a global reward, enabling decentralized decision-making. We use publicly available implementations of both baselines, carefully tuning their hyperparameters to align with our method’s training conditions.

\paragraph{Evaluation Metrics.}
We employ four metrics to assess final performance and learning dynamics. Win Rate (\%) measures the percentage of episodes in which the agent defeats its opponent. Cumulative Reward summarizes partial achievements (e.g., resources gathered, enemy units eliminated). Exploration Efficiency counts how many steps are required to reach 80\% of the best final score. Stability Index represents the standard deviation of rewards or win rates over the last 50 episodes, indicating the agent’s consistency once trained.

\paragraph{Implementation Details.}
Our experiments run under Python 3.9 and PyTorch 1.10 on an NVIDIA RTX 3090 GPU. The Adam optimizer is used with a learning rate of $5\times 10^{-4}$, mini-batch size of 64, and weight decay set to $1\times 10^{-5}$. The strategic layer follows an $\epsilon$-greedy exploration schedule decaying from 1.0 to 0.1 over 50k steps, while the tactical layer applies a combination of Boltzmann and uniform exploration~\cite{haarnoja2018soft}. We adopt curriculum training~\cite{bengio2009curriculum}, starting with simplified sub-tasks and progressively increasing difficulty until the full environment complexity is reached. Each run trains for up to 300k environment steps or stops early if validation rewards stagnate for 10k steps. We repeat all experiments with three random seeds (42, 123, 999), and we report the mean and standard deviation for each scenario.

\subsection{Main Results}
\label{sec:main_results}

\input{tables/baselines_comparison}

Table~\ref{tab:baselines_comparison} summarizes the performance of \textit{HierDRL}, \textit{FlatDRL}, and \textit{MARL} across Scenarios~A, B, and C. Our hierarchical framework clearly outperforms the baselines in the larger, more complex environments. Notably, \textit{HierDRL} improves over \textit{FlatDRL} by 5--10\% and \textit{MARL} by up to 15\% in win rate for Scenarios B/C (p-value < 0.05). In Scenario A, where the task complexity is lower, all methods perform comparably, suggesting that flat models can suffice when the action and state spaces remain small.

\input{figures/efficiency}

Figure~\ref{fig:efficiency} highlights exploration efficiency, where \textit{HierDRL} consistently converges faster. By separating macro-level strategy from micro-level tactics, the agent requires fewer steps to reach competitive performance. Additionally, \textit{HierDRL} exhibits lower variance in late-stage rewards, indicating enhanced training stability.

\subsection{Ablation Study}
\label{sec:ablation}

\input{tables/ablation_comparsion}

To isolate the contributions of each architectural choice, we conduct ablation experiments as shown in Table~\ref{tab:ablation_comparison}. Removing the strategic layer reduces final performance substantially in Scenarios B/C, implying that macro-level planning is essential in complex tasks. Eliminating the tactical layer degrades all scenarios, underlining the importance of refined micromanagement. Skipping curriculum training increases the required steps by around 30\% in Scenario C. Moreover, replacing our selective state-sharing mechanism with naive concatenation lowers win rates across the board, demonstrating that structured information flow between layers is beneficial.

\subsection{Discussion}
\label{sec:discussion}

We conduct error analysis by classifying unsuccessful episodes, revealing that most failures arise from adaptive enemy strategies, such as sudden resource contention or simultaneous attacks on multiple fronts. Appendix~A shows how the tactical layer’s attention maps focus heavily on regions with immediate threats while drawing cues from the strategic policy for expansions and resource management. Detailed GPU memory usage (Appendix~B) indicates that our hierarchical approach requires around 10\% more memory than \textit{FlatDRL}, a modest overhead given the performance gains.

In summary, the hierarchical design accelerates exploration, improves late-stage stability, and achieves higher final win rates in challenging RTS environments. Although simpler tasks do not benefit as markedly from the added complexity, the synergy of macro-level strategies and micro-level control is instrumental in scaling to multi-faceted domains. Our results suggest that further enhancements in adaptive strategic planning—potentially incorporating opponent modeling—could mitigate vulnerabilities to adversarial tactics. Ultimately, this work demonstrates the viability of hierarchical DRL for real-time decision-making under partial observability, non-stationarity, and large-scale action spaces.
(this is the end of sections/empirical_results.tex)

** Example tables/baselines_compariso.tex **
(this is the start of tables/baselines_comparison.tex)
\begin{table}[t]
    \centering
    \caption{Performance Comparison on Three RTS Scenarios. Mean (std) values over three runs.}
    \label{tab:baselines_comparison}
    \begin{tabular}{lccc}
    \toprule
    \textbf{Method} & \textbf{Win Rate (\%)} & \textbf{Cumulative Reward} & \textbf{Exploration Efficiency} \\
    \midrule
    FlatDRL~\cite{schulman2017proximal} & 72.3 (2.1) & 312.5 (10.3) & 45k steps \\
    MARL~\cite{silver2016mastering} & 68.1 (3.2) & 294.0 (12.7) & 52k steps \\
    \textbf{HierDRL (Ours)} & \textbf{80.5 (1.6)} & \textbf{355.2 (9.1)} & \textbf{39k steps} \\
    \bottomrule
    \end{tabular}
\end{table}
(this is the end of tables/baselines_comparison.tex)

** Example figures/efficiency.tex **
(this is the start of figures/efficiency.tex)
\begin{figure}[t]
    \centering
    \begin{tikzpicture}
    \begin{axis}[
        ybar,
        width=7.0cm,
        height=5.0cm,
        bar width=0.35cm,
        ymin=0,
        ymax=60,
        enlarge x limits=0.3,
        axis x line=bottom,
        axis y line=left,
        ymajorgrids=true,
        xtick=data,
        symbolic x coords={FlatDRL, MARL, HierDRL},
        xlabel={Method},
        ylabel={\#Steps to 80\% Performance (k)},
        xlabel style={font=\footnotesize},
        ylabel style={font=\footnotesize},
        ticklabel style={font=\footnotesize},
        legend style={
            at={(0.5,1.03)},
            anchor=south,
            font=\footnotesize,
            cells={anchor=west},
            legend columns=1
        },
        nodes near coords,
        every node near coord/.append style={
            font=\footnotesize,
            rotate=90,
            anchor=west,
        },
    ]
    % Use a custom or cycle list color scheme:
    % e.g., fill=blue!50 or fill=red!50 for each bar.
    \addplot[fill=blue!50] coordinates {
        (FlatDRL,45)
        (MARL,52)
        (HierDRL,39)
    };

    \legend{Exploration Efficiency}
    \end{axis}
    \end{tikzpicture}
    \caption{Comparison of exploration efficiency, measured by the number of steps required to reach 80\% of the best final performance.}
    \label{fig:exploration_efficiency}
\end{figure}
(this is the end of figures/efficiency.tex)

** Example tables/ablation_comparison.tex **
(this is the start of tables/ablation_comparison.tex)
\begin{table}[t]
    \centering
    \caption{Ablation Study: Impact of Removing or Modifying Key Components in the Hierarchical Framework.}
    \label{tab:ablation_comparison}
    \begin{tabular}{lccc}
    \toprule
    \textbf{Ablation Variant} & \textbf{Win Rate (\%)} & \textbf{Cumulative Reward} & \textbf{Steps to 80\% Perf.} \\
    \midrule
    \textbf{Full HierDRL (Ours)} & \textbf{80.5 (1.6)} & \textbf{355.2 (9.1)} & \textbf{39k} \\
    No-Strategic-Layer & 72.0 (2.8) & 310.1 (11.2) & 45k \\
    No-Tactical-Layer & 70.1 (3.5) & 300.5 (12.3) & 48k \\
    No-Curriculum & 74.5 (2.6) & 320.2 (10.7) & 50k \\
    Simplified State-Sharing & 75.2 (2.1) & 325.4 (10.2) & 44k \\
    \bottomrule
    \end{tabular}
\end{table}
(this is the end of tables/ablation_comparison.tex)

** Example main.bib **
(this is the start of main.bib)
@article{silver2016mastering,
    title={Mastering the game of Go with deep neural networks and tree search},
    author={Silver, David and Huang, Aja and Maddison, Chris J and Guez, Arthur and Sifre, Laurent and van den Driessche, George and Schrittwieser, Julian and Antonoglou, Ioannis and Panneershelvam, Veda and Lanctot, Marc and others},
    journal={nature},
    volume={529},
    number={7587},
    pages={484--489},
    year={2016},
    publisher={Nature Publishing Group}
}
...
(this is the end of main.bib)
"""

_PARAMETER_COMMAND_DESCRIPTION = (
    'The bash command to generate the empirical results in the terminal.'
)

LatexEmpiricalResultsTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='latex_empirical_results',
        description=_LATEX_EMPIRICAL_RESULTS_DESCRIPTION,
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
