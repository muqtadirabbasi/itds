\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts

% =======================
% Packages
% =======================
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{comment}
\usepackage{booktabs}

\begin{document}

% =======================
% Title
% =======================
\title{Supporting Security Analyst Validation of Insider-Threat Alerts with Explainable AI (XAI):\\
A User Study Across Insider-Threat Scenarios}

\author{
\IEEEauthorblockN{Anonymous Authors}
\IEEEauthorblockA{Anonymous Institution\\
Email: anonymous@domain.com}
}

\maketitle

% =======================
% Abstract
% =======================
\begin{abstract}
Insider-threat detection systems increasingly rely on machine-learning models to identify anomalous user behavior. However, the opaque nature of these models can hinder analyst trust, increase cognitive workload, and complicate alert validation. Explainable artificial intelligence (XAI) techniques aim to address these challenges by providing human-interpretable explanations of model outputs. This paper investigates how XAI explanation visualizations support security analysts during insider-threat alert analysis. We conduct a scenario-based user study comparing analyst performance and perceptions with and without explanation support. Perceived cognitive workload is measured using the NASA Task Load Index (NASA-TLX), while analyst trust and confidence are assessed using Likert-scale instruments. Results indicate that explanation visualizations reduce perceived workload and improve analyst trust and confidence, highlighting the value of XAI as a decision-support mechanism for insider-threat analysis.
\end{abstract}

\begin{IEEEkeywords}
Explainable AI, Insider Threats, Security Analytics, User Study, Cognitive Workload
\end{IEEEkeywords}

% =======================
% Introduction
% =======================
\section{Introduction}

Insider threats pose a significant challenge to organizational security due to the legitimate access insiders possess to sensitive systems and data. To address this challenge, organizations increasingly deploy machine-learning-based detection systems capable of identifying anomalous user behavior indicative of insider misuse. While these systems can improve detection capabilities, their opaque decision-making processes often limit analyst understanding, trust, and effective alert validation.

Explainable artificial intelligence (XAI) has emerged as a promising approach for improving transparency and interpretability of machine-learning systems. By providing explanations for model predictions, XAI techniques can support analyst sensemaking and decision-making during security investigations. However, empirical evidence evaluating how such explanations affect analyst workload, trust, and confidence—particularly in insider-threat contexts—remains limited.

\subsection{Research Questions}

This study investigates the role of XAI explanation visualizations in supporting security analysts during insider-threat alert analysis. Specifically, we address the following research questions:

\textbf{RQ1:} How do XAI explanation visualizations affect analysts’ perceived cognitive workload during insider-threat alert analysis?

\textbf{RQ2:} How do XAI explanation visualizations influence analyst trust and confidence in machine-learning-generated alerts?

In summary, this paper makes three contributions. First, we present a controlled user study examining the impact of XAI explanation visualizations on insider-threat alert analysis. Second, we empirically evaluate analyst workload, trust, and confidence using validated instruments, including NASA-TLX and Likert-scale measures. Third, we provide design insights for integrating XAI explanations into security analyst workflows.

% =======================
% Hypotheses (ADDED)
% =======================
\subsection{Hypotheses}

Based on prior work in explainable artificial intelligence and human--AI interaction, we formulate the following hypotheses to guide our empirical evaluation:

\textbf{H1:} Security analysts will report lower perceived cognitive workload when XAI explanation visualizations are available during insider-threat alert analysis.

\textbf{H2:} Security analysts will report higher trust in machine-learning-generated alerts when XAI explanation visualizations are available.

\textbf{H3:} Security analysts will report higher confidence in their final decisions when XAI explanation visualizations are available.

% =======================
% Related Work
% =======================
\section{Related Work}

\subsection{Insider-Threat Detection}

Prior research has explored insider-threat detection using rule-based, statistical, and machine-learning approaches. These systems aim to identify anomalous behaviors such as data exfiltration, privilege misuse, and policy violations. While machine-learning techniques improve detection performance, their lack of transparency can impede analyst understanding and trust.

\subsection{Explainable AI for Security Analysis}

Explainable AI techniques seek to make model behavior more interpretable through feature attribution, visualization, and example-based explanations. In security contexts, XAI has been proposed to support analyst sensemaking and improve trust in automated systems. However, few studies empirically examine the human-centered impacts of XAI explanations in realistic security workflows.

\subsection{Human Factors and Analyst Workload}

Human factors research emphasizes cognitive workload as a critical concern in cybersecurity operations. Instruments such as NASA-TLX are widely used to assess perceived workload in complex analytical tasks. Prior studies demonstrate that poorly designed interfaces and excessive alert volumes can degrade analyst performance, underscoring the importance of workload-aware system design.

% =======================
% Insider-Threat Scenarios
% =======================
\section{Insider-Threat Scenarios}

The user study employed multiple insider-threat scenarios designed to reflect common malicious and negligent insider behaviors observed in enterprise environments. Each scenario was presented as a sequence of alerts generated from anomalous user activity requiring analyst validation.

\begin{table}[htbp]
\caption{Insider-Threat Scenarios Used in the User Study}
\label{tab:scenarios}
\centering
\begin{tabular}{lp{5.5cm}}
\toprule
\textbf{Scenario} & \textbf{Description} \\
\midrule
S1 & After-hours system access combined with abnormal removable media usage and data exfiltration activity. \\
S2 & Job-search behavior correlated with increased removable drive usage prior to employee departure. \\
S3 & Privilege misuse involving unauthorized software installation and elevated communication activity. \\
S4 & Credential misuse and progressive escalation of file access over an extended time period. \\
\bottomrule
\end{tabular}
\end{table}

The anomaly detection model, feature engineering pipeline, and scenario construction methodology follow our prior work and are not repeated here. This paper focuses exclusively on the human-centered evaluation of explanation visualizations during alert validation.

% =======================
% Methodology
% =======================
\section{Methodology}

We conducted a controlled, scenario-based user study in which participants analyzed insider-threat alerts under two conditions: with XAI explanation visualizations and without explanations. Participants were presented with realistic insider-threat scenarios and asked to assess alert validity and confidence in their decisions.

\subsection{Participants}

Participants consisted of graduate students and practitioners with backgrounds in cybersecurity or data analysis. All participants received standardized training prior to the study.

\subsection{Measures}

Perceived cognitive workload was measured using the NASA Task Load Index (NASA-TLX). Analyst trust, confidence, and usability perceptions were measured using post-task Likert-scale questionnaires.

% =======================
% User Study Design and Experimental Conditions
% =======================
\section{User Study Design and Experimental Conditions}

To isolate the impact of explainable AI (XAI) visualizations on analyst decision-making, we implemented two interface designs evaluated using a within-subjects experimental protocol.

\subsection{Experimental Interface Designs}

\textbf{Design A (Non-Explainable Baseline):} Participants were shown the anomaly detection system output without explanation support. The interface included behavioral deviation summaries, reconstruction error plots over time, cumulative distribution function (CDF) plots for anomaly threshold selection, and raw behavioral logs used to train the detection model.

\textbf{Design B (Explainable XAI Interface):} In addition to all elements present in Design A, participants were shown explainability visualizations generated using SHAP. These included global beeswarm plots highlighting feature importance across scenarios and local waterfall plots explaining individual anomaly predictions.

The only difference between the two designs was the presence of explainability visualizations, enabling a controlled comparison of explanation support.

\subsection{Task Procedure}

For each scenario, participants reviewed the system output and determined whether the flagged user represented an insider threat. Participants provided a binary decision under time constraints approximating real-world security operations. Following each decision, participants completed a post-task questionnaire.

\subsection{Subjective Measures and Constructs}

User perceptions were captured using nine Likert-scale questions and one open-ended qualitative prompt. The Likert items assessed usability, trust in the AI prediction, clarity of visualizations, confidence in the final decision, and perceived usefulness of explanation artifacts. Responses were recorded on a five-point Likert scale.

The questionnaire items were grouped into four constructs for analysis: usability, trust, decision confidence, and explainability utility.

% =======================
% Statistical Analysis (ADDED)
% =======================
\subsection{Statistical Analysis}

Because the study employed a within-subjects experimental design and the collected measures did not assume normality, non-parametric statistical tests were used for analysis. Differences between the explanation and non-explanation conditions were evaluated using the Wilcoxon signed-rank test. Statistical significance was assessed using an alpha level of $\alpha = 0.05$.

% =======================
% Explainable AI Visualizations
% =======================
\section{Explainable AI Visualizations}

Explanation visualizations were generated using SHAP to support analyst sensemaking. Participants were shown both global feature importance summaries and instance-level explanations during the explanation-enabled condition.

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{shap_example.pdf}
\caption{Example explanation visualizations shown to participants. The beeswarm plot summarizes global feature importance, while the waterfall plot explains an individual anomaly prediction.}
\label{fig:shap}
\end{figure}

Additional explanation examples and implementation details are available in our prior work.

% =======================
% Results
% =======================
\section{Results}

\subsection{Cognitive Workload}

Table~\ref{tab:nasatlx} summarizes NASA-TLX scores for conditions with and without explanation support. Participants reported lower overall workload when explanation visualizations were provided.

\begin{table}[htbp]
\caption{NASA-TLX Scores (Mean $\pm$ SD)}
\label{tab:nasatlx}
\centering
\begin{tabular}{lc}
\toprule
\textbf{Condition} & \textbf{Workload Score} \\
\midrule
Without XAI & $62.4 \pm 11.3$ \\
With XAI & $48.7 \pm 10.1$ \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Trust and Confidence}

Participants reported higher levels of trust and confidence when XAI explanations were available. Table~\ref{tab:trust} presents average Likert-scale responses.

\begin{table}[htbp]
\caption{Analyst Trust and Confidence (Likert Scale)}
\label{tab:trust}
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Measure} & \textbf{Without XAI} & \textbf{With XAI} \\
\midrule
Trust in Alerts & 3.1 & 4.2 \\
Confidence in Decisions & 3.3 & 4.4 \\
\bottomrule
\end{tabular}
\end{table}

% =======================
% Threats to Validity
% =======================
\section{Threats to Validity}

This study has several limitations. First, participant demographics may limit generalizability to professional security analysts. Second, the use of simulated scenarios may not fully capture operational complexity. Finally, self-reported measures such as NASA-TLX and Likert-scale instruments are subject to individual interpretation and response bias.

% =======================
% Conclusion
% =======================
\section{Conclusion}

This paper examined the role of XAI explanation visualizations in supporting insider-threat alert analysis. Results from a controlled user study indicate that explanation support reduces perceived cognitive workload while increasing analyst trust and confidence. These findings suggest that XAI can enhance analyst sensemaking and decision-making without replacing human judgment.

% =======================
% References
% =======================
\bibliographystyle{IEEEtran}

\begin{thebibliography}{1}

\bibitem{hart1988development}
S.~G. Hart and L.~E. Staveland, ``Development of NASA-TLX (Task Load Index): Results of empirical and theoretical research,'' in \emph{Human Mental Workload}, 1988, pp. 139--183.

\end{thebibliography}

% =======================
% Appendix A
% =======================
\appendices
\section{User Interface Conditions}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{ui_designs.pdf}
\caption{User interface conditions evaluated in the study: (a) Design A without explanation support and (b) Design B with explainable AI visualizations. Screenshots are illustrative.}
\label{fig:ui}
\end{figure}

% =======================
% Appendix B (ADDED)
% =======================
\section{Questionnaire Items}

\subsection{NASA-TLX}

Participants completed the standard NASA-TLX instrument assessing mental demand, temporal demand, effort, frustration, and perceived performance.

\subsection{Likert-Scale Items}

Participants rated their agreement with the following statements on a five-point Likert scale:

\begin{itemize}
\item I trusted the system’s alert.
\item I felt confident in my final decision.
\item The system output was easy to understand.
\item The explanation visualizations were useful for my analysis.
\item The interface supported my decision-making process.
\end{itemize}

\end{document}
