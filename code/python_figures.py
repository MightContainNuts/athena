from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from pytrends.request import TrendReq
import seaborn as sns
import pandas as pd
import numpy as np
from graphviz import Digraph
import os

def plot_ai_awareness_uk():
    """ plot the UK awareness for AI in the public sector"""
    categories = ["Aware of GenAI use in their area",
            "Actively use GenAI", "Trust GenAI outputs",
            "Fear of being replaced",
            "See clear guidance at work",
            "Feel UK is missing AI opportunities"]
    percentages = [45, 22, 61, 16, 32, 76]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(categories, percentages, color="#4C72B0")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width}%', va='center')

    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of Respondents")
    ax.set_title(
        "Public Sector Survey on Generative AI (UK, 2024)")
    ax.invert_yaxis()  # Highest percentage on top
    plt.tight_layout()

    plt.savefig("../fig/ai_awareness_uk.png")


def plot_rigid_use_frameworks():
    G = nx.DiGraph()

    # Central node (cluster)
    central = "Impact Measurement Frameworks"
    G.add_node(central, size=3000, color='skyblue')

    # Framework nodes
    frameworks = ["Logical Framework Approach (LogFrame)",
            "Theory of Change",
            "Social Return on Investment (SROI)",
            "Balanced Scorecard",
            "Results-Based Management (RBM)",
            "Outcome Mapping",
            "Most Significant Change (MSC)",
            "Custom/Hybrid Models"]

    # Add framework nodes
    for f in frameworks:
        G.add_node(f, size=1500, color='lightgreen')
        G.add_edge(central, f)

    # Outcome nodes
    rigid = "Rigid Use of Single Framework"
    adaptive = "Adaptive, Intelligent Systems"
    G.add_node(rigid, size=2000, color='salmon')
    G.add_node(adaptive, size=2000, color='lightcoral')

    # Connect frameworks to outcomes
    for f in frameworks:
        G.add_edge(f, rigid)
        G.add_edge(f, adaptive)

    # Positions for better layout
    pos = nx.spring_layout(G, k=1.5, iterations=100)

    # Draw nodes with colors and sizes
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrowsize=20,
                           arrowstyle='-|>')
    nx.draw_networkx_labels(G, pos, font_size=9,
                            font_weight='bold',
                            font_family='sans-serif')

    plt.title(
        "Impact Measurement Frameworks and Usage Patterns",
        fontsize=16)
    plt.axis('off')
    plt.savefig("../fig/rigid_use_frameworks.png")


def google_trend_impact():
    # Initialize
    pytrends = TrendReq()

    # Keywords
    kw_list = ["impact investing"]

    # Build payload
    pytrends.build_payload(kw_list,
                           timeframe='2015-01-01 2025-07-01',
                           geo='', gprop='')

    # Retrieve interest over time
    df = pytrends.interest_over_time()

    # Plot
    if not df.empty:
        df = df[kw_list]
        df.plot(figsize=(12, 6), linewidth=2)
        plt.title(
            "Google Trends: Impact Investing (2015–2025)")
        plt.xlabel("Year")
        plt.ylabel("Relative Search Volume")
        plt.legend(kw_list, loc='upper left')
        plt.savefig("../fig/google_trend_impact_investing.png")
    else:
        print(
            "No trend data available. Try adjusting keywords or timeframe.")



def fig_synthesis_gaps():
    import numpy as np
    import matplotlib.pyplot as plt

    categories = ["Unstructured Data Handling",
            "Stakeholder Engagement",
            "Normative Commitments", "AI Scalability",
            "Policy Alignment"]
    N = len(categories)

    imm_scores = [2, 3, 2, 1, 3]
    proposed_scores = [4, 5, 5, 4, 5]

    # Calculate angle for each category on the plot
    angles = np.linspace(0, 2 * np.pi, N,
                         endpoint=False).tolist()

    # Complete the loop for plotting by appending the first element to the end
    imm_scores += imm_scores[:1]
    proposed_scores += proposed_scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw=dict(polar=True))

    # Draw the outline and fill for IMM framework
    ax.plot(angles, imm_scores, color='blue', linewidth=2,
            label='Existing IMM')
    ax.fill(angles, imm_scores, color='blue', alpha=0.25)

    # Draw the outline and fill for proposed framework
    ax.plot(angles, proposed_scores, color='red',
            linewidth=2, label='Proposed Framework')
    ax.fill(angles, proposed_scores, color='red',
            alpha=0.25)

    # Set the category labels at the appropriate angles
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11,
                       color='black')

    # Set radial ticks
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'],
                       fontsize=10)
    ax.set_ylim(0, 5)

    # Add title and legend
    plt.title(
        'Comparison of IMM Frameworks Across Key Criteria',
        size=14, y=1.08)
    plt.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.1))

    plt.savefig('../fig/imm_comparison.png')



def clustering_grouping():
    """methodology figures"""

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')

    steps = [("Narrative Inputs",
              "Project proposals,\nworkshop transcripts"),
            ("Embedding Generation",
             "text-embedding-ada-002"),
            ("Dimensionality Reduction", "UMAP or t-SNE"),
            ("Clustering", "HDBSCAN"),
            ("Cluster Interpretation",
             "GPT-4 Summarization"), ("Thematic Outputs",
                                      "For KPI generation,\nreporting, reflection")]

    box_width = 0.8
    box_height = 0.12
    spacing = 0.06

    y = 1.0
    for i, (title, desc) in enumerate(steps):
        box = FancyBboxPatch((0.1, y - box_height),
            box_width, box_height,
            boxstyle="round,pad=0.02", edgecolor="black",
            facecolor="lightblue")
        ax.add_patch(box)

        text = f"{title}\n{desc}"
        ax.text(0.5, y - box_height / 2, text, ha="center",
                va="center", fontsize=10)

        if i < len(steps) - 1:
            ax.annotate("",
                        xy=(0.5, y - box_height - 0.005),
                        xytext=(0.5,
                                y - box_height - spacing + 0.01),
                        arrowprops=dict(arrowstyle="->",
                                        lw=1.5))

        y -= (box_height + spacing)

    # Increase the y limits to add padding top and bottom
    ax.set_ylim(y - 0.05, 1.05)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig("../fig/clustering_pipeline.png",
                bbox_inches='tight')
    plt.show()


def langgraph_flow():
    """LangGraph KPI Derivation Pipeline with conditional audit loops"""

    dot = Digraph(
        comment="LangGraph KPI Derivation Pipeline")
    dot.attr(rankdir='TB', fontsize='8', ranksep='0.3',
             nodesep='0.2')
    dot.graph_attr.update(margin='0.5', size='8,10!')

    # Main nodes
    dot.node('A',
             'Input Normalization\n(Scope, Context, Constraints)',
             shape='box', style='filled',
             fillcolor='lightblue')
    dot.node('B', 'SDG Mapping\n(GPT-4 Classifier)',
             shape='box', style='filled',
             fillcolor='lightblue')
    dot.node('C', 'Audit: SDG Justification',
             shape='ellipse', style='filled',
             fillcolor='lightgray')
    dot.node('C_decide', 'Pass Justification?',
             shape='diamond', style='filled',
             fillcolor='white')

    dot.node('D',
             'Indicator Retrieval\n(Vector Similarity Search)',
             shape='box', style='filled',
             fillcolor='lightblue')
    dot.node('E', 'Audit: Indicator Fit Check',
             shape='ellipse', style='filled',
             fillcolor='lightgray')
    dot.node('E_decide', 'Pass Fit Check?', shape='diamond',
             style='filled', fillcolor='white')

    dot.node('F', 'KPI Generation\n(Name, Logic, Baseline)',
             shape='box', style='filled',
             fillcolor='lightblue')
    dot.node('G', 'Audit: KPI Quality Scoring',
             shape='ellipse', style='filled',
             fillcolor='lightgray')
    dot.node('G_decide', 'Pass Quality? (>,80%)',
             shape='diamond', style='filled',
             fillcolor='white')

    dot.node('H', 'Transparency Trace\n(Optional)',
             shape='note', style='filled',
             fillcolor='lightyellow')

    # Flow edges
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'C_decide')
    dot.edge('C_decide', 'D', label='Yes', color='green')
    dot.edge('C_decide', 'B', label='No', style='dashed',
             color='red')

    dot.edge('D', 'E')
    dot.edge('E', 'E_decide')
    dot.edge('E_decide', 'F', label='Yes', color='green')
    dot.edge('E_decide', 'D', label='No', style='dashed',
             color='red')

    dot.edge('F', 'G')
    dot.edge('G', 'G_decide')
    dot.edge('G_decide', 'H', label='Yes', color='green')
    dot.edge('G_decide', 'F', label='No', style='dashed',
             color='red')

    # Output
    output_file = dot.render(
        '../fig/langgraph_pipeline', format='png',
        cleanup=True)
    abs_path = os.path.abspath(output_file)
    print(f"✅ Rendered corrected diagram to: {abs_path}")

if __name__ == '__main__':
    langgraph_flow()