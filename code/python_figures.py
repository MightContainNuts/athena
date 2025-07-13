from matplotlib import pyplot as plt
import networkx as nx
from pytrends.request import TrendReq
import seaborn as sns
import pandas as pd
import numpy as np


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

if __name__ == '__main__':
    fig_synthesis_gaps()