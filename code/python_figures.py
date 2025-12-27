from __future__ import annotations
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from pytrends.request import TrendReq

import pandas as pd
import numpy as np
from graphviz import Digraph
import os


import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

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


@dataclass(frozen=True)
class PipelineConfig:
    out_dir: str = "../fig/"
    seed: int = 42

    # UMAP
    n_neighbors: int = 15
    min_dist: float = 0.10
    n_components: int = 2
    umap_metric: str = "cosine"

    # Clustering (HDBSCAN if available; else KMeans)
    hdbscan_min_cluster_size: int = 15
    kmeans_k: int = 4

    # Summary
    top_terms: int = 7
    quotes_per_cluster: int = 2


# ----------------------------
# 1) Example data generator (swap with real text later)
# ----------------------------

BASE_RESPONSES = [
    # Transparency / trust / ethics
    "I scanned the code to see where the product came from and whether it was local.",
    "Origin info makes me trust the brand more, especially for meat and dairy.",
    "Knowing the farm location helps me decide which product to buy.",
    "I like seeing certifications and dates; it feels more honest.",
    "Traceability is important to me because of food safety.",
    "I scanned to check if it matches the sustainability claims on the label.",
    "If I can verify the origin, I'm more willing to pay a bit extra.",

    # Low awareness / low relevance
    "I didn't notice the QR code at all.",
    "I saw it but I was in a hurry and didn't scan.",
    "I don't really care where it comes from if the price is right.",
    "I never scan codes on food packaging.",
    "I assume the information is marketing anyway.",
    "It doesn't change my decision; I buy the same brand out of habit.",

    # Usability / friction / skepticism
    "The page loaded slowly and I gave up.",
    "The link opened a long page; it was too much information.",
    "The text was hard to read on my phone.",
    "The scan didn't work under the supermarket lighting.",
    "I scanned but the website felt confusing and I couldn't find the key details.",
    "I worry about tracking/privacy when scanning product codes.",
    "The information looked generic, not specific to the product I bought.",

    # Mixed / practical use
    "I scanned after purchase at home, mostly out of curiosity.",
    "I would scan more often if the info was short and clear.",
    "I want a quick summary first, then details if needed.",
    "I scan when buying for kids; I want to avoid certain ingredients.",
    "I use it mainly for new brands, not for products I know already.",
]


def generate_responses(n: int, seed: int) -> List[str]:
    """Create a realistic sample by mixing base responses with template-based paraphrases."""
    rng = random.Random(seed)
    templates = [
        "I scanned the code to check {topic}.",
        "I didn't scan because {reason}.",
        "The information {effect} my purchase decision.",
        "Scanning was {ux}; I {outcome}.",
        "I care about {topic} because {value}.",
    ]
    topics = ["origin", "farm location", "animal welfare", "certifications", "ingredients", "food safety", "sustainability claims"]
    reasons = ["I was in a hurry", "I didn't notice it", "my phone had no signal", "I don't usually scan codes", "I wasn't sure it was safe"]
    effects = ["did not change", "strongly influenced", "slightly influenced", "could influence in the future"]
    uxs = ["easy", "slow", "confusing", "straightforward", "frustrating"]
    outcomes = ["finished quickly", "stopped halfway", "gave up", "found what I needed", "got more confused"]
    values = ["I want to support local producers", "I avoid greenwashing", "I want transparency", "I care about health", "I want ethical products"]

    out = list(BASE_RESPONSES)
    while len(out) < n:
        t = rng.choice(templates)
        out.append(t.format(
            topic=rng.choice(topics),
            reason=rng.choice(reasons),
            effect=rng.choice(effects),
            ux=rng.choice(uxs),
            outcome=rng.choice(outcomes),
            value=rng.choice(values),
        ))
    rng.shuffle(out)
    return out[:n]


def make_dataframe_from_texts(texts: List[str]) -> pd.DataFrame:
    return pd.DataFrame({"response_id": range(1, len(texts) + 1), "text": texts})


# ----------------------------
# 2) Embeddings (preferred: sentence-transformers; fallback: TF-IDF)
# ----------------------------

def embed_texts(texts: List[str]) -> Tuple[np.ndarray, str]:
    """Return (embeddings, method_name)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(texts, show_progress_bar=False)
        return np.asarray(emb), "sentence-transformers (all-MiniLM-L6-v2)"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vect = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        emb = vect.fit_transform(texts).toarray()
        return np.asarray(emb), "TF-IDF (fallback)"


# ----------------------------
# 3) UMAP projection
# ----------------------------

def umap_project(embeddings: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    try:
        import umap
    except Exception as e:
        raise RuntimeError("UMAP not installed. Install with: pip install umap-learn") from e

    reducer = umap.UMAP(
        n_neighbors=cfg.n_neighbors,
        min_dist=cfg.min_dist,
        n_components=cfg.n_components,
        metric=cfg.umap_metric,
        random_state=cfg.seed,
    )
    return reducer.fit_transform(embeddings)


# ----------------------------
# 4) Clustering (HDBSCAN preferred; fallback to KMeans)
# ----------------------------

def cluster_points(points_2d: np.ndarray, cfg: PipelineConfig) -> Tuple[np.ndarray, str]:
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=cfg.hdbscan_min_cluster_size)
        labels = clusterer.fit_predict(points_2d)
        return labels, "HDBSCAN"
    except Exception:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=cfg.kmeans_k, random_state=cfg.seed, n_init="auto")
        labels = km.fit_predict(points_2d)
        return labels, f"KMeans(k={cfg.kmeans_k})"


def format_cluster_name(label: int) -> str:
    return "Noise" if label == -1 else f"C{int(label) + 1}"


# ----------------------------
# 5) Cluster summaries (keywords + representative quotes)
# ----------------------------

def top_keywords(texts: List[str], top_n: int) -> str:
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(stop_words="english", max_features=2000)
    X = vec.fit_transform(texts)
    freqs = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    top_idx = freqs.argsort()[::-1][:top_n]
    return ", ".join(vocab[top_idx])


def build_cluster_summary(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    rows = []
    for cname, g in df.groupby("cluster_name"):
        texts = g["text"].tolist()
        n = len(texts)

        kw = top_keywords(texts, cfg.top_terms) if n >= 5 else ""
        quotes = texts[: cfg.quotes_per_cluster]

        rows.append({
            "ID": cname,
            "Size": n,
            "Cluster label/theme (auto)": f"Top terms: {kw}" if kw else "",
            "Representative quotes": " | ".join([f"“{q}”" for q in quotes]),
        })

    out = pd.DataFrame(rows).sort_values("Size", ascending=False).reset_index(drop=True)
    return out


# ----------------------------
# 6) Plotting helpers
# ----------------------------

def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_umap_plot(df: pd.DataFrame, cfg: PipelineConfig, filename_base: str) -> Tuple[str, str]:
    plt.figure(figsize=(8, 5))
    for cname, g in df.groupby("cluster_name"):
        plt.scatter(g["umap_x"], g["umap_y"], s=14, alpha=0.85, label=cname)

    # centroid labels
    for cname, g in df.groupby("cluster_name"):
        cx, cy = g[["umap_x", "umap_y"]].mean().values
        plt.text(cx, cy, cname, fontsize=10, weight="bold", ha="center", va="center")

    plt.title("UMAP projection of qualitative responses")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(title="Cluster", fontsize=9, title_fontsize=9, loc="best")
    plt.tight_layout()

    png_path = os.path.join(cfg.out_dir, f"{filename_base}.png")
    pdf_path = os.path.join(cfg.out_dir, f"{filename_base}.pdf")
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()
    return png_path, pdf_path


def save_table_figure(summary: pd.DataFrame, cfg: PipelineConfig, filename_base: str) -> Tuple[str, str]:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.axis("off")

    col_labels = ["ID", "Size", "Cluster label/theme (auto)", "Representative quotes"]
    cell_text = summary[col_labels].values.tolist()

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # column widths (relative)
    col_widths = [0.06, 0.08, 0.34, 0.52]
    for (row, col), cell in table.get_celld().items():
        if col < len(col_widths):
            cell.set_width(col_widths[col])
        if row == 0:
            cell.set_text_props(weight="bold")

    fig.tight_layout(pad=0.6)

    png_path = os.path.join(cfg.out_dir, f"{filename_base}.png")
    pdf_path = os.path.join(cfg.out_dir, f"{filename_base}.pdf")
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


# ----------------------------
# 7) End-to-end pipeline
# ----------------------------

def run_pipeline(texts: List[str], cfg: PipelineConfig) -> Dict[str, str]:
    ensure_out_dir(cfg.out_dir)

    df = make_dataframe_from_texts(texts)

    emb, emb_method = embed_texts(df["text"].tolist())
    coords = umap_project(emb, cfg)
    labels, cluster_method = cluster_points(coords, cfg)

    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]
    df["cluster"] = labels
    df["cluster_name"] = df["cluster"].apply(format_cluster_name)

    summary = build_cluster_summary(df, cfg)

    # Save summary CSV
    summary_csv = os.path.join(cfg.out_dir, "cluster_summary.csv")
    summary.to_csv(summary_csv, index=False)

    # Save figures
    umap_png, umap_pdf = save_umap_plot(df, cfg, "umap_plot")
    tab_png, tab_pdf = save_table_figure(summary, cfg, "cluster_summary_table")

    return {
        "embedding_method": emb_method,
        "clustering_method": cluster_method,
        "umap_png": umap_png,
        "umap_pdf": umap_pdf,
        "table_png": tab_png,
        "table_pdf": tab_pdf,
        "summary_csv": summary_csv,
    }


def umap():
    cfg = PipelineConfig(out_dir="../fig/", seed=42)
    texts = generate_responses(n=150, seed=cfg.seed)
    outputs = run_pipeline(texts, cfg)
    for k, v in outputs.items():
        print(f"{k}: {v}")

@dataclass(frozen=True)
class SdgTableConfig:
    out_dir: str = "../fig/"
    filename_base: str = "todo_sdg_mapping_table"
    max_rows: int = 8  # keep it thesis-friendly



def build_sdg_mappings() -> pd.DataFrame:
    """
    Realistic example table for SDG mappings & justifications.
    Replace/extend rows with your real artefact outputs later.
    """
    rows: List[Dict[str, str]] = [
        {
            "KPI / Indicator": "Consumer scan rate per 100 units sold",
            "Category": "Agrar & Agrar Tech",
            "Subcategory": "\"Farm-to-Fork\" Transparency",
            "Primary SDG target": "12.8 (Ensure people have relevant information for sustainable lifestyles)",
            "Secondary SDG target(s)": "9.c (Access to ICT)",
            "Justification (short)": "Scanning indicates consumer engagement with origin data. Increasing uptake improves informed purchase decisions and supports transparency along supply chains.",
            "Evidence / Data source": "QR/NFC analytics + units sold (SKU-level)",
            "Framework reference": "SDG 12.8; GS1 Digital Link; IRIS+ (Product/Service Users)",
        },
        {
            "KPI / Indicator": "Share of transactions with an origin-data view",
            "Category": "Agrar & Agrar Tech",
            "Subcategory": "\"Farm-to-Fork\" Transparency",
            "Primary SDG target": "12.8",
            "Secondary SDG target(s)": "12.6 (Encourage companies to adopt sustainable practices and reporting)",
            "Justification (short)": "Views during shopping reflect practical use of transparency information, not just curiosity. Can be linked to sustainable consumption behaviour and transparency reporting.",
            "Evidence / Data source": "Linked scans to batch/SKU sales within 24–72h window",
            "Framework reference": "SDG 12.8; SDG 12.6; GS1 EPCIS (traceability event model)",
        },
        {
            "KPI / Indicator": "Share of products with verified traceability record (batch-level)",
            "Category": "Agrar & Agrar Tech",
            "Subcategory": "Traceability Infrastructure",
            "Primary SDG target": "12.6",
            "Secondary SDG target(s)": "9.4 (Upgrade infrastructure and retrofit industries for sustainability)",
            "Justification (short)": "Batch-level traceability improves accountability in production systems and enables more credible sustainability disclosures across the value chain.",
            "Evidence / Data source": "EPCIS event completeness checks; supplier onboarding logs",
            "Framework reference": "SDG 12.6; SDG 9.4; GS1 EPCIS 2.0; ISO 22005 (food chain traceability)",
        },
        {
            "KPI / Indicator": "Reduction in customer-reported information gaps (survey scale 1–5)",
            "Category": "Consumer Empowerment",
            "Subcategory": "Informed Decision-Making",
            "Primary SDG target": "12.8",
            "Secondary SDG target(s)": "16.6 (Effective, accountable and transparent institutions)",
            "Justification (short)": "Perceived information sufficiency is a direct user-level outcome. Improved transparency can reduce uncertainty and build trust in responsible market practices.",
            "Evidence / Data source": "Consumer survey (pre/post) + qualitative comments",
            "Framework reference": "SDG 12.8; SDG 16.6 (transparency); survey instrument",
        },
        {
            "KPI / Indicator": "Active monthly users among impact startups (platform)",
            "Category": "Platform Adoption",
            "Subcategory": "Ecosystem Enablement",
            "Primary SDG target": "9.3 (Increase access of small enterprises to services/value chains)",
            "Secondary SDG target(s)": "8.3 (Support productive activities, entrepreneurship)",
            "Justification (short)": "Active usage by early-stage ventures indicates adoption of enabling infrastructure. This supports entrepreneurship and access to tools for responsible innovation.",
            "Evidence / Data source": "Platform analytics (MAU), cohort membership, retention",
            "Framework reference": "SDG 9.3; SDG 8.3; IRIS+ (Business Support Services)",
        },
        {
            "KPI / Indicator": "Share of SMEs reporting at least one impact metric quarterly",
            "Category": "Impact Measurement",
            "Subcategory": "Reporting & Learning",
            "Primary SDG target": "12.6",
            "Secondary SDG target(s)": "17.17 (Partnerships for sustainable development)",
            "Justification (short)": "Regular reporting operationalises sustainability management and supports learning loops with stakeholders and partners.",
            "Evidence / Data source": "Submitted reports per quarter + completeness checks",
            "Framework reference": "SDG 12.6; SDG 17.17; impact reporting practice",
        },
        {
            "KPI / Indicator": "Decrease in process energy use per unit output (pilot SMEs)",
            "Category": "Environmental Impact",
            "Subcategory": "Energy Efficiency",
            "Primary SDG target": "7.3 (Double the global rate of improvement in energy efficiency)",
            "Secondary SDG target(s)": "13.2 (Integrate climate measures into policies/planning)",
            "Justification (short)": "Energy intensity improvements reduce emissions and operational costs. KPI supports climate action through measurable efficiency gains.",
            "Evidence / Data source": "Electricity bills + output volume (normalised)",
            "Framework reference": "SDG 7.3; SDG 13.2; GHG Protocol (contextual reference)",
        },
        {
            "KPI / Indicator": "Share of participants from underrepresented groups in workshops",
            "Category": "Social Impact",
            "Subcategory": "Inclusion & Participation",
            "Primary SDG target": "10.2 (Promote social, economic and political inclusion)",
            "Secondary SDG target(s)": "5.5 (Ensure women’s full participation and equal opportunities)",
            "Justification (short)": "Participation rates indicate whether capacity-building formats reach diverse groups and reduce access barriers.",
            "Evidence / Data source": "Registration data (consent-based) + attendance logs",
            "Framework reference": "SDG 10.2; SDG 5.5; DEI reporting practice",
        },
    ]
    return pd.DataFrame(rows)


def save_sdg_table(df: pd.DataFrame, cfg: SdgTableConfig) -> Dict[str, str]:
    ensure_out_dir(cfg.out_dir)

    df_out = df.head(cfg.max_rows).copy()

    csv_path = os.path.join(cfg.out_dir, f"{cfg.filename_base}.csv")
    png_path = os.path.join(cfg.out_dir, f"{cfg.filename_base}.png")
    pdf_path = os.path.join(cfg.out_dir, f"{cfg.filename_base}.pdf")

    df_out.to_csv(csv_path, index=False)

    # Render as an image table (easy to include in LaTeX)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axis("off")

    col_order = [
        "KPI / Indicator",
        "Primary SDG target",
        "Secondary SDG target(s)",
        "Justification (short)",
        "Framework reference",
    ]

    cell_text = df_out[col_order].values.tolist()
    col_labels = col_order

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.45)

    # column widths (relative)
    col_widths = [0.22, 0.14, 0.14, 0.34, 0.16]
    for (row, col), cell in table.get_celld().items():
        if col < len(col_widths):
            cell.set_width(col_widths[col])
        if row == 0:
            cell.set_text_props(weight="bold")

    fig.tight_layout(pad=0.8)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    return {"csv": csv_path, "png": png_path, "pdf": pdf_path}


def sdg_tables():
    cfg = SdgTableConfig(out_dir="../fig/", filename_base="sdg_mapping_table", max_rows=8)
    df = build_sdg_mappings()
    outputs = save_sdg_table(df, cfg)
    print("Saved:", outputs)



if __name__ == '__main__':
    sdg_tables()