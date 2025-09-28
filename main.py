import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
from google_play_scraper import Sort, reviews, app
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
import aiohttp
from tqdm import tqdm
import logging
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import gc
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class LlamaModel:
    """Singleton class to manage Llama 3.2 3B model"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LlamaModel, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self, device: str = None):
        """Initialize the Llama model"""
        if not self.initialized:
            logger.info("Loading Llama 3.2 3B Instruct model...")

            # Determine device
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            logger.info(f"Using device: {self.device}")

            # Model name - Llama 3.2 3B Instruct
            model_name = "meta-llama/Llama-3.2-3B-Instruct"

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with optimization for memory
            if self.device == "cuda":
                # Use 8-bit quantization for GPU to save memory
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                  load_in_4bit=True,
                  bnb_4bit_use_double_quant=True,
                  bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=torch.float16
              )

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            else:
                # CPU loading
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                self.model = self.model.to(self.device)

            self.model.eval()
            self.initialized = True
            logger.info("Model loaded successfully!")

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
        """Generate response from the model"""
        # Format prompt for Llama 3.2 chat format
        messages = [
            {"role": "system", "content": "You are an expert at analyzing customer reviews and extracting specific issues."},
            {"role": "user", "content": prompt}
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Clean up GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return response

  @dataclass
class Review:
    """Data class for storing review information"""
    content: str
    rating: int
    date: datetime
    review_id: str
    user_name: str

@dataclass
class TopicCluster:
    """Data class for topic clusters"""
    canonical_name: str
    variations: List[str]
    example_reviews: List[str]
    count: int = 0

class ReviewScraper:
    """Handles scraping reviews from Google Play Store"""

    def __init__(self, app_id: str):
        self.app_id = app_id

    def fetch_reviews_batch(self, start_date: datetime, end_date: datetime) -> List[Review]:
        """Fetch reviews for a specific date range"""
        all_reviews = []
        continuation_token = None

        try:
            while True:
                result, continuation_token = reviews(
                    self.app_id,
                    lang='en',
                    country='in',
                    sort=Sort.NEWEST,  # Sort by newest first
                    count=200,
                    continuation_token=continuation_token
                )

                for r in result:
                    review_date = r['at']
                    if start_date <= review_date <= end_date:
                        all_reviews.append(Review(
                            content=r['content'],
                            rating=r['score'],
                            date=review_date,
                            review_id=r['reviewId'],
                            user_name=r['userName']
                        ))
                    elif review_date < start_date:
                        return all_reviews

                if not continuation_token:
                    break

        except Exception as e:
            logger.error(f"Error fetching reviews: {e}")

        return all_reviews

  class TopicExtractor:
    """AI Agent for extracting negative issues from reviews using Llama 3.3"""

    def __init__(self):
        self.llama = LlamaModel()
        self.llama.initialize()

    async def extract_topics_batch(self, reviews: List[Review]) -> Dict[str, List[str]]:
        """Extract negative issues from a batch of reviews using Llama 3.3"""

        # Group reviews into chunks for processing
        chunk_size = 10
        review_chunks = [reviews[i:i+chunk_size] for i in range(0, len(reviews), chunk_size)]

        all_extractions = {}

        for chunk in tqdm(review_chunks, desc="Processing review chunks"):
            # Filter to focus on negative reviews (rating <= 3)
            negative_reviews = [r for r in chunk if r.rating <= 3]
            if not negative_reviews:
                continue

            reviews_text = "\n".join([f"Review {i+1} (Rating: {r.rating}/5): {r.content}"
                                    for i, r in enumerate(negative_reviews)])

            prompt = f"""
            Analyze the following customer reviews and extract ONLY negative issues, problems, complaints, and pain points.

            IMPORTANT RULES:
            1. Extract ONLY negative feedback - ignore all positive comments
            2. Focus on problems, issues, complaints, bugs, and failures
            3. Each review can have multiple distinct problems
            4. DO NOT include any positive feedback like "good service", "nice app", etc.

            Reviews to analyze:
            {reviews_text}

            Return a JSON object with review numbers as keys and lists of specific NEGATIVE issues as values.
            Only include reviews that have actual problems.

            Example output format:
            {{"1": ["order delivered 3 hours late", "food was cold and stale"], "2": ["payment failed multiple times", "app crashed during checkout"]}}

            If a review has no negative issues, don't include it in the output.
            """

            try:
                response = await asyncio.to_thread(self.llama.generate, prompt, max_new_tokens=256)
                extracted = self._parse_llm_response(response)

                # Map back to original review IDs
                for review_idx, topics in extracted.items():
                    try:
                        idx = int(review_idx) - 1
                        if idx < len(negative_reviews):
                            review_id = negative_reviews[idx].review_id
                            # Filter out any accidentally included positive topics
                            negative_topics = [t for t in topics if self._is_negative_topic(t)]
                            if negative_topics:
                                all_extractions[review_id] = negative_topics
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error mapping review index: {e}")

            except Exception as e:
                logger.error(f"Error in topic extraction: {e}")

        return all_extractions

    def _is_negative_topic(self, topic: str) -> bool:
        """Check if a topic is actually negative"""
        positive_keywords = ['good', 'great', 'excellent', 'awesome', 'love', 'best',
                           'perfect', 'amazing', 'fantastic', 'wonderful', 'satisfied']
        topic_lower = topic.lower()
        return not any(word in topic_lower for word in positive_keywords)

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM response to extract JSON"""
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            return {}
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return {}

      class LLMTopicConsolidator:
    """LLM-based agent for consolidating similar negative issues into categories"""

    def __init__(self):
        self.llama = LlamaModel()
        self.llama.initialize()
        self.category_cache = {}

    async def consolidate_topics(self, topics_with_counts: Dict[str, int]) -> Dict[str, str]:
        """Use Llama 3.3 to group similar negative issues into coherent categories"""

        if not topics_with_counts:
            return {}

        # Sort topics by frequency for better processing
        sorted_topics = sorted(topics_with_counts.items(), key=lambda x: x[1], reverse=True)
        topics_list = [topic for topic, _ in sorted_topics[:100]]  # Process top 100

        prompt = f"""
        Group the following customer complaints and issues into logical problem categories.
        These are negative feedback extracted from app reviews.

        Issues to categorize:
        {json.dumps(topics_list, indent=2)}

        RULES:
        1. Group similar problems together (e.g., "delivery was 2 hours late", "delayed delivery", "order took forever" → "Delivery delays")
        2. Create clear, problem-focused category names
        3. Each issue must map to exactly ONE category
        4. Keep categories specific and actionable for fixing problems
        5. Category names should clearly indicate the problem area

        Return ONLY a JSON object mapping each issue to its problem category.

        Example output:
        {{
            "delivery was 2 hours late": "Delivery delays",
            "order took forever": "Delivery delays",
            "delivery partner was rude": "Delivery partner behavior issues",
            "driver didn't follow instructions": "Delivery partner behavior issues",
            "app crashes during payment": "Payment system failures",
            "payment failed but money deducted": "Payment system failures"
        }}
        """

        try:
            response = await asyncio.to_thread(self.llama.generate, prompt, max_new_tokens=512)
            mappings = self._parse_llm_response(response)

            # Handle topics not in the top 100
            if len(topics_with_counts) > 100:
                remaining_topics = list(topics_with_counts.keys())[100:]
                for topic in remaining_topics:
                    mappings[topic] = await self._find_best_category(topic, mappings)

            return mappings

        except Exception as e:
            logger.error(f"Error in LLM consolidation: {e}")
            return {topic: topic for topic in topics_list}

    async def _find_best_category(self, topic: str, existing_mappings: Dict[str, str]) -> str:
        """Find the best matching category for a topic"""
        categories = list(set(existing_mappings.values()))

        prompt = f"""
        Which problem category best fits this issue: "{topic}"

        Available problem categories:
        {json.dumps(categories, indent=2)}

        Return ONLY the category name that best matches, or return "{topic}" if none fit well.
        """

        try:
            response = await asyncio.to_thread(self.llama.generate, prompt, max_new_tokens=50, temperature=0.1)
            return response.strip().strip('"')
        except:
            return topic

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM response to extract JSON"""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            return {}
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return {}

      class ClusteringTopicConsolidator:
    """Clustering-based consolidator using embeddings and unsupervised learning"""

    def __init__(self, method: str = "hierarchical", threshold: float = 0.3):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.method = method
        self.threshold = threshold

    def consolidate_topics(self, topics_with_counts: Dict[str, int]) -> Dict[str, str]:
        """Consolidate topics using clustering algorithms"""

        if not topics_with_counts:
            return {}

        topics_list = list(topics_with_counts.keys())

        if len(topics_list) == 1:
            return {topics_list[0]: topics_list[0]}

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(topics_list)} topics...")
        embeddings = self.embedding_model.encode(topics_list)

        # Apply clustering
        if self.method == "dbscan":
            labels = self._dbscan_clustering(embeddings)
        else:  # hierarchical
            labels = self._hierarchical_clustering(embeddings, len(topics_list))

        # Create mappings
        mappings = {}
        clusters = defaultdict(list)

        for topic, label in zip(topics_list, labels):
            clusters[label].append((topic, topics_with_counts[topic]))

        # For each cluster, select the canonical name
        for label, cluster_topics in clusters.items():
            if label == -1:  # DBSCAN noise points
                for topic, _ in cluster_topics:
                    mappings[topic] = topic
            else:
                # Sort by frequency and length (prefer shorter, more frequent)
                cluster_topics.sort(key=lambda x: (-x[1], len(x[0])))
                canonical = cluster_topics[0][0]

                for topic, _ in cluster_topics:
                    mappings[topic] = canonical

        logger.info(f"Consolidated {len(topics_list)} topics into {len(set(mappings.values()))} categories")

        return mappings

    def _dbscan_clustering(self, embeddings):
        """Apply DBSCAN clustering"""
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix

        clustering = DBSCAN(
            eps=self.threshold,
            min_samples=2,
            metric='precomputed'
        )

        return clustering.fit_predict(distance_matrix)

    def _hierarchical_clustering(self, embeddings, n_topics):
        """Apply Hierarchical/Agglomerative clustering"""
        n_clusters = max(2, min(int(np.sqrt(n_topics / 2)), n_topics // 3))

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )

        return clustering.fit_predict(embeddings)

      class HybridTopicConsolidator:
    """Hybrid approach: Clustering + LLM refinement"""

    def __init__(self):
        self.clustering_consolidator = ClusteringTopicConsolidator(method="hierarchical")
        self.llm_consolidator = LLMTopicConsolidator()

    async def consolidate_topics(self, topics_with_counts: Dict[str, int]) -> Dict[str, str]:
        """Two-step consolidation: clustering followed by LLM refinement"""

        # Step 1: Initial clustering
        logger.info("Step 1: Clustering similar topics...")
        cluster_mappings = self.clustering_consolidator.consolidate_topics(topics_with_counts)

        # Get unique clusters and their total counts
        cluster_counts = defaultdict(int)
        for topic, cluster in cluster_mappings.items():
            cluster_counts[cluster] += topics_with_counts[topic]

        # Step 2: LLM refinement on cluster representatives
        logger.info("Step 2: LLM refinement of categories...")
        refined_mappings = await self.llm_consolidator.consolidate_topics(cluster_counts)

        # Combine mappings
        final_mappings = {}
        for topic, cluster in cluster_mappings.items():
            final_mappings[topic] = refined_mappings.get(cluster, cluster)

        return final_mappings

      class TrendAnalyzer:
    """Agent for analyzing trends and generating reports with visualizations"""

    def __init__(self):
        self.trends_data = defaultdict(lambda: defaultdict(int))
        self.category_evolution = defaultdict(list)
        self.daily_totals = defaultdict(int)

    def update_trends(self, date: datetime, topic_counts: Dict[str, int]):
        """Update trend data with daily topic counts"""
        date_str = date.strftime('%Y-%m-%d')
        for topic, count in topic_counts.items():
            self.trends_data[topic][date_str] = count
            self.daily_totals[date_str] += count

        self.category_evolution[date_str] = list(topic_counts.keys())

    def generate_trend_report(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate trend report for the specified date range"""

        # Create date range
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)

        if not dates:
            return pd.DataFrame()

        # Create DataFrame
        report_data = {}
        for topic in self.trends_data.keys():
            topic_trend = []
            for date in dates:
                topic_trend.append(self.trends_data[topic].get(date, 0))

            # Only include topics with at least some activity
            if sum(topic_trend) > 0:
                report_data[topic] = topic_trend

        if not report_data:
            return pd.DataFrame()

        df = pd.DataFrame(report_data, index=dates).T
        df.columns = [datetime.strptime(d, '%Y-%m-%d').strftime('%b %d') for d in dates]

        # Sort by total frequency
        df['Total'] = df.sum(axis=1)
        df = df.sort_values('Total', ascending=False)
        df = df.drop('Total', axis=1)

        return df

    def generate_visualizations(self, start_date: datetime, end_date: datetime, output_dir: str = "output"):
        """Generate trend visualization plots"""

        # Ensure output directory exists
        os.makedirs(f"{output_dir}/plots", exist_ok=True)

        # Get report data
        report_df = self.generate_trend_report(start_date, end_date)

        if report_df.empty:
            logger.warning("No data to visualize")
            return

        # 1. Top Issues Bar Chart
        self._plot_top_issues_bar(report_df, output_dir)

        # 2. Trend Lines for Top Issues
        self._plot_trend_lines(report_df, output_dir)

        # 3. Heatmap
        self._plot_heatmap(report_df, output_dir)

        # 4. Daily Issue Volume
        self._plot_daily_volume(start_date, end_date, output_dir)

        # 5. Interactive Plotly Dashboard (without kaleido)
        self._create_interactive_dashboard(report_df, output_dir)

        logger.info(f"Visualizations saved to {output_dir}/plots/")

    def _plot_top_issues_bar(self, df: pd.DataFrame, output_dir: str):
        """Create bar chart of top issues"""
        plt.figure(figsize=(12, 6))

        # Get top 15 issues
        top_issues = df.sum(axis=1).sort_values(ascending=False).head(15)

        ax = top_issues.plot(kind='barh', color='coral')
        plt.title('Top 15 Issues by Frequency', fontsize=16, fontweight='bold')
        plt.xlabel('Total Occurrences', fontsize=12)
        plt.ylabel('Issue Category', fontsize=12)
        plt.tight_layout()

        # Add value labels
        for i, v in enumerate(top_issues.values):
            ax.text(v + 0.5, i, str(int(v)), va='center')

        plt.savefig(f"{output_dir}/plots/top_issues_bar.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_trend_lines(self, df: pd.DataFrame, output_dir: str):
        """Create trend lines for top issues"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Top 5 issues
        top_5 = df.sum(axis=1).sort_values(ascending=False).head(5).index

        for issue in top_5:
            axes[0].plot(df.columns, df.loc[issue], marker='o', label=issue[:30] + '...' if len(issue) > 30 else issue)

        axes[0].set_title('Trend Lines - Top 5 Issues', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)

        # Issues 6-10
        issues_6_10 = df.sum(axis=1).sort_values(ascending=False).iloc[5:10].index

        for issue in issues_6_10:
            axes[1].plot(df.columns, df.loc[issue], marker='s', label=issue[:30] + '...' if len(issue) > 30 else issue)

        axes[1].set_title('Trend Lines - Issues 6-10', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)

        # Rotate x-axis labels
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/trend_lines.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_heatmap(self, df: pd.DataFrame, output_dir: str):
        """Create heatmap of issues over time"""
        plt.figure(figsize=(16, 10))

        # Get top 20 issues
        top_20 = df.sum(axis=1).sort_values(ascending=False).head(20).index

        # Create heatmap
        sns.heatmap(df.loc[top_20],
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Frequency'},
                   linewidths=0.5,
                   linecolor='gray')

        plt.title('Issue Frequency Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Issue Category', fontsize=12)
        plt.tight_layout()

        plt.savefig(f"{output_dir}/plots/heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_daily_volume(self, start_date: datetime, end_date: datetime, output_dir: str):
        """Plot daily issue volume"""
        plt.figure(figsize=(14, 6))

        # Prepare data
        dates = []
        volumes = []
        current = start_date
        while current <= end_date:
            date_str = current.strftime('%Y-%m-%d')
            dates.append(current)
            volumes.append(self.daily_totals.get(date_str, 0))
            current += timedelta(days=1)

        # Create plot
        plt.plot(dates, volumes, marker='o', linewidth=2, markersize=6, color='steelblue')
        plt.fill_between(dates, volumes, alpha=0.3, color='steelblue')

        plt.title('Daily Issue Volume Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Issues Reported', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Format x-axis
        plt.gca().xaxis.set_major_formatter(DateFormatter('%b %d'))
        plt.xticks(rotation=45)

        # Add average line
        avg_volume = np.mean(volumes) if volumes else 0
        plt.axhline(y=avg_volume, color='red', linestyle='--', alpha=0.7, label=f'Average: {avg_volume:.1f}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/daily_volume.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_interactive_dashboard(self, df: pd.DataFrame, output_dir: str):
        """Create interactive Plotly dashboard without kaleido"""

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top 10 Issues', 'Trend Lines', 'Daily Distribution', 'Issue Evolution'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'box'}, {'type': 'scatter'}]]
        )

        # 1. Top 10 Issues Bar
        top_10 = df.sum(axis=1).sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=top_10.values, y=top_10.index, orientation='h', name='Frequency'),
            row=1, col=1
        )

        # 2. Trend Lines for Top 5
        top_5 = df.sum(axis=1).sort_values(ascending=False).head(5).index
        for issue in top_5:
            fig.add_trace(
                go.Scatter(x=df.columns, y=df.loc[issue],
                          mode='lines+markers',
                          name=issue[:20] + '...' if len(issue) > 20 else issue),
                row=1, col=2
            )

        # 3. Box plot distribution
        for issue in top_5:
            fig.add_trace(
                go.Box(y=df.loc[issue], name=issue[:15] + '...' if len(issue) > 15 else issue),
                row=2, col=1
            )

        # 4. Cumulative trend
        cumulative_data = df.loc[top_5].cumsum(axis=1)
        for issue in top_5:
            fig.add_trace(
                go.Scatter(x=cumulative_data.columns, y=cumulative_data.loc[issue],
                          mode='lines',
                          name=issue[:20] + '...' if len(issue) > 20 else issue,
                          stackgroup='one'),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title_text="Issue Analysis Dashboard",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )

        # Save interactive HTML
        fig.write_html(f"{output_dir}/plots/interactive_dashboard.html")

        # For Google Colab, you can display the figure directly instead of saving as image
        # Optionally save as static matplotlib figure instead
        self._save_dashboard_as_matplotlib(df, output_dir)

        # Log information about the saved dashboard
        logger.info(f"Interactive dashboard saved as HTML at {output_dir}/plots/interactive_dashboard.html")
        logger.info("Static version saved as dashboard.png using matplotlib")

    def _save_dashboard_as_matplotlib(self, df: pd.DataFrame, output_dir: str):
        """Create a static matplotlib version of the dashboard as alternative to Plotly static export"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Issue Analysis Dashboard', fontsize=16, fontweight='bold')

        # Top 10 Issues Bar
        top_10 = df.sum(axis=1).sort_values(ascending=False).head(10)
        axes[0, 0].barh(range(len(top_10)), top_10.values, color='steelblue')
        axes[0, 0].set_yticks(range(len(top_10)))
        axes[0, 0].set_yticklabels([label[:25] + '...' if len(label) > 25 else label
                                     for label in top_10.index])
        axes[0, 0].set_xlabel('Frequency')
        axes[0, 0].set_title('Top 10 Issues')
        axes[0, 0].grid(True, alpha=0.3)

        # Trend Lines for Top 5
        top_5 = df.sum(axis=1).sort_values(ascending=False).head(5).index
        for issue in top_5:
            axes[0, 1].plot(df.columns, df.loc[issue], marker='o',
                           label=issue[:20] + '...' if len(issue) > 20 else issue)
        axes[0, 1].set_title('Trend Lines - Top 5')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Box plot distribution
        box_data = [df.loc[issue].values for issue in top_5]
        bp = axes[1, 0].boxplot(box_data, labels=[issue[:15] + '...' if len(issue) > 15 else issue
                                                   for issue in top_5])
        axes[1, 0].set_title('Daily Distribution - Top 5')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Cumulative trend
        cumulative_data = df.loc[top_5].cumsum(axis=1)
        for issue in top_5:
            axes[1, 1].fill_between(range(len(cumulative_data.columns)),
                                   cumulative_data.loc[issue].values,
                                   alpha=0.5,
                                   label=issue[:20] + '...' if len(issue) > 20 else issue)
        axes[1, 1].set_title('Cumulative Issue Evolution')
        axes[1, 1].set_xlabel('Days')
        axes[1, 1].set_ylabel('Cumulative Frequency')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()

    def get_emerging_topics(self, target_date: datetime, days_back: int = 7) -> List[str]:
        """Identify newly emerging topics in recent days"""
        recent_dates = [(target_date - timedelta(days=i)).strftime('%Y-%m-%d')
                       for i in range(days_back)]
        older_dates = [(target_date - timedelta(days=i)).strftime('%Y-%m-%d')
                      for i in range(days_back, days_back * 2)]

        recent_topics = set()
        older_topics = set()

        for date_str in recent_dates:
            if date_str in self.category_evolution:
                recent_topics.update(self.category_evolution[date_str])

        for date_str in older_dates:
            if date_str in self.category_evolution:
                older_topics.update(self.category_evolution[date_str])

        emerging = recent_topics - older_topics
        return list(emerging)

    def display_in_colab(self, start_date: datetime, end_date: datetime):
        """Helper method to display Plotly figures directly in Google Colab"""
        report_df = self.generate_trend_report(start_date, end_date)

        if report_df.empty:
            print("No data to visualize")
            return

        # Create the interactive dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top 10 Issues', 'Trend Lines', 'Daily Distribution', 'Issue Evolution'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'box'}, {'type': 'scatter'}]]
        )

        # Add all the traces (same as in _create_interactive_dashboard)
        top_10 = report_df.sum(axis=1).sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=top_10.values, y=top_10.index, orientation='h', name='Frequency'),
            row=1, col=1
        )

        top_5 = report_df.sum(axis=1).sort_values(ascending=False).head(5).index
        for issue in top_5:
            fig.add_trace(
                go.Scatter(x=report_df.columns, y=report_df.loc[issue],
                          mode='lines+markers',
                          name=issue[:20] + '...' if len(issue) > 20 else issue),
                row=1, col=2
            )

        for issue in top_5:
            fig.add_trace(
                go.Box(y=report_df.loc[issue], name=issue[:15] + '...' if len(issue) > 15 else issue),
                row=2, col=1
            )

        cumulative_data = report_df.loc[top_5].cumsum(axis=1)
        for issue in top_5:
            fig.add_trace(
                go.Scatter(x=cumulative_data.columns, y=cumulative_data.loc[issue],
                          mode='lines',
                          name=issue[:20] + '...' if len(issue) > 20 else issue,
                          stackgroup='one'),
                row=2, col=2
            )

        fig.update_layout(
            title_text="Issue Analysis Dashboard",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )

        # Display the figure directly in Colab
        fig.show()

        return fig

      class ReviewAnalysisOrchestrator:
    """Main orchestrator for the review analysis pipeline"""

    def __init__(self, app_id: str, consolidation_method: str = "hybrid"):
        """
        Initialize orchestrator

        Args:
            app_id: Google Play Store app ID
            consolidation_method: "llm", "clustering", or "hybrid"
        """
        self.app_id = app_id
        self.scraper = ReviewScraper(app_id)
        self.topic_extractor = TopicExtractor()

        # Choose consolidation method
        if consolidation_method == "llm":
            self.consolidator = LLMTopicConsolidator()
        elif consolidation_method == "clustering":
            self.consolidator = ClusteringTopicConsolidator()
        else:  # hybrid
            self.consolidator = HybridTopicConsolidator()

        self.trend_analyzer = TrendAnalyzer()
        self.processed_dates = set()
        self.all_topics_extracted = defaultdict(int)

    async def process_daily_batch(self, date: datetime) -> Dict[str, int]:
        """Process reviews for a single day"""

        logger.info(f"Processing reviews for {date.strftime('%Y-%m-%d')}")

        # Fetch reviews for the day
        start_date = datetime.combine(date.date(), datetime.min.time())
        end_date = datetime.combine(date.date(), datetime.max.time())
        reviews = self.scraper.fetch_reviews_batch(start_date, end_date)

        if not reviews:
            logger.info(f"No reviews found for {date.strftime('%Y-%m-%d')}")
            return {}

        logger.info(f"Found {len(reviews)} reviews")

        # Filter for negative reviews (rating <= 3)
        negative_reviews = [r for r in reviews if r.rating <= 3]
        logger.info(f"Found {len(negative_reviews)} negative reviews (rating <= 3)")

        if not negative_reviews:
            return {}

        # Extract topics (only negative issues)
        topic_extractions = await self.topic_extractor.extract_topics_batch(negative_reviews)

        # Count raw topics before consolidation
        raw_topic_counts = defaultdict(int)
        for topics in topic_extractions.values():
            for topic in topics:
                raw_topic_counts[topic] += 1
                self.all_topics_extracted[topic] += 1

        logger.info(f"Extracted {len(raw_topic_counts)} unique negative issues")

        # Consolidate topics
        if isinstance(self.consolidator, (LLMTopicConsolidator, HybridTopicConsolidator)):
            topic_mappings = await self.consolidator.consolidate_topics(raw_topic_counts)
        else:
            topic_mappings = self.consolidator.consolidate_topics(raw_topic_counts)

        # Apply consolidation
        consolidated_counts = defaultdict(int)
        for topic, count in raw_topic_counts.items():
            category = topic_mappings.get(topic, topic)
            consolidated_counts[category] += count

        logger.info(f"Consolidated to {len(consolidated_counts)} problem categories")

        # Update trends
        self.trend_analyzer.update_trends(date, consolidated_counts)
        self.processed_dates.add(date.strftime('%Y-%m-%d'))

        return consolidated_counts

    async def run_batch_processing(self, start_date: datetime, end_date: datetime):
        """Run batch processing for date range"""

        current_date = start_date
        while current_date <= end_date:
            await self.process_daily_batch(current_date)
            current_date += timedelta(days=1)

            # Periodic re-consolidation for consistency
            if len(self.processed_dates) % 7 == 0:
                await self._global_reconsolidation()

    async def _global_reconsolidation(self):
        """Periodically reconsolidate all topics for consistency"""
        logger.info("Performing global reconsolidation...")

        if isinstance(self.consolidator, (LLMTopicConsolidator, HybridTopicConsolidator)):
            global_mappings = await self.consolidator.consolidate_topics(self.all_topics_extracted)
        else:
            global_mappings = self.consolidator.consolidate_topics(self.all_topics_extracted)

        # Update mappings for consistency across dates
        logger.info(f"Global consolidation complete: {len(global_mappings)} mappings")

    def generate_report(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate trend analysis report for specified date range"""
        report = self.trend_analyzer.generate_trend_report(start_date, end_date)

        # Add emerging topics information
        emerging = self.trend_analyzer.get_emerging_topics(end_date)
        if emerging:
            logger.info(f"Emerging issues: {emerging[:5]}")

        return report

    def generate_visualizations(self, start_date: datetime, end_date: datetime, output_dir: str = "output"):
        """Generate trend visualizations"""
        self.trend_analyzer.generate_visualizations(start_date, end_date, output_dir)

    def get_analytics(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get analytics about the processing"""
        total_issues = sum(self.trend_analyzer.daily_totals.values())
        avg_daily = total_issues / max(len(self.processed_dates), 1)

        # Get top issues for the period
        report = self.generate_report(start_date, end_date)
        top_issues = []
        if not report.empty:
            top_5 = report.sum(axis=1).sort_values(ascending=False).head(5)
            top_issues = [(issue, int(count)) for issue, count in top_5.items()]

        return {
            "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "total_dates_processed": len(self.processed_dates),
            "unique_issues_extracted": len(self.all_topics_extracted),
            "unique_problem_categories": len(set(self.trend_analyzer.trends_data.keys())),
            "total_issues_reported": total_issues,
            "average_issues_per_day": round(avg_daily, 2),
            "top_5_issues": top_issues,
            "emerging_issues": self.trend_analyzer.get_emerging_topics(end_date)[:5]
        }

  async def main():
    """Main execution function"""

    # Configuration
    APP_ID = "com.application.zomato"
    os.environ["HF_TOKEN"] =  'YOUR_HUGGINGFACE_TOKEN_HERE'
    CONSOLIDATION_METHOD = "hybrid"  # llm, clustering, or hybrid

    # Date range configuration
    START_DATE_STR = os.environ.get("START_DATE", "2025-09-21")
    END_DATE_STR = os.environ.get("END_DATE", "2025-09-28")


    # Parse dates
    start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE_STR, "%Y-%m-%d")

    # Initialize orchestrator
    logger.info(f"Initializing with {CONSOLIDATION_METHOD} consolidation method")
    logger.info(f"Using Llama 3.1 via HuggingFace API")
    orchestrator = ReviewAnalysisOrchestrator(APP_ID, CONSOLIDATION_METHOD)

    # Process reviews in batches
    logger.info(f"Starting batch processing from {start_date} to {end_date}")
    logger.info("Note: Only extracting NEGATIVE issues (complaints, problems, bugs)")
    await orchestrator.run_batch_processing(start_date, end_date)

    # Generate report for the date range
    report = orchestrator.generate_report(start_date, end_date)

    # Save report
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    report_filename = f"{output_dir}/trend_report_{end_date.strftime('%Y%m%d')}.csv"
    report.to_csv(report_filename)

    # Save detailed Excel with multiple sheets
    excel_filename = f"{output_dir}/trend_report_{end_date.strftime('%Y%m%d')}.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Main trend report
        report.to_excel(writer, sheet_name='Issue Trends')

        # Analytics summary
        analytics = orchestrator.get_analytics(start_date, end_date)

        # Format analytics for Excel
        analytics_df = pd.DataFrame([
            ['Date Range', analytics['date_range']],
            ['Total Days Processed', analytics['total_dates_processed']],
            ['Unique Issues Extracted', analytics['unique_issues_extracted']],
            ['Problem Categories', analytics['unique_problem_categories']],
            ['Total Issues Reported', analytics['total_issues_reported']],
            ['Average Issues/Day', analytics['average_issues_per_day']]
        ], columns=['Metric', 'Value'])
        analytics_df.to_excel(writer, sheet_name='Analytics', index=False)

        # Top issues
        if analytics['top_5_issues']:
            top_issues_df = pd.DataFrame(analytics['top_5_issues'],
                                        columns=['Issue', 'Count'])
            top_issues_df.to_excel(writer, sheet_name='Top Issues', index=False)

        # Emerging issues
        if analytics['emerging_issues']:
            emerging_df = pd.DataFrame(analytics['emerging_issues'], columns=['Emerging Issues'])
            emerging_df.to_excel(writer, sheet_name='Emerging Issues', index=False)

    logger.info(f"Reports saved to {report_filename} and {excel_filename}")

    # Generate visualizations
    logger.info("Generating trend visualizations...")
    orchestrator.generate_visualizations(start_date, end_date, output_dir)

    # Display report
    print("\n" + "="*80)
    print(f"NEGATIVE ISSUE TREND ANALYSIS REPORT")
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"App: {APP_ID}")
    print(f"Consolidation Method: {CONSOLIDATION_METHOD}")
    print(f"LLM: Llama 3.3 70B via HuggingFace")
    print("="*80)

    if not report.empty:
        print("\nTop 15 Problem Categories:")
        print("-"*40)
        print(report.head(15).to_string())
    else:
        print("No negative issues found in the date range")

    # Display analytics
    print("\n" + "="*80)
    print("ANALYTICS SUMMARY")
    print("="*80)
    for key, value in analytics.items():
        if key == 'top_5_issues':
            if value:
                print(f"\nTop 5 Issues:")
                for issue, count in value:
                    print(f"  • {issue}: {count} occurrences")
        elif key == 'emerging_issues':
            if value:
                print(f"\nEmerging Issues:")
                for issue in value:
                    print(f"  • {issue}")
        elif isinstance(value, list):
            print(f"{key}: {', '.join(str(v) for v in value[:5])}")
        else:
            print(f"{key}: {value}")

    print("\n✅ Processing complete!")
    print(f"📊 Visualizations saved to {output_dir}/plots/")
    print(f"📈 Interactive dashboard available at {output_dir}/plots/interactive_dashboard.html")

    return report

await(main())
