# AI-Powered Review Analysis System

An advanced system for extracting, categorizing, and analyzing negative issues from Google Play Store reviews using local Llama models and machine learning techniques.

## Features

- **Automated Review Scraping**: Fetches reviews from Google Play Store for specified date ranges
- **AI-Powered Issue Extraction**: Uses Llama 3.2 3B Instruct model to identify negative issues and complaints
- **Smart Topic Consolidation**: Three consolidation methods (LLM-based, clustering-based, or hybrid)
- **Trend Analysis**: Tracks issue evolution over time with comprehensive analytics
- **Interactive Visualizations**: Generates static plots and interactive HTML dashboards
- **Comprehensive Reporting**: Exports data to CSV and Excel with multiple analysis sheets

## System Requirements

### Hardware
- **RAM**: Minimum 8GB, recommended 16GB+ for local LLM inference
- **GPU**: Optional but recommended - NVIDIA GPU with 6GB+ VRAM for faster processing
- **Storage**: 10GB+ free space for model downloads and data

### Software
- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd review-analysis-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Hugging Face Token
```bash
# Get token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_huggingface_token_here"
```

## Quick Start

### Basic Usage
```python
import asyncio
from datetime import datetime

# Initialize the system
APP_ID = "com.application.zomato"  # Replace with your app ID
orchestrator = ReviewAnalysisOrchestrator(APP_ID, "hybrid")

# Process reviews for a date range
start_date = datetime(2025, 9, 21)
end_date = datetime(2025, 9, 28)

# Run the analysis
await orchestrator.run_batch_processing(start_date, end_date)

# Generate reports
report = orchestrator.generate_report(start_date, end_date)
orchestrator.generate_visualizations(start_date, end_date)
```

### Environment Variables
```bash
export START_DATE="2025-09-21"
export END_DATE="2025-09-28"
export HF_TOKEN="your_token"
```

## Architecture

### Core Components

1. **ReviewScraper**: Fetches reviews from Google Play Store
2. **TopicExtractor**: AI agent using Llama 3.2 3B for issue extraction
3. **Topic Consolidators**: 
   - LLM-based consolidation
   - Clustering-based consolidation
   - Hybrid approach (clustering + LLM refinement)
4. **TrendAnalyzer**: Analyzes patterns and generates visualizations
5. **ReviewAnalysisOrchestrator**: Main coordinator

### Data Flow

```
Reviews → AI Extraction → Topic Consolidation → Trend Analysis → Reports & Visualizations
```

## Configuration Options

### Consolidation Methods
- `"llm"`: Pure LLM-based consolidation using Llama 3.2
- `"clustering"`: Embedding-based clustering with semantic similarity
- `"hybrid"`: Two-step process (clustering then LLM refinement)

### Model Configuration
```python
# GPU with quantization (recommended)
device = "cuda"  # Automatic 4-bit quantization

# CPU (slower but works without GPU)
device = "cpu"   # Full precision on CPU
```

## Output Files

### Reports
- `trend_report_YYYYMMDD.csv`: Daily issue trends
- `trend_report_YYYYMMDD.xlsx`: Multi-sheet Excel with analytics

### Visualizations
- `plots/top_issues_bar.png`: Bar chart of most frequent issues
- `plots/trend_lines.png`: Time series trends for top issues
- `plots/heatmap.png`: Issue frequency heatmap
- `plots/daily_volume.png`: Daily issue volume over time
- `plots/dashboard.png`: Static dashboard overview
- `plots/interactive_dashboard.html`: Interactive Plotly dashboard

## Usage Examples

### 1. Basic Analysis
```python
# Analyze Zomato app reviews
APP_ID = "com.application.zomato"
orchestrator = ReviewAnalysisOrchestrator(APP_ID, "hybrid")

start_date = datetime(2025, 9, 20)
end_date = datetime(2025, 9, 25)

await orchestrator.run_batch_processing(start_date, end_date)
report = orchestrator.generate_report(start_date, end_date)
print(report.head(10))
```

### 2. Custom Date Range with Environment Variables
```bash
export START_DATE="2025-08-01"
export END_DATE="2025-08-31" 
export HF_TOKEN="your_token"
python main.py
```

### 3. Analytics Summary
```python
analytics = orchestrator.get_analytics(start_date, end_date)
print(f"Total issues: {analytics['total_issues_reported']}")
print(f"Top issue: {analytics['top_5_issues'][0]}")
```

## Performance Optimization

### GPU Setup
```python
# Ensure CUDA is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### Memory Management
- Uses 4-bit quantization for GPU inference
- Automatic memory cleanup after processing
- Batch processing to handle large datasets

### Processing Tips
- Start with smaller date ranges (7-14 days) for testing
- Use `"clustering"` method for faster processing on CPU
- Use `"llm"` or `"hybrid"` for better accuracy with GPU

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size or use CPU
   device = "cpu"
   ```

2. **Slow Processing**
   ```python
   # Use clustering method for speed
   consolidation_method = "clustering"
   ```

3. **No Reviews Found**
   - Check app ID is correct
   - Verify date range has reviews
   - App might not be available in specified country

4. **Model Download Issues**
   ```bash
   # Set HF_HOME for model cache
   export HF_HOME="/path/to/cache"
   ```

### Model Access
- Requires Hugging Face account and token
- First run downloads ~6GB model files
- Models cached locally after first download

## Data Privacy

- Reviews are processed locally on your machine
- No data sent to external APIs except for model downloads
- All processing happens offline after initial setup

## Limitations

- Focuses only on negative reviews (rating ≤ 3)
- English language reviews only
- Requires significant computational resources for local LLM
- Processing speed depends on hardware capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request

## License

[Specify your license here]

## Acknowledgments

- Hugging Face Transformers for model infrastructure
- Google Play Scraper for review data access
- Meta AI for Llama model architecture

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review system requirements
3. Open an issue with detailed error logs

---

**Note**: This system is designed for research and business intelligence purposes. Ensure compliance with Google Play Store terms of service and applicable data protection regulations.
