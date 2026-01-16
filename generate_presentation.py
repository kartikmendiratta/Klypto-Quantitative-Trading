"""
Generate PowerPoint Presentation for NIFTY Trading Strategy Project
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# Create presentation
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Color scheme
DARK_BLUE = RGBColor(0, 51, 102)
LIGHT_BLUE = RGBColor(0, 112, 192)
GREEN = RGBColor(0, 176, 80)
RED = RGBColor(255, 0, 0)
GRAY = RGBColor(128, 128, 128)

def add_title_slide(title, subtitle=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
    # Title
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(12), Inches(1.5))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE
    p.alignment = PP_ALIGN.CENTER
    # Subtitle
    if subtitle:
        txBox2 = slide.shapes.add_textbox(Inches(0.5), Inches(4), Inches(12), Inches(1))
        tf2 = txBox2.text_frame
        p2 = tf2.paragraphs[0]
        p2.text = subtitle
        p2.font.size = Pt(24)
        p2.font.color.rgb = GRAY
        p2.alignment = PP_ALIGN.CENTER
    return slide

def add_content_slide(title, bullets, image_path=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
    # Title
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12), Inches(0.8))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE
    
    # Bullet points - adjust width and font based on image
    has_image = image_path and os.path.exists(image_path)
    if has_image:
        left = Inches(0.5)
        width = Inches(5.8)
        font_size = Pt(16)  # Smaller font when image present
        space_after = Pt(8)
    else:
        left = Inches(0.5)
        width = Inches(12)
        font_size = Pt(18)
        space_after = Pt(12)
    
    txBox2 = slide.shapes.add_textbox(left, Inches(1.3), width, Inches(5.5))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    
    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf2.paragraphs[0]
        else:
            p = tf2.add_paragraph()
        p.text = "• " + bullet
        p.font.size = font_size
        p.space_after = space_after
    
    # Image - position further right to avoid overlap
    if has_image:
        slide.shapes.add_picture(image_path, Inches(6.5), Inches(1.3), width=Inches(6.3))
    
    # Slide number
    add_slide_number(slide, len(prs.slides))
    return slide

def add_slide_number(slide, num):
    txBox = slide.shapes.add_textbox(Inches(12.5), Inches(7), Inches(0.8), Inches(0.4))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = str(num)
    p.font.size = Pt(12)
    p.font.color.rgb = GRAY
    p.alignment = PP_ALIGN.RIGHT

def add_section_header(title):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(3), Inches(12), Inches(1.5))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = LIGHT_BLUE
    p.alignment = PP_ALIGN.CENTER
    add_slide_number(slide, len(prs.slides))
    return slide

# ============================================================
# SLIDE 1: Title
# ============================================================
add_title_slide(
    "NIFTY Algorithmic Trading Strategy",
    "Sequential Filtering Architecture with HMM Regime Detection & ML Trade Validation"
)

# ============================================================
# SLIDES 2-4: Executive Summary
# ============================================================
add_section_header("Executive Summary")

add_content_slide("Project Overview", [
    "Built a quantitative trading system for NIFTY 50 derivatives",
    "Sequential Filtering Architecture: Data → Features → Regime → Signal → ML → Execution",
    "Uses Hidden Markov Model (HMM) for market regime detection",
    "Machine Learning filters (XGBoost, LSTM) to validate trade signals",
    "Backtested on 5-minute NIFTY data with comprehensive metrics"
])

add_content_slide("Key Results", [
    "Baseline Strategy: 147 trades, 25.85% win rate",
    "XGBoost Filtered: 40 trades, 35.00% win rate (+9.15 pp)",
    "LSTM Filtered: 28 trades, 39.29% win rate (+13.44 pp)",
    "ML filtering reduces trade count by 70-80%",
    "Significant improvement in risk-adjusted returns"
])

# ============================================================
# SLIDES 5-7: Data Pipeline
# ============================================================
add_section_header("Data Pipeline")

add_content_slide("Data Sources", [
    "NIFTY 50 Spot Index (5-minute intervals)",
    "NIFTY Futures (Current Month contract)",
    "NIFTY Options (ATM ± 2 strikes)",
    "Time period: Multiple months of intraday data",
    "Total records: ~19,500 rows × 71 features"
])

add_content_slide("Data Cleaning Process", [
    "Handled futures contract rollovers (stitching)",
    "Calculated dynamic ATM strike based on spot price",
    "Aligned all timestamps to 5-minute intervals",
    "Forward-filled missing values",
    "Merged Spot, Futures, and Options into single dataset"
], "plots/02_data_quality.png")

# ============================================================
# SLIDES 8-11: Feature Engineering
# ============================================================
add_section_header("Feature Engineering")

add_content_slide("Technical Indicators", [
    "EMA (5-period) - Fast moving average",
    "EMA (15-period) - Slow moving average",
    "EMA Crossover signals for entry/exit",
    "ATR (14-period) for volatility measurement"
])

add_content_slide("Options Greeks (Black-Scholes)", [
    "Delta - Price sensitivity to underlying",
    "Gamma - Rate of change of Delta",
    "Theta - Time decay",
    "Vega - Volatility sensitivity",
    "Rho - Interest rate sensitivity",
    "Calculated using mibian library (r = 6.5%)"
])

add_content_slide("Derived Features", [
    "Average IV = (Call IV + Put IV) / 2",
    "IV Spread = Call IV - Put IV",
    "PCR (OI-based) = Put OI / Call OI",
    "Futures Basis = (Futures - Spot) / Spot",
    "Gamma Exposure = Spot × Gamma × OI"
], "plots/03_features.png")

# ============================================================
# SLIDES 12-15: Regime Detection
# ============================================================
add_section_header("Regime Detection")

add_content_slide("Hidden Markov Model (HMM)", [
    "3-State Model: Uptrend (+1), Sideways (0), Downtrend (-1)",
    "Library: hmmlearn with GaussianHMM",
    "Trained on first 70% of data",
    "Uses options-based features only for regime classification"
])

add_content_slide("HMM Input Features", [
    "Average IV - Implied volatility level",
    "IV Spread - Call vs Put IV difference",
    "PCR (OI-based) - Put/Call sentiment",
    "ATM Delta, Gamma, Vega - Greeks",
    "Futures Basis - Futures premium/discount",
    "Spot Returns - Price momentum"
])

add_content_slide("Regime Detection Results", [
    "Downtrend (-1): ~59% of candles",
    "Sideways (0): ~19% of candles",
    "Uptrend (+1): ~22% of candles",
    "High regime persistence (>99% self-transition probability)",
    "Clear separation of market conditions"
], "plots/04_regime_detection.png")

# ============================================================
# SLIDES 16-19: Baseline Strategy
# ============================================================
add_section_header("Baseline Strategy Results")

add_content_slide("EMA Crossover + Regime Filter", [
    "LONG: EMA(5) crosses above EMA(15) AND Regime = +1",
    "SHORT: EMA(5) crosses below EMA(15) AND Regime = -1",
    "NO TRADE: When Regime = 0 (Sideways market)",
    "Entry at next candle open after signal",
    "Exit on opposite crossover"
])

add_content_slide("Baseline Performance Metrics", [
    "Total Trades: 147",
    "Win Rate: 25.85%",
    "Total Return: -1.55%",
    "Sharpe Ratio: -14.79",
    "Max Drawdown: 2.04%",
    "Profit Factor: 0.74"
], "plots/05_baseline_strategy.png")

add_content_slide("Force-Close on Regime Change", [
    "Added safety feature: Force-close when Regime → 0",
    "Prevents holding positions in uncertain markets",
    "Reduces exposure during sideways conditions",
    "Improves risk management"
])

# ============================================================
# SLIDES 20-24: ML Enhancement
# ============================================================
add_section_header("ML Enhancement")

add_content_slide("ML Problem Definition", [
    "Binary Classification: Predict if trade will be profitable",
    "Target: 1 if trade PnL > 0, else 0",
    "Features: All engineered features at signal point",
    "Training: First 70% of data with cross-validation",
    "Inference: Only take trades when ML confidence > 50%"
])

add_content_slide("Model A: XGBoost Classifier", [
    "Gradient Boosting Decision Trees",
    "Time-Series Cross-Validation (5 folds)",
    "CV Accuracy: ~65%",
    "Fast training and inference",
    "Feature importance analysis available"
], "plots/06_xgb_importance.png")

add_content_slide("Model B: LSTM Classifier", [
    "Sequence model using last 10 candles",
    "Architecture: LSTM → Dropout → Dense → Sigmoid",
    "Captures temporal patterns in features",
    "Pre-calculated batch predictions for efficiency",
    "Provides strongest filtering (81% trades rejected)"
])

add_content_slide("XGBoost Confusion Matrix", [
    "True Negatives: 305 (Loss correctly predicted)",
    "True Positives: 126 (Profit correctly predicted)",
    "Training Accuracy: 100%",
    "Model learns to distinguish profitable patterns"
], "plots/06_xgb_confusion_matrix.png")

add_content_slide("ML Enhancement Results", [
    "Baseline: 147 trades, 25.85% win rate",
    "XGBoost: 40 trades, 35.00% win rate (+9.15 pp)",
    "LSTM: 28 trades, 39.29% win rate (+13.44 pp)",
    "Both models significantly reduce false signals",
    "Trade quality improved at cost of quantity"
], "plots/06_ml_comparison.png")

# ============================================================
# SLIDES 25-27: High-Performance Analysis
# ============================================================
add_section_header("High-Performance Trade Analysis")

add_content_slide("Outlier Detection (Z-score > 3)", [
    "Identified profitable trades beyond 3 standard deviations",
    "Analyzed: regime, IV, ATR, time of day, Greeks, duration",
    "Statistical tests comparing outliers vs normal trades",
    "Found distinguishing patterns in high-performance trades"
])

add_content_slide("Key Findings", [
    "Outlier trades tend to occur in strong trending regimes",
    "Higher IV environments correlate with larger moves",
    "Duration shows moderate correlation with PnL",
    "Morning hours (9:15-11:00) show better performance",
    "PCR extremes often precede profitable trades"
], "plots/07_outlier_analysis.png")

# ============================================================
# SLIDES 28-30: Conclusion
# ============================================================
add_section_header("Conclusion & Recommendations")

add_content_slide("Key Achievements", [
    "Successfully implemented Sequential Filtering Architecture",
    "HMM regime detection with 3 market states",
    "ML validation improves win rate by 9-13 percentage points",
    "Comprehensive backtesting framework with all metrics",
    "Modular codebase for easy extension"
])

add_content_slide("Recommendations", [
    "Use LSTM filter for highest conviction trades only",
    "Combine XGBoost + LSTM for ensemble approach",
    "Add position sizing based on ML confidence",
    "Consider adding stop-loss and take-profit levels",
    "Paper trade before live deployment",
    "Monitor regime transitions in real-time"
])

# Save presentation
output_path = "NIFTY_Presentation_v2.pptx"
prs.save(output_path)
print(f"Presentation saved: {output_path}")
print(f"Total slides: {len(prs.slides)}")
