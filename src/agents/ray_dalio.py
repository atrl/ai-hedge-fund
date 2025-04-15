"""Ray Dalio agent implementation."""
import json
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate

from tools.api import get_financial_metrics, get_market_cap, search_line_items, get_economic_indicators
from utils.llm import call_llm
from utils.progress import AgentProgress as Progress
from utils.models import RayDalioSignal


def ray_dalio_agent(
    tickers: List[str],
    end_date: str,
    model_name: str,
    model_provider: str,
    progress: Progress,
) -> Dict[str, Any]:
    """Ray Dalio agent implementation."""
    results = {}
    
    for ticker in tickers:
        progress.update_status("ray_dalio_agent", ticker, "Fetching financial metrics")
        # Fetch required data
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5)
        
        progress.update_status("ray_dalio_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "net_income",
                "total_assets",
                "total_liabilities",
                "total_debt",
                "cash_and_equivalents",  # Changed from cash_and_cash_equivalents
                "operating_cashflow",    # Changed from operating_cash_flow
                "free_cashflow",
                "revenue",
                "ebitda",
                "interest_expense",
            ],
            end_date,
            period="ttm",
            limit=5,
        )
        
        progress.update_status("ray_dalio_agent", ticker, "Analyzing economic environment")
        economic_indicators = get_economic_indicators(end_date)
        
        # Determine current economic environment
        environment_type = determine_economic_environment(economic_indicators)
        
        # Analyze company fundamentals
        progress.update_status("ray_dalio_agent", ticker, "Analyzing fundamentals")
        latest_metrics = metrics[0] if metrics else None
        latest_financials = financial_line_items[0] if financial_line_items else None
        
        # Prepare analysis data
        analysis_data = {
            "economic_environment": {
                "type": environment_type,
                "indicators": economic_indicators.get("indicators", {}) if economic_indicators else {},
            },
            "fundamentals": {
                "debt_to_equity": getattr(latest_metrics, "debt_to_equity", None),
                "current_ratio": getattr(latest_metrics, "current_ratio", None),
                "return_on_equity": getattr(latest_metrics, "return_on_equity", None),
                "price_to_earnings": getattr(latest_metrics, "price_to_earnings", None),
                "dividend_yield": getattr(latest_metrics, "dividend_yield", None),
                "market_cap": get_market_cap(ticker, end_date),
            },
            "financial_health": analyze_financial_health(latest_metrics, latest_financials),
            "cash_flow_stability": analyze_cash_flow_stability(latest_financials),
            "environment_fit": analyze_environment_fit(environment_type, latest_metrics, latest_financials),
        }
        
        # Generate output
        progress.update_status("ray_dalio_agent", ticker, "Generating investment signal")
        signal = generate_dalio_output(ticker, analysis_data, model_name, model_provider)
        
        results[ticker] = signal.model_dump()
    
    return results


def determine_economic_environment(economic_indicators: Optional[Dict[str, Any]]) -> str:
    """Determine the current economic environment based on indicators."""
    if not economic_indicators or "indicators" not in economic_indicators:
        return "unknown"
    
    indicators = economic_indicators["indicators"]
    gdp_growth = indicators.get("gdp_growth", 0)
    inflation_rate = indicators.get("inflation_rate", 0)
    
    # Simplified environment determination based on growth and inflation
    if gdp_growth >= 2.0:
        if inflation_rate >= 2.5:
            return "rising growth, rising inflation"
        else:
            return "rising growth, falling inflation"
    else:
        if inflation_rate >= 2.5:
            return "falling growth, rising inflation"
        else:
            return "falling growth, falling inflation"


def analyze_financial_health(latest_metrics: Any, latest_financials: Any) -> Dict[str, Any]:
    """Analyze the financial health of the company."""
    score = 0
    max_score = 3
    reasoning = []
    
    # Check debt to equity ratio
    if latest_metrics and hasattr(latest_metrics, "debt_to_equity") and latest_metrics.debt_to_equity is not None:
        if latest_metrics.debt_to_equity < 0.5:
            score += 1
            reasoning.append(f"Low debt-to-equity ratio ({latest_metrics.debt_to_equity:.2f}) indicates strong balance sheet")
        elif latest_metrics.debt_to_equity < 1.0:
            score += 0.5
            reasoning.append(f"Moderate debt-to-equity ratio ({latest_metrics.debt_to_equity:.2f}) indicates acceptable balance sheet")
        else:
            reasoning.append(f"High debt-to-equity ratio ({latest_metrics.debt_to_equity:.2f}) indicates potential financial risk")
    else:
        reasoning.append("Debt-to-equity data not available")
    
    # Check interest coverage ratio
    if latest_financials and hasattr(latest_financials, "ebitda") and latest_financials.ebitda is not None and hasattr(latest_financials, "interest_expense") and latest_financials.interest_expense is not None and latest_financials.interest_expense != 0:
        interest_coverage = latest_financials.ebitda / latest_financials.interest_expense
        if interest_coverage > 5:
            score += 1
            reasoning.append(f"Strong interest coverage ratio ({interest_coverage:.2f}) indicates ability to service debt")
        elif interest_coverage > 2:
            score += 0.5
            reasoning.append(f"Adequate interest coverage ratio ({interest_coverage:.2f}) indicates reasonable ability to service debt")
        else:
            reasoning.append(f"Low interest coverage ratio ({interest_coverage:.2f}) indicates potential debt servicing issues")
    else:
        reasoning.append("Interest coverage data not available")
    
    # Check current ratio
    if latest_metrics and hasattr(latest_metrics, "current_ratio") and latest_metrics.current_ratio is not None:
        if latest_metrics.current_ratio > 1.5:
            score += 1
            reasoning.append(f"Strong current ratio ({latest_metrics.current_ratio:.2f}) indicates good short-term liquidity")
        elif latest_metrics.current_ratio > 1.0:
            score += 0.5
            reasoning.append(f"Adequate current ratio ({latest_metrics.current_ratio:.2f}) indicates acceptable short-term liquidity")
        else:
            reasoning.append(f"Low current ratio ({latest_metrics.current_ratio:.2f}) indicates potential short-term liquidity issues")
    else:
        reasoning.append("Current ratio data not available")
    
    return {
        "score": score,
        "max_score": max_score,
        "details": "; ".join(reasoning),
    }


def analyze_cash_flow_stability(latest_financials: Any) -> Dict[str, Any]:
    """Analyze the stability of the company's cash flows."""
    score = 0
    max_score = 2
    reasoning = []
    
    # Check free cash flow to revenue ratio
    if latest_financials and hasattr(latest_financials, "free_cashflow") and latest_financials.free_cashflow is not None and hasattr(latest_financials, "revenue") and latest_financials.revenue is not None and latest_financials.revenue > 0:
        fcf_to_revenue = latest_financials.free_cashflow / latest_financials.revenue
        if fcf_to_revenue > 0.15:
            score += 1
            reasoning.append(f"Strong free cash flow to revenue ratio ({fcf_to_revenue:.1%}) indicates excellent cash generation")
        elif fcf_to_revenue > 0.08:
            score += 0.5
            reasoning.append(f"Good free cash flow to revenue ratio ({fcf_to_revenue:.1%}) indicates solid cash generation")
        else:
            reasoning.append(f"Low free cash flow to revenue ratio ({fcf_to_revenue:.1%}) indicates potential cash flow concerns")
    else:
        reasoning.append("Free cash flow to revenue data not available")
    
    # Check operating cash flow to net income ratio
    if latest_financials and hasattr(latest_financials, "operating_cashflow") and latest_financials.operating_cashflow is not None and hasattr(latest_financials, "net_income") and latest_financials.net_income is not None and latest_financials.net_income > 0:
        ocf_to_ni = latest_financials.operating_cashflow / latest_financials.net_income
        if ocf_to_ni > 1.2:
            score += 1
            reasoning.append(f"Strong operating cash flow to net income ratio ({ocf_to_ni:.2f}) indicates high earnings quality")
        elif ocf_to_ni > 0.9:
            score += 0.5
            reasoning.append(f"Adequate operating cash flow to net income ratio ({ocf_to_ni:.2f}) indicates reasonable earnings quality")
        else:
            reasoning.append(f"Low operating cash flow to net income ratio ({ocf_to_ni:.2f}) indicates potential earnings quality issues")
    else:
        reasoning.append("Operating cash flow to net income data not available")
    
    return {
        "score": score,
        "max_score": max_score,
        "details": "; ".join(reasoning),
    }


def analyze_environment_fit(environment_type: str, latest_metrics: Any, latest_financials: Any) -> Dict[str, Any]:
    """Analyze how well the company fits the current economic environment."""
    score = 0
    max_score = 1
    reasoning = []
    
    if environment_type == "rising growth, rising inflation":
        if latest_metrics and hasattr(latest_metrics, "return_on_equity") and latest_metrics.return_on_equity is not None:
            if latest_metrics.return_on_equity > 0.15:
                score += 1
                reasoning.append(f"High ROE ({latest_metrics.return_on_equity:.1%}) beneficial in growth environment")
            else:
                reasoning.append(f"Lower ROE ({latest_metrics.return_on_equity:.1%}) less optimal in growth environment")
        else:
            reasoning.append("ROE data not available for environment fit analysis")
    
    elif environment_type == "rising growth, falling inflation":
        if latest_metrics and hasattr(latest_metrics, "price_to_earnings") and latest_metrics.price_to_earnings is not None:
            if latest_metrics.price_to_earnings < 20:
                score += 1
                reasoning.append(f"Reasonable P/E ({latest_metrics.price_to_earnings:.1f}) attractive in growth with low inflation")
            else:
                reasoning.append(f"Higher P/E ({latest_metrics.price_to_earnings:.1f}) less attractive even in favorable environment")
        else:
            reasoning.append("P/E data not available for environment fit analysis")
    
    elif environment_type == "falling growth, falling inflation":
        if latest_metrics and hasattr(latest_metrics, "dividend_yield") and latest_metrics.dividend_yield is not None:
            if latest_metrics.dividend_yield > 0.03:
                score += 1
                reasoning.append(f"Higher dividend yield ({latest_metrics.dividend_yield:.1%}) valuable in slowing economy")
            else:
                reasoning.append(f"Lower dividend yield ({latest_metrics.dividend_yield:.1%}) less attractive in slowing economy")
        else:
            reasoning.append("Dividend yield data not available for environment fit analysis")
    
    elif environment_type == "falling growth, rising inflation":
        if latest_financials and hasattr(latest_financials, "cash_and_equivalents") and latest_financials.cash_and_equivalents is not None and hasattr(latest_financials, "total_assets") and latest_financials.total_assets is not None and latest_financials.total_assets > 0:
            cash_ratio = latest_financials.cash_and_equivalents / latest_financials.total_assets
            if cash_ratio > 0.2:
                score += 1
                reasoning.append(f"Strong cash position ({cash_ratio:.1%} of assets) provides stagflation protection")
            else:
                reasoning.append(f"Limited cash reserves ({cash_ratio:.1%} of assets) may be vulnerable in stagflation")
        else:
            reasoning.append("Cash position data not available for environment fit analysis")
    
    return {
        "score": score,
        "max_score": 1,
        "details": "; ".join(reasoning),
    }


def generate_dalio_output(
    ticker: str,
    analysis_data: dict,
    model_name: str,
    model_provider: str,
) -> RayDalioSignal:
    """Get investment decision from LLM with Ray Dalio's principles"""
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Ray Dalio AI agent. Decide on investment signals based on Ray Dalio's principles:
                - All Weather Strategy: Balance risk across different economic environments
                - Economic Machine: Understand how the economy works as a machine with key drivers
                - Risk Parity: Focus on risk allocation rather than capital allocation
                - Diversification: Seek uncorrelated return streams
                - Debt Cycles: Recognize long-term and short-term debt cycles
                - Cash Flow: Focus on sustainable and predictable cash flows
                - Balance Sheet Strength: Prefer companies with strong balance sheets
                - Economic Environment Fit: Assess how well assets perform in different economic regimes
                When providing your reasoning, be thorough and specific by:
                1. Explaining how the current economic environment impacts this investment
                2. Highlighting the company's strengths and vulnerabilities in this environment
                3. Analyzing balance sheet strength, cash flow stability, and debt levels
                4. Providing quantitative evidence where relevant
                5. Concluding with a Dalio-style assessment of the investment opportunity
                6. Using Ray Dalio's voice and conversational style in your explanation
                For example, if bullish: "In this economic environment of [specific condition], this company's [specific strength] positions it well because..."
                For example, if bearish: "The combination of [economic factor] and the company's [specific weakness] creates significant risk because..."
                Follow these guidelines strictly.
                """,
            ),
            (
                "human",
                """Based on the following data, create the investment signal as Ray Dalio would:
                Analysis Data for {ticker}:
                {analysis_data}
                Return the trading signal in the following JSON format exactly:
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float between 0 and 100,
                  "reasoning": "string"
                }}
                """,
            ),
        ]
    )
    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})
    
    # Default fallback signal in case parsing fails
    def create_default_ray_dalio_signal():
        return RayDalioSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")
    
    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=RayDalioSignal,
        agent_name="ray_dalio_agent",
        default_factory=create_default_ray_dalio_signal,
    )
