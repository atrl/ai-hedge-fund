from graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from tools.api import get_financial_metrics, get_market_cap, search_line_items, get_economic_indicators
from utils.llm import call_llm
from utils.progress import progress

class RayDalioSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

def ray_dalio_agent(state: AgentState):
    """Analyzes stocks using Ray Dalio's All Weather principles and LLM reasoning."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    
    # Collect all analysis for LLM reasoning
    analysis_data = {}
    dalio_analysis = {}
    
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
                "cash_and_cash_equivalents",
                "operating_cash_flow",
                "revenue",
            ],
            end_date,
        )
        
        progress.update_status("ray_dalio_agent", ticker, "Getting market cap")
        # Get current market cap
        market_cap = get_market_cap(ticker, end_date)
        
        progress.update_status("ray_dalio_agent", ticker, "Getting economic indicators")
        # Get economic indicators
        economic_indicators = get_economic_indicators(end_date)
        
        progress.update_status("ray_dalio_agent", ticker, "Analyzing economic environment")
        # Analyze economic environment
        environment_analysis = analyze_economic_environment(economic_indicators)
        
        progress.update_status("ray_dalio_agent", ticker, "Analyzing balance sheet strength")
        # Analyze balance sheet strength
        balance_sheet_analysis = analyze_balance_sheet(financial_line_items)
        
        progress.update_status("ray_dalio_agent", ticker, "Analyzing cash flow stability")
        # Analyze cash flow stability
        cash_flow_analysis = analyze_cash_flow_stability(financial_line_items)
        
        progress.update_status("ray_dalio_agent", ticker, "Analyzing debt levels")
        # Analyze debt levels
        debt_analysis = analyze_debt_levels(financial_line_items)
        
        progress.update_status("ray_dalio_agent", ticker, "Analyzing environment fit")
        # Analyze how well the stock fits the current economic environment
        environment_fit_analysis = analyze_environment_fit(environment_analysis, metrics, financial_line_items)
        
        # Calculate total score
        total_score = (
            balance_sheet_analysis["score"] + 
            cash_flow_analysis["score"] + 
            debt_analysis["score"] + 
            environment_fit_analysis["score"]
        )
        
        max_possible_score = (
            balance_sheet_analysis["max_score"] + 
            cash_flow_analysis["max_score"] + 
            debt_analysis["max_score"] + 
            environment_fit_analysis["max_score"]
        )
        
        # Generate trading signal based on total score
        if total_score >= 0.75 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"
        
        # Combine all analysis results
        analysis_data[ticker] = {
            "economic_environment": environment_analysis,
            "balance_sheet_strength": balance_sheet_analysis,
            "cash_flow_stability": cash_flow_analysis,
            "debt_levels": debt_analysis,
            "environment_fit": environment_fit_analysis,
            "total_score": total_score,
            "max_possible_score": max_possible_score,
            "score_percentage": total_score / max_possible_score if max_possible_score > 0 else 0,
            "preliminary_signal": signal,
            "market_cap": market_cap,
        }
        
        # Get LLM reasoning
        metadata = state["metadata"]
        model_name = metadata.get("model_name", "gpt-4o")
        model_provider = metadata.get("model_provider", "OpenAI")
        
        progress.update_status("ray_dalio_agent", ticker, "Generating investment signal")
        dalio_signal = generate_dalio_output(ticker, analysis_data[ticker], model_name, model_provider)
        dalio_analysis[ticker] = dalio_signal
    
    # Add to analyst signals
    state["data"]["analyst_signals"]["ray_dalio"] = dalio_analysis
    
    # Show reasoning if requested
    if state["metadata"].get("show_reasoning", False):
        show_agent_reasoning(state, "ray_dalio", dalio_analysis)
    
    return state

def analyze_economic_environment(economic_indicators: dict) -> dict:
    """Analyzes the current economic environment based on indicators."""
    # Default values
    growth_trend = "stable"
    inflation_trend = "stable"
    environment_type = "stable growth, stable inflation"
    
    # Extract indicators
    indicators = economic_indicators.get("indicators", {})
    gdp_growth = indicators.get("gdp_growth", 0)
    inflation_rate = indicators.get("inflation_rate", 0)
    interest_rate = indicators.get("interest_rate", 0)
    unemployment_rate = indicators.get("unemployment_rate", 0)
    
    # Determine growth trend
    if gdp_growth > 3:
        growth_trend = "rising"
    elif gdp_growth < 1:
        growth_trend = "falling"
    
    # Determine inflation trend
    if inflation_rate > 3:
        inflation_trend = "rising"
    elif inflation_rate < 1:
        inflation_trend = "falling"
    
    # Determine environment type
    if growth_trend == "rising" and inflation_trend == "rising":
        environment_type = "rising growth, rising inflation"
    elif growth_trend == "rising" and inflation_trend == "falling":
        environment_type = "rising growth, falling inflation"
    elif growth_trend == "falling" and inflation_trend == "rising":
        environment_type = "falling growth, rising inflation"
    elif growth_trend == "falling" and inflation_trend == "falling":
        environment_type = "falling growth, falling inflation"
    
    # Determine risk parity allocation based on environment
    risk_parity = {
        "equities": 0.25,
        "bonds": 0.25,
        "commodities": 0.25,
        "cash": 0.25
    }
    
    if environment_type == "rising growth, rising inflation":
        risk_parity = {
            "equities": 0.30,
            "bonds": 0.15,
            "commodities": 0.40,
            "cash": 0.15
        }
    elif environment_type == "rising growth, falling inflation":
        risk_parity = {
            "equities": 0.40,
            "bonds": 0.30,
            "commodities": 0.15,
            "cash": 0.15
        }
    elif environment_type == "falling growth, rising inflation":
        risk_parity = {
            "equities": 0.15,
            "bonds": 0.15,
            "commodities": 0.40,
            "cash": 0.30
        }
    elif environment_type == "falling growth, falling inflation":
        risk_parity = {
            "equities": 0.15,
            "bonds": 0.40,
            "commodities": 0.15,
            "cash": 0.30
        }
    
    return {
        "growth_trend": growth_trend,
        "inflation_trend": inflation_trend,
        "environment_type": environment_type,
        "risk_parity": risk_parity,
        "indicators": indicators
    }

def analyze_balance_sheet(financial_line_items: list) -> dict:
    """Analyzes balance sheet strength based on financial line items."""
    score = 0
    reasoning = []
    
    if not financial_line_items:
        return {
            "score": 0,
            "max_score": 3,
            "details": "No financial data available for balance sheet analysis",
        }
    
    latest = financial_line_items[0]
    
    # Check cash position
    if hasattr(latest, "cash_and_cash_equivalents") and hasattr(latest, "total_assets") and latest.cash_and_cash_equivalents is not None and latest.total_assets is not None and latest.total_assets > 0:
        cash_to_assets_ratio = latest.cash_and_cash_equivalents / latest.total_assets
        if cash_to_assets_ratio > 0.15:
            score += 1
            reasoning.append(f"Strong cash position ({cash_to_assets_ratio:.1%} of assets)")
        elif cash_to_assets_ratio > 0.08:
            score += 0.5
            reasoning.append(f"Adequate cash position ({cash_to_assets_ratio:.1%} of assets)")
        else:
            reasoning.append(f"Limited cash reserves ({cash_to_assets_ratio:.1%} of assets)")
    else:
        reasoning.append("Cash position data not available")
    
    # Check for balance sheet growth
    if len(financial_line_items) >= 3 and all(hasattr(item, "total_assets") for item in financial_line_items[:3]):
        assets_values = [item.total_assets for item in financial_line_items[:3] if item.total_assets is not None]
        if len(assets_values) >= 2 and assets_values[0] > 0 and assets_values[-1] > 0:
            assets_growth = (assets_values[0] - assets_values[-1]) / assets_values[-1]
            if assets_growth > 0.1:
                score += 1
                reasoning.append(f"Strong balance sheet growth of {assets_growth:.1%} over recent periods")
            elif assets_growth > 0:
                score += 0.5
                reasoning.append(f"Modest balance sheet growth of {assets_growth:.1%}")
            else:
                reasoning.append(f"Declining balance sheet with {assets_growth:.1%} change")
        else:
            reasoning.append("Insufficient data for balance sheet growth analysis")
    else:
        reasoning.append("Insufficient data for balance sheet growth analysis")
    
    # Check asset-to-liability ratio
    if hasattr(latest, "total_assets") and hasattr(latest, "total_liabilities") and latest.total_assets is not None and latest.total_liabilities is not None and latest.total_liabilities > 0:
        asset_to_liability_ratio = latest.total_assets / latest.total_liabilities
        if asset_to_liability_ratio > 2:
            score += 1
            reasoning.append(f"Strong asset-to-liability ratio of {asset_to_liability_ratio:.2f}")
        elif asset_to_liability_ratio > 1.5:
            score += 0.5
            reasoning.append(f"Adequate asset-to-liability ratio of {asset_to_liability_ratio:.2f}")
        else:
            reasoning.append(f"Concerning asset-to-liability ratio of {asset_to_liability_ratio:.2f}")
    else:
        reasoning.append("Asset-to-liability ratio data not available")
    
    return {
        "score": score,
        "max_score": 3,
        "details": "; ".join(reasoning),
    }

def analyze_cash_flow_stability(financial_line_items: list) -> dict:
    """Analyzes cash flow stability based on financial line items."""
    score = 0
    reasoning = []
    
    if not financial_line_items:
        return {
            "score": 0,
            "max_score": 3,
            "details": "No financial data available for cash flow analysis",
        }
    
    latest = financial_line_items[0]
    
    # Check operating cash flow to net income ratio
    if hasattr(latest, "operating_cash_flow") and hasattr(latest, "net_income") and latest.operating_cash_flow is not None and latest.net_income is not None and latest.net_income != 0:
        ocf_to_ni_ratio = latest.operating_cash_flow / latest.net_income if latest.net_income != 0 else 0
        if ocf_to_ni_ratio > 1.2:
            score += 1
            reasoning.append(f"Strong cash flow to earnings quality with ratio of {ocf_to_ni_ratio:.2f}")
        elif ocf_to_ni_ratio > 0.8:
            score += 0.5
            reasoning.append(f"Adequate cash flow to earnings quality with ratio of {ocf_to_ni_ratio:.2f}")
        elif ocf_to_ni_ratio > 0:
            reasoning.append(f"Poor cash flow to earnings quality with ratio of {ocf_to_ni_ratio:.2f}")
        else:
            reasoning.append("Negative cash flow to earnings ratio")
    else:
        reasoning.append("Cash flow to earnings ratio data not available")
    
    # Check cash flow consistency
    if len(financial_line_items) >= 3:
        ocf_values = [item.operating_cash_flow for item in financial_line_items[:3] if hasattr(item, "operating_cash_flow") and item.operating_cash_flow is not None]
        if len(ocf_values) >= 3:
            if all(ocf > 0 for ocf in ocf_values):
                score += 1
                reasoning.append("Consistently positive operating cash flow")
            elif sum(1 for ocf in ocf_values if ocf > 0) >= 2:
                score += 0.5
                reasoning.append("Mostly positive operating cash flow")
            else:
                reasoning.append("Inconsistent or negative operating cash flow")
        else:
            reasoning.append("Insufficient data for cash flow consistency analysis")
    else:
        reasoning.append("Insufficient data for cash flow consistency analysis")
    
    # Check cash flow growth
    if len(financial_line_items) >= 2:
        ocf_values = [item.operating_cash_flow for item in financial_line_items[:2] if hasattr(item, "operating_cash_flow") and item.operating_cash_flow is not None]
        if len(ocf_values) >= 2 and ocf_values[1] != 0:
            ocf_growth = (ocf_values[0] - ocf_values[1]) / abs(ocf_values[1])
            if ocf_growth > 0.1:
                score += 1
                reasoning.append(f"Strong cash flow growth of {ocf_growth:.1%}")
            elif ocf_growth > 0:
                score += 0.5
                reasoning.append(f"Modest cash flow growth of {ocf_growth:.1%}")
            else:
                reasoning.append(f"Declining cash flow with {ocf_growth:.1%} change")
        else:
            reasoning.append("Cash flow growth data not available")
    else:
        reasoning.append("Cash flow growth data not available")
    
    return {
        "score": score,
        "max_score": 3,
        "details": "; ".join(reasoning),
    }

def analyze_debt_levels(financial_line_items: list) -> dict:
    """Analyzes debt levels based on financial line items."""
    score = 0
    reasoning = []
    
    if not financial_line_items:
        return {
            "score": 0,
            "max_score": 2,
            "details": "No financial data available for debt analysis",
        }
    
    latest = financial_line_items[0]
    
    # Check debt to asset ratio
    if hasattr(latest, "total_debt") and hasattr(latest, "total_assets") and latest.total_debt is not None and latest.total_assets is not None and latest.total_assets > 0:
        debt_to_asset_ratio = latest.total_debt / latest.total_assets
        if debt_to_asset_ratio < 0.2:
            score += 1
            reasoning.append(f"Low debt to asset ratio of {debt_to_asset_ratio:.1%}")
        elif debt_to_asset_ratio < 0.4:
            score += 0.5
            reasoning.append(f"Moderate debt to asset ratio of {debt_to_asset_ratio:.1%}")
        else:
            reasoning.append(f"High debt to asset ratio of {debt_to_asset_ratio:.1%}")
    else:
        reasoning.append("Debt to asset ratio data not available")
    
    # Check debt serviceability
    if hasattr(latest, "total_debt") and hasattr(latest, "operating_cash_flow") and latest.total_debt is not None and latest.operating_cash_flow is not None and latest.operating_cash_flow > 0:
        years_to_repay = latest.total_debt / latest.operating_cash_flow
        if years_to_repay < 3:
            score += 1
            reasoning.append(f"Strong debt serviceability ({years_to_repay:.1f} years to repay)")
        elif years_to_repay < 5:
            score += 0.5
            reasoning.append(f"Adequate debt serviceability ({years_to_repay:.1f} years to repay)")
        else:
            reasoning.append(f"Poor debt serviceability ({years_to_repay:.1f} years to repay)")
    else:
        reasoning.append("Debt serviceability data not available")
    
    return {
        "score": score,
        "max_score": 2,
        "details": "; ".join(reasoning),
    }

def analyze_environment_fit(environment_analysis: dict, metrics: list, financial_line_items: list) -> dict:
    """Analyzes how well the stock fits the current economic environment."""
    score = 0
    reasoning = []
    
    if not environment_analysis or not metrics or not financial_line_items:
        return {
            "score": 0,
            "max_score": 4,
            "details": "Insufficient data for environment fit analysis",
        }
    
    environment_type = environment_analysis.get("environment_type", "stable growth, stable inflation")
    growth_trend = environment_analysis.get("growth_trend", "stable")
    inflation_trend = environment_analysis.get("inflation_trend", "stable")
    indicators = environment_analysis.get("indicators", {})
    gdp_growth = indicators.get("gdp_growth", 0)
    
    latest_metrics = metrics[0] if metrics else None
    latest_financials = financial_line_items[0] if financial_line_items else None
    
    # Check profit margins for inflation resilience
    if latest_metrics and hasattr(latest_metrics, "gross_margin") and latest_metrics.gross_margin is not None:
        if inflation_trend == "rising" and latest_metrics.gross_margin > 0.4:
            score += 1
            reasoning.append(f"Strong gross margins ({latest_metrics.gross_margin:.1%}) provide inflation protection")
        elif inflation_trend == "rising" and latest_metrics.gross_margin > 0.25:
            score += 0.5
            reasoning.append(f"Moderate gross margins ({latest_metrics.gross_margin:.1%}) provide some inflation protection")
        elif inflation_trend == "falling" and latest_metrics.gross_margin < 0.25:
            score += 0.5
            reasoning.append(f"Lower margins ({latest_metrics.gross_margin:.1%}) may benefit in deflationary environment")
    else:
        reasoning.append("Margin data not available for inflation analysis")
    
    # Check debt structure for interest rate environment
    if latest_financials and hasattr(latest_financials, "total_debt") and latest_financials.total_debt is not None:
        interest_rate = environment_analysis["indicators"].get("interest_rate", 0)
        
        if interest_rate > 3 and (not latest_financials.total_debt or latest_financials.total_debt == 0):
            score += 1
            reasoning.append("Low/no debt position is advantageous in high interest rate environment")
        elif interest_rate < 2 and latest_financials.total_debt > 0:
            score += 0.5
            reasoning.append("Debt utilization may be beneficial in low interest rate environment")
    else:
        reasoning.append("Debt structure data not available for interest rate analysis")
    
    # Check revenue growth relative to economic growth
    if len(financial_line_items) >= 2:
        revenue_values = []
        for item in financial_line_items[:2]:
            if hasattr(item, "revenue") and item.revenue is not None:
                revenue_values.append(item.revenue)
        
        if len(revenue_values) >= 2 and revenue_values[1] > 0:
            revenue_growth = (revenue_values[0] - revenue_values[1]) / revenue_values[1]
            
            if growth_trend == "rising" and revenue_growth > gdp_growth:
                score += 1
                reasoning.append(f"Revenue growth ({revenue_growth:.1%}) outpaces economic growth in rising economy")
            elif growth_trend == "falling" and revenue_growth > 0:
                score += 1
                reasoning.append(f"Positive revenue growth ({revenue_growth:.1%}) despite slowing economy")
            elif growth_trend == "stable" and revenue_growth > gdp_growth:
                score += 0.5
                reasoning.append(f"Revenue growth ({revenue_growth:.1%}) exceeds economic growth")
        else:
            reasoning.append("Revenue growth data not available")
    else:
        reasoning.append("Revenue growth data not available")
    
    # Check specific environment type fit
    if environment_type == "rising growth, rising inflation":
        if latest_metrics and hasattr(latest_metrics, "beta") and latest_metrics.beta is not None:
            if latest_metrics.beta > 1:
                score += 1
                reasoning.append(f"Higher beta ({latest_metrics.beta:.2f}) beneficial in growth environment")
            else:
                reasoning.append(f"Lower beta ({latest_metrics.beta:.2f}) may limit upside in growth environment")
        else:
            reasoning.append("Beta data not available for environment fit analysis")
    
    elif environment_type == "rising growth, falling inflation":
        if latest_metrics and hasattr(latest_metrics, "pe_ratio") and latest_metrics.pe_ratio is not None:
            if latest_metrics.pe_ratio > 0 and latest_metrics.pe_ratio < 20:
                score += 1
                reasoning.append(f"Reasonable valuation ({latest_metrics.pe_ratio:.1f} P/E) in growth environment")
            elif latest_metrics.pe_ratio >= 20:
                reasoning.append(f"Higher valuation ({latest_metrics.pe_ratio:.1f} P/E) may limit upside")
            else:
                reasoning.append("Negative P/E ratio indicates earnings challenges")
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
        if latest_financials and hasattr(latest_financials, "cash_and_cash_equivalents") and latest_financials.cash_and_cash_equivalents is not None and hasattr(latest_financials, "total_assets") and latest_financials.total_assets is not None and latest_financials.total_assets > 0:
            cash_ratio = latest_financials.cash_and_cash_equivalents / latest_financials.total_assets
            if cash_ratio > 0.2:
                score += 1
                reasoning.append(f"Strong cash position ({cash_ratio:.1%} of assets) provides stagflation protection")
            else:
                reasoning.append(f"Limited cash reserves ({cash_ratio:.1%} of assets) may be vulnerable in stagflation")
        else:
            reasoning.append("Cash position data not available for environment fit analysis")
    
    return {
        "score": score,
        "max_score": 4,
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
