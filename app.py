import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import json
import logging
from pydantic import BaseModel, Field
import csv
from io import StringIO
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_NAME = "finance_advisor"
USER_ID = "default_user"

# Pydantic models for output schemas
class SpendingCategory(BaseModel):
    category: str = Field(..., description="Expense category name")
    amount: float = Field(..., description="Amount spent in this category")
    percentage: Optional[float] = Field(None, description="Percentage of total spending")

class SpendingRecommendation(BaseModel):
    category: str = Field(..., description="Category for recommendation")
    recommendation: str = Field(..., description="Recommendation details")
    potential_savings: Optional[float] = Field(None, description="Estimated monthly savings")

class BudgetAnalysis(BaseModel):
    total_expenses: float = Field(..., description="Total monthly expenses")
    monthly_income: Optional[float] = Field(None, description="Monthly income")
    spending_categories: List[SpendingCategory] = Field(..., description="Breakdown of spending by category")
    recommendations: List[SpendingRecommendation] = Field(..., description="Spending recommendations")

class EmergencyFund(BaseModel):
    recommended_amount: float = Field(..., description="Recommended emergency fund size")
    current_amount: Optional[float] = Field(None, description="Current emergency fund (if any)")
    current_status: str = Field(..., description="Status assessment of emergency fund")

class SavingsRecommendation(BaseModel):
    category: str = Field(..., description="Savings category")
    amount: float = Field(..., description="Recommended monthly amount")
    rationale: Optional[str] = Field(None, description="Explanation for this recommendation")

class AutomationTechnique(BaseModel):
    name: str = Field(..., description="Name of automation technique")
    description: str = Field(..., description="Details of how to implement")

class SavingsStrategy(BaseModel):
    emergency_fund: EmergencyFund = Field(..., description="Emergency fund recommendation")
    recommendations: List[SavingsRecommendation] = Field(..., description="Savings allocation recommendations")
    automation_techniques: Optional[List[AutomationTechnique]] = Field(None, description="Automation techniques to help save")

class Debt(BaseModel):
    name: str = Field(..., description="Name of debt")
    amount: float = Field(..., description="Current balance")
    interest_rate: float = Field(..., description="Annual interest rate (%)")
    min_payment: Optional[float] = Field(None, description="Minimum monthly payment")

class PayoffPlan(BaseModel):
    total_interest: float = Field(..., description="Total interest paid")
    months_to_payoff: int = Field(..., description="Months until debt-free")
    monthly_payment: Optional[float] = Field(None, description="Recommended monthly payment")

class PayoffPlans(BaseModel):
    avalanche: PayoffPlan = Field(..., description="Highest interest first method")
    snowball: PayoffPlan = Field(..., description="Smallest balance first method")

class DebtRecommendation(BaseModel):
    title: str = Field(..., description="Title of recommendation")
    description: str = Field(..., description="Details of recommendation")
    impact: Optional[str] = Field(None, description="Expected impact of this action")

class DebtReduction(BaseModel):
    total_debt: float = Field(..., description="Total debt amount")
    debts: List[Debt] = Field(..., description="List of all debts")
    payoff_plans: PayoffPlans = Field(..., description="Debt payoff strategies")
    recommendations: Optional[List[DebtRecommendation]] = Field(None, description="Recommendations for debt reduction")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_chat_response(user_message: str, financial_data: Dict[str, Any], analysis_results: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
    """Generate a chat response based on the financial analysis context"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Build context from financial data and analysis
    context = f"""You are a helpful financial advisor assistant. The user has already received a financial analysis. 
Here is their financial data and analysis results for context:

FINANCIAL DATA:
- Monthly Income: ${financial_data.get('monthly_income', 0)}
- Dependants: {financial_data.get('dependants', 0)}
- Expenses: {json.dumps(financial_data.get('manual_expenses', {}), indent=2) if financial_data.get('manual_expenses') else 'From CSV transactions'}
- Debts: {json.dumps(financial_data.get('debts', []), indent=2)}

ANALYSIS RESULTS:
Budget Analysis: {json.dumps(analysis_results.get('budget_analysis', {}), indent=2)}

Savings Strategy: {json.dumps(analysis_results.get('savings_strategy', {}), indent=2)}

Debt Reduction Plan: {json.dumps(analysis_results.get('debt_reduction', {}), indent=2)}

Based on this context, answer the user's questions helpfully and specifically. Provide actionable advice when possible.
If asked about something not in the analysis, you can provide general financial advice but note that it's general guidance."""

    # Build messages list
    messages = [{"role": "system", "content": context}]
    
    # Add chat history
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def parse_json_safely(data: str, default_value: Any = None) -> Any:
    """Safely parse JSON data with error handling"""
    try:
        return json.loads(data) if isinstance(data, str) else data
    except json.JSONDecodeError:
        return default_value

class FinanceAdvisorSystem:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-4o"  # Using GPT-4o for best results
        
        self.budget_analysis_instruction = """You are a Budget Analysis Agent. Analyze the user's financial data and return a JSON object with EXACTLY this structure:

{
  "total_expenses": <number - sum of all expenses>,
  "monthly_income": <number - user's monthly income>,
  "spending_categories": [
    {"category": "<string - category name>", "amount": <number>, "percentage": <number>}
  ],
  "recommendations": [
    {"category": "<string - expense category>", "recommendation": "<string - specific advice>", "potential_savings": <number>}
  ]
}

Guidelines:
- Include ALL expense categories from the user's data in spending_categories
- Percentages should add up to 100
- Provide 3-5 specific recommendations with estimated savings amounts
- Consider dependants when evaluating household expenses
- Focus on actionable advice with specific implementation steps"""

        self.savings_strategy_instruction = """You are a Savings Strategy Agent. Based on the financial data and budget analysis, return a JSON object with EXACTLY this structure:

{
  "emergency_fund": {
    "recommended_amount": <number - typically 3-6 months of expenses>,
    "current_amount": <number - 0 if not specified>,
    "current_status": "<string - assessment like 'Not started', 'Building', 'Adequate'>"
  },
  "recommendations": [
    {"category": "<string - savings goal like 'Emergency Fund', 'Retirement', 'Vacation'>", "amount": <number - monthly amount>, "rationale": "<string - reason for this recommendation>"}
  ],
  "automation_techniques": [
    {"name": "<string - technique name>", "description": "<string - how to implement>"}
  ]
}

Guidelines:
- Emergency fund should be 3-6 months of total expenses (more for dependants)
- Include at least 3 savings recommendations
- Provide practical automation techniques for consistent saving"""

        self.debt_reduction_instruction = """You are a Debt Reduction Agent. Based on the financial data, budget analysis, and savings strategy, return a JSON object with EXACTLY this structure:

{
  "total_debt": <number - sum of all debts>,
  "debts": [
    {"name": "<string>", "amount": <number>, "interest_rate": <number>, "min_payment": <number>}
  ],
  "payoff_plans": {
    "avalanche": {
      "total_interest": <number - total interest paid with this method>,
      "months_to_payoff": <number - months to become debt-free>,
      "monthly_payment": <number - recommended monthly payment>
    },
    "snowball": {
      "total_interest": <number - total interest paid with this method>,
      "months_to_payoff": <number - months to become debt-free>,
      "monthly_payment": <number - recommended monthly payment>
    }
  },
  "recommendations": [
    {"title": "<string - recommendation title>", "description": "<string - detailed advice>", "impact": "<string - expected outcome>"}
  ]
}

Guidelines:
- Avalanche method: pay highest interest rate debts first (saves more money)
- Snowball method: pay smallest balance first (psychological wins)
- Include at least 3 actionable recommendations
- If no debts provided, use empty arrays and zero values"""

    async def analyze_finances(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Step 1: Budget Analysis
            budget_analysis = await self._run_budget_analysis(financial_data)
            
            # Step 2: Savings Strategy (using budget analysis)
            savings_strategy = await self._run_savings_strategy(financial_data, budget_analysis)
            
            # Step 3: Debt Reduction (using both previous analyses)
            debt_reduction = await self._run_debt_reduction(financial_data, budget_analysis, savings_strategy)
            
            return {
                "budget_analysis": budget_analysis,
                "savings_strategy": savings_strategy,
                "debt_reduction": debt_reduction
            }
            
        except Exception as e:
            logger.exception(f"Error during finance analysis: {str(e)}")
            # Return default results on error
            return self._create_default_results(financial_data)

    async def _run_budget_analysis(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the budget analysis agent"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.budget_analysis_instruction},
                    {"role": "user", "content": f"Analyze this financial data and provide budget analysis:\n{json.dumps(financial_data, indent=2)}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            logger.error(f"Budget analysis error: {e}")
            return self._create_default_results(financial_data)["budget_analysis"]

    async def _run_savings_strategy(self, financial_data: Dict[str, Any], budget_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Run the savings strategy agent"""
        try:
            context = {
                "financial_data": financial_data,
                "budget_analysis": budget_analysis
            }
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.savings_strategy_instruction},
                    {"role": "user", "content": f"Based on this financial data and budget analysis, create a savings strategy:\n{json.dumps(context, indent=2)}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            logger.error(f"Savings strategy error: {e}")
            return self._create_default_results(financial_data)["savings_strategy"]

    async def _run_debt_reduction(self, financial_data: Dict[str, Any], budget_analysis: Dict[str, Any], savings_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Run the debt reduction agent"""
        try:
            context = {
                "financial_data": financial_data,
                "budget_analysis": budget_analysis,
                "savings_strategy": savings_strategy
            }
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.debt_reduction_instruction},
                    {"role": "user", "content": f"Based on this financial data, budget analysis, and savings strategy, create a debt reduction plan:\n{json.dumps(context, indent=2)}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            logger.error(f"Debt reduction error: {e}")
            return self._create_default_results(financial_data)["debt_reduction"]

    def _create_default_results(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        monthly_income = financial_data.get("monthly_income", 0)
        expenses = financial_data.get("manual_expenses", {})
        
        # Ensure expenses is not None
        if expenses is None:
            expenses = {}
        
        if not expenses and financial_data.get("transactions"):
            expenses = {}
            for transaction in financial_data["transactions"]:
                category = transaction.get("Category", "Uncategorized")
                amount = transaction.get("Amount", 0)
                expenses[category] = expenses.get(category, 0) + amount
        
        total_expenses = sum(expenses.values()) if expenses else 0
        
        return {
            "budget_analysis": {
                "total_expenses": total_expenses,
                "monthly_income": monthly_income,
                "spending_categories": [
                    {"category": cat, "amount": amt, "percentage": (amt / total_expenses * 100) if total_expenses > 0 else 0}
                    for cat, amt in expenses.items()
                ],
                "recommendations": [
                    {"category": "General", "recommendation": "Consider reviewing your expenses carefully", "potential_savings": total_expenses * 0.1}
                ]
            },
            "savings_strategy": {
                "emergency_fund": {
                    "recommended_amount": total_expenses * 6,
                    "current_amount": 0,
                    "current_status": "Not started"
                },
                "recommendations": [
                    {"category": "Emergency Fund", "amount": total_expenses * 0.1, "rationale": "Build emergency fund first"},
                    {"category": "Retirement", "amount": monthly_income * 0.15, "rationale": "Long-term savings"}
                ],
                "automation_techniques": [
                    {"name": "Automatic Transfer", "description": "Set up automatic transfers on payday"}
                ]
            },
            "debt_reduction": {
                "total_debt": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])),
                "debts": financial_data.get("debts", []),
                "payoff_plans": {
                    "avalanche": {
                        "total_interest": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) * 0.2,
                        "months_to_payoff": 24,
                        "monthly_payment": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) / 24 if financial_data.get("debts") else 0
                    },
                    "snowball": {
                        "total_interest": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) * 0.25,
                        "months_to_payoff": 24,
                        "monthly_payment": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) / 24 if financial_data.get("debts") else 0
                    }
                },
                "recommendations": [
                    {"title": "Increase Payments", "description": "Increase your monthly payments", "impact": "Reduces total interest paid"}
                ]
            }
        }

def display_budget_analysis(analysis: Dict[str, Any]):
    if isinstance(analysis, str):
        try:
            analysis = json.loads(analysis)
        except json.JSONDecodeError:
            st.error("Failed to parse budget analysis results")
            return
    
    if not isinstance(analysis, dict):
        st.error("Invalid budget analysis format")
        return
    
    # Handle nested response (OpenAI might wrap in a key)
    if "budget_analysis" in analysis:
        analysis = analysis["budget_analysis"]
    
    spending_categories = analysis.get("spending_categories", [])
    if spending_categories:
        st.subheader("Spending by Category")
        try:
            values = []
            names = []
            for cat in spending_categories:
                values.append(cat.get("amount", 0))
                names.append(cat.get("category", "Unknown"))
            fig = px.pie(values=values, names=names, title="Your Spending Breakdown")
            st.plotly_chart(fig)
        except Exception as e:
            st.warning(f"Could not display spending chart: {e}")
    
    total_expenses = analysis.get("total_expenses", 0) or 0
    monthly_income = analysis.get("monthly_income", 0) or 0
    
    if total_expenses > 0 or monthly_income > 0:
        st.subheader("Income vs. Expenses")
        surplus_deficit = monthly_income - total_expenses
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Income", "Expenses"], 
                            y=[monthly_income, total_expenses],
                            marker_color=["green", "red"]))
        fig.update_layout(title="Monthly Income vs. Expenses")
        st.plotly_chart(fig)
        
        st.metric("Monthly Surplus/Deficit", 
                  f"${surplus_deficit:.2f}", 
                  delta=f"{surplus_deficit:.2f}")
    
    recommendations = analysis.get("recommendations", [])
    if recommendations:
        st.subheader("Spending Reduction Recommendations")
        for rec in recommendations:
            category = rec.get("category", "General")
            recommendation = rec.get("recommendation", "No details provided")
            st.markdown(f"**{category}**: {recommendation}")
            potential_savings = rec.get("potential_savings")
            if potential_savings:
                st.metric("Potential Monthly Savings", f"${float(potential_savings):.2f}")

def display_savings_strategy(strategy: Dict[str, Any]):
    if isinstance(strategy, str):
        try:
            strategy = json.loads(strategy)
        except json.JSONDecodeError:
            st.error("Failed to parse savings strategy results")
            return
    
    if not isinstance(strategy, dict):
        st.error("Invalid savings strategy format")
        return
    
    # Handle nested response
    if "savings_strategy" in strategy:
        strategy = strategy["savings_strategy"]
    
    st.subheader("Savings Recommendations")
    
    emergency_fund = strategy.get("emergency_fund", {})
    if emergency_fund:
        st.markdown("### Emergency Fund")
        recommended = emergency_fund.get("recommended_amount", 0) or 0
        st.markdown(f"**Recommended Size**: ${float(recommended):.2f}")
        status = emergency_fund.get("current_status", "Not assessed")
        st.markdown(f"**Current Status**: {status}")
        
        current = emergency_fund.get("current_amount", 0) or 0
        if recommended > 0:
            progress = current / recommended
            st.progress(min(progress, 1.0))
            st.markdown(f"${float(current):.2f} of ${float(recommended):.2f}")
    
    recommendations = strategy.get("recommendations", [])
    if recommendations:
        st.markdown("### Recommended Savings Allocations")
        for rec in recommendations:
            amount = rec.get("amount", 0) or 0
            category = rec.get("category", "Savings")
            st.markdown(f"**{category}**: ${float(amount):.2f}/month")
            rationale = rec.get("rationale", "")
            if rationale:
                st.markdown(f"_{rationale}_")
    
    automation_techniques = strategy.get("automation_techniques", [])
    if automation_techniques:
        st.markdown("### Automation Techniques")
        for technique in automation_techniques:
            name = technique.get("name", "Technique")
            description = technique.get("description", "No details")
            st.markdown(f"**{name}**: {description}")

def display_debt_reduction(plan: Dict[str, Any]):
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except json.JSONDecodeError:
            st.error("Failed to parse debt reduction results")
            return
    
    if not isinstance(plan, dict):
        st.error("Invalid debt reduction format")
        return
    
    # Handle nested response
    if "debt_reduction" in plan:
        plan = plan["debt_reduction"]
    
    total_debt = plan.get("total_debt", 0) or 0
    if total_debt > 0:
        st.metric("Total Debt", f"${total_debt:.2f}")
    
    debts = plan.get("debts", [])
    if debts:
        st.subheader("Your Debts")
        debt_df = pd.DataFrame(debts)
        st.dataframe(debt_df)
        
        if len(debt_df) > 0 and "name" in debt_df.columns and "amount" in debt_df.columns:
            fig = px.bar(debt_df, x="name", y="amount", color="interest_rate",
                        labels={"name": "Debt", "amount": "Amount ($)", "interest_rate": "Interest Rate (%)"},
                        title="Debt Breakdown")
            st.plotly_chart(fig)
    
    payoff_plans = plan.get("payoff_plans", {})
    if payoff_plans:
        st.subheader("Debt Payoff Plans")
        tabs = st.tabs(["Avalanche Method", "Snowball Method", "Comparison"])
        
        avalanche = payoff_plans.get("avalanche", {})
        snowball = payoff_plans.get("snowball", {})
        
        with tabs[0]:
            st.markdown("### Avalanche Method (Highest Interest First)")
            if avalanche:
                total_interest = avalanche.get("total_interest", 0) or 0
                months = avalanche.get("months_to_payoff", 0) or 0
                monthly = avalanche.get("monthly_payment", 0) or 0
                st.markdown(f"**Total Interest Paid**: ${total_interest:.2f}")
                st.markdown(f"**Time to Debt Freedom**: {months} months")
                if monthly > 0:
                    st.markdown(f"**Recommended Monthly Payment**: ${monthly:.2f}")
        
        with tabs[1]:
            st.markdown("### Snowball Method (Smallest Balance First)")
            if snowball:
                total_interest = snowball.get("total_interest", 0) or 0
                months = snowball.get("months_to_payoff", 0) or 0
                monthly = snowball.get("monthly_payment", 0) or 0
                st.markdown(f"**Total Interest Paid**: ${total_interest:.2f}")
                st.markdown(f"**Time to Debt Freedom**: {months} months")
                if monthly > 0:
                    st.markdown(f"**Recommended Monthly Payment**: ${monthly:.2f}")
        
        with tabs[2]:
            st.markdown("### Method Comparison")
            if avalanche and snowball:
                comparison_data = {
                    "Method": ["Avalanche", "Snowball"],
                    "Total Interest": [avalanche.get("total_interest", 0), snowball.get("total_interest", 0)],
                    "Months to Payoff": [avalanche.get("months_to_payoff", 0), snowball.get("months_to_payoff", 0)]
                }
                comparison_df = pd.DataFrame(comparison_data)
                
                st.dataframe(comparison_df)
                
                fig = go.Figure(data=[
                    go.Bar(name="Total Interest", x=comparison_df["Method"], y=comparison_df["Total Interest"]),
                    go.Bar(name="Months to Payoff", x=comparison_df["Method"], y=comparison_df["Months to Payoff"])
                ])
                fig.update_layout(barmode='group', title="Debt Payoff Method Comparison")
                st.plotly_chart(fig)
    
    recommendations = plan.get("recommendations", [])
    if recommendations:
        st.subheader("Debt Reduction Recommendations")
        for rec in recommendations:
            title = rec.get("title", "Recommendation")
            description = rec.get("description", "No details")
            st.markdown(f"**{title}**: {description}")
            impact = rec.get("impact", "")
            if impact:
                st.markdown(f"_Impact: {impact}_")

def parse_csv_transactions(file_content) -> List[Dict[str, Any]]:
    """Parse CSV file content into a list of transactions"""
    try:
        # Read CSV content
        df = pd.read_csv(StringIO(file_content.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['Date', 'Category', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert date strings to datetime and then to string format YYYY-MM-DD
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # Convert amount strings to float, handling currency symbols and commas
        df['Amount'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)
        
        # Group by category and calculate totals
        category_totals = df.groupby('Category')['Amount'].sum().reset_index()
        
        # Convert to list of dictionaries
        transactions = df.to_dict('records')
        
        return {
            'transactions': transactions,
            'category_totals': category_totals.to_dict('records')
        }
    except Exception as e:
        raise ValueError(f"Error parsing CSV file: {str(e)}")

def validate_csv_format(file) -> bool:
    """Validate CSV file format and content"""
    try:
        content = file.read().decode('utf-8')
        dialect = csv.Sniffer().sniff(content)
        has_header = csv.Sniffer().has_header(content)
        file.seek(0)  # Reset file pointer
        
        if not has_header:
            return False, "CSV file must have headers"
            
        df = pd.read_csv(StringIO(content))
        required_columns = ['Date', 'Category', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
            
        # Validate date format
        try:
            pd.to_datetime(df['Date'])
        except:
            return False, "Invalid date format in Date column"
            
        # Validate amount format (should be numeric after removing currency symbols)
        try:
            df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)
        except:
            return False, "Invalid amount format in Amount column"
            
        return True, "CSV format is valid"
    except Exception as e:
        return False, f"Invalid CSV format: {str(e)}"

def display_csv_preview(df: pd.DataFrame):
    """Display a preview of the CSV data with basic statistics"""
    st.subheader("CSV Data Preview")
    
    # Show basic statistics
    total_transactions = len(df)
    total_amount = df['Amount'].sum()
    
    # Convert dates for display
    df_dates = pd.to_datetime(df['Date'])
    date_range = f"{df_dates.min().strftime('%Y-%m-%d')} to {df_dates.max().strftime('%Y-%m-%d')}"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", total_transactions)
    with col2:
        st.metric("Total Amount", f"${total_amount:,.2f}")
    with col3:
        st.metric("Date Range", date_range)
    
    # Show category breakdown
    st.subheader("Spending by Category")
    category_totals = df.groupby('Category')['Amount'].agg(['sum', 'count']).reset_index()
    category_totals.columns = ['Category', 'Total Amount', 'Transaction Count']
    st.dataframe(category_totals)
    
    # Show sample transactions
    st.subheader("Sample Transactions")
    st.dataframe(df.head())

def main():
    st.set_page_config(
        page_title="AI Finance Analyzer",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar with CSV template
    with st.sidebar:
        st.title("📊 Templates")
        st.caption("This application uses AI to provide personalized financial advice through multiple specialized agents.")
        
        st.divider()
        
        # Add CSV template download
        st.subheader("📥 CSV Template")
        st.markdown("""
        Download the template CSV file with the required format:
        - Date (YYYY-MM-DD)
        - Category
        - Amount (numeric)
        """)
        
        # Create sample CSV content
        sample_csv = """Date,Category,Amount
2024-01-01,Housing,1200.00
2024-01-02,Food,150.50
2024-01-03,Transportation,45.00"""
        
        st.download_button(
            label="📥 Download CSV Template",
            data=sample_csv,
            file_name="expense_template.csv",
            mime="text/csv"
        )
    
    # Main content
    st.title("📊 AI Finance Analyzer")
    st.caption("Powered by OpenAI GPT-4o")
    st.info("This tool analyzes your financial data and provides tailored recommendations for budgeting, savings, and debt management using multiple specialized AI agents.")
    st.divider()
    
    # Create tabs for different sections
    input_tab, about_tab = st.tabs(["💼 Financial Information", "ℹ️ About"])
    
    with input_tab:
        st.header("Enter Your Financial Information")
        st.caption("All data is processed locally and not stored anywhere.")
        
        # Income and Dependants section in a container
        with st.container():
            st.subheader("💰 Income & Household")
            income_col, dependants_col = st.columns([2, 1])
            with income_col:
                monthly_income = st.number_input(
                    "Monthly Income ($)",
                    min_value=0.0,
                    step=100.0,
                    value=3000.0,
                    key="income",
                    help="Enter your total monthly income after taxes"
                )
            with dependants_col:
                dependants = st.number_input(
                    "Number of Dependants",
                    min_value=0,
                    step=1,
                    value=0,
                    key="dependants",
                    help="Include all dependants in your household"
                )
        
        st.divider()
        
        # Expenses section
        with st.container():
            st.subheader("💳 Expenses")
            expense_option = st.radio(
                "How would you like to enter your expenses?",
                ("📤 Upload CSV Transactions", "✍️ Enter Manually"),
                key="expense_option",
                horizontal=True
            )
            
            transaction_file = None
            manual_expenses = {}
            use_manual_expenses = False
            transactions_df = None

            if expense_option == "📤 Upload CSV Transactions":
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("""
                    #### Upload your transaction data
                    Your CSV file should have these columns:
                    - 📅 Date (YYYY-MM-DD)
                    - 📝 Category
                    - 💲 Amount
                    """)
                    
                    transaction_file = st.file_uploader(
                        "Choose your CSV file",
                        type=["csv"],
                        key="transaction_file",
                        help="Upload a CSV file containing your transactions"
                    )
                
                if transaction_file is not None:
                    # Validate CSV format
                    is_valid, message = validate_csv_format(transaction_file)
                    
                    if is_valid:
                        try:
                            # Parse CSV content
                            transaction_file.seek(0)
                            file_content = transaction_file.read()
                            parsed_data = parse_csv_transactions(file_content)
                            
                            # Create DataFrame
                            transactions_df = pd.DataFrame(parsed_data['transactions'])
                            
                            # Display preview
                            display_csv_preview(transactions_df)
                            
                            st.success("✅ Transaction file uploaded and validated successfully!")
                        except Exception as e:
                            st.error(f"❌ Error processing CSV file: {str(e)}")
                            transactions_df = None
                    else:
                        st.error(message)
                        transactions_df = None
            else:
                use_manual_expenses = True
                st.markdown("#### Enter your monthly expenses by category")
                
                # Define expense categories with emojis
                categories = [
                    ("🏠 Housing", "Housing"),
                    ("🔌 Utilities", "Utilities"),
                    ("🍽️ Food", "Food"),
                    ("🚗 Transportation", "Transportation"),
                    ("🏥 Healthcare", "Healthcare"),
                    ("🎭 Entertainment", "Entertainment"),
                    ("👤 Personal", "Personal"),
                    ("💰 Savings", "Savings"),
                    ("📦 Other", "Other")
                ]
                
                # Create three columns for better layout
                col1, col2, col3 = st.columns(3)
                cols = [col1, col2, col3]
                
                # Distribute categories across columns
                for i, (emoji_cat, cat) in enumerate(categories):
                    with cols[i % 3]:
                        manual_expenses[cat] = st.number_input(
                            emoji_cat,
                            min_value=0.0,
                            step=50.0,
                            value=0.0,
                            key=f"manual_{cat}",
                            help=f"Enter your monthly {cat.lower()} expenses"
                        )
                
                if manual_expenses and any(manual_expenses.values()):
                    st.markdown("#### 📊 Summary of Entered Expenses")
                    manual_df_disp = pd.DataFrame({
                        'Category': list(manual_expenses.keys()),
                        'Amount': list(manual_expenses.values())
                    })
                    manual_df_disp = manual_df_disp[manual_df_disp['Amount'] > 0]
                    if not manual_df_disp.empty:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.dataframe(
                                manual_df_disp,
                                column_config={
                                    "Category": "Category",
                                    "Amount": st.column_config.NumberColumn(
                                        "Amount",
                                        format="$%.2f"
                                    )
                                },
                                hide_index=True
                            )
                        with col2:
                            st.metric(
                                "Total Monthly Expenses",
                                f"${manual_df_disp['Amount'].sum():,.2f}"
                            )
        
        st.divider()
        
        # Debt Information section
        with st.container():
            st.subheader("🏦 Debt Information")
            st.info("Enter your debts to get personalized payoff strategies using both avalanche and snowball methods.")
            
            num_debts = st.number_input(
                "How many debts do you have?",
                min_value=0,
                max_value=10,
                step=1,
                value=0,
                key="num_debts"
            )
            
            debts = []
            if num_debts > 0:
                # Create columns for debts
                cols = st.columns(min(num_debts, 3))  # Max 3 columns per row
                for i in range(num_debts):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.markdown(f"##### Debt #{i+1}")
                        debt_name = st.text_input(
                            "Name",
                            value=f"Debt {i+1}",
                            key=f"debt_name_{i}",
                            help="Enter a name for this debt (e.g., Credit Card, Student Loan)"
                        )
                        debt_amount = st.number_input(
                            "Amount ($)",
                            min_value=0.01,
                            step=100.0,
                            value=1000.0,
                            key=f"debt_amount_{i}",
                            help="Enter the current balance of this debt"
                        )
                        interest_rate = st.number_input(
                            "Interest Rate (%)",
                            min_value=0.0,
                            max_value=100.0,
                            step=0.1,
                            value=5.0,
                            key=f"debt_rate_{i}",
                            help="Enter the annual interest rate"
                        )
                        min_payment = st.number_input(
                            "Minimum Payment ($)",
                            min_value=0.0,
                            step=10.0,
                            value=50.0,
                            key=f"debt_min_payment_{i}",
                            help="Enter the minimum monthly payment required"
                        )
                        
                        debts.append({
                            "name": debt_name,
                            "amount": debt_amount,
                            "interest_rate": interest_rate,
                            "min_payment": min_payment
                        })
                        
                        if col_idx == 2 or i == num_debts - 1:  # Add spacing after every 3 debts or last debt
                            st.markdown("---")
        
        st.divider()
        
        # Analysis button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🔄 Analyze My Finances",
                key="analyze_button",
                use_container_width=True,
                help="Click to get your personalized financial analysis"
            )
        
        if analyze_button:
            if expense_option == "Upload CSV Transactions" and transactions_df is None:
                st.error("Please upload a valid transaction CSV file or choose manual entry.")
                return
            if use_manual_expenses and (not manual_expenses or not any(manual_expenses.values())):
                st.warning("No manual expenses entered. Analysis might be limited.")

            with st.spinner("🤖 AI agents are analyzing your financial data..."): 
                financial_data = {
                    "monthly_income": monthly_income,
                    "dependants": dependants,
                    "transactions": transactions_df.to_dict('records') if transactions_df is not None else None,
                    "manual_expenses": manual_expenses if use_manual_expenses else None,
                    "debts": debts
                }
                
                finance_system = FinanceAdvisorSystem()
                
                try:
                    results = asyncio.run(finance_system.analyze_finances(financial_data))
                    
                    # Store results in session state for persistence
                    st.session_state.analysis_results = results
                    st.session_state.financial_data = financial_data
                    st.session_state.analysis_complete = True
                    # Clear chat history on new analysis
                    st.session_state.chat_messages = []
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
        
        # Display results from session state (persists across reruns)
        if st.session_state.get("analysis_complete", False) and st.session_state.get("analysis_results"):
            st.header("Financial Analysis Results")
            results = st.session_state.analysis_results
            
            tabs = st.tabs(["💰 Budget Analysis", "📈 Savings Strategy", "💳 Debt Reduction"])
            
            with tabs[0]:
                st.subheader("Budget Analysis")
                if "budget_analysis" in results and results["budget_analysis"]:
                    display_budget_analysis(results["budget_analysis"])
                else:
                    st.write("No budget analysis available.")
            
            with tabs[1]:
                st.subheader("Savings Strategy")
                if "savings_strategy" in results and results["savings_strategy"]:
                    display_savings_strategy(results["savings_strategy"])
                else:
                    st.write("No savings strategy available.")
            
            with tabs[2]:
                st.subheader("Debt Reduction Plan")
                if "debt_reduction" in results and results["debt_reduction"]:
                    display_debt_reduction(results["debt_reduction"])
                else:
                    st.write("No debt reduction plan available.")
        
        # Chat section - appears after analysis or if previous analysis exists
        if st.session_state.get("analysis_complete", False):
            st.divider()
            col1, col2 = st.columns([3, 1])
            with col1:
                st.header("💬 Chat with Your Finance Advisor")
            with col2:
                if st.button("🗑️ Clear Chat", key="clear_chat"):
                    st.session_state.chat_messages = []
                    st.rerun()
            
            st.caption("Ask follow-up questions about your financial analysis")
            
            # Initialize chat history
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []
            
            # Display chat history
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your finances..."):
                # Add user message to history
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = get_chat_response(
                            prompt,
                            st.session_state.financial_data,
                            st.session_state.analysis_results,
                            st.session_state.chat_messages[:-1]  # Exclude current message
                        )
                        st.markdown(response)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
    
    with about_tab:
        st.markdown("""
        ### About AI Finance Analyzer
        
        This application uses OpenAI's GPT-4o to provide comprehensive financial analysis and advice through multiple specialized AI agents:
        
        1. **🔍 Budget Analysis Agent**
           - Analyzes spending patterns
           - Identifies areas for cost reduction
           - Provides actionable recommendations
        
        2. **💰 Savings Strategy Agent**
           - Creates personalized savings plans
           - Calculates emergency fund requirements
           - Suggests automation techniques
        
        3. **💳 Debt Reduction Agent**
           - Develops optimal debt payoff strategies
           - Compares different repayment methods
           - Provides actionable debt reduction tips
        
        ### Privacy & Security
        
        - All data is processed locally
        - No financial information is stored or transmitted
        - Secure API communication with OpenAI's services
        
        ### Need Help?
        
        For support or questions:
        - Check the [documentation](https://github.com/Shubhamsaboo/awesome-llm-apps)
        - Report issues on [GitHub](https://github.com/Shubhamsaboo/awesome-llm-apps/issues)
        """)

if __name__ == "__main__":
    main()
