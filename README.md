# 📊 AI Finance Analyzer

An intelligent personal finance advisor powered by OpenAI GPT-4o that provides comprehensive financial analysis, savings strategies, and debt reduction plans through a modern Streamlit interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
## 🌟 Features
Streamlit App: https://ai-finance-analyzer-lswergsvmj9bcrqyt7ajpe.streamlit.app/

### 🤖 Multi-Agent AI System
- **Budget Analysis Agent**: Analyzes spending patterns and identifies cost-reduction opportunities
- **Savings Strategy Agent**: Creates personalized savings plans and emergency fund recommendations
- **Debt Reduction Agent**: Develops optimal debt payoff strategies using avalanche and snowball methods

### 💬 Interactive Chat
- Follow-up questions after analysis
- Context-aware responses based on your financial data
- Persistent conversation history

### 📈 Visual Analytics
- Interactive pie charts for spending breakdown
- Bar charts comparing income vs expenses
- Debt comparison visualizations
- Method comparison tables

### 📤 Flexible Data Input
- Manual expense entry by category
- CSV file upload for transaction data
- Support for multiple debts with interest rates

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Project 2"
   ```

2. **Install dependencies**
   ```bash
   cd code
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the `code` folder:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application**
   ```bash
   cd code
   streamlit run app.py
   ```

5. **Open in browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
AI-Finance-Analyzer/
├── code/
│   ├── app.py              # Main Streamlit application
│   ├── requirements.txt    # Python dependencies
│   └── .env               # Environment variables (API keys)
├── README.md              # This file
└── PROJECT_REPORT.md      # Detailed technical documentation
```

## 🎯 Usage Guide

### Step 1: Enter Financial Information
1. Input your **monthly income**
2. Specify number of **dependants**

### Step 2: Add Expenses
Choose one of two methods:
- **Manual Entry**: Enter expenses by category (Housing, Food, Transportation, etc.)
- **CSV Upload**: Upload a CSV file with columns: `Date`, `Category`, `Amount`

### Step 3: Add Debts (Optional)
For each debt, provide:
- Debt name
- Current balance
- Interest rate (%)
- Minimum monthly payment

### Step 4: Analyze
Click **"Analyze My Finances"** to receive:
- Budget analysis with spending breakdown
- Personalized savings strategy
- Debt reduction plan with avalanche vs snowball comparison

### Step 5: Chat
Use the chat interface to ask follow-up questions about your analysis.

## 📊 Sample CSV Format

```csv
Date,Category,Amount
2024-01-01,Housing,1200.00
2024-01-02,Food,150.50
2024-01-03,Transportation,45.00
2024-01-05,Entertainment,75.00
2024-01-10,Utilities,120.00
```

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |

### Supported Models
The application uses `gpt-4o` by default. To change the model, modify the `self.model` variable in the `FinanceAdvisorSystem` class.

## ☁️ Streamlit Cloud Deployment

To deploy on Streamlit Cloud:

1. **Fork/Push** the repository to your GitHub account
2. **Connect** your GitHub repo to [Streamlit Cloud](https://streamlit.io/cloud)
3. **Set the main file path** to `code/app.py`
4. **Add secrets** in Streamlit Cloud dashboard:
   - Go to App Settings → Secrets
   - Add your OpenAI API key:
     ```toml
     OPENAI_API_KEY = "sk-your-api-key-here"
     ```
5. **Deploy** and your app will be live!

The app automatically detects whether it's running locally (uses `.env`) or on Streamlit Cloud (uses secrets).

---

## 🛡️ Privacy & Security

- **Local Processing**: All data is processed locally on your machine
- **No Data Storage**: Financial information is not stored or persisted
- **Secure API Communication**: Uses HTTPS for OpenAI API calls
- **Session-Based**: Data exists only during your browser session

## 📋 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.15.0
python-dotenv>=1.0.0
pydantic>=2.0.0
openai>=1.0.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) for GPT-4o API
- [Streamlit](https://streamlit.io/) for the web framework
- [Plotly](https://plotly.com/) for interactive visualizations

## 📧 Support

For support or questions:
- Open an issue on GitHub
- Check the [PROJECT_REPORT.md](PROJECT_REPORT.md) for technical details

