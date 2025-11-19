# 📈 Exponential Smoothing FX Trading – Streamlit App  
### **α–β Grid Search Optimization + Full Trading Backtester**

This project provides a complete **FX trading research environment** built using **Streamlit, Python, Plotly, and yFinance**. It includes:

- A **main trading page** (`st_app.py`)  
- A **parameter optimization suite** (`alpha_beta_optimization.py`)  
- Full **grid search** over smoothing parameters  
- Threshold-based entry logic, deceleration-based exits  
- Sharpe/Return/Trades heatmaps  
- 3D Sharpe surface  
- One-click integration back into the main page  
- Complete **backtesting engine + trade log**  

---

## 🚀 Features

### **1. Trading Strategy Based on Double Exponential Smoothing**
The strategy computes two exponential smoothing series:

- **α (slow smoothing)**  
- **β (fast smoothing)**  

The difference  
\[
d_t = ES_β(t) - ES_α(t)
\]  
generates **BUY**, **SELL**, and **EXIT** signals based on:

- Zero/threshold crosses  
- Optional **deceleration exit rule** using slope  
- Customizable `x` (threshold) and `θ` (slope sensitivity)

---

### **2. Fully Integrated Grid Search Optimizer**
The optimization app performs:

- **α grid** (slow ES)
- **β grid** (fast ES)
- **x grid** (threshold)
- **θ grid** (deceleration slope)
- **Both exit rules**: `cross` and `deceleration`

For each (α, β), it picks the best (x, θ, exit rule) by **Sharpe Ratio**.

Includes:

- Sharpe heatmap  
- Total Return heatmap  
- #Trades heatmap  
- Interactive 3D Sharpe surface  
- Top 15 best configurations  
- **One-click “Use in main page” button**  
  - Updates `st.session_state`  
  - Immediately available in the main trading app  

---

## 🔧 Installation & Setup

### **1. Clone the repo**
```bash
git clone <your-repo-url>
cd <project-folder>
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

Typical requirements include:

```
streamlit
numpy
pandas
yfinance
plotly
```

### **3. Run Streamlit**
To launch the **main trading page**:

```bash
streamlit run st_app.py
```

To launch the **optimizer page**:

```bash
streamlit run alpha_beta_optimization.py
```

---

## 🔍 How the Strategy Works

### **1. Exponential Smoothing**
Each price series produces:

\[
ES_α(t), \quad ES_β(t)
\]

with α < β so that:

- ESα = slow trend  
- ESβ = fast trend  

### **2. Trading Logic**
Long signal when:

\[
d_{t-1} < 0 \quad \text{and} \quad d_t > x
\]

Short signal when:

\[
d_{t-1} > 0 \quad \text{and} \quad d_t < -x
\]

Optional **deceleration exit**:

\[
\text{if } (d_t - d_{t-1}) < -θ \Rightarrow \text{close position}
\]

---

## 📊 Visual Outputs

### **In st_app.py**
- Price chart with ES(α) and ES(β)
- Buy/Sell markers
- Equity curve
- Trade log table
- JSON metrics panel

### **In alpha_beta_optimization.py**
- Sharpe heatmap  
- Total return heatmap  
- Trades heatmap  
- 3D Sharpe surface  
- Table of best parameter combinations  
- “Use in main page” buttons  

---

## 🧪 Example Workflow

1. Open `alpha_beta_optimization.py`
2. Run full grid search  
3. Review Sharpe heatmap  
4. Click **Use in main page** on a top configuration  
5. Open `st_app.py`  
6. View updated:
   - Signals  
   - Trades  
   - Equity curve  
   - Metrics  
7. Adjust cost/thresholds if needed  

---

## 🧩 Customization

You can easily extend:

- Add new exit rules  
- Add drawdown-based stop-loss  
- Add trailing exit  
- Add volatility filters  
- Replace ES with Holt-Winters or EMA  
- Add portfolio support  

---

## 📬 Contact
Maintainer: **Tanish Patel**

