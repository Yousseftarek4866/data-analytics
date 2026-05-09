"""
FinTech Stock Market Intelligence Dashboard
============================================
Run with: streamlit run app.py
"""
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
yf.set_tz_cache_location("custom_cache_dir")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinTech Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0a0e1a; }

    .metric-card {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1221 100%);
        border: 1px solid #2a3050;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #6b7db3;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'Space Mono', monospace;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        margin-top: 2px;
    }
    .metric-positive { color: #00e676; }
    .metric-negative { color: #ff5252; }
    .metric-neutral  { color: #82b1ff; }

    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        color: #6b7db3;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #1e2540;
        padding-bottom: 0.4rem;
    }
    div[data-testid="stMetric"] { background: transparent !important; }
    .stSelectbox label, .stSlider label, .stMultiSelect label {
        color: #6b7db3 !important;
        font-size: 0.8rem !important;
        font-family: 'Space Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
FINTECH_TICKERS = {
    'JPM': 'JPMorgan Chase', 'V': 'Visa', 'MA': 'Mastercard',
    'PYPL': 'PayPal', 'XYZ': 'Block (Square)', 'COIN': 'Coinbase',
    'HOOD': 'Robinhood', 'AFRM': 'Affirm'
}
COLORS = ['#00e5ff', '#00e676', '#ff6b6b', '#ffd740', '#ea80fc',
          '#69f0ae', '#ff5252', '#82b1ff']

PLOT_THEME = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,14,26,0.8)',
    font=dict(family='Inter', color='#c0caf5'),
    margin=dict(l=20, r=20, t=40, b=20)
)

# ─── Helper Functions ────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_stock_data(ticker, period_days):
    end   = datetime.today()
    start = end - timedelta(days=period_days)
    
    try:
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            return df
    except Exception:
        pass

    # Fallback: stooq (no geo-block)
    try:
        import pandas_datareader.data as web
        df = web.DataReader(ticker, 'stooq', start=start, end=end).sort_index()
        if not df.empty:
            return df
    except Exception:
        pass

    st.error(f"Could not load data for {ticker}.")
    st.stop()

def add_indicators(df):
    c = df['Close']
    df['SMA_20']  = c.rolling(20).mean()
    df['SMA_50']  = c.rolling(50).mean()
    df['EMA_12']  = c.ewm(span=12, adjust=False).mean()
    df['EMA_26']  = c.ewm(span=26, adjust=False).mean()
    df['MACD']    = df['EMA_12'] - df['EMA_26']
    df['MACD_Sig']= df['MACD'].ewm(span=9, adjust=False).mean()
    delta         = c.diff()
    gain          = delta.clip(lower=0).rolling(14).mean()
    loss          = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI']     = 100 - (100 / (1 + gain / loss))
    df['BB_Mid']  = c.rolling(20).mean()
    std           = c.rolling(20).std()
    df['BB_Up']   = df['BB_Mid'] + 2*std
    df['BB_Lo']   = df['BB_Mid'] - 2*std
    df['Return']  = c.pct_change()
    df['Vol_20d'] = df['Return'].rolling(20).std() * np.sqrt(252)
    return df

def compute_risk(returns):
    r          = returns.dropna()
    ann_ret    = r.mean() * 252 * 100
    ann_vol    = r.std() * np.sqrt(252) * 100
    sharpe     = ((r.mean() - 0.05/252) / r.std()) * np.sqrt(252)
    var95      = np.percentile(r, 5) * 100
    cum        = (1+r).cumprod()
    max_dd     = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    return ann_ret, ann_vol, sharpe, var95, max_dd

@st.cache_data(ttl=3600)
def compute_portfolio_optimization(returns_json):
    returns     = pd.read_json(returns_json)
    mean_r      = returns.mean()
    cov         = returns.cov()
    n           = len(mean_r)
    tickers_opt = list(returns.columns)

    def perf(w):
        r = np.dot(w, mean_r) * 252
        v = np.sqrt(w @ (cov.values * 252) @ w)
        s = (r - 0.05) / v if v > 1e-10 else 0.0
        return r, v, s

    x0          = np.full(n, 1.0 / n)
    bounds      = tuple((0.0, 1.0) for _ in range(n))
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    res_ms = minimize(lambda w: -perf(w)[2], x0,
                      method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-12})
    res_mv = minimize(lambda w:  perf(w)[1], x0,
                      method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-12})
    w_ms = res_ms.x;  ms_r, ms_v, ms_s = perf(w_ms)
    w_mv = res_mv.x;  mv_r, mv_v, mv_s = perf(w_mv)

    np.random.seed(42)
    N_SIM = 6000
    mc_r, mc_v, mc_s = [], [], []
    for _ in range(N_SIM):
        w = np.random.random(n); w /= w.sum()
        r, v, s = perf(w)
        mc_r.append(r); mc_v.append(v); mc_s.append(s)

    target_rets = np.linspace(min(mc_r), max(mc_r), 60)
    ef_v, ef_r  = [], []
    for target in target_rets:
        cons_ef = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w, t=target: perf(w)[0] - t}
        ]
        res = minimize(lambda w: perf(w)[1], x0, method='SLSQP',
                       bounds=bounds, constraints=cons_ef,
                       options={'maxiter': 500, 'ftol': 1e-10})
        if res.success:
            r2, v2, _ = perf(res.x)
            ef_r.append(r2 * 100); ef_v.append(v2 * 100)

    return {
        'tickers': tickers_opt,
        'mc':  {'r': [x*100 for x in mc_r], 'v': [x*100 for x in mc_v], 's': mc_s},
        'ef':  {'r': ef_r, 'v': ef_v},
        'max_sharpe': {'w': w_ms, 'r': ms_r*100, 'v': ms_v*100, 's': ms_s},
        'min_var':    {'w': w_mv, 'r': mv_r*100, 'v': mv_v*100, 's': mv_s},
    }

@st.cache_data(ttl=300)
def run_gbm_simulation(mu, sigma, S0, horizon=252, n_paths=400):
    dt = 1.0 / 252
    np.random.seed(42)
    paths    = np.zeros((horizon + 1, n_paths))
    paths[0] = S0
    for t in range(1, horizon + 1):
        Z        = np.random.standard_normal(n_paths)
        paths[t] = paths[t-1] + mu*paths[t-1]*dt + sigma*paths[t-1]*np.sqrt(dt)*Z
    return paths

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 FinTech Intelligence")
    st.markdown("<p style='color:#6b7db3;font-size:0.75rem;'>Stock Market Analysis Dashboard</p>", unsafe_allow_html=True)
    st.divider()

    selected_ticker = st.selectbox(
        "Primary Ticker",
        options=list(FINTECH_TICKERS.keys()),
        format_func=lambda x: f"{x} — {FINTECH_TICKERS[x]}"
    )

    _default_compare = [t for t in ['V', 'MA'] if t != selected_ticker]
    compare_tickers = st.multiselect(
        "Compare With",
        options=[t for t in FINTECH_TICKERS if t != selected_ticker],
        default=_default_compare,
        format_func=lambda x: f"{x} — {FINTECH_TICKERS[x]}"
    )

    period_days = st.select_slider(
        "Time Period",
        options=[90, 180, 365, 730, 1095],
        value=365,
        format_func=lambda x: f"{x//365}Y" if x >= 365 else f"{x//30}M"
    )

    st.divider()
    show_sma    = st.checkbox("Show SMA Lines", value=True)
    show_bb     = st.checkbox("Show Bollinger Bands", value=True)
    show_volume = st.checkbox("Show Volume", value=True)

    st.divider()
    st.markdown("<p style='color:#6b7db3;font-size:0.7rem;font-family:Space Mono;'>DATA SOURCE: Yahoo Finance<br>REFRESH: Every 5 minutes</p>", unsafe_allow_html=True)

# ─── Load Data ───────────────────────────────────────────────────────────────
with st.spinner(f"Loading {selected_ticker} data..."):
    df_main = load_stock_data(selected_ticker, period_days)
    df_main = add_indicators(df_main)

all_tickers = [selected_ticker] + compare_tickers
all_data = {selected_ticker: df_main}
for t in compare_tickers:
    d = load_stock_data(t, period_days)
    all_data[t] = add_indicators(d)

# ─── Header ──────────────────────────────────────────────────────────────────
current_price = df_main['Close'].iloc[-1]
prev_price    = df_main['Close'].iloc[-2]
day_change    = ((current_price - prev_price) / prev_price) * 100
day_change_abs = current_price - prev_price
color_class   = "metric-positive" if day_change >= 0 else "metric-negative"
arrow         = "▲" if day_change >= 0 else "▼"

st.markdown(f"""
<div style='display:flex; align-items:baseline; gap:1.5rem; margin-bottom:1.5rem;'>
    <h1 style='font-family:Space Mono,monospace; font-size:2rem; margin:0; color:#e0e8ff;'>
        {selected_ticker}
    </h1>
    <span style='color:#6b7db3; font-size:1rem;'>{FINTECH_TICKERS[selected_ticker]}</span>
    <span style='font-family:Space Mono,monospace; font-size:1.8rem; color:#e0e8ff; margin-left:auto;'>
        ${current_price:.2f}
    </span>
    <span class='{color_class}' style='font-family:Space Mono,monospace; font-size:1.1rem;'>
        {arrow} {abs(day_change_abs):.2f} ({abs(day_change):.2f}%)
    </span>
</div>
""", unsafe_allow_html=True)

# ─── Key Metrics Row ─────────────────────────────────────────────────────────
ann_ret, ann_vol, sharpe, var95, max_dd = compute_risk(df_main['Return'])

c1, c2, c3, c4, c5, c6 = st.columns(6)

def metric_card(col, label, value, color_class="metric-neutral"):
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value {color_class}'>{value}</div>
    </div>""", unsafe_allow_html=True)

metric_card(c1, "Current Price", f"${current_price:.2f}")
metric_card(c2, "Ann. Return", f"{ann_ret:+.1f}%",
            "metric-positive" if ann_ret > 0 else "metric-negative")
metric_card(c3, "Volatility", f"{ann_vol:.1f}%")
metric_card(c4, "Sharpe Ratio", f"{sharpe:.2f}",
            "metric-positive" if sharpe > 1 else ("metric-neutral" if sharpe > 0 else "metric-negative"))
metric_card(c5, "VaR (95%)", f"{var95:.2f}%", "metric-negative")
metric_card(c6, "Max Drawdown", f"{max_dd:.1f}%", "metric-negative")

st.divider()

# ─── Main Chart ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Price Chart & Technical Indicators</div>", unsafe_allow_html=True)

rows = 3 if show_volume else 2
heights = [0.6, 0.2, 0.2] if show_volume else [0.7, 0.3]
subtitles = ['Candlestick', 'RSI', 'Volume'] if show_volume else ['Candlestick', 'RSI']
if not show_volume:
    subtitles = ['Candlestick', 'RSI']

fig_main = make_subplots(
    rows=rows, cols=1, shared_xaxes=True,
    row_heights=heights,
    vertical_spacing=0.03,
    subplot_titles=subtitles
)

# Candlestick
fig_main.add_trace(go.Candlestick(
    x=df_main.index, open=df_main['Open'], high=df_main['High'],
    low=df_main['Low'], close=df_main['Close'],
    name='OHLC', increasing_line_color='#00e676', decreasing_line_color='#ff5252'
), row=1, col=1)

# Bollinger Bands
if show_bb:
    fig_main.add_trace(go.Scatter(
        x=df_main.index, y=df_main['BB_Up'], name='BB Upper',
        line=dict(color='rgba(255,215,64,0.5)', dash='dash', width=1), showlegend=False
    ), row=1, col=1)
    fig_main.add_trace(go.Scatter(
        x=df_main.index, y=df_main['BB_Lo'], name='BB Lower',
        line=dict(color='rgba(255,215,64,0.5)', dash='dash', width=1),
        fill='tonexty', fillcolor='rgba(255,215,64,0.05)', showlegend=False
    ), row=1, col=1)

# SMAs
if show_sma:
    fig_main.add_trace(go.Scatter(x=df_main.index, y=df_main['SMA_20'],
        name='SMA 20', line=dict(color='#00e5ff', width=1.2)), row=1, col=1)
    fig_main.add_trace(go.Scatter(x=df_main.index, y=df_main['SMA_50'],
        name='SMA 50', line=dict(color='#ea80fc', width=1.2)), row=1, col=1)

# RSI
rsi_row = 2
fig_main.add_trace(go.Scatter(x=df_main.index, y=df_main['RSI'],
    name='RSI', line=dict(color='#ffd740', width=1.5)), row=rsi_row, col=1)
fig_main.add_hline(y=70, line_dash='dash', line_color='#ff5252',
                    opacity=0.7, annotation_text='Overbought', row=rsi_row, col=1)
fig_main.add_hline(y=30, line_dash='dash', line_color='#00e676',
                    opacity=0.7, annotation_text='Oversold', row=rsi_row, col=1)
fig_main.update_yaxes(range=[0, 100], row=rsi_row, col=1)

# Volume
if show_volume:
    colors_vol = ['#00e676' if r >= 0 else '#ff5252' for r in df_main['Return']]
    fig_main.add_trace(go.Bar(x=df_main.index, y=df_main['Volume'],
        marker_color=colors_vol, name='Volume', showlegend=False), row=3, col=1)

fig_main.update_layout(
    **PLOT_THEME, height=650,
    xaxis_rangeslider_visible=False,
    legend=dict(orientation='h', y=1.02, x=0, font=dict(size=11))
)
st.plotly_chart(fig_main, use_container_width=True)

# ─── MACD ───────────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>MACD — Momentum Indicator</div>", unsafe_allow_html=True)

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df_main.index, y=df_main['MACD'],
    name='MACD', line=dict(color='#00e5ff', width=1.5)))
fig_macd.add_trace(go.Scatter(x=df_main.index, y=df_main['MACD_Sig'],
    name='Signal', line=dict(color='#ff6b6b', width=1.5)))
macd_hist = df_main['MACD'] - df_main['MACD_Sig']
fig_macd.add_trace(go.Bar(x=df_main.index, y=macd_hist,
    name='Histogram',
    marker_color=['#00e676' if v > 0 else '#ff5252' for v in macd_hist]))
fig_macd.add_hline(y=0, line_color='white', opacity=0.2)
fig_macd.update_layout(**PLOT_THEME, height=280)
st.plotly_chart(fig_macd, use_container_width=True)

st.divider()

# ─── Comparison Section ──────────────────────────────────────────────────────
if compare_tickers:
    st.markdown("<div class='section-title'>Multi-Stock Comparison</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Normalized Performance", "⚖️ Risk vs Return", "🔥 Correlation"])

    with tab1:
        fig_norm = go.Figure()
        for i, ticker in enumerate(all_tickers):
            closes = all_data[ticker]['Close'].dropna()
            norm   = (closes / closes.iloc[0]) * 100
            fig_norm.add_trace(go.Scatter(
                x=norm.index, y=norm,
                name=ticker, line=dict(color=COLORS[i % len(COLORS)], width=2)
            ))
        fig_norm.add_hline(y=100, line_dash='dash', line_color='white', opacity=0.2)
        fig_norm.update_layout(**PLOT_THEME, height=420,
                                yaxis_title='Indexed Price (Base=100)',
                                hovermode='x unified')
        st.plotly_chart(fig_norm, use_container_width=True)

    with tab2:
        risk_rows = []
        for ticker in all_tickers:
            r = all_data[ticker]['Return'].dropna()
            ann_r, ann_v, sh, v95, mdd = compute_risk(r)
            risk_rows.append({'Ticker': ticker, 'Ann. Return (%)': ann_r,
                               'Ann. Volatility (%)': ann_v, 'Sharpe': sh})
        risk_comp = pd.DataFrame(risk_rows)

        fig_rv = go.Figure()
        for i, row in risk_comp.iterrows():
            fig_rv.add_trace(go.Scatter(
                x=[row['Ann. Volatility (%)']],
                y=[row['Ann. Return (%)']],
                mode='markers+text',
                marker=dict(size=20, color=COLORS[i % len(COLORS)]),
                text=[row['Ticker']], textposition='top center',
                name=row['Ticker']
            ))
        fig_rv.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.2)
        fig_rv.update_layout(**PLOT_THEME, height=420, showlegend=False,
                              xaxis_title='Annualized Volatility (%)',
                              yaxis_title='Annualized Return (%)')
        st.plotly_chart(fig_rv, use_container_width=True)

    with tab3:
        all_returns = pd.DataFrame({
            t: all_data[t]['Return'] for t in all_tickers
        }).dropna()
        corr = all_returns.corr()

        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale='RdBu_r', zmin=-1, zmax=1,
            text=corr.round(2).values, texttemplate='%{text}',
            textfont={'size': 13}
        ))
        fig_corr.update_layout(**PLOT_THEME, height=420)
        st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

# ─── Risk Metrics Table ──────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Risk Metrics Summary</div>", unsafe_allow_html=True)

summary_rows = []
for ticker in all_tickers:
    r = all_data[ticker]['Return'].dropna()
    ann_r, ann_v, sh, v95, mdd = compute_risk(r)
    last = all_data[ticker]['Close'].iloc[-1]
    summary_rows.append({
        'Ticker': ticker,
        'Company': FINTECH_TICKERS[ticker],
        'Price ($)': round(last, 2),
        'Ann. Return (%)': round(ann_r, 2),
        'Ann. Volatility (%)': round(ann_v, 2),
        'Sharpe Ratio': round(sh, 3),
        'VaR 95% (%)': round(v95, 3),
        'Max Drawdown (%)': round(mdd, 2)
    })

summary_df = pd.DataFrame(summary_rows).set_index('Ticker')

def color_value(val, col):
    if col in ['Ann. Return (%)', 'Sharpe Ratio']:
        return 'color: #00e676' if val > 0 else 'color: #ff5252'
    if col in ['Ann. Volatility (%)', 'VaR 95% (%)', 'Max Drawdown (%)']:
        return 'color: #ff8a80'
    return ''

st.dataframe(
    summary_df.style.apply(
        lambda col: [color_value(v, col.name) for v in col], axis=0
    ).format({
        'Price ($)': '${:.2f}',
        'Ann. Return (%)': '{:+.2f}%',
        'Ann. Volatility (%)': '{:.2f}%',
        'Sharpe Ratio': '{:.3f}',
        'VaR 95% (%)': '{:.3f}%',
        'Max Drawdown (%)': '{:.2f}%'
    }),
    use_container_width=True,
    height=200
)

# ─── Optimization Techniques ─────────────────────────────────────────────────
st.divider()
st.markdown("<div class='section-title'>Optimization Techniques</div>", unsafe_allow_html=True)

opt_tab1, opt_tab2 = st.tabs([
    "📊 Portfolio Optimization (Markowitz)",
    "📉 GBM Price Simulation (Euler Method)"
])

# ── Tab 1: Portfolio Optimization ────────────────────────────────────────────
with opt_tab1:
    if len(all_tickers) < 2:
        st.info("Select at least 2 stocks in the sidebar to run portfolio optimization.")
    else:
        all_returns_df = pd.DataFrame(
            {t: all_data[t]['Return'] for t in all_tickers}
        ).dropna()

        if len(all_returns_df) < 60:
            st.warning("Not enough overlapping data to run optimization.")
        else:
            with st.spinner("Running Markowitz optimization…"):
                opt = compute_portfolio_optimization(all_returns_df.to_json())

            tickers_opt = opt['tickers']
            ms          = opt['max_sharpe']
            mv          = opt['min_var']

            # Efficient frontier chart
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(
                x=opt['mc']['v'], y=opt['mc']['r'],
                mode='markers',
                marker=dict(size=3, opacity=0.4,
                            color=opt['mc']['s'], colorscale='Viridis',
                            colorbar=dict(title='Sharpe', x=1.02)),
                name='Random Portfolios',
                hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
            ))
            if opt['ef']['v']:
                fig_ef.add_trace(go.Scatter(
                    x=opt['ef']['v'], y=opt['ef']['r'],
                    mode='lines', line=dict(color='white', width=3),
                    name='Efficient Frontier'
                ))
            fig_ef.add_trace(go.Scatter(
                x=[ms['v']], y=[ms['r']],
                mode='markers+text',
                marker=dict(size=18, color='#ffd740', symbol='star'),
                text=[f"Max Sharpe ({ms['s']:.2f})"], textposition='top right',
                name='Max Sharpe'
            ))
            fig_ef.add_trace(go.Scatter(
                x=[mv['v']], y=[mv['r']],
                mode='markers+text',
                marker=dict(size=18, color='#00e676', symbol='diamond'),
                text=['Min Variance'], textposition='top right',
                name='Min Variance'
            ))
            fig_ef.update_layout(
                **PLOT_THEME, height=480,
                title='Efficient Frontier — Mean-Variance Optimization',
                xaxis_title='Annualized Volatility (%)',
                yaxis_title='Annualized Return (%)'
            )
            st.plotly_chart(fig_ef, use_container_width=True)

            # Optimal weights side by side
            col_ms, col_mv = st.columns(2)
            with col_ms:
                fig_w1 = go.Figure(go.Bar(
                    x=tickers_opt, y=[w * 100 for w in ms['w']],
                    marker_color=COLORS[:len(tickers_opt)]
                ))
                fig_w1.update_layout(
                    **PLOT_THEME, height=300,
                    title=f"Max Sharpe Weights  (Sharpe = {ms['s']:.2f})",
                    yaxis_title='Allocation (%)'
                )
                st.plotly_chart(fig_w1, use_container_width=True)
            with col_mv:
                fig_w2 = go.Figure(go.Bar(
                    x=tickers_opt, y=[w * 100 for w in mv['w']],
                    marker_color=COLORS[:len(tickers_opt)]
                ))
                fig_w2.update_layout(
                    **PLOT_THEME, height=300,
                    title=f"Min Variance Weights  (Vol = {mv['v']:.1f}%)",
                    yaxis_title='Allocation (%)'
                )
                st.plotly_chart(fig_w2, use_container_width=True)

# ── Tab 2: GBM Price Simulation ───────────────────────────────────────────────
with opt_tab2:
    ret_series = df_main['Return'].dropna()
    mu_gbm     = float(ret_series.mean())
    sigma_gbm  = float(ret_series.std())
    S0_gbm     = float(df_main['Close'].iloc[-1])

    with st.spinner(f"Simulating {selected_ticker} price paths…"):
        paths = run_gbm_simulation(mu_gbm, sigma_gbm, S0_gbm)

    t_axis = np.arange(paths.shape[0])
    p5  = np.percentile(paths,  5, axis=1)
    p25 = np.percentile(paths, 25, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p75 = np.percentile(paths, 75, axis=1)
    p95 = np.percentile(paths, 95, axis=1)

    fig_gbm = go.Figure()
    for i in range(min(100, paths.shape[1])):
        fig_gbm.add_trace(go.Scatter(
            x=t_axis, y=paths[:, i], mode='lines',
            line=dict(color='rgba(0,229,255,0.05)', width=1),
            showlegend=False, hoverinfo='skip'
        ))
    fig_gbm.add_trace(go.Scatter(x=t_axis, y=p95, mode='lines',
        line=dict(color='rgba(255,215,64,0)'), showlegend=False))
    fig_gbm.add_trace(go.Scatter(x=t_axis, y=p5, mode='lines',
        fill='tonexty', fillcolor='rgba(255,215,64,0.12)',
        line=dict(color='rgba(255,215,64,0)'), name='90% Confidence Band'))
    fig_gbm.add_trace(go.Scatter(x=t_axis, y=p75, mode='lines',
        line=dict(color='rgba(100,255,150,0)'), showlegend=False))
    fig_gbm.add_trace(go.Scatter(x=t_axis, y=p25, mode='lines',
        fill='tonexty', fillcolor='rgba(100,255,150,0.18)',
        line=dict(color='rgba(100,255,150,0)'), name='50% Confidence Band'))
    fig_gbm.add_trace(go.Scatter(x=t_axis, y=p50, mode='lines',
        line=dict(color='#ffd740', width=2.5), name='Median Path'))
    fig_gbm.add_hline(y=S0_gbm, line_dash='dash', line_color='white',
                      opacity=0.35, annotation_text=f'Today: ${S0_gbm:.2f}')
    fig_gbm.update_layout(
        **PLOT_THEME, height=500,
        title=(f'{selected_ticker} — GBM Monte Carlo Simulation'
               f' (Euler-Maruyama, {paths.shape[1]} paths, dt=1/252)'),
        xaxis_title='Trading Days Forward',
        yaxis_title='Simulated Price ($)'
    )
    st.plotly_chart(fig_gbm, use_container_width=True)

    gc1, gc2, gc3, gc4 = st.columns(4)
    metric_card(gc1, "Starting Price",    f"${S0_gbm:.2f}")
    metric_card(gc2, "Annual Drift (μ)",  f"{mu_gbm*252*100:+.1f}%",
                "metric-positive" if mu_gbm > 0 else "metric-negative")
    metric_card(gc3, "Annual Vol (σ)",    f"{sigma_gbm*np.sqrt(252)*100:.1f}%")
    metric_card(gc4, "1Y Range (90%CI)",
                f"${p5[-1]:.0f} – ${p95[-1]:.0f}")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#2a3050; font-family:Space Mono,monospace; font-size:0.7rem; padding:1rem;'>
    FinTech Stock Market Intelligence Dashboard · Data via Yahoo Finance · For Educational Purposes Only
</div>
""", unsafe_allow_html=True)
