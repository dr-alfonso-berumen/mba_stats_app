import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import textwrap

# -------------------------------------------------------------------
# Pepperdine colors (web palette)
# -------------------------------------------------------------------
PEP_BLUE = "#00205c"
PEP_ORANGE = "#c25700"

# -------------------------------------------------------------------
# Helper: build prompt for Gemini (Core, copy-paste)
# -------------------------------------------------------------------
def build_prompt(analysis_type: str, context: dict, results: dict) -> str:
    base = f"""
    You are helping a business analytics MBA student interpret a {analysis_type} result.

    COURSE CONTEXT:
    - Course: Business Analytics / Applied Statistics (MBA)
    - Goal: Explain the statistical result in clear, non-technical business language.

    BUSINESS CONTEXT:
    {context.get('description', '(No extra context provided.)')}

    VARIABLES:
    {context.get('variables', '')}

    NUMERICAL RESULTS:
    {textwrap.indent(results.get('summary', ''), '    ')}

    TASK:
    1. Explain what these results mean for a business stakeholder who is not technical.
    2. Clearly state whether the evidence is strong enough to support the conclusion (e.g., reject or fail to reject the null at alpha = 0.05 if applicable).
    3. Connect the statistics back to the business question (2–3 sentences).
    4. Keep the explanation under 250 words.
    """
    return textwrap.dedent(base).strip()


# -------------------------------------------------------------------
# Helper CI / stats functions
# -------------------------------------------------------------------
def ci_mean_t(data, alpha=0.05):
    data = pd.Series(data).dropna()
    n = len(data)
    xbar = data.mean()
    s = data.std(ddof=1)
    se = s / np.sqrt(n)
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    lower = xbar - t_crit * se
    upper = xbar + t_crit * se
    return xbar, (lower, upper), se, df


def ci_diff_means_welch(x1, x2, alpha=0.05):
    x1 = pd.Series(x1).dropna()
    x2 = pd.Series(x2).dropna()
    n1, n2 = len(x1), len(x2)
    m1, m2 = x1.mean(), x2.mean()
    s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
    se = np.sqrt(s1**2 / n1 + s2**2 / n2)
    df = (s1**2 / n1 + s2**2 / n2) ** 2 / (
        (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
    )
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    diff = m1 - m2
    lower = diff - t_crit * se
    upper = diff + t_crit * se
    return diff, (lower, upper), se, df


def ci_proportion(phat, n, alpha=0.05):
    z_crit = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(phat * (1 - phat) / n)
    lower = phat - z_crit * se
    upper = phat + z_crit * se
    return (lower, upper), se


def regression_prediction_with_intervals(model, new_df, alpha=0.05):
    pred_res = model.get_prediction(new_df)
    frame = pred_res.summary_frame(alpha=alpha)
    return frame  # mean, mean_ci_lower, mean_ci_upper, obs_ci_lower, obs_ci_upper


def compute_vif(data, predictors, var_types):
    # Build X with numeric predictors + dummies for categoricals
    X_parts = []
    for p in predictors:
        if var_types[p] == "Categorical":
            dummies = pd.get_dummies(data[p], prefix=p, drop_first=True)
            X_parts.append(dummies)
        else:
            X_parts.append(data[[p]])
    if not X_parts:
        return pd.DataFrame(columns=["Variable", "VIF"])
    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X, has_constant="add")

    vif_rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif_val = variance_inflation_factor(X.values, i)
        vif_rows.append({"Variable": col, "VIF": vif_val})
    return pd.DataFrame(vif_rows)


def sanitize_header(row) -> list:
    """
    Take a pandas Series (one row) and return a list of
    unique, non-empty string column names suitable for st.data_editor.
    """
    cols = []
    seen = {}
    for val in row:
        if pd.isna(val) or str(val).strip() == "":
            base = "col"
        else:
            base = str(val).strip()
        count = seen.get(base, 0)
        if count == 0:
            name = base
        else:
            name = f"{base}_{count+1}"
        seen[base] = count + 1
        cols.append(name)
    return cols


def interpret_pvalue(p, alpha):
    if p < alpha:
        return f"At α = {alpha}, the result is statistically significant (reject H₀)."
    else:
        return f"At α = {alpha}, the result is not statistically significant (fail to reject H₀)."


def infer_var_type(series: pd.Series) -> str:
    """
    Heuristic to infer 'Quantitative' vs 'Categorical'.

    - Tries to coerce to numeric.
    - If mostly non-numeric -> Categorical.
    - If numeric with few distinct values -> Categorical (e.g., 0/1, 1-5 Likert).
    - Otherwise -> Quantitative.
    """
    s = series.dropna()

    if s.empty:
        return "Categorical"

    num = pd.to_numeric(s, errors="coerce")
    non_numeric_frac = num.isna().mean()

    if non_numeric_frac > 0.5:
        return "Categorical"

    num = num.dropna()
    nunique = num.nunique()

    if nunique <= 10:
        return "Categorical"

    return "Quantitative"


# -------------------------------------------------------------------
# Streamlit layout
# -------------------------------------------------------------------
st.set_page_config(page_title="MBA Stats Lab", layout="wide")
st.title("MBA Stats Lab: Spreadsheet + Drop-Down Analysis")

# 1. Upload data (no headers yet)
st.sidebar.header("Step 1: Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

if uploaded_file is None:
    st.info("Upload a spreadsheet to get started.")
    st.stop()

# Read raw file with NO header so user can choose header row and data start row
if uploaded_file.name.endswith(".csv"):
    raw_df = pd.read_csv(uploaded_file, header=None)
else:
    raw_df = pd.read_excel(uploaded_file, header=None)

st.subheader("Raw file preview (first 15 rows)")
st.caption("Row numbers on the left will help you choose the header and data start rows.")
st.dataframe(raw_df.head(15))

# Header & data row settings
st.sidebar.subheader("Header & Data Row Settings")
max_row = len(raw_df)

if max_row < 2:
    st.error("The file must have at least 2 rows (one for header, one for data).")
    st.stop()

max_header_row = max_row - 1

header_row = st.sidebar.number_input(
    "Row that contains column names (1 = first row)",
    min_value=1,
    max_value=max_header_row,
    value=1,
    step=1,
)

min_data_start = header_row + 1
data_start_row = st.sidebar.number_input(
    "Row where data starts (1-based)",
    min_value=min_data_start,
    max_value=max_row,
    value=min_data_start,
    step=1,
)

# Convert to 0-based indexes
header_idx = int(header_row - 1)
data_start_idx = int(data_start_row - 1)

# Build clean df: use chosen header row as column names, drop rows above data_start
raw_header_row = raw_df.iloc[header_idx]
columns = sanitize_header(raw_header_row)

df = raw_df.iloc[data_start_idx:].copy()
df.columns = columns
df.reset_index(drop=True, inplace=True)

st.subheader("Processed data (this is what the analysis will use)")
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# Allow students to download current edited data (for saving work)
csv_bytes = edited_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download current data as CSV (for saving & re-uploading later)",
    csv_bytes,
    "mba_stats_data.csv",
    "text/csv",
)

# 2. Choose columns + variable types
st.sidebar.header("Step 2: Columns & Variable Types")

columns_for_analysis = st.sidebar.multiselect(
    "Columns to include in analysis (uncheck to 'delete' from analysis):",
    list(edited_df.columns),
    default=list(edited_df.columns),
)

if not columns_for_analysis:
    st.warning("Select at least one column to include in the analysis.")
    st.stop()

data_df = edited_df[columns_for_analysis].copy()

st.sidebar.caption("Variable types are auto-detected; you can override them at any time.")

var_types = {}
for col in data_df.columns:
    inferred = infer_var_type(data_df[col])
    var_types[col] = st.sidebar.selectbox(
        f"{col} (variable type) [auto: {inferred}]",
        ["Quantitative", "Categorical"],
        index=0 if inferred == "Quantitative" else 1,
        key=f"vartype_{col}",
    )

quant_vars = [c for c in data_df.columns if var_types[c] == "Quantitative"]
cat_vars = [c for c in data_df.columns if var_types[c] == "Categorical"]

# 2b. Data explorer & filters
with st.expander("Explore & filter data"):
    st.write("### Full data (after column selection)")
    st.dataframe(data_df, use_container_width=True, height=300)

    st.write("### Optional filters")
    filtered_df = data_df.copy()

    for col in data_df.columns:
        col_data = filtered_df[col]
        if col_data.dropna().empty:
            continue

        if var_types[col] == "Quantitative":
            # Only apply slider if column is numeric after coercion
            numeric_col = pd.to_numeric(col_data, errors="coerce")
            if numeric_col.dropna().empty:
                continue
            col_min = float(numeric_col.min())
            col_max = float(numeric_col.max())
            if col_min == col_max:
                continue  # no range to filter on
            f_min, f_max = st.slider(
                f"{col} range filter",
                min_value=col_min,
                max_value=col_max,
                value=(col_min, col_max),
            )
            mask = numeric_col.between(f_min, f_max)
            filtered_df = filtered_df[mask]

        else:
            categories = sorted(col_data.dropna().unique().astype(str))
            if not categories:
                continue
            selected = st.multiselect(
                f"{col} values to include",
                categories,
                default=categories,
            )
            filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]

    st.write("### Filtered data preview")
    st.dataframe(filtered_df, use_container_width=True, height=300)

# Use filtered data for all subsequent analyses
data_df = filtered_df.copy()

st.sidebar.markdown("---")

# 3. Main analysis choice
analysis_choice = st.sidebar.selectbox(
    "Step 3: What would you like to do?",
    [
        "Descriptive statistics",
        "Normal probabilities & critical values",
        "One-sample inference (means/proportions)",
        "Paired or two-sample comparisons",
        "ANOVA (compare 3+ group means)",
        "Categorical association (chi-square)",
        "Correlation & regression"
    ]
)

st.sidebar.markdown("---")

# -------------------------------------------------------------------
# A. Descriptive statistics
# -------------------------------------------------------------------
if analysis_choice == "Descriptive statistics":
    st.header("Descriptive Statistics")

    quant_vars = [c for c in data_df.columns if var_types[c] == "Quantitative"]
    cat_vars = [c for c in data_df.columns if var_types[c] == "Categorical"]

    if quant_vars:
        with st.expander("Show descriptive statistics for ALL quantitative variables"):
            num_df = data_df[quant_vars].apply(pd.to_numeric, errors="coerce")
            desc_all = num_df.describe().T
            desc_all["variance"] = num_df.var()
            desc_all["skewness"] = num_df.skew()
            desc_all["kurtosis"] = num_df.kurtosis()
            st.dataframe(desc_all)

    target = st.radio("Focus on:", ["One variable", "Two categorical variables (crosstab)"])

    if target == "One variable":
        var = st.selectbox("Pick a variable", data_df.columns)
        series_raw = data_df[var]

        if var_types[var] == "Quantitative":
            series = pd.to_numeric(series_raw, errors="coerce").dropna()
            n_total = len(series_raw)
            n_numeric = len(series)
            if n_numeric < n_total:
                st.caption(
                    f"Note: {n_total - n_numeric} non-numeric values were ignored for numeric summaries and plots."
                )

            if len(series) == 0:
                st.warning("No numeric data available for this variable.")
            else:
                st.write("### Detailed summary statistics")
                desc = series.describe()
                extra = pd.Series(
                    {
                        "variance": series.var(),
                        "skewness": series.skew(),
                        "kurtosis": series.kurtosis(),
                    }
                )
                full_desc = pd.concat([desc, extra]).to_frame("Value")
                st.dataframe(full_desc)

                st.write("### Histogram")
                fig, ax = plt.subplots()
                ax.hist(series, bins="auto", color=PEP_BLUE, edgecolor="black")
                ax.set_xlabel(var)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

                st.write("### Boxplot")
                fig, ax = plt.subplots()
                bp = ax.boxplot(series, vert=True, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set(facecolor=PEP_ORANGE, edgecolor=PEP_BLUE)
                for whisker in bp['whiskers']:
                    whisker.set(color=PEP_BLUE)
                for cap in bp['caps']:
                    cap.set(color=PEP_BLUE)
                for median in bp['medians']:
                    median.set(color=PEP_BLUE)
                ax.set_ylabel(var)
                st.pyplot(fig)

                st.write("### QQ plot (Normality check)")
                if len(series) >= 3:
                    fig = sm.ProbPlot(series, dist=stats.norm, fit=True).qqplot(line="45")
                    st.pyplot(fig)
                else:
                    st.info("Not enough observations to draw a QQ plot (need at least 3).")

        else:
            series = series_raw.dropna()
            st.write("### Frequency table (counts)")
            freq = series.value_counts()
            st.dataframe(freq.to_frame("Count"))

            st.write("### Proportions")
            st.dataframe((freq / freq.sum()).to_frame("Proportion"))

            st.write("### Bar chart")
            fig, ax = plt.subplots()
            ax.bar(freq.index.astype(str), freq.values, color=PEP_ORANGE, edgecolor=PEP_BLUE)
            ax.set_xlabel(var)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    else:
        if len(cat_vars) < 2:
            st.warning("You need at least two categorical variables.")
        else:
            c1 = st.selectbox("Row variable", cat_vars)
            c2 = st.selectbox("Column variable", [c for c in cat_vars if c != c1])
            data = data_df[[c1, c2]].dropna()
            ct = pd.crosstab(data[c1], data[c2])

            st.write("### Crosstab (counts)")
            st.dataframe(ct)

            row_prop = ct.div(ct.sum(axis=1), axis=0)
            st.write("### Row percentages")
            st.dataframe((row_prop * 100).round(2))

            st.write("### Clustered bar chart (row % by column category)")
            row_categories = row_prop.index.astype(str)
            col_categories = row_prop.columns.astype(str)
            x = np.arange(len(row_categories))
            total_bars = len(col_categories)
            width = 0.8 / max(total_bars, 1)

            fig, ax = plt.subplots()
            for i, col in enumerate(col_categories):
                offsets = x + (i - (total_bars - 1) / 2) * width
                color = PEP_BLUE if i % 2 == 0 else PEP_ORANGE
                ax.bar(
                    offsets,
                    (row_prop[col].values * 100),
                    width=width,
                    label=str(col),
                    color=color,
                    edgecolor="white",
                )

            ax.set_xticks(x)
            ax.set_xticklabels(row_categories, rotation=45)
            ax.set_ylabel("Row percentage (%)")
            ax.set_xlabel(c1)
            ax.legend(title=c2)
            st.pyplot(fig)

# -------------------------------------------------------------------
# B. Normal probabilities & critical values (with curve + shaded area)
# -------------------------------------------------------------------
elif analysis_choice == "Normal probabilities & critical values":
    st.header("Normal Probabilities & Critical Values")

    col1, col2 = st.columns(2)
    with col1:
        mu = st.number_input("Mean (μ)", value=0.0)
    with col2:
        sigma = st.number_input("Standard deviation (σ)", value=1.0, min_value=0.0001)

    mode = st.radio("What would you like to compute?", ["Probability given X", "X value given probability"])

    def plot_normal_shaded(mu, sigma, tail_type, x=None, a=None, b=None, p=None, from_prob=False):
        xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
        ys = norm.pdf(xs, loc=mu, scale=sigma)

        fig, ax = plt.subplots()
        ax.plot(xs, ys, color=PEP_BLUE)
        ax.set_xlabel("X")
        ax.set_ylabel("Density")
        ax.set_title("Normal distribution with shaded probability region")

        mask = np.zeros_like(xs, dtype=bool)

        if not from_prob:
            if tail_type == "P(X ≤ x)":
                mask = xs <= x
            elif tail_type == "P(X ≥ x)":
                mask = xs >= x
            else:
                mask = (xs >= a) & (xs <= b)
        else:
            if tail_type.startswith("Lower"):
                mask = xs <= x
            else:
                mask = xs >= x

        ax.fill_between(xs[mask], ys[mask], alpha=0.3, color=PEP_ORANGE)
        return fig

    if mode == "Probability given X":
        prob_type = st.radio("Tail type", ["P(X ≤ x)", "P(X ≥ x)", "P(a ≤ X ≤ b)"])
        if prob_type in ["P(X ≤ x)", "P(X ≥ x)"]:
            x = st.number_input("x value", value=0.0)
        else:
            a = st.number_input("Lower bound a", value=-1.0)
            b = st.number_input("Upper bound b", value=1.0)

        if st.button("Compute probability"):
            if prob_type == "P(X ≤ x)":
                p = norm.cdf(x, loc=mu, scale=sigma)
                st.write(f"P(X ≤ {x}) = {p:.4f}")
                fig = plot_normal_shaded(mu, sigma, prob_type, x=x)
            elif prob_type == "P(X ≥ x)":
                p = 1 - norm.cdf(x, loc=mu, scale=sigma)
                st.write(f"P(X ≥ {x}) = {p:.4f}")
                fig = plot_normal_shaded(mu, sigma, prob_type, x=x)
            else:
                p = norm.cdf(b, loc=mu, scale=sigma) - norm.cdf(a, loc=mu, scale=sigma)
                st.write(f"P({a} ≤ X ≤ {b}) = {p:.4f}")
                fig = plot_normal_shaded(mu, sigma, prob_type, a=a, b=b)

            st.pyplot(fig)

            results = {"summary": f"mu={mu}, sigma={sigma}, type={prob_type}, probability={p:.4f}"}
            context = {
                "description": "Evaluate a normal probability for a business-related metric.",
                "variables": f"X ~ Normal({mu}, {sigma}^2)."
            }
            prompt = build_prompt("normal probability calculation", context, results)
            st.subheader("Optional: AI Explanation Prompt (Gemini)")
            st.code(prompt, language="markdown")

    else:
        prob_type = st.radio("Tail type", ["Lower tail (P(X ≤ x))", "Upper tail (P(X ≥ x))"])
        p = st.number_input("Probability (0 < p < 1)", value=0.95, min_value=0.0001, max_value=0.9999)

        if st.button("Compute x value"):
            if prob_type == "Lower tail (P(X ≤ x))":
                x = norm.ppf(p, loc=mu, scale=sigma)
                st.write(f"x such that P(X ≤ x) = {p} is x = {x:.4f}")
                fig = plot_normal_shaded(mu, sigma, prob_type, x=x, p=p, from_prob=True)
            else:
                x = norm.ppf(1 - p, loc=mu, scale=sigma)
                st.write(f"x such that P(X ≥ x) = {p} is x = {x:.4f}")
                fig = plot_normal_shaded(mu, sigma, prob_type, x=x, p=p, from_prob=True)

            st.pyplot(fig)

            results = {"summary": f"mu={mu}, sigma={sigma}, tail={prob_type}, p={p}, x={x:.4f}"}
            context = {
                "description": "Find a critical value on a normal distribution for a business threshold.",
                "variables": f"X ~ Normal({mu}, {sigma}^2)."
            }
            prompt = build_prompt("normal critical value", context, results)
            st.subheader("Optional: AI Explanation Prompt (Gemini)")
            st.code(prompt, language="markdown")

# -------------------------------------------------------------------
# C. One-sample inference (means/proportions)
# -------------------------------------------------------------------
elif analysis_choice == "One-sample inference (means/proportions)":
    st.header("One-Sample Inference (Means / Proportions)")

    test_kind = st.radio("Choose test type", ["Mean (z or t)", "Proportion (z)"])

    if test_kind == "Mean (z or t)":
        quant_vars = [c for c in data_df.columns if var_types[c] == "Quantitative"]
        if not quant_vars:
            st.warning("You need at least one quantitative variable.")
        else:
            dv = st.selectbox("Outcome variable (numeric)", quant_vars)
            data = pd.to_numeric(data_df[dv], errors="coerce").dropna()
            test_value = st.number_input(
                "Null hypothesis mean (H₀: μ = ?)",
                value=float(data.mean()) if len(data) > 0 else 0.0
            )
            sigma_known = st.checkbox("Use z-test (σ known)? Otherwise t-test.", value=False)
            alpha = st.number_input("Significance level α", value=0.05,
                                    min_value=0.0001, max_value=0.5)

            if st.button("Run one-sample test"):
                n = len(data)
                if n == 0:
                    st.error("No numeric data available for this variable.")
                else:
                    xbar = data.mean()
                    s = data.std(ddof=1)

                    if sigma_known:
                        sigma = st.number_input("Population σ (known)", value=float(s))
                        se = sigma / np.sqrt(n)
                        z_stat = (xbar - test_value) / se
                        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                        z_crit = stats.norm.ppf(1 - alpha / 2)
                        ci_lower = xbar - z_crit * se
                        ci_upper = xbar + z_crit * se

                        out = {
                            "Test": "One-sample z-test (mean)",
                            "n": n,
                            "x̄": round(xbar, 3),
                            "σ": round(sigma, 3),
                            "z": round(z_stat, 3),
                            "p-value": round(p_val, 4),
                            "CI lower": round(ci_lower, 3),
                            "CI upper": round(ci_upper, 3),
                            "α": alpha,
                        }
                        st.write("### Test results (structured)")
                        st.table(pd.DataFrame([out]))
                        msg = interpret_pvalue(p_val, alpha)
                        if p_val < alpha:
                            st.success(msg)
                        else:
                            st.warning(msg)

                        summary = (
                            f"n={n}, x̄={xbar:.3f}, σ={sigma:.3f}, "
                            f"z={z_stat:.3f}, p={p_val:.4f}, "
                            f"{100*(1-alpha):.1f}% CI=({ci_lower:.3f}, {ci_upper:.3f})"
                        )
                        results = {"summary": summary}
                        context = {
                            "description": "Test whether a sample mean differs from a hypothesized population mean using a z-test.",
                            "variables": f"Outcome variable: {dv}. Null mean: {test_value}."
                        }
                        prompt = build_prompt("one-sample z-test for a mean", context, results)

                    else:
                        xbar, (ci_lower, ci_upper), se, df_ = ci_mean_t(data, alpha=alpha)
                        t_stat = (xbar - test_value) / se
                        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_))

                        out = {
                            "Test": "One-sample t-test (mean)",
                            "n": n,
                            "df": df_,
                            "x̄": round(xbar, 3),
                            "s": round(s, 3),
                            "t": round(t_stat, 3),
                            "p-value": round(p_val, 4),
                            "CI lower": round(ci_lower, 3),
                            "CI upper": round(ci_upper, 3),
                            "α": alpha,
                        }
                        st.write("### Test results (structured)")
                        st.table(pd.DataFrame([out]))
                        msg = interpret_pvalue(p_val, alpha)
                        if p_val < alpha:
                            st.success(msg)
                        else:
                            st.warning(msg)

                        summary = (
                            f"n={n}, x̄={xbar:.3f}, s={s:.3f}, t({df_})={t_stat:.3f}, p={p_val:.4f}, "
                            f"{100*(1-alpha):.1f}% CI=({ci_lower:.3f}, {ci_upper:.3f})"
                        )
                        results = {"summary": summary}
                        context = {
                            "description": "Test whether a sample mean differs from a hypothesized population mean using a t-test.",
                            "variables": f"Outcome variable: {dv}. Null mean: {test_value}."
                        }
                        prompt = build_prompt("one-sample t-test for a mean", context, results)

                    st.subheader("Optional: AI Explanation Prompt (Gemini)")
                    st.code(prompt, language="markdown")

    else:  # Proportion (z)
        var = st.selectbox("Binary outcome variable", data_df.columns)
        data = data_df[var].dropna()
        if data.dtype == "bool":
            x = data.astype(int)
        else:
            uniq = data.unique()
            if len(uniq) < 2:
                st.warning("Variable must have at least two distinct values.")
                st.stop()
            mapping = {uniq[0]: 0, uniq[1]: 1}
            x = data.map(mapping)
        p_hat = x.mean()
        n = x.count()
        p0 = st.number_input("Null proportion (H₀: p = ?)", value=0.5)
        alpha = st.number_input("Significance level α", value=0.05,
                                min_value=0.0001, max_value=0.5)

        if st.button("Run one-sample z-test for proportion"):
            se0 = np.sqrt(p0 * (1 - p0) / n)
            z_stat = (p_hat - p0) / se0
            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            (ci_lower, ci_upper), se_hat = ci_proportion(p_hat, n, alpha=alpha)

            out = {
                "Test": "One-sample z-test (proportion)",
                "n": n,
                "p̂": round(p_hat, 3),
                "p0": round(p0, 3),
                "z": round(z_stat, 3),
                "p-value": round(p_val, 4),
                "CI lower": round(ci_lower, 3),
                "CI upper": round(ci_upper, 3),
                "α": alpha,
            }
            st.write("### Test results (structured)")
            st.table(pd.DataFrame([out]))
            msg = interpret_pvalue(p_val, alpha)
            if p_val < alpha:
                st.success(msg)
            else:
                st.warning(msg)

            summary = (
                f"n={n}, p̂={p_hat:.3f}, p0={p0:.3f}, z={z_stat:.3f}, p={p_val:.4f}, "
                f"{100*(1-alpha):.1f}% CI=({ci_lower:.3f}, {ci_upper:.3f})"
            )

            results = {"summary": summary}
            context = {
                "description": "Test whether the true proportion differs from a hypothesized proportion.",
                "variables": f"Binary variable: {var}. Null proportion: {p0}."
            }
            prompt = build_prompt("one-sample z-test for a proportion", context, results)
            st.subheader("Optional: AI Explanation Prompt (Gemini)")
            st.code(prompt, language="markdown")

# -------------------------------------------------------------------
# D. Paired or two-sample comparisons
# -------------------------------------------------------------------
elif analysis_choice == "Paired or two-sample comparisons":
    st.header("Paired or Two-Sample Comparisons")

    mode = st.radio("Choose", ["Paired (matched) t-test", "Independent two-sample t-test"])

    if mode == "Paired (matched) t-test":
        quant_vars = [c for c in data_df.columns if var_types[c] == "Quantitative"]
        if len(quant_vars) < 2:
            st.warning("Need two quantitative variables (e.g., before/after).")
        else:
            v1 = st.selectbox("First measure (e.g., Before)", quant_vars)
            v2 = st.selectbox("Second measure (e.g., After)", [v for v in quant_vars if v != v1])
            data = data_df[[v1, v2]].dropna()
            diff = pd.to_numeric(data[v2], errors="coerce") - pd.to_numeric(data[v1], errors="coerce")
            diff = diff.dropna()
            n = len(diff)
            if n == 0:
                st.error("No numeric paired data available.")
            else:
                dbar = diff.mean()
                s_d = diff.std(ddof=1)
                se = s_d / np.sqrt(n)
                df_ = n - 1
                t_stat = dbar / se
                alpha = st.number_input("Significance level α", value=0.05,
                                        min_value=0.0001, max_value=0.5)
                t_crit = stats.t.ppf(1 - alpha / 2, df_)
                ci_lower = dbar - t_crit * se
                ci_upper = dbar + t_crit * se
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_))

                out = {
                    "Test": "Paired t-test",
                    "n": n,
                    "df": df_,
                    "mean diff": round(dbar, 3),
                    "sd diff": round(s_d, 3),
                    "t": round(t_stat, 3),
                    "p-value": round(p_val, 4),
                    "CI lower": round(ci_lower, 3),
                    "CI upper": round(ci_upper, 3),
                    "α": alpha,
                }
                st.write("### Test results (structured)")
                st.table(pd.DataFrame([out]))
                msg = interpret_pvalue(p_val, alpha)
                if p_val < alpha:
                    st.success(msg)
                else:
                    st.warning(msg)

                summary = (
                    f"n={n}, mean difference={dbar:.3f}, sd_diff={s_d:.3f}, "
                    f"t({df_})={t_stat:.3f}, p={p_val:.4f}, "
                    f"{100*(1-alpha):.1f}% CI=({ci_lower:.3f}, {ci_upper:.3f})"
                )

                results = {"summary": summary}
                context = {
                    "description": "Compare before/after or matched pairs using a paired t-test.",
                    "variables": f"Before: {v1}, After: {v2}."
                }
                prompt = build_prompt("paired t-test", context, results)
                st.subheader("Optional: AI Explanation Prompt (Gemini)")
                st.code(prompt, language="markdown")

    else:
        quant_vars = [c for c in data_df.columns if var_types[c] == "Quantitative"]
        cat_vars = [c for c in data_df.columns if var_types[c] == "Categorical"]
        if not quant_vars or not cat_vars:
            st.warning("Need a quantitative outcome and a categorical grouping variable.")
        else:
            dv = st.selectbox("Outcome (numeric)", quant_vars)
            iv = st.selectbox("Grouping variable (2 groups)", cat_vars)
            data = data_df[[dv, iv]].dropna()
            groups = data[iv].unique()
            if len(groups) != 2:
                st.warning("Grouping variable must have exactly 2 groups.")
            else:
                g1, g2 = groups[0], groups[1]
                x1 = pd.to_numeric(data[data[iv] == g1][dv], errors="coerce").dropna()
                x2 = pd.to_numeric(data[data[iv] == g2][dv], errors="coerce").dropna()
                equal_var = st.checkbox("Assume equal variances", value=True)
                alpha = st.number_input("Significance level α", value=0.05,
                                        min_value=0.0001, max_value=0.5)

                if len(x1) == 0 or len(x2) == 0:
                    st.error("Not enough numeric data in one or both groups.")
                else:
                    t_stat, p_val = stats.ttest_ind(x1, x2, equal_var=equal_var)
                    diff, (ci_lower, ci_upper), se, df_ = ci_diff_means_welch(x1, x2, alpha=alpha)

                    st.write("### Group summaries")
                    desc = pd.DataFrame({
                        "group": [g1, g2],
                        "n": [len(x1), len(x2)],
                        "mean": [x1.mean(), x2.mean()],
                        "sd": [x1.std(ddof=1), x2.std(ddof=1)],
                    })
                    st.table(desc)

                    out = {
                        "Test": "Independent two-sample t-test",
                        "df (Welch)": round(df_, 1),
                        "mean1 - mean2": round(diff, 3),
                        "t": round(t_stat, 3),
                        "p-value": round(p_val, 4),
                        "CI lower": round(ci_lower, 3),
                        "CI upper": round(ci_upper, 3),
                        "α": alpha,
                    }
                    st.write("### Test results (structured)")
                    st.table(pd.DataFrame([out]))
                    msg = interpret_pvalue(p_val, alpha)
                    if p_val < alpha:
                        st.success(msg)
                    else:
                        st.warning(msg)

                    st.write("### Boxplot by group")
                    fig, ax = plt.subplots()
                    bp = ax.boxplot([x1, x2], labels=[str(g1), str(g2)],
                                    patch_artist=True)
                    for i, box in enumerate(bp['boxes']):
                        box.set(facecolor=PEP_ORANGE if i == 1 else PEP_BLUE,
                                edgecolor="black")
                    ax.set_ylabel(dv)
                    st.pyplot(fig)

                    summary = (
                        f"Group 1 ({g1}): n={len(x1)}, mean={x1.mean():.3f}, sd={x1.std(ddof=1):.3f}\n"
                        f"Group 2 ({g2}): n={len(x2)}, mean={x2.mean():.3f}, sd={x2.std(ddof=1):.3f}\n"
                        f"Difference (mean1 - mean2)={diff:.3f}, t({df_:.1f})={t_stat:.3f}, p={p_val:.4f}, "
                        f"{100*(1-alpha):.1f}% CI=({ci_lower:.3f}, {ci_upper:.3f})"
                    )

                    results = {"summary": summary}
                    context = {
                        "description": "Compare the average outcome between two independent groups using a two-sample t-test.",
                        "variables": f"Outcome: {dv}. Grouping: {iv} with levels {g1} and {g2}."
                    }
                    prompt = build_prompt("two-sample t-test", context, results)
                    st.subheader("Optional: AI Explanation Prompt (Gemini)")
                    st.code(prompt, language="markdown")

# -------------------------------------------------------------------
# E. ANOVA
# -------------------------------------------------------------------
elif analysis_choice == "ANOVA (compare 3+ group means)":
    st.header("ANOVA: Compare 3+ Group Means")

    quant_vars = [c for c in data_df.columns if var_types[c] == "Quantitative"]
    cat_vars = [c for c in data_df.columns if var_types[c] == "Categorical"]

    if not quant_vars or not cat_vars:
        st.warning("Need a quantitative outcome and a categorical grouping variable.")
    else:
        dv = st.selectbox("Outcome (numeric)", quant_vars)
        iv = st.selectbox("Grouping variable (category)", cat_vars)
        data = data_df[[dv, iv]].dropna()
        groups = data[iv].unique()
        if len(groups) < 3:
            st.warning("ANOVA requires at least 3 groups.")
        else:
            samples = [
                pd.to_numeric(data[data[iv] == g][dv], errors="coerce").dropna()
                for g in groups
            ]
            if any(len(s) == 0 for s in samples):
                st.error("One or more groups have no numeric data.")
            else:
                f_stat, p_val = stats.f_oneway(*samples)

                desc = data.groupby(iv)[dv].agg(["count", "mean", "std"])
                st.write("### Group summaries")
                st.dataframe(desc)

                out = {
                    "Test": "One-way ANOVA",
                    "F": round(f_stat, 3),
                    "p-value": round(p_val, 4),
                }
                st.write("### Test results (structured)")
                st.table(pd.DataFrame([out]))
                msg = interpret_pvalue(p_val, alpha=0.05)
                if p_val < 0.05:
                    st.success(msg)
                else:
                    st.warning(msg)

                st.write("### Boxplot by group")
                fig, ax = plt.subplots()
                bp = ax.boxplot(samples, labels=[str(g) for g in groups],
                                patch_artist=True)
                for i, box in enumerate(bp['boxes']):
                    color = PEP_BLUE if i % 2 == 0 else PEP_ORANGE
                    box.set(facecolor=color, edgecolor="black")
                ax.set_ylabel(dv)
                st.pyplot(fig)

                summary = f"F-statistic={f_stat:.3f}, p-value={p_val:.4f}"

                results = {"summary": summary + "\n" + desc.to_string()}
                context = {
                    "description": "Compare mean outcomes across multiple groups using one-way ANOVA.",
                    "variables": f"Outcome: {dv}. Grouping variable: {iv} with groups {list(groups)}."
                }
                prompt = build_prompt("one-way ANOVA", context, results)
                st.subheader("Optional: AI Explanation Prompt (Gemini)")
                st.code(prompt, language="markdown")

# -------------------------------------------------------------------
# F. Chi-square (categorical association)
# -------------------------------------------------------------------
elif analysis_choice == "Categorical association (chi-square)":
    st.header("Chi-Square Test of Independence")

    cat_vars = [c for c in data_df.columns if var_types[c] == "Categorical"]
    if len(cat_vars) < 2:
        st.warning("Need two categorical variables.")
    else:
        c1 = st.selectbox("First categorical variable", cat_vars)
        c2 = st.selectbox("Second categorical variable", [c for c in cat_vars if c != c1])
        data = data_df[[c1, c2]].dropna()
        contingency = pd.crosstab(data[c1], data[c2])

        st.write("### Contingency table (counts)")
        st.dataframe(contingency)

        chi2, p, dof, expected = stats.chi2_contingency(contingency)

        out = {
            "Test": "Chi-square test of independence",
            "Chi-square": round(chi2, 3),
            "df": dof,
            "p-value": round(p, 4),
        }
        st.write("### Test results (structured)")
        st.table(pd.DataFrame([out]))
        msg = interpret_pvalue(p, alpha=0.05)
        if p < 0.05:
            st.success(msg)
        else:
            st.warning(msg)

        summary = f"Chi-square={chi2:.3f}, df={dof}, p-value={p:.4f}"

        results = {"summary": summary + "\nObserved:\n" + contingency.to_string()}
        context = {
            "description": "Assess whether two categorical variables are associated using a chi-square test of independence.",
            "variables": f"Categorical variables: {c1} and {c2}."
        }
        prompt = build_prompt("chi-square test of independence", context, results)
        st.subheader("Optional: AI Explanation Prompt (Gemini)")
        st.code(prompt, language="markdown")

# -------------------------------------------------------------------
# G. Correlation & regression
# -------------------------------------------------------------------
elif analysis_choice == "Correlation & regression":
    st.header("Correlation & Regression")

    mode = st.radio("Choose", ["Correlation (two numeric variables)", "Regression (simple or multiple)"])

    if mode == "Correlation (two numeric variables)":
        quant_vars = [c for c in data_df.columns if var_types[c] == "Quantitative"]
        if len(quant_vars) < 2:
            st.warning("Need at least two quantitative variables.")
        else:
            x_var = st.selectbox("X (numeric)", quant_vars)
            y_var = st.selectbox("Y (numeric)", [v for v in quant_vars if v != x_var])
            data = data_df[[x_var, y_var]].dropna()
            x_num = pd.to_numeric(data[x_var], errors="coerce")
            y_num = pd.to_numeric(data[y_var], errors="coerce")
            df_corr = pd.DataFrame({x_var: x_num, y_var: y_num}).dropna()

            # FIX: need at least 2 pairs for Pearson r
            if len(df_corr) < 2:
                st.error(
                    "Need at least 2 non-missing numeric pairs to compute Pearson correlation. "
                    "Try relaxing your filters or checking for missing data."
                )
            else:
                r, p_val = stats.pearsonr(df_corr[x_var], df_corr[y_var])

                out = {
                    "Test": "Pearson correlation",
                    "r": round(r, 3),
                    "p-value": round(p_val, 4),
                    "n": len(df_corr),
                }
                st.write("### Correlation results (structured)")
                st.table(pd.DataFrame([out]))
                msg = interpret_pvalue(p_val, alpha=0.05)
                if p_val < 0.05:
                    st.success(msg)
                else:
                    st.warning(msg)

                st.write("### Scatterplot with regression line")
                fig, ax = plt.subplots()
                ax.scatter(df_corr[x_var], df_corr[y_var], color=PEP_BLUE)
                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)

                slope, intercept = np.polyfit(df_corr[x_var], df_corr[y_var], 1)
                x_vals = np.linspace(df_corr[x_var].min(), df_corr[x_var].max(), 100)
                y_vals = intercept + slope * x_vals
                ax.plot(x_vals, y_vals, color=PEP_ORANGE, linewidth=2)
                st.pyplot(fig)

                summary = f"Correlation r={r:.3f}, p-value={p_val:.4f}"

                results = {"summary": summary}
                context = {
                    "description": "Measure the strength and direction of the linear relationship between two quantitative variables.",
                    "variables": f"X: {x_var}, Y: {y_var}."
                }
                prompt = build_prompt("Pearson correlation", context, results)
                st.subheader("Optional: AI Explanation Prompt (Gemini)")
                st.code(prompt, language="markdown")

    else:
        # Regression
        quant_vars = [c for c in data_df.columns if var_types[c] == "Quantitative"]
        if not quant_vars:
            st.warning("Need at least one quantitative outcome.")
        else:
            dv = st.selectbox("Outcome (Y, numeric)", quant_vars)
            predictors = st.multiselect("Predictors (X)", [c for c in data_df.columns if c != dv])

            if predictors:
                # Safe formula using Q("col") so arbitrary names are allowed
                def q_name(col):
                    safe = str(col).replace('"', '\\"')
                    return f'Q("{safe}")'

                lhs = q_name(dv)
                terms = []
                for p in predictors:
                    if var_types[p] == "Categorical":
                        terms.append(f"C({q_name(p)})")
                    else:
                        terms.append(q_name(p))
                formula = lhs + " ~ " + " + ".join(terms)

                data = data_df[[dv] + predictors].dropna()

                for c in [dv] + predictors:
                    if var_types.get(c) == "Quantitative":
                        data[c] = pd.to_numeric(data[c], errors="coerce")
                data = data.dropna()

                if len(data) == 0:
                    st.error("No complete numeric data for regression.")
                else:
                    model = smf.ols(formula, data=data).fit()

                    st.write("### Regression formula")
                    st.code(formula)

                    st.write("### Coefficients with confidence intervals")
                    coef = model.params
                    conf = model.conf_int()
                    coef_df = pd.DataFrame({
                        "Estimate": coef,
                        "CI Lower": conf[0],
                        "CI Upper": conf[1]
                    })
                    st.dataframe(coef_df)

                    overall = {
                        "R-squared": round(model.rsquared, 3),
                        "Adj R-squared": round(model.rsquared_adj, 3),
                        "F-statistic": round(model.fvalue, 3) if model.fvalue is not None else None,
                        "F p-value": round(model.f_pvalue, 4) if model.f_pvalue is not None else None,
                    }
                    st.write("### Model fit (overall)")
                    st.table(pd.DataFrame([overall]))
                    if model.f_pvalue is not None:
                        msg = interpret_pvalue(model.f_pvalue, alpha=0.05)
                        if model.f_pvalue < 0.05:
                            st.success("Overall model significance: " + msg)
                        else:
                            st.warning("Overall model significance: " + msg)

                    if st.checkbox("Show regression diagnostics (residuals & plots)"):
                        fitted = model.fittedvalues
                        resid = model.resid

                        diag_df = pd.DataFrame({"fitted": fitted, "residual": resid})
                        st.write("### Residuals (first 50)")
                        st.dataframe(diag_df.head(50))

                        st.write("### Residuals vs Fitted")
                        fig, ax = plt.subplots()
                        ax.scatter(fitted, resid, color=PEP_BLUE)
                        ax.axhline(0, linestyle="--", color=PEP_ORANGE)
                        ax.set_xlabel("Fitted values")
                        ax.set_ylabel("Residuals")
                        st.pyplot(fig)

                        st.write("### Histogram of residuals")
                        fig, ax = plt.subplots()
                        ax.hist(resid, bins="auto", color=PEP_BLUE, edgecolor="black")
                        ax.set_xlabel("Residual")
                        ax.set_ylabel("Frequency")
                        st.pyplot(fig)

                        st.write("### QQ plot of residuals")
                        if len(resid) >= 3:
                            fig = sm.ProbPlot(resid, dist=stats.norm, fit=True).qqplot(line="45")
                            st.pyplot(fig)
                        else:
                            st.info("Not enough residuals to make a QQ plot (need at least 3).")

                    if st.checkbox("Show multicollinearity diagnostics (VIF)"):
                        vif_df = compute_vif(data, predictors, var_types)
                        st.write("### VIF for predictors")
                        st.dataframe(vif_df)

                    st.write("### Prediction at specified values")
                    user_inputs = {}
                    for p in predictors:
                        if var_types[p] == "Quantitative":
                            default_val = float(data[p].mean())
                            user_inputs[p] = st.number_input(f"Value for {p}", value=default_val)
                        else:
                            levels = data[p].unique()
                            user_inputs[p] = st.selectbox(f"Category for {p}", options=levels)

                    alpha = st.number_input(
                        "Significance level α for intervals",
                        value=0.05,
                        min_value=0.0001,
                        max_value=0.5
                    )

                    if st.button("Compute prediction and intervals"):
                        new_df = pd.DataFrame([user_inputs])
                        pred_frame = regression_prediction_with_intervals(model, new_df, alpha=alpha)
                        st.dataframe(pred_frame[["mean", "mean_ci_lower", "mean_ci_upper",
                                                 "obs_ci_lower", "obs_ci_upper"]])

                        summary = (
                            f"R-squared={model.rsquared:.3f}, Adj R-squared={model.rsquared_adj:.3f}\n"
                            f"Coefficients:\n{model.params.to_string()}\n\n"
                            f"Prediction at X={user_inputs}:\n"
                            f"Predicted mean={pred_frame['mean'].iloc[0]:.3f}, "
                            f"{100*(1-alpha):.1f}% CI for mean=({pred_frame['mean_ci_lower'].iloc[0]:.3f}, "
                            f"{pred_frame['mean_ci_upper'].iloc[0]:.3f}), "
                            f"{100*(1-alpha):.1f}% prediction interval=({pred_frame['obs_ci_lower'].iloc[0]:.3f}, "
                            f"{pred_frame['obs_ci_upper'].iloc[0]:.3f})"
                        )

                        st.write(summary.replace("\n", "  \n"))

                        results = {"summary": summary}
                        context = {
                            "description": "Use a regression model to predict the outcome for a specific business scenario and interpret the strength and uncertainty of the prediction.",
                            "variables": f"Outcome: {dv}. Predictors: {predictors}."
                        }
                        prompt = build_prompt("linear regression with prediction and diagnostics", context, results)
                        st.subheader("Optional: AI Explanation Prompt (Gemini)")
                        st.code(prompt, language="markdown")
