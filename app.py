import gradio as gr
import pandas as pd
import os
import glob
import re

# ==========================================
# 0. Helper: Robust Column Resolution
# ==========================================

def resolve_column(columns, exact_target, fallback_keywords):
    """
    Intelligently finds a column in the DataFrame. 
    Prevents KeyErrors if an Excel file has slightly altered column names.
    """
    if exact_target in columns:
        return exact_target
    
    # Search for fallback keywords (case-insensitive)
    for col in columns:
        if all(kw.lower() in str(col).lower() for kw in fallback_keywords):
            return col
            
    # Absolute fallback to prevent crashes: return the last column (often a metric)
    return columns[-1] if len(columns) > 0 else exact_target

# ==========================================
# 1. Agnostic Data Ingestion & State Contract
# ==========================================

def parse_perf_sweep(file_path):
    """
    Parses a single Excel performance sweep dynamically.
    Extracts the model identification string from the file architecture.
    """
    try:
        df = pd.read_excel(file_path)
        
        # Hardened Whitespace Normalization: converts newlines, tabs, and double spaces to a single space
        df.columns = [" ".join(str(col).split()) for col in df.columns]
        
        # Dynamic model resolution from filename
        base_name = os.path.basename(file_path)
        match = re.match(r"([A-Za-z0-9_-]+)_profile", base_name)
        if not match:
            match = re.match(r"([A-Za-z0-9\s-]+) profile", base_name)
            
        model_id = match.group(1).strip() if match else base_name.split(".")[0]
        
        df["Model_ID"] = model_id
        return df
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return pd.DataFrame()

def initialize_default_dataset(directory="perf_data"):
    """Pre-loads initial performance sweeps if present in the local directory."""
    if not os.path.exists(directory):
        return pd.DataFrame()
    files = glob.glob(os.path.join(directory, "**/*.xlsx"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(directory, "*.xlsx"))
        
    dfs = [parse_perf_sweep(f) for f in files]
    valid_dfs = [d for d in dfs if not d.empty]
    return pd.concat(valid_dfs, ignore_index=True) if valid_dfs else pd.DataFrame()

def process_new_uploads(uploaded_files, master_df):
    """Ingests fresh sweeps live and returns updated choices."""
    if not uploaded_files:
        current_models = master_df["Model_ID"].unique().tolist() if not master_df.empty else []
        return master_df, gr.update(choices=current_models)
    
    new_dfs = [parse_perf_sweep(f.name) for f in uploaded_files]
    valid_new_dfs = [d for d in new_dfs if not d.empty]
    
    if not valid_new_dfs:
        current_models = master_df["Model_ID"].unique().tolist() if not master_df.empty else []
        return master_df, gr.update(choices=current_models)
        
    combined_new = pd.concat(valid_new_dfs, ignore_index=True)
    
    if master_df.empty:
        updated_df = combined_new
    else:
        updated_df = pd.concat([master_df, combined_new], ignore_index=True).drop_duplicates()
        
    all_models = updated_df["Model_ID"].unique().tolist()
    return updated_df, gr.update(choices=all_models, value=all_models)

# ==========================================
# 2. Audience Analytics Engines
# ==========================================

def update_customer_analytics(df, selected_models, target_input, target_output):
    """Generates high-level business go/no-go performance insights."""
    if df.empty or not selected_models:
        return "No data available. Please upload files or select models.", None, None
    
    filtered = df[df["Model_ID"].isin(selected_models)]
    
    # Resolve columns safely
    input_col = resolve_column(filtered.columns, "Input Length", ["input"])
    output_col = resolve_column(filtered.columns, "Output Length", ["output"])
    tp_col = resolve_column(filtered.columns, "Throughput (t/s)", ["throughput", "t/s"])
    ttft_col = resolve_column(filtered.columns, "TTFT (ms)", ["ttft"])
    gen_col = resolve_column(filtered.columns, "Gen Speed (t/s/user)", ["gen", "speed"])
    rpm_col = resolve_column(filtered.columns, "RPM", ["rpm"])

    # Filter for workload parameters
    if input_col in filtered.columns and output_col in filtered.columns:
        exact_match = filtered[
            pd.to_numeric(filtered[input_col], errors='coerce').fillna(0) >= target_input & 
            pd.to_numeric(filtered[output_col], errors='coerce').fillna(0) >= target_output
        ]
        if not exact_match.empty:
            filtered = exact_match

    summary_data = []
    for model in selected_models:
        m_df = filtered[filtered["Model_ID"] == model]
        if m_df.empty:
            continue
            
        # Ensure the target column is numeric before finding max to avoid errors
        m_df[tp_col] = pd.to_numeric(m_df[tp_col], errors='coerce')
        best_idx = m_df[tp_col].idxmax()
        
        if pd.isna(best_idx):
            continue # Skip if all values are NaN
            
        best_row = m_df.loc[best_idx]
        
        summary_data.append({
            "Model": model,
            "Max Total Throughput (t/s)": best_row.get(tp_col, 0),
            "Optimal TTFT (ms)": best_row.get(ttft_col, 0),
            "User Gen Speed (t/s)": best_row.get(gen_col, 0),
            "Requests/Min (RPM)": best_row.get(rpm_col, 0)
        })
        
    summary_df = pd.DataFrame(summary_data)
    if summary_df.empty:
        return "No numeric performance match found for the selected parameters.", None, None
        
    verdict = "### 📋 Workload Sizing Insights\n"
    for _, row in summary_df.iterrows():
        verdict += f"- **{row['Model']}**: Delivers a peak generation performance of **{row['User Gen Speed (t/s)']:.1f} t/s per user** with a Time-to-First-Token latency of **{row['Optimal TTFT (ms)']:.1f} ms**. Sustains up to **{row['Requests/Min (RPM)']} RPM**.\n"

    plot_throughput = gr.BarPlot(
        summary_df, x="Model", y="Max Total Throughput (t/s)", 
        title="Sustained Cluster Throughput Capacity (Tokens/Second)", color="Model", vertical=False
    )
    
    plot_ttft = gr.BarPlot(
        summary_df, x="Model", y="Optimal TTFT (ms)", 
        title="Responsiveness Profile: Lower is Better (TTFT in ms)", color="Model", vertical=False
    )
    
    return verdict, plot_throughput, plot_ttft

def update_engineer_analytics(df, selected_models):
    """Surfaces structural data points, scaling curves, and hardware efficiency."""
    if df.empty or not selected_models:
        return pd.DataFrame(), None, None
        
    filtered = df[df["Model_ID"].isin(selected_models)]
    
    # Resolve columns safely
    batch_col = resolve_column(filtered.columns, "Batch Size", ["batch"])
    tp_col = resolve_column(filtered.columns, "Throughput (t/s)", ["throughput", "t/s"])
    hardware_col = resolve_column(filtered.columns, "Throughput / box (t/s/hardware)", ["box", "hardware"])
    
    # Ensure numeric types for plotting
    for col in [batch_col, tp_col, hardware_col]:
        if col in filtered.columns:
            filtered[col] = pd.to_numeric(filtered[col], errors='coerce')

    scaling_plot = gr.LinePlot(
        filtered, x=batch_col, y=tp_col, color="Model_ID",
        title="Throughput Scaling Properties Across Batch Sizes",
        tooltip=["Model_ID", batch_col, tp_col]
    )
    
    efficiency_plot = gr.LinePlot(
        filtered, x=batch_col, y=hardware_col, color="Model_ID",
        title="Hardware Efficiency Factor (Throughput per Node vs. Batch)",
        tooltip=["Model_ID", batch_col, hardware_col]
    )
    
    # Clean UI Table Output
    core_cols = ["Model_ID", batch_col, "Cache %", tp_col, "TTFT (ms)", "RPM", "Max number of milliseconds"]
    existing_cols = [c for c in core_cols if c in filtered.columns]
    remaining_cols = [c for c in filtered.columns if c not in existing_cols]
    
    return filtered[existing_cols + remaining_cols], scaling_plot, efficiency_plot

# ==========================================
# 3. Gradio Interface Layout
# ==========================================

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", secondary_hue="slate")) as app:
    gr.Markdown("# 🚀 Cerebras Performance Analytics Lab\n*Transforming complex hardware projection metrics into clear runtime decisions.*")
    
    initial_dataset = initialize_default_dataset()
    known_models = initial_dataset["Model_ID"].unique().tolist() if not initial_dataset.empty else []
    state_store = gr.State(initial_dataset)
    
    with gr.Row():
        with gr.Column(scale=2):
            uploader = gr.File(
                label="Drop New Performance Sweeps here (.xlsx)", 
                file_count="multiple", 
                file_types=[".xlsx"]
            )
        with gr.Column(scale=2):
            comparer = gr.Dropdown(
                choices=known_models,
                value=known_models[:3] if len(known_models) >= 3 else known_models,
                multiselect=True,
                label="Active Models to Compare"
            )
            
    uploader.upload(
        fn=process_new_uploads,
        inputs=[uploader, state_store],
        outputs=[state_store, comparer]
    )
    
    with gr.Tabs():
        with gr.Tab("🎯 Customer / PM Decision Matrix"):
            gr.Markdown("### Application Performance Validation\n*Determine whether a selected infrastructure setup meets enterprise application latency constraints.*")
            
            with gr.Row():
                input_len_slider = gr.Slider(minimum=128, maximum=32768, step=128, value=2048, label="Target Input Context Length")
                output_len_slider = gr.Slider(minimum=128, maximum=8192, step=128, value=1024, label="Target Output Generation Length")
            
            trigger_customer = gr.Button("Evaluate Customer Alignment Metrics", variant="primary")
            insights_box = gr.Markdown("*Click evaluation button to run comparative analysis.*")
            
            with gr.Row():
                chart_cust_1 = gr.Plot(label="Total Sizing Capacity")
                chart_cust_2 = gr.Plot(label="User Experience Latency")
                
            trigger_customer.click(
                fn=update_customer_analytics,
                inputs=[state_store, comparer, input_len_slider, output_len_slider],
                outputs=[insights_box, chart_cust_1, chart_cust_2]
            )
            
        with gr.Tab("🛠️ Deployment Engineer Deep Dive"):
            gr.Markdown("### Scaling Properties & Performance Isolation\n*Audit saturation cliffs, memory allocation thresholds, cache hits, and runtime variances.*")
            
            trigger_engineer = gr.Button("Generate Deep Investigation Matrix", variant="secondary")
            
            with gr.Row():
                chart_eng_1 = gr.Plot(label="Sustained Saturation Curve")
                chart_eng_2 = gr.Plot(label="Hardware Multi-Node Scaling Quotient")
                
            gr.Markdown("#### Complete Raw Engineering Log")
            data_matrix = gr.DataFrame(interactive=False, wrap=True)
            
            trigger_engineer.click(
                fn=update_engineer_analytics,
                inputs=[state_store, comparer],
                outputs=[data_matrix, chart_eng_1, chart_eng_2]
            )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)