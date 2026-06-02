import gradio as gr
import pandas as pd
import os
import glob
import re

# ==========================================
# 0. Helper: Robust Column Resolution
# ==========================================

def resolve_column(columns, exact_target, fallback_keywords):
    """Intelligently finds a column, prioritizing exact matches and shorter names."""
    if exact_target in columns:
        return exact_target
    
    clean_target = " ".join(exact_target.lower().split())
    for col in columns:
        if " ".join(str(col).lower().split()) == clean_target:
            return col
            
    # Fallback keyword match - prioritize the shortest matching column name
    # This prevents grabbing "Prompt only Throughput" when we want "Throughput"
    matches = []
    for col in columns:
        if all(kw.lower() in str(col).lower() for kw in fallback_keywords):
            matches.append(col)
            
    if matches:
        return min(matches, key=len) # Returns the most direct match
        
    return columns[-1] if len(columns) > 0 else exact_target

# ==========================================
# 1. Agnostic Data Ingestion & State Contract
# ==========================================

def parse_perf_sweep(file_path):
    """
    Parses a single Excel performance sweep dynamically.
    Automatically hunts for the true header row to bypass metadata/summaries.
    """
    try:
        # 1. Read the raw file without assuming row 0 is the header
        temp_df = pd.read_excel(file_path, header=None)
        
        # 2. Hunt for the real header row by looking for standard perf keywords
        header_idx = 0
        for i, row in temp_df.iterrows():
            row_text = " ".join(str(val).lower() for val in row.values)
            if "throughput" in row_text or "input length" in row_text or "batch size" in row_text:
                header_idx = i
                break
                
        # 3. Re-read the dataframe using the correct starting row
        df = pd.read_excel(file_path, header=header_idx)
        
        # 4. Hardened Whitespace Normalization
        df.columns = [" ".join(str(col).split()) for col in df.columns]
        
        # 5. Dynamic model resolution from filename
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
    """Pre-loads initial performance sweeps if present."""
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
    
    input_col = resolve_column(filtered.columns, "Input Length", ["input"])
    output_col = resolve_column(filtered.columns, "Output Length", ["output"])
    tp_col = resolve_column(filtered.columns, "Throughput (t/s)", ["throughput", "t/s"])
    ttft_col = resolve_column(filtered.columns, "TTFT (ms)", ["ttft"])
    gen_col = resolve_column(filtered.columns, "Gen Speed (t/s/user)", ["gen", "speed"])
    rpm_col = resolve_column(filtered.columns, "RPM", ["rpm"])

    # Helper to safely convert Excel string-numbers (like "1,200.5") into floats
    def safe_numeric(series):
        return pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce')

    # Filter for workload parameters
    warning_message = ""
    if input_col in filtered.columns and output_col in filtered.columns:
        exact_match = filtered[
            (safe_numeric(filtered[input_col]).fillna(0) >= target_input) & 
            (safe_numeric(filtered[output_col]).fillna(0) >= target_output)
        ]
        if not exact_match.empty:
            filtered = exact_match
        else:
            # FALLBACK FIX: Instead of returning an error and hiding the data, 
            # we just warn the user and show them the highest data available.
            warning_message = f"*(⚠️ Note: The selected models do not contain test data reaching {target_input} Input / {target_output} Output. Showing highest available performance instead.)*\n\n"

    summary_data = []
    for model in selected_models:
        m_df = filtered[filtered["Model_ID"] == model].copy()
        if m_df.empty:
            continue
            
        m_df[tp_col] = safe_numeric(m_df[tp_col])
        
        if m_df[tp_col].isna().all():
            continue 
            
        best_idx = m_df[tp_col].idxmax()
        best_row = m_df.loc[best_idx]
        
        def get_val(col_name):
            val = best_row.get(col_name, 0)
            try:
                return float(str(val).replace(',', ''))
            except (ValueError, TypeError):
                return 0.0
                
        summary_data.append({
            "Model": model,
            "Max Total Throughput (t/s)": get_val(tp_col),
            "Optimal TTFT (ms)": get_val(ttft_col),
            "User Gen Speed (t/s)": get_val(gen_col),
            "Requests/Min (RPM)": get_val(rpm_col)
        })
        
    summary_df = pd.DataFrame(summary_data)
    if summary_df.empty:
        return "No numeric performance match found for the selected parameters.", None, None
        
    verdict = f"### 📋 Workload Sizing Insights\n{warning_message}"
    for _, row in summary_df.iterrows():
        verdict += f"- **{row['Model']}**: Delivers a peak generation performance of **{row['User Gen Speed (t/s)']:.1f} t/s per user** with a Time-to-First-Token latency of **{row['Optimal TTFT (ms)']:.1f} ms**. Sustains up to **{row['Requests/Min (RPM)']} RPM**.\n"

    plot_throughput = gr.BarPlot(
        summary_df, 
        x="Max Total Throughput (t/s)", 
        y="Model", 
        title="Sustained Cluster Throughput Capacity (Tokens/Second)", 
        color="Model"
    )
    
    plot_ttft = gr.BarPlot(
        summary_df, 
        x="Optimal TTFT (ms)", 
        y="Model", 
        title="Responsiveness Profile: Lower is Better (TTFT in ms)", 
        color="Model"
    )
    
    return verdict, plot_throughput, plot_ttft

def update_engineer_analytics(df, selected_models):
    """Surfaces structural data points, scaling curves, and hardware efficiency."""
    if df.empty or not selected_models:
        return pd.DataFrame(), None, None
        
    filtered = df[df["Model_ID"].isin(selected_models)].copy()
    
    batch_col = resolve_column(filtered.columns, "Batch Size", ["batch"])
    tp_col = resolve_column(filtered.columns, "Throughput (t/s)", ["throughput", "t/s"])
    hardware_col = resolve_column(filtered.columns, "Throughput / box (t/s/hardware)", ["box", "hardware"])
    
    def safe_numeric(series):
        return pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce')
    
    for col in [batch_col, tp_col, hardware_col]:
        if col in filtered.columns:
            filtered[col] = safe_numeric(filtered[col])

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
    
    core_cols = ["Model_ID", batch_col, "Cache %", tp_col, "TTFT (ms)", "RPM", "Max number of milliseconds"]
    existing_cols = [c for c in core_cols if c in filtered.columns]
    remaining_cols = [c for c in filtered.columns if c not in existing_cols]
    
    return filtered[existing_cols + remaining_cols], scaling_plot, efficiency_plot

# ==========================================
# 3. Gradio Interface Layout
# ==========================================

# FIX: Removed the theme argument from gr.Blocks() for Gradio 6.0 compatibility
with gr.Blocks() as app:
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
            comparer = gr.CheckboxGroup(
                choices=known_models,
                value=known_models,
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
                chart_cust_1 = gr.BarPlot(label="Total Sizing Capacity")
                chart_cust_2 = gr.BarPlot(label="User Experience Latency")
                
            trigger_customer.click(
                fn=update_customer_analytics,
                inputs=[state_store, comparer, input_len_slider, output_len_slider],
                outputs=[insights_box, chart_cust_1, chart_cust_2]
            )
            
        with gr.Tab("🛠️ Deployment Engineer Deep Dive"):
            gr.Markdown("### Scaling Properties & Performance Isolation\n*Audit saturation cliffs, memory allocation thresholds, cache hits, and runtime variances.*")
            
            trigger_engineer = gr.Button("Generate Deep Investigation Matrix", variant="secondary")
            
            with gr.Row():
                chart_eng_1 = gr.LinePlot(label="Sustained Saturation Curve")
                chart_eng_2 = gr.LinePlot(label="Hardware Multi-Node Scaling Quotient")
                
            gr.Markdown("#### Complete Raw Engineering Log")
            data_matrix = gr.DataFrame(interactive=False, wrap=True)
            
            trigger_engineer.click(
                fn=update_engineer_analytics,
                inputs=[state_store, comparer],
                outputs=[data_matrix, chart_eng_1, chart_eng_2]
            )

if __name__ == "__main__":
    # FIX: Moved the theme assignment down here to the launch method
    app.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="slate")
    )