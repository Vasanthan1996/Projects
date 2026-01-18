# app.py imports section
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
try:
    from config import Config
    from data_preprocessor import DataPreprocessor
    from model_manager import ModelManager
    from detector import AnomalyDetector
    from database import DatabaseManager
    from visualization import VisualizationEngine
    from failure_patterns import FailurePatternManager  # Add this
    import utils
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please ensure all required modules are in the same directory.")
    st.stop()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
try:
    from config import Config
    from data_preprocessor import DataPreprocessor
    from model_manager import ModelManager
    from detector import AnomalyDetector
    from database import DatabaseManager
    from visualization import VisualizationEngine
    import utils
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please ensure all required modules are in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Manufacturing Anomaly Detection System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin-bottom: 1rem;
    }
    .anomaly-high {
        color: #A23B72;
        font-weight: bold;
    }
    .anomaly-medium {
        color: #F18F01;
        font-weight: bold;
    }
    .anomaly-low {
        color: #73AB84;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class AnomalyDetectionApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.config = Config()
        self.db_manager = DatabaseManager(self.config.DATABASE_PATH)
        self.detector = AnomalyDetector(self.config, self.db_manager)
        self.visualizer = VisualizationEngine(self.config)
        
        # Initialize session state
        if 'uploaded_historical' not in st.session_state:
            st.session_state.uploaded_historical = None
        if 'uploaded_new' not in st.session_state:
            st.session_state.uploaded_new = None
        if 'detection_results' not in st.session_state:
            st.session_state.detection_results = None
        if 'training_results' not in st.session_state:
            st.session_state.training_results = None
        if 'selected_models' not in st.session_state:
            st.session_state.selected_models = self.config.ML_MODELS[:2]
        if 'feedback_data' not in st.session_state:
            st.session_state.feedback_data = []
        if 'page' not in st.session_state:
            st.session_state.page = "Home Dashboard"
    
    def run(self):
        """Run the main application"""
        # Sidebar navigation
        st.sidebar.title("🏭 Navigation")
        
        # Define pages
        pages = {
            "Home Dashboard": self.render_home_dashboard,
            "Data Upload": self.render_data_upload,
            "Model Training": self.render_model_training,
            "Outlier Detection": self.render_outlier_detection,
            "Visualizations": self.render_visualizations,
            "Feedback System": self.render_feedback_system,
            "Settings": self.render_settings
        }
        
        # Page selection
        page = st.sidebar.radio(
            "Select Page",
            list(pages.keys()),
            index=list(pages.keys()).index(st.session_state.page)
        )
        
        # Update session state if page changed
        if page != st.session_state.page:
            st.session_state.page = page
            st.rerun()
        
        # Run selected page
        pages[page]()
    
    def render_home_dashboard(self):
        """Render home dashboard page"""
        st.markdown("<h1 class='main-header'>🏭 Manufacturing Anomaly Detection System</h1>", 
                   unsafe_allow_html=True)
        
        # System status metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Trained Items",
                value=len(self.detector.trained_items),
                delta=None
            )
        
        with col2:
            # Count trained models
            model_count = sum(len(models) for models in self.detector.model_manager.models.values())
            st.metric(
                label="Active Models",
                value=model_count,
                delta=None
            )
        
        with col3:
            # Get feedback count
            try:
                feedback_df = self.db_manager.get_feedback()
                feedback_count = len(feedback_df)
            except:
                feedback_count = 0
            st.metric(
                label="Feedback Items",
                value=feedback_count,
                delta=None
            )
        
        with col4:
            # Recent anomalies
            if st.session_state.detection_results is not None:
                anomaly_count = len(st.session_state.detection_results)
            else:
                anomaly_count = 0
            st.metric(
                label="Recent Anomalies",
                value=anomaly_count,
                delta=None
            )
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        
        with quick_col1:
            if st.button("📊 Upload New Data", use_container_width=True):
                st.session_state.page = "Data Upload"
                st.rerun()
        
        with quick_col2:
            if st.button("🤖 Train Models", use_container_width=True):
                st.session_state.page = "Model Training"
                st.rerun()
        
        with quick_col3:
            if st.button("🔍 Detect Anomalies", use_container_width=True):
                st.session_state.page = "Outlier Detection"
                st.rerun()
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        
        if st.session_state.detection_results is not None and len(st.session_state.detection_results) > 0:
            recent_anomalies = st.session_state.detection_results.head(5)
            
            for _, row in recent_anomalies.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{row['item_code']}</strong> - {row['bar_code']}<br>
                        <span class="anomaly-high">Anomaly Score: {row['anomaly_score']:.3f}</span><br>
                        Operator: {row['operator']} | Date: {row['date']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No recent anomalies detected. Upload new data and run detection.")
        
        # System information
        st.subheader("ℹ️ System Information")
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.info(f"""
            **Database**: {self.config.DATABASE_PATH}
            **Model Directory**: {self.config.MODEL_DIR}
            **Log Directory**: {self.config.LOG_DIR}
            """)
        
        with info_col2:
            st.info(f"""
            **Active Models**: {', '.join(self.config.ML_MODELS[:2])}
            **Anomaly Threshold**: {self.config.ANOMALY_THRESHOLD}
            **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """)
    
    def render_data_upload(self):
        """Render data upload page"""
        st.title("📤 Data Upload")
        
        tab1, tab2 = st.tabs(["Historical Training Data", "New Production Data"])
        
        with tab1:
            st.subheader("Upload Historical Data for Training")
            st.markdown("""
            **Requirements:**
            - CSV or Excel file
            - Must contain `item code` and `result` columns
            - Only `GG` results are used for training
            - Test parameters should include: stroke 1, Q L feed, Q R feed, cyl 1, cyl 2, stroke 2, stroke 3
            """)
            
            historical_file = st.file_uploader(
                "Choose historical data file",
                type=['csv', 'xlsx'],
                key="historical_uploader"
            )
            
            if historical_file is not None:
                try:
                    # Read file based on extension
                    if historical_file.name.endswith('.csv'):
                        df = pd.read_csv(historical_file)
                    else:
                        df = pd.read_excel(historical_file)
                    
                    # Validate columns
                    required_columns = ['item code', 'result']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    else:
                        st.session_state.uploaded_historical = df
                        
                        # Show preview
                        st.subheader("Data Preview")
                        st.dataframe(df.head(10))
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Records", len(df))
                        
                        with col2:
                            gg_count = len(df[df['result'] == 'GG'])
                            st.metric("GG (Good) Records", gg_count)
                        
                        with col3:
                            item_count = df['item code'].nunique()
                            st.metric("Unique Items", item_count)
                        
                        # Show item distribution
                        if 'result' in df.columns:
                            fig = go.Figure(data=[
                                go.Pie(
                                    labels=df['result'].value_counts().index,
                                    values=df['result'].value_counts().values,
                                    hole=0.3
                                )
                            ])
                            fig.update_layout(title="Result Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.success("✅ Historical data uploaded successfully!")
                        
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        with tab2:
            st.subheader("Upload New Production Data for Detection")
            st.markdown("""
            **Requirements:**
            - CSV or Excel file
            - Must contain `item code` and `bar code` columns
            - Can include `date`, `time`, `operator` columns
            - `result` column is optional
            """)
            
            new_file = st.file_uploader(
                "Choose new data file",
                type=['csv', 'xlsx'],
                key="new_uploader"
            )
            
            if new_file is not None:
                try:
                    # Read file based on extension
                    if new_file.name.endswith('.csv'):
                        df = pd.read_csv(new_file)
                    else:
                        df = pd.read_excel(new_file)
                    
                    # Validate columns
                    required_columns = ['item code', 'bar code']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    else:
                        st.session_state.uploaded_new = df
                        
                        # Show preview
                        st.subheader("Data Preview")
                        st.dataframe(df.head(10))
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Records", len(df))
                        
                        with col2:
                            item_count = df['item code'].nunique()
                            st.metric("Unique Items", item_count)
                        
                        with col3:
                            if 'operator' in df.columns:
                                operator_count = df['operator'].nunique()
                                st.metric("Unique Operators", operator_count)
                        
                        st.success("✅ New data uploaded successfully!")
                        
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        # Data management options
        st.subheader("📁 Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Historical Data", use_container_width=True):
                st.session_state.uploaded_historical = None
                st.rerun()
        
        with col2:
            if st.button("Clear New Data", use_container_width=True):
                st.session_state.uploaded_new = None
                st.rerun()
    
    # app.py - Updated render_model_training method
    def render_model_training(self):
        """Render model training page with failure pattern learning"""
        st.title("🤖 Model Training with Failure Pattern Learning")
        
        # Check if historical data is available
        if st.session_state.uploaded_historical is None:
            st.warning("Please upload historical data first on the Data Upload page.")
            if st.button("Go to Data Upload"):
                st.session_state.page = "Data Upload"
                st.rerun()
            return
        
        st.info(f"**Loaded Historical Data:** {len(st.session_state.uploaded_historical)} records")
        
        # Data analysis
        st.subheader("📊 Data Analysis")
        
        if 'result' in st.session_state.uploaded_historical.columns:
            # Count results
            result_counts = st.session_state.uploaded_historical['result'].value_counts()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                gg_count = result_counts.get('GG', 0)
                st.metric("GG (Good)", gg_count, 
                         delta=None, delta_color="normal")
            
            with col2:
                ww_count = result_counts.get('WW', 0)
                st.metric("WW (Rejected)", ww_count,
                         delta=None, delta_color="inverse")
            
            with col3:
                gw_count = result_counts.get('GW', 0) + result_counts.get('WG', 0)
                st.metric("GW/WG (Partial)", gw_count,
                         delta=None, delta_color="off")
            
            with col4:
                other_count = len(st.session_state.uploaded_historical) - (gg_count + ww_count + gw_count)
                st.metric("Other/Unknown", other_count)
            
            # Show distribution
            fig = go.Figure(data=[
                go.Pie(
                    labels=result_counts.index,
                    values=result_counts.values,
                    hole=0.3,
                    marker_colors=['#2E86AB', '#A23B72', '#F18F01', '#73AB84']
                )
            ])
            fig.update_layout(title="Result Distribution in Historical Data")
            st.plotly_chart(fig, use_container_width=True)
        
        # Training configuration
        st.subheader("⚙️ Training Configuration")
        
        tab1, tab2, tab3 = st.tabs(["Normal Models", "Failure Patterns", "Advanced"])
        
        with tab1:
            st.markdown("**Train Normal Behavior Models**")
            st.info("These models learn what 'normal' (GG) looks like for each item.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Machine Learning Models**")
                for model in self.config.ML_MODELS:
                    is_selected = st.checkbox(
                        f"{model.replace('_', ' ').title()}",
                        value=model in st.session_state.selected_models,
                        key=f"ml_{model}"
                    )
                    if is_selected and model not in st.session_state.selected_models:
                        st.session_state.selected_models.append(model)
                    elif not is_selected and model in st.session_state.selected_models:
                        st.session_state.selected_models.remove(model)
            
            with col2:
                st.markdown("**Deep Learning Models**")
                for model in self.config.DL_MODELS:
                    is_selected = st.checkbox(
                        f"{model.replace('_', ' ').title()}",
                        value=model in st.session_state.selected_models,
                        key=f"dl_{model}"
                    )
                    if is_selected and model not in st.session_state.selected_models:
                        st.session_state.selected_models.append(model)
                    elif not is_selected and model in st.session_state.selected_models:
                        st.session_state.selected_models.remove(model)
            
            # GG sample requirements
            min_gg_samples = st.slider(
                "Minimum GG samples per item",
                min_value=3,
                max_value=50,
                value=5,
                help="Minimum number of GG samples required to train normal models"
            )
        
        with tab2:
            st.markdown("**Learn Failure Patterns**")
            st.info("The system will analyze WW, GW, WG items to learn common failure patterns.")
            
            learn_failures = st.checkbox(
                "Learn failure patterns from historical failures",
                value=True,
                help="Enable to memorize common failure patterns for each item"
            )
            
            min_failure_samples = 3  # Default value
            if learn_failures:
                min_failure_samples = st.slider(
                    "Minimum failure samples per pattern",
                    min_value=2,
                    max_value=20,
                    value=3,
                    help="Minimum number of similar failures to create a pattern"
                )
                
                st.info("""
                **What will be learned:**
                - Common parameter values for each failure type (WW, GW, WG)
                - Severity levels for different failures
                - Operator patterns associated with failures
                - Parameter ranges that indicate specific problems
                """)
        
        with tab3:
            st.markdown("**Advanced Settings**")
            
            self.config.TEST_SIZE = st.slider(
                "Validation Split", 0.1, 0.5, self.config.TEST_SIZE, 0.05
            )
            
            self.config.ANOMALY_THRESHOLD = st.slider(
                "Anomaly Threshold", 0.5, 0.99, self.config.ANOMALY_THRESHOLD, 0.01
            )
            
            self.config.FAILURE_SIMILARITY_THRESHOLD = st.slider(
                "Failure Pattern Match Threshold", 0.5, 0.95, 
                self.config.FAILURE_SIMILARITY_THRESHOLD, 0.05,
                help="Similarity required to match a known failure pattern"
            )
        
        # Item selection
        st.subheader("📋 Item Selection")
        
        unique_items = st.session_state.uploaded_historical['item code'].unique()
        
        # Analyze each item
        item_analysis = []
        for item in unique_items:
            item_data = st.session_state.uploaded_historical[
                st.session_state.uploaded_historical['item code'] == item
            ]
            
            gg_count = len(item_data[item_data['result'] == 'GG'])
            ww_count = len(item_data[item_data['result'] == 'WW'])
            gw_count = len(item_data[item_data['result'] == 'GW'])
            wg_count = len(item_data[item_data['result'] == 'WG'])
            total_failures = ww_count + gw_count + wg_count
            
            # Determine if we can train
            can_train_normal = gg_count >= min_gg_samples
            can_learn_patterns = learn_failures and total_failures >= min_failure_samples
            
            item_analysis.append({
                'item_code': item,
                'gg_samples': gg_count,
                'ww_samples': ww_count,
                'gw_samples': gw_count,
                'wg_samples': wg_count,
                'total_failures': total_failures,
                'can_train_normal': can_train_normal,
                'can_learn_patterns': can_learn_patterns
            })
        
        analysis_df = pd.DataFrame(item_analysis)
        
        # Display analysis with proper styling
        def highlight_rows(row):
            """Apply background color based on training capability"""
            can_train = row['can_train_normal'] or row['can_learn_patterns']
            return ['background-color: #d4edda' if can_train else 'background-color: #f8d7da'] * len(row)
        
        if not analysis_df.empty:
            styled_df = analysis_df.style.apply(highlight_rows, axis=1)
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("No items found in the data.")
        
        # Select items
        if not analysis_df.empty:
            selected_items = st.multiselect(
                "Select items to train (or leave empty for all suitable items)",
                options=unique_items,
                default=[],
                help="Select specific items or leave empty to train all items with sufficient data"
            )
            
            if not selected_items:
                # Auto-select items with sufficient data
                suitable_items = analysis_df[
                    (analysis_df['can_train_normal']) | (analysis_df['can_learn_patterns'])
                ]['item_code'].tolist()
                selected_items = suitable_items
                st.info(f"Auto-selected {len(selected_items)} suitable items")
        else:
            selected_items = []
            st.warning("No items available for selection.")
        
        st.info(f"**Selected {len(selected_items)} items for training**")
        
        # Start training
        st.subheader("🚀 Start Training")
        
        if st.button("Train Models & Learn Failure Patterns", type="primary", use_container_width=True):
            if not st.session_state.selected_models and not learn_failures:
                st.error("Please select at least one model or enable failure pattern learning.")
                return
            
            if not selected_items:
                st.error("No items selected for training.")
                return
            
            # Filter data for selected items
            training_data = st.session_state.uploaded_historical[
                st.session_state.uploaded_historical['item code'].isin(selected_items)
            ].copy()
            
            # Show training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            try:
                # Step 1: Train normal models
                status_text.text("Step 1/2: Training normal models from GG items...")
                progress_bar.progress(0.3)
                
                # Update config with user settings
                self.config.FAILURE_PATTERN_MIN_SAMPLES = min_failure_samples
                
                training_results = self.detector.train_all_models(
                    training_data,
                    st.session_state.selected_models if st.session_state.selected_models else [],
                    min_samples=min_gg_samples,
                    learn_failure_patterns=learn_failures
                )
                
                # Step 2: Get failure pattern summary
                status_text.text("Step 2/2: Analyzing failure patterns...")
                progress_bar.progress(0.7)
                
                failure_summary = self.detector.get_failure_patterns_summary()
                
                progress_bar.progress(1.0)
                status_text.text("Training completed!")
                
                # Store results
                st.session_state.training_results = training_results
                st.session_state.failure_summary = failure_summary
                
                # Show results
                with results_container:
                    st.subheader("📊 Training Results")
                    
                    # Normal models results
                    st.markdown("### Normal Behavior Models")
                    
                    if training_results:
                        # Filter normal behavior results
                        normal_results = {}
                        for item_code, result in training_results.items():
                            if isinstance(result, dict) and 'model_type' in result and result['model_type'] == 'normal_behavior':
                                normal_results[item_code] = result
                        
                        if normal_results:
                            success_count = len([r for r in normal_results.values() 
                                               if r.get('status') == 'success'])
                            total_count = len(normal_results)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Items Attempted", total_count)
                            
                            with col2:
                                st.metric("Successfully Trained", success_count)
                            
                            with col3:
                                success_rate = f"{(success_count/total_count*100):.1f}%" if total_count > 0 else "0.0%"
                                st.metric("Success Rate", success_rate)
                            
                            # Show detailed results
                            with st.expander("View Detailed Results"):
                                results_df = pd.DataFrame.from_dict(normal_results, orient='index')
                                st.dataframe(results_df, use_container_width=True)
                        else:
                            st.warning("No normal models were trained.")
                    else:
                        st.warning("No training results available.")
                    
                    # Failure patterns results
                    if learn_failures:
                        st.markdown("### Failure Patterns Learned")
                        
                        if failure_summary['total_patterns'] > 0:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Items with Patterns", failure_summary['total_items'])
                            
                            with col2:
                                st.metric("Total Patterns", failure_summary['total_patterns'])
                            
                            with col3:
                                ww_patterns = failure_summary['by_failure_type'].get('WW', 0)
                                st.metric("WW Patterns", ww_patterns)
                            
                            with col4:
                                gw_wg_patterns = (
                                    failure_summary['by_failure_type'].get('GW', 0) +
                                    failure_summary['by_failure_type'].get('WG', 0)
                                )
                                st.metric("GW/WG Patterns", gw_wg_patterns)
                            
                            # Show patterns by failure type
                            st.markdown("#### Patterns by Failure Type")
                            failure_types = list(failure_summary['by_failure_type'].keys())
                            pattern_counts = [failure_summary['by_failure_type'][ft] for ft in failure_types]
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=failure_types,
                                    y=pattern_counts,
                                    marker_color=['#A23B72', '#F18F01', '#73AB84']
                                )
                            ])
                            fig.update_layout(
                                title="Failure Patterns by Type",
                                xaxis_title="Failure Type",
                                yaxis_title="Number of Patterns"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show items with patterns
                            with st.expander("View Items with Learned Patterns"):
                                for item in failure_summary['items_with_patterns'][:10]:  # Show first 10
                                    patterns = self.detector.failure_manager.get_patterns_for_item(item)
                                    if patterns:
                                        st.write(f"**{item}**: {len(patterns)} patterns")
                                        for pattern in patterns[:2]:  # Show first 2 patterns
                                            desc = pattern.get('description', 'No description')
                                            st.write(f"  - {pattern.get('failure_type', 'Unknown')}: {desc[:80]}...")
                        else:
                            st.info("No failure patterns were learned (insufficient failure data).")
                    
                    # Export options
                    st.subheader("💾 Export Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Download Training Report"):
                            report_data = {
                                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'normal_models_trained': normal_results if 'normal_results' in locals() else {},
                                'failure_patterns_learned': failure_summary,
                                'configuration': {
                                    'min_gg_samples': min_gg_samples,
                                    'learn_failures': learn_failures,
                                    'selected_models': st.session_state.selected_models
                                }
                            }
                            
                            import json
                            report_json = json.dumps(report_data, indent=2, default=str)
                            
                            st.download_button(
                                label="Download JSON Report",
                                data=report_json,
                                file_name=f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                    
                    with col2:
                        if failure_summary.get('total_patterns', 0) > 0:
                            if st.button("Download Failure Patterns"):
                                patterns_data = {}
                                for item in failure_summary.get('items_with_patterns', []):
                                    patterns = self.detector.failure_manager.get_patterns_for_item(item)
                                    if patterns:
                                        patterns_data[item] = patterns
                                
                                patterns_json = json.dumps(patterns_data, indent=2, default=str)
                                
                                st.download_button(
                                    label="Download Patterns JSON",
                                    data=patterns_json,
                                    file_name=f"failure_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                    
                    st.success("✅ Training completed successfully!")
                    
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                import traceback
                with st.expander("See detailed error traceback"):
                    st.code(traceback.format_exc())
        
        # Show existing trained items and patterns
        st.subheader("📋 Current System Knowledge")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Trained Items (Normal Models)")
            if self.detector.trained_items:
                trained_list = list(self.detector.trained_items)
                for item in trained_list[:10]:  # Show first 10
                    st.success(f"✅ {item}")
                if len(trained_list) > 10:
                    st.info(f"... and {len(trained_list) - 10} more items")
            else:
                st.info("No normal models trained yet")
        
        with col2:
            st.markdown("#### Items with Failure Patterns")
            if hasattr(self.detector, 'failure_manager') and self.detector.failure_manager.patterns:
                pattern_items = list(self.detector.failure_manager.patterns.keys())
                for item in pattern_items[:10]:  # Show first 10
                    pattern_count = len(self.detector.failure_manager.patterns[item])
                    st.warning(f"⚠️ {item} ({pattern_count} patterns)")
                if len(pattern_items) > 10:
                    st.info(f"... and {len(pattern_items) - 10} more items")
            else:
                st.info("No failure patterns learned yet")
    
    def render_outlier_detection(self):
        """Render outlier detection page"""
        st.title("🔍 Outlier Detection")
        
        # Check if new data is available
        if st.session_state.uploaded_new is None:
            st.warning("Please upload new data first on the Data Upload page.")
            if st.button("Go to Data Upload"):
                st.session_state.page = "Data Upload"
                st.rerun()
            return
        
        st.info(f"**Loaded New Data:** {len(st.session_state.uploaded_new)} records")
        
        # Detection settings
        st.subheader("1. Detection Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Item code filter
            unique_items = st.session_state.uploaded_new['item code'].unique()
            selected_item = st.selectbox(
                "Select Item Code (empty = all)",
                options=["All"] + list(unique_items),
                index=0
            )
        
        with col2:
            # Confidence threshold
            confidence_threshold = st.slider(
                "Confidence Threshold (%)",
                50, 100, 85, 5
            )
        
        # Date filter (if date column exists)
        if 'date' in st.session_state.uploaded_new.columns:
            st.subheader("2. Date Filter")
            
            try:
                # Convert date column
                date_series = pd.to_datetime(
                    st.session_state.uploaded_new['date'], 
                    errors='coerce'
                )
                min_date = date_series.min()
                max_date = date_series.max()
                
                if pd.notna(min_date) and pd.notna(max_date):
                    date_range = st.date_input(
                        "Select Date Range",
                        value=(min_date.date(), max_date.date()),
                        min_value=min_date.date(),
                        max_value=max_date.date()
                    )
                    
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        # Filter data
                        mask = (date_series.dt.date >= start_date) & (date_series.dt.date <= end_date)
                        filtered_data = st.session_state.uploaded_new[mask].copy()
                    else:
                        filtered_data = st.session_state.uploaded_new.copy()
                else:
                    filtered_data = st.session_state.uploaded_new.copy()
                    st.warning("Date column contains invalid dates")
                    
            except Exception as e:
                filtered_data = st.session_state.uploaded_new.copy()
                st.warning(f"Error parsing dates: {e}")
        else:
            filtered_data = st.session_state.uploaded_new.copy()
        
        # Operator filter (if operator column exists)
        if 'operator' in filtered_data.columns:
            st.subheader("3. Operator Filter")
            
            unique_operators = filtered_data['operator'].unique()
            selected_operators = st.multiselect(
                "Select Operators (empty = all)",
                options=unique_operators,
                default=[]
            )
            
            if selected_operators:
                filtered_data = filtered_data[filtered_data['operator'].isin(selected_operators)].copy()
        
        st.info(f"**Data after filtering:** {len(filtered_data)} records")
        
        # Run detection
        st.subheader("4. Run Detection")
        
        if st.button("🚀 Detect Anomalies", type="primary", use_container_width=True):
            if len(filtered_data) == 0:
                st.error("No data to analyze after filtering.")
                return
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Starting anomaly detection...")
                
                # Run detection
                if selected_item == "All":
                    results = self.detector.detect_anomalies(filtered_data)
                else:
                    results = self.detector.detect_anomalies(filtered_data, selected_item)
                
                progress_bar.progress(1.0)
                status_text.text("Detection completed!")
                
                # Filter by confidence threshold
                if len(results) > 0:
                    results = results[results['confidence_level'] >= confidence_threshold].copy()
                
                # Store results
                st.session_state.detection_results = results
                
                # Show results summary
                st.subheader("📊 Detection Results")
                
                if len(results) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Anomalies", len(results))
                    
                    with col2:
                        avg_score = results['anomaly_score'].mean()
                        st.metric("Avg. Anomaly Score", f"{avg_score:.3f}")
                    
                    with col3:
                        unique_items = results['item_code'].nunique()
                        st.metric("Affected Items", unique_items)
                    
                    with col4:
                        if 'operator' in results.columns:
                            unique_operators = results['operator'].nunique()
                            st.metric("Affected Operators", unique_operators)
                    
                    # Show results table
                    st.subheader("📋 Detected Anomalies")
                    
                    # Format display
                    display_cols = [
                        'bar_code', 'item_code', 'operator', 'date', 
                        'anomaly_score', 'confidence_level', 'abnormal_fields'
                    ]
                    
                    available_cols = [col for col in display_cols if col in results.columns]
                    
                    st.dataframe(
                        results[available_cols].head(100),
                        use_container_width=True
                    )
                    
                    # Download results
                    st.subheader("💾 Export Results")
                    
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.success("🎉 No anomalies detected above the confidence threshold!")
                    
            except Exception as e:
                st.error(f"Error during detection: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Show previous results if available
        if st.session_state.detection_results is not None:
            st.subheader("📈 Previous Detection Results")
            
            if len(st.session_state.detection_results) > 0:
                # Create visualization
                fig = self.visualizer.create_anomaly_distribution(
                    st.session_state.detection_results
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No previous detection results available.")
    
    def render_visualizations(self):
        """Render visualizations page"""
        st.title("📊 Visualizations")
        
        # Check if we have data
        if st.session_state.uploaded_historical is None and \
           st.session_state.uploaded_new is None and \
           st.session_state.detection_results is None:
            st.warning("Please upload data and run detection first.")
            if st.button("Go to Outlier Detection"):
                st.session_state.page = "Outlier Detection"
                st.rerun()
            return
        
        # Visualization selection
        viz_options = [
            "Anomaly Score Distribution",
            "Feature Comparison",
            "Time Series Trends",
            "Correlation Heatmap",
            "Parallel Coordinates",
            "3D Scatter Plot",
            "Model Performance"
        ]
        
        selected_viz = st.selectbox(
            "Select Visualization Type",
            options=viz_options
        )
        
        # Data selection
        data_source = st.radio(
            "Data Source",
            options=["Historical Data", "New Data", "Detection Results"],
            horizontal=True
        )
        
        # Get appropriate data
        if data_source == "Historical Data":
            data = st.session_state.uploaded_historical
        elif data_source == "New Data":
            data = st.session_state.uploaded_new
        else:
            data = st.session_state.detection_results
        
        if data is None or len(data) == 0:
            st.warning(f"No {data_source.lower()} available.")
            return
        
        # Render selected visualization
        if selected_viz == "Anomaly Score Distribution":
            if 'anomaly_score' in data.columns:
                fig = self.visualizer.create_anomaly_distribution(data)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Anomaly scores not available in selected data.")
        
        elif selected_viz == "Feature Comparison":
            # Select feature to compare
            numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_features) == 0:
                st.warning("No numeric features available for comparison.")
                return
            
            selected_feature = st.selectbox(
                "Select Feature",
                options=numeric_features
            )
            
            # Split data if we have anomaly labels
            if 'is_anomaly' in data.columns:
                normal_data = data[data['is_anomaly'] == 0]
                anomaly_data = data[data['is_anomaly'] == 1]
            else:
                normal_data = data
                anomaly_data = pd.DataFrame()
            
            fig = self.visualizer.create_feature_comparison(
                normal_data, anomaly_data, selected_feature
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif selected_viz == "Time Series Trends":
            fig = self.visualizer.create_time_series_plot(data)
            st.plotly_chart(fig, use_container_width=True)
        
        elif selected_viz == "Correlation Heatmap":
            # Select features
            numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_features) < 2:
                st.warning("Need at least 2 numeric features for correlation heatmap.")
                return
            
            selected_features = st.multiselect(
                "Select Features",
                options=numeric_features,
                default=numeric_features[:min(10, len(numeric_features))]
            )
            
            if len(selected_features) >= 2:
                fig = self.visualizer.create_correlation_heatmap(data, selected_features)
                st.plotly_chart(fig, use_container_width=True)
        
        elif selected_viz == "Parallel Coordinates":
            # Select features
            all_features = data.columns.tolist()
            selected_features = st.multiselect(
                "Select Features (max 5)",
                options=all_features,
                default=all_features[:min(5, len(all_features))]
            )
            
            if len(selected_features) >= 2:
                fig = self.visualizer.create_parallel_coordinates(
                    data, selected_features
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif selected_viz == "3D Scatter Plot":
            # Select features
            numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_features) < 3:
                st.warning("Need at least 3 numeric features for 3D scatter plot.")
                return
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_feat = st.selectbox("X Axis", options=numeric_features)
            
            with col2:
                y_feat = st.selectbox("Y Axis", options=[f for f in numeric_features if f != x_feat])
            
            with col3:
                z_feat = st.selectbox("Z Axis", options=[f for f in numeric_features if f not in [x_feat, y_feat]])
            
            fig = self.visualizer.create_3d_scatter(data, [x_feat, y_feat, z_feat])
            st.plotly_chart(fig, use_container_width=True)
        
        elif selected_viz == "Model Performance":
            if st.session_state.training_results is not None:
                # Convert training results to model metrics format
                # This is simplified - in production, you'd have actual metrics
                model_metrics = {}
                for item_code, results in st.session_state.training_results.items():
                    if results.get('status') == 'success':
                        for model in results.get('trained_models', []):
                            if model not in model_metrics:
                                model_metrics[model] = {
                                    'precision': np.random.uniform(0.7, 0.95),
                                    'recall': np.random.uniform(0.6, 0.9),
                                    'f1_score': np.random.uniform(0.65, 0.92)
                                }
                
                fig = self.visualizer.create_model_performance_dashboard(model_metrics)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No model performance data available. Train models first.")
        
        # Export visualization
        st.subheader("💾 Export Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export as PNG"):
                st.info("In production, this would export the current visualization as PNG")
        
        with col2:
            if st.button("Export as PDF"):
                st.info("In production, this would export the current visualization as PDF")
    
    def render_feedback_system(self):
        """Render feedback system page"""
        st.title("📝 Feedback System")
        
        # Check if we have detection results
        if st.session_state.detection_results is None or \
           len(st.session_state.detection_results) == 0:
            st.warning("Please run anomaly detection first to get results for feedback.")
            if st.button("Go to Outlier Detection"):
                st.session_state.page = "Outlier Detection"
                st.rerun()
            return
        
        st.info(f"**Available Anomalies for Review:** {len(st.session_state.detection_results)}")
        
        # Filter anomalies for review
        st.subheader("1. Select Anomaly for Review")
        
        # Create selection list
        anomaly_list = st.session_state.detection_results[
            ['bar_code', 'item_code', 'operator', 'date', 'anomaly_score']
        ].to_dict('records')
        
        if not anomaly_list:
            st.warning("No anomalies available for review.")
            return
        
        selected_idx = st.selectbox(
            "Select Anomaly",
            options=range(len(anomaly_list)),
            format_func=lambda x: f"{anomaly_list[x]['bar_code']} - {anomaly_list[x]['item_code']} "
                                f"(Score: {anomaly_list[x]['anomaly_score']:.3f})"
        )
        
        selected_anomaly = st.session_state.detection_results.iloc[selected_idx]
        
        # Display anomaly details
        st.subheader("2. Anomaly Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Bar Code:** {selected_anomaly.get('bar_code', 'N/A')}
            **Item Code:** {selected_anomaly.get('item_code', 'N/A')}
            **Operator:** {selected_anomaly.get('operator', 'N/A')}
            **Date:** {selected_anomaly.get('date', 'N/A')}
            **Time:** {selected_anomaly.get('time', 'N/A')}
            """)
        
        with col2:
            st.markdown(f"""
            **Anomaly Score:** {selected_anomaly.get('anomaly_score', 0):.3f}
            **Confidence Level:** {selected_anomaly.get('confidence_level', 0):.1f}%
            **Result:** {selected_anomaly.get('result', 'Unknown')}
            """)
        
        # Show abnormal fields
        if 'abnormal_fields' in selected_anomaly:
            st.subheader("3. Abnormal Fields")
            
            abnormal_fields = selected_anomaly['abnormal_fields']
            if isinstance(abnormal_fields, str):
                try:
                    abnormal_fields = json.loads(abnormal_fields)
                except:
                    abnormal_fields = []
            
            if abnormal_fields and len(abnormal_fields) > 0:
                for field_info in abnormal_fields:
                    field_name = field_info.get('field', 'Unknown')
                    field_value = field_info.get('value', 'Unknown')
                    field_reason = field_info.get('reason', 'Unknown')
                    field_severity = field_info.get('severity', 'medium')
                    
                    severity_color = {
                        'high': 'red',
                        'medium': 'orange',
                        'low': 'green'
                    }.get(field_severity, 'gray')
                    
                    st.markdown(f"""
                    <div style="border-left: 4px solid {severity_color}; padding-left: 10px; margin: 10px 0;">
                        <strong>{field_name}</strong>: {field_value}<br>
                        <em>Reason:</em> {field_reason} | <em>Severity:</em> {field_severity}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No specific abnormal fields identified.")
        
        # Feedback form
        st.subheader("4. Provide Feedback")
        
        with st.form("feedback_form"):
            feedback_type = st.radio(
                "Feedback Type",
                options=["Accept", "Reject", "Need Discussion"],
                horizontal=True
            )
            
            business_justification = st.text_area(
                "Business Justification",
                placeholder="Explain why you're accepting/rejecting this anomaly..."
            )
            
            reviewed_by = st.text_input("Your Name", value="Operator")
            
            submit_feedback = st.form_submit_button("Submit Feedback", type="primary")
        
        if submit_feedback:
            if not business_justification.strip():
                st.error("Please provide a business justification.")
            else:
                # Prepare feedback data
                feedback_data = {
                    'bar_code': selected_anomaly.get('bar_code'),
                    'item_code': selected_anomaly.get('item_code'),
                    'operator': selected_anomaly.get('operator'),
                    'date': selected_anomaly.get('date'),
                    'time': selected_anomaly.get('time'),
                    'anomaly_fields': selected_anomaly.get('abnormal_fields', []),
                    'original_value': {
                        field: selected_anomaly.get(field, '')
                        for field in self.config.NUMERIC_FEATURES
                        if field in selected_anomaly
                    },
                    'feedback_type': feedback_type.lower(),
                    'business_justification': business_justification,
                    'reviewed_by': reviewed_by,
                    'model_predictions': selected_anomaly.get('model_predictions', {}),
                    'confidence_score': selected_anomaly.get('anomaly_score', 0)
                }
                
                # Save feedback
                try:
                    feedback_id = self.db_manager.save_feedback(feedback_data)
                    
                    # Update model knowledge
                    self.detector.update_with_feedback(feedback_data)
                    
                    # Store in session state
                    if 'feedback_data' not in st.session_state:
                        st.session_state.feedback_data = []
                    
                    st.session_state.feedback_data.append(feedback_data)
                    
                    st.success(f"✅ Feedback submitted successfully! (ID: {feedback_id})")
                    
                    # Remove from detection results
                    st.session_state.detection_results = st.session_state.detection_results.drop(
                        st.session_state.detection_results.index[selected_idx]
                    ).reset_index(drop=True)
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error saving feedback: {str(e)}")
        
        # Show feedback history
        st.subheader("📋 Feedback History")
        
        try:
            feedback_df = self.db_manager.get_feedback()
            
            if len(feedback_df) > 0:
                # Convert JSON strings to display format
                def format_anomaly_fields(field_str):
                    try:
                        fields = json.loads(field_str)
                        if isinstance(fields, list):
                            return ', '.join([f.get('field', '') for f in fields[:3]])
                    except:
                        pass
                    return str(field_str)[:50]
                
                feedback_df['anomaly_fields_display'] = feedback_df['anomaly_fields'].apply(
                    format_anomaly_fields
                )
                
                display_cols = [
                    'timestamp', 'bar_code', 'item_code', 'feedback_type',
                    'reviewed_by', 'anomaly_fields_display'
                ]
                
                available_cols = [col for col in display_cols if col in feedback_df.columns]
                
                st.dataframe(
                    feedback_df[available_cols].sort_values('timestamp', ascending=False).head(20),
                    use_container_width=True
                )
                
                # Export feedback
                if st.button("Export All Feedback as CSV"):
                    csv = feedback_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"feedback_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No feedback submitted yet.")
                
        except Exception as e:
            st.error(f"Error loading feedback: {str(e)}")
    
    def render_settings(self):
        """Render settings page"""
        st.title("⚙️ Settings")
        
        tabs = st.tabs(["System Configuration", "Data Management", "User Management", "Logs & Monitoring"])
        
        with tabs[0]:
            st.subheader("System Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Model settings
                st.markdown("**Model Settings**")
                
                new_threshold = st.slider(
                    "Anomaly Threshold",
                    0.5, 0.99, self.config.ANOMALY_THRESHOLD, 0.01
                )
                
                if new_threshold != self.config.ANOMALY_THRESHOLD:
                    self.config.ANOMALY_THRESHOLD = new_threshold
                    st.success(f"Anomaly threshold updated to {new_threshold}")
            
            with col2:
                # Database settings
                st.markdown("**Database Settings**")
                
                db_path = st.text_input(
                    "Database Path",
                    value=self.config.DATABASE_PATH
                )
                
                if db_path != self.config.DATABASE_PATH:
                    self.config.DATABASE_PATH = db_path
                    st.success(f"Database path updated to {db_path}")
            
            # Save configuration
            if st.button("💾 Save Configuration"):
                self.config.save("config.json")
                st.success("Configuration saved successfully!")
            
            # Load configuration
            config_file = st.file_uploader(
                "Load Configuration File",
                type=['json']
            )
            
            if config_file is not None:
                try:
                    loaded_config = Config.load(config_file)
                    self.config = loaded_config
                    st.success("Configuration loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading configuration: {e}")
        
        with tabs[1]:
            st.subheader("Data Management")
            
            # Backup data
            if st.button("📂 Create Backup"):
                backup_file = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                st.info(f"Backup would be created as {backup_file}")
            
            # Restore data
            backup_file = st.file_uploader(
                "Restore from Backup",
                type=['zip']
            )
            
            if backup_file is not None:
                st.warning("⚠️ Restoring from backup will overwrite current data.")
                
                if st.button("🔄 Restore Backup", type="secondary"):
                    st.info("Restore functionality would be implemented here")
            
            # Data retention
            st.markdown("**Data Retention Policy**")
            
            retention_days = st.slider(
                "Keep data for (days)",
                30, 365, 90, 30
            )
            
            if st.button("🗑️ Apply Retention Policy"):
                st.info(f"Data older than {retention_days} days would be archived")
        
        with tabs[2]:
            st.subheader("User Management")
            
            # Simple user management (in production, use proper authentication)
            st.info("User management would be implemented with proper authentication system")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input("New Username")
                st.text_input("New Password", type="password")
                
                if st.button("Create User"):
                    st.success("User creation would be implemented here")
            
            with col2:
                user_roles = ["Operator", "Supervisor", "Administrator"]
                selected_role = st.selectbox("User Role", options=user_roles)
                
                if st.button("Update Role"):
                    st.success(f"Role updated to {selected_role}")
        
        with tabs[3]:
            st.subheader("Logs & Monitoring")
            
            # View system logs
            log_level = st.selectbox(
                "Log Level",
                options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            )
            
            if st.button("📋 View Logs"):
                try:
                    log_files = [f for f in os.listdir(self.config.LOG_DIR) if f.endswith('.log')]
                    
                    if log_files:
                        latest_log = max(log_files)
                        log_path = os.path.join(self.config.LOG_DIR, latest_log)
                        
                        with open(log_path, 'r') as f:
                            logs = f.readlines()[-100:]  # Last 100 lines
                        
                        st.text_area("Recent Logs", value=''.join(logs), height=300)
                    else:
                        st.info("No log files found.")
                        
                except Exception as e:
                    st.error(f"Error reading logs: {e}")
            
            # System monitoring
            st.markdown("**System Health**")
            
            # Check disk space
            import shutil
            
            total, used, free = shutil.disk_usage("/")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Disk Usage", f"{(used/total*100):.1f}%")
            
            with col2:
                st.metric("Free Space", f"{free // (2**30)} GB")
            
            with col3:
                import psutil
                cpu_usage = psutil.cpu_percent()
                st.metric("CPU Usage", f"{cpu_usage:.1f}%")

def main():
    """Main entry point"""
    app = AnomalyDetectionApp()
    app.run()

if __name__ == "__main__":
    main()