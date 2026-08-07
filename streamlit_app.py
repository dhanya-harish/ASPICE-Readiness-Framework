from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import shap
except Exception:
    shap = None

st.set_page_config(
    page_title='Repository-derived ASPICE Readiness Dashboard',
    page_icon='📊',
    layout='wide',
)

st.title('Repository-derived Automotive SPICE Readiness Dashboard')
st.caption(
    'Continuous repository analytics, Machine Learning prediction, '
    'process-area proxy scores, and explainable AI.'
)
st.info(
    'Research prototype: the readiness score and prediction are repository-derived '
    'proxies and do not replace a formal Automotive SPICE assessment performed by '
    'certified assessors.'
)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / 'data' / 'repository_derived_aspice_dataset.csv'
MODEL_FILE = ROOT / 'models' / 'best_repository_readiness_model.joblib'
METADATA_FILE = ROOT / 'models' / 'model_metadata.json'

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'observation_period' in df.columns:
        df['observation_period'] = pd.to_datetime(df['observation_period'], errors='coerce')
    return df

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    return joblib.load(path)

def load_metadata(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)

missing_files = [str(p) for p in [DATA_FILE, MODEL_FILE, METADATA_FILE] if not p.exists()]
if missing_files:
    st.error('Required files are missing:')
    for item in missing_files:
        st.write(f'- `{item}`')
    st.code(
        'ASPICE-Readiness-Framework/\n'
        '├── streamlit_app.py\n'
        '├── requirements.txt\n'
        '├── data/\n'
        '│   └── repository_derived_aspice_dataset.csv\n'
        '└── models/\n'
        '    ├── best_repository_readiness_model.joblib\n'
        '    └── model_metadata.json'
    )
    st.stop()

df = load_data(DATA_FILE)
model = load_model(MODEL_FILE)
metadata = load_metadata(METADATA_FILE)

feature_columns = metadata.get('feature_columns', [])
target_column = metadata.get('target_column', 'aspice_readiness_level')
score_column = metadata.get('score_column', 'repository_aspice_readiness_score')
best_model_name = metadata.get('best_model_name', 'Selected Model')
class_order = metadata.get('class_order', ['Low', 'Moderate', 'High'])

if not feature_columns:
    st.error('`feature_columns` is missing from models/model_metadata.json.')
    st.stop()

missing_features = [f for f in feature_columns if f not in df.columns]
if missing_features:
    st.error('Dataset is missing model features: ' + ', '.join(missing_features))
    st.stop()

st.sidebar.header('Dashboard Filters')
repositories = sorted(df['repository_name'].dropna().astype(str).unique().tolist())
selected_repository = st.sidebar.selectbox('Repository', ['All'] + repositories)

filtered_df = df.copy()
if selected_repository != 'All':
    filtered_df = filtered_df[filtered_df['repository_name'].astype(str) == selected_repository].copy()

if target_column in filtered_df.columns:
    available_levels = [x for x in class_order if x in filtered_df[target_column].astype(str).unique()]
    selected_levels = st.sidebar.multiselect('Readiness Level', available_levels, default=available_levels)
    if selected_levels:
        filtered_df = filtered_df[filtered_df[target_column].astype(str).isin(selected_levels)].copy()

if filtered_df.empty:
    st.warning('No observations match the selected filters.')
    st.stop()

if 'observation_period' in filtered_df.columns:
    labels = (
        filtered_df['repository_name'].astype(str)
        + ' | '
        + filtered_df['observation_period'].dt.strftime('%Y-%m').fillna('Unknown Period')
    )
else:
    labels = filtered_df['repository_name'].astype(str)

selected_label = st.sidebar.selectbox('Observation', labels.tolist())
selected_position = labels.tolist().index(selected_label)
selected_index = filtered_df.index[selected_position]
selected_row = df.loc[selected_index]

X_selected = pd.DataFrame([selected_row[feature_columns]], columns=feature_columns)
prediction_raw = model.predict(X_selected)[0]
try:
    prediction_index = int(prediction_raw)
    prediction_label = class_order[prediction_index] if 0 <= prediction_index < len(class_order) else str(prediction_raw)
except Exception:
    prediction_label = str(prediction_raw)

prediction_confidence = None
prediction_probabilities = None
if hasattr(model, 'predict_proba'):
    prediction_probabilities = model.predict_proba(X_selected)[0]
    prediction_confidence = float(np.max(prediction_probabilities))

observed_score = float(selected_row[score_column]) if score_column in selected_row.index and pd.notna(selected_row[score_column]) else np.nan
observed_level = str(selected_row[target_column]) if target_column in selected_row.index and pd.notna(selected_row[target_column]) else 'Unknown'

st.subheader('Executive Summary')
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric('Readiness Score', f'{observed_score:.1f}' if not np.isnan(observed_score) else 'N/A')
k2.metric('Observed Readiness', observed_level)
k3.metric('ML Prediction', prediction_label)
k4.metric('Prediction Confidence', f'{prediction_confidence * 100:.1f}%' if prediction_confidence is not None else 'N/A')
k5.metric('Best Model', best_model_name)

left, right = st.columns([1, 2])
with left:
    st.subheader('Readiness Gauge')
    gauge_value = 0 if np.isnan(observed_score) else observed_score
    gauge = go.Figure(go.Indicator(
        mode='gauge+number',
        value=gauge_value,
        title={'text': 'Repository-derived Readiness'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#1f77b4'},
            'steps': [
                {'range': [0, 40], 'color': '#f8d7da'},
                {'range': [40, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#d1e7dd'},
            ],
            'threshold': {'line': {'color': 'black', 'width': 4}, 'thickness': 0.75, 'value': gauge_value},
        },
    ))
    gauge.update_layout(height=330, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(gauge, use_container_width=True)

with right:
    st.subheader('Readiness Trend')
    if score_column in filtered_df.columns and 'observation_period' in filtered_df.columns:
        trend_df = filtered_df.dropna(subset=[score_column, 'observation_period']).sort_values('observation_period')
        trend = px.line(
            trend_df,
            x='observation_period',
            y=score_column,
            color='repository_name',
            markers=True,
            labels={score_column: 'Readiness Score', 'observation_period': 'Observation Period', 'repository_name': 'Repository'},
        )
        trend.add_hline(y=40, line_dash='dash', annotation_text='Low / Moderate')
        trend.add_hline(y=70, line_dash='dash', annotation_text='Moderate / High')
        trend.update_layout(height=330)
        st.plotly_chart(trend, use_container_width=True)
    else:
        st.info('Readiness trend requires observation_period and readiness score columns.')

st.subheader('Prediction Results')
if prediction_probabilities is not None:
    probability_labels = class_order[:len(prediction_probabilities)]
    probability_df = pd.DataFrame({
        'Readiness Level': probability_labels,
        'Probability (%)': prediction_probabilities * 100,
    })
    probability_chart = px.bar(probability_df, x='Readiness Level', y='Probability (%)', text='Probability (%)')
    probability_chart.update_traces(texttemplate='%{text:.1f}%')
    probability_chart.update_layout(height=350, yaxis_range=[0, 100])
    st.plotly_chart(probability_chart, use_container_width=True)
else:
    st.info('Prediction probabilities are not available for the selected model.')

left, right = st.columns(2)
with left:
    st.subheader('Process-Area Proxy Scores')
    proxy_columns = [c for c in df.columns if c.startswith('proxy_') and c.endswith('_score')]
    if proxy_columns:
        radar_values = [float(selected_row[c]) if pd.notna(selected_row[c]) else 0.0 for c in proxy_columns]
        radar_labels = [c.replace('proxy_', '').replace('_score', '').replace('_', '.') for c in proxy_columns]
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_labels + [radar_labels[0]],
            fill='toself',
            name='Process Proxy Score',
        ))
        radar.update_layout(polar={'radialaxis': {'visible': True, 'range': [0, 100]}}, showlegend=False, height=430)
        st.plotly_chart(radar, use_container_width=True)
    else:
        st.info('No process-area proxy score columns were found.')

with right:
    st.subheader('Repository Metrics')
    metric_df = pd.DataFrame({
        'Metric': [f.replace('_', ' ').title() for f in feature_columns],
        'Value': [selected_row[f] for f in feature_columns],
    }).dropna()
    metric_chart = px.bar(metric_df, x='Value', y='Metric', orientation='h')
    metric_chart.update_layout(height=430, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(metric_chart, use_container_width=True)

st.subheader('Global Model Feature Importance')
classifier = model.named_steps.get('classifier', list(model.named_steps.values())[-1]) if hasattr(model, 'named_steps') else model
importance_values = None
if hasattr(classifier, 'feature_importances_'):
    importance_values = np.asarray(classifier.feature_importances_)
elif hasattr(classifier, 'coef_'):
    coef = np.asarray(classifier.coef_)
    importance_values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

if importance_values is not None and len(importance_values) == len(feature_columns):
    importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importance_values}).sort_values('Importance', ascending=False)
    importance_chart = px.bar(importance_df.head(10).sort_values('Importance'), x='Importance', y='Feature', orientation='h')
    importance_chart.update_layout(height=430)
    st.plotly_chart(importance_chart, use_container_width=True)
else:
    st.info('Native feature importance is not available for this model.')

st.subheader('SHAP Explanation')
if shap is None:
    st.warning('SHAP is not installed. Add `shap` to requirements.txt.')
else:
    try:
        if hasattr(model, 'named_steps'):
            preprocessor = model.named_steps.get('preprocessor')
            classifier = model.named_steps.get('classifier')

            X_background = df[feature_columns].copy()
            X_background = X_background.fillna(X_background.median(numeric_only=True))
            X_selected_filled = X_selected.fillna(X_background.median(numeric_only=True))

            X_background_transformed = preprocessor.transform(X_background) if preprocessor is not None else X_background.values
            X_selected_transformed = preprocessor.transform(X_selected_filled) if preprocessor is not None else X_selected_filled.values

            if X_background_transformed.shape[0] > 100:
                X_background_transformed = X_background_transformed[:100]

            explainer = shap.Explainer(classifier, X_background_transformed)
            explanation = explainer(X_selected_transformed)
            values = np.asarray(explanation.values)

            if values.ndim == 3:
                class_idx = int(np.argmax(prediction_probabilities)) if prediction_probabilities is not None else 0
                local_values = values[0, :, class_idx]
            elif values.ndim == 2:
                local_values = values[0, :]
            else:
                local_values = values.reshape(-1)

            if len(local_values) == len(feature_columns):
                shap_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'SHAP Value': local_values,
                    'Absolute SHAP': np.abs(local_values),
                }).sort_values('Absolute SHAP', ascending=False).head(10)
                shap_chart = px.bar(shap_df.sort_values('SHAP Value'), x='SHAP Value', y='Feature', orientation='h')
                shap_chart.add_vline(x=0, line_width=1)
                shap_chart.update_layout(height=430)
                st.plotly_chart(shap_chart, use_container_width=True)
                st.caption('Positive SHAP values increase the selected-class prediction; negative values reduce it.')
            else:
                st.info('SHAP values could not be aligned with the model feature set.')
        else:
            st.info('SHAP explanation requires the saved preprocessing/model pipeline.')
    except Exception as error:
        st.warning('SHAP explanation could not be generated for this observation.')
        st.caption(f'Technical detail: {error}')

st.subheader('Repository-derived Recommendations')
higher_is_better = {'commit_frequency', 'unique_authors', 'author_diversity_ratio'}
lower_is_better = {'avg_commit_size', 'code_churn', 'avg_files_per_commit', 'avg_dmm_unit_size', 'avg_dmm_complexity'}
recommendations = []

for feature in feature_columns:
    value = selected_row.get(feature, np.nan)
    if pd.isna(value):
        continue
    q25 = df[feature].quantile(0.25)
    q75 = df[feature].quantile(0.75)
    friendly_name = feature.replace('_', ' ').title()

    if feature in higher_is_better and value < q25:
        recommendations.append(
            f'Improve **{friendly_name}**: the selected observation is below the lower quartile of the repository-derived reference dataset.'
        )
    elif feature in lower_is_better and value > q75:
        recommendations.append(
            f'Review **{friendly_name}**: the selected observation is above the upper quartile and may indicate larger or less manageable changes.'
        )

if not recommendations:
    recommendations.append(
        'No major repository-metric warning was identified for this observation. Continue monitoring trends and validate the results against formal process evidence.'
    )

for recommendation in recommendations[:5]:
    st.markdown(f'- {recommendation}')

with st.expander('Selected Observation Details'):
    st.dataframe(selected_row.to_frame('Value'), use_container_width=True)

with st.expander('Filtered Repository Dataset'):
    st.dataframe(filtered_df, use_container_width=True)

st.download_button(
    label='Download Filtered Data',
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='filtered_repository_readiness_data.csv',
    mime='text/csv',
)

st.divider()
st.caption(
    'AI-Driven Repository Analytics Framework for Continuous Automotive SPICE '
    'Readiness Assessment Using Explainable Machine Learning.'
)
