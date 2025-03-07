import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from ..utils.logging import AutoTrainerLogger

class ExplainabilityDashboard:
    def __init__(self, model, data):
        self.logger = AutoTrainerLogger()
        self.model = model
        self.data = data
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Model Explainability Dashboard"),
            dcc.Tabs(id="tabs", value='feature-importance', children=[
                dcc.Tab(label='Feature Importance', value='feature-importance'),
                dcc.Tab(label='Partial Dependence', value='partial-dependence'),
                dcc.Tab(label='SHAP Values', value='shap-values'),
            ]),
            html.Div(id='tabs-content')
        ])
        
    def _setup_callbacks(self):
        """Setup dashboard interactivity"""
        @self.app.callback(
            Output('tabs-content', 'children'),
            Input('tabs', 'value')
        )
        def render_content(tab):
            if tab == 'feature-importance':
                return self._create_feature_importance_view()
            elif tab == 'partial-dependence':
                return self._create_partial_dependence_view()
            elif tab == 'shap-values':
                return self._create_shap_values_view()
                
    def _create_feature_importance_view(self):
        """Create feature importance visualization"""
        importances = self.model.feature_importances_
        features = self.data.columns
        df = pd.DataFrame({'Feature': features, 'Importance': importances})
        fig = px.bar(df, x='Feature', y='Importance', title='Feature Importance')
        return dcc.Graph(figure=fig)
        
    def _create_partial_dependence_view(self):
        """Create partial dependence plots"""
        from sklearn.inspection import partial_dependence
        features = self.data.columns[:2]  # Show first two features
        pdp, axes = partial_dependence(self.model, self.data, features=features)
        fig = px.imshow(pdp[0], x=axes[0], y=axes[1], 
                       labels={'x': features[0], 'y': features[1], 'color': 'Partial Dependence'},
                       title='Partial Dependence Plot')
        return dcc.Graph(figure=fig)
        
    def _create_shap_values_view(self):
        """Create SHAP values visualization"""
        import shap
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.data)
        fig = shap.summary_plot(shap_values, self.data, plot_type="bar")
        return dcc.Graph(figure=fig)
        
    def run(self, port: int = 8050):
        """Run the dashboard"""
        self.logger.log(f"Starting explainability dashboard on port {port}")
        self.app.run_server(port=port)