"""
Interactive Option Greeks Visualization App using Bokeh

Visualize Delta, Gamma, and Vega as functions of strike or maturity
for European options using Black-Scholes formulas.

Run with: uv run bokeh serve --show theme_05/option_greeks_app.py
"""

import numpy as np
from scipy.stats import norm
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Slider, RadioButtonGroup, Div, ColumnDataSource, Legend, LegendItem, Range1d
)
from bokeh.plotting import figure
from bokeh.palettes import Category10_3


# =============================================================================
# Translations
# =============================================================================

TRANSLATIONS = {
    'en': {
        'title': "<h2>Option Greeks Visualization</h2>",
        'language_label': "<b>Language / Langue:</b>",
        'option_type_label': "<b>Option Type:</b>",
        'call': "Call",
        'put': "Put",
        'mode_label': "<b>Plot as function of:</b>",
        'mode_spot': "Spot Price (S)",
        'mode_maturity': "Maturity (T)",
        'params_label': "<b>Option Parameters:</b>",
        'strike_price': "Strike Price (K)",
        'volatility': "Volatility (σ)",
        'risk_free_rate': "Risk-free Rate (r)",
        'dividend_yield': "Dividend Yield (δ)",
        'maturities_label': "<b>Maturities (years):</b>",
        'strikes_label': "<b>Strikes (as K/S moneyness):</b>",
        't1_label': "T₁ (blue)",
        't2_label': "T₂ (orange)",
        't3_label': "T₃ (green)",
        'k1_label': "K₁/S (blue)",
        'k2_label': "K₂/S (orange)",
        'k3_label': "K₃/S (green)",
        'delta_title': "Delta (Δ)",
        'gamma_title': "Gamma (Γ)",
        'vega_title': "Vega (ν) per 1% vol",
        'spot_axis': "Spot Price (S)",
        'maturity_axis': "Maturity (T)",
        'doc_title': "Option Greeks Visualization",
    },
    'fr': {
        'title': "<h2>Visualisation des Grecques</h2>",
        'language_label': "<b>Language / Langue:</b>",
        'option_type_label': "<b>Type d'option:</b>",
        'call': "Call",
        'put': "Put",
        'mode_label': "<b>Tracer en fonction de:</b>",
        'mode_spot': "Prix spot (S)",
        'mode_maturity': "Maturité (T)",
        'params_label': "<b>Paramètres de l'option:</b>",
        'strike_price': "Prix d'exercice (K)",
        'volatility': "Volatilité (σ)",
        'risk_free_rate': "Taux sans risque (r)",
        'dividend_yield': "Rendement en dividendes (δ)",
        'maturities_label': "<b>Maturités (années):</b>",
        'strikes_label': "<b>Strikes (moneyness K/S):</b>",
        't1_label': "T₁ (bleu)",
        't2_label': "T₂ (orange)",
        't3_label': "T₃ (vert)",
        'k1_label': "K₁/S (bleu)",
        'k2_label': "K₂/S (orange)",
        'k3_label': "K₃/S (vert)",
        'delta_title': "Delta (Δ)",
        'gamma_title': "Gamma (Γ)",
        'vega_title': "Vega (ν) par 1% de vol",
        'spot_axis': "Prix spot (S)",
        'maturity_axis': "Maturité (T)",
        'doc_title': "Visualisation des Grecques",
    }
}


# =============================================================================
# Black-Scholes Greeks Formulas
# =============================================================================

def d1(S, K, r, delta, sigma, T):
    """Calculate d1 in Black-Scholes formula"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (np.log(S / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return np.where(T > 0, result, 0)


def d2(S, K, r, delta, sigma, T):
    """Calculate d2 in Black-Scholes formula"""
    return d1(S, K, r, delta, sigma, T) - sigma * np.sqrt(T)


def bs_delta_call(S, K, r, delta, sigma, T):
    """Delta for a European call option"""
    d1_val = d1(S, K, r, delta, sigma, T)
    return np.exp(-delta * T) * norm.cdf(d1_val)


def bs_delta_put(S, K, r, delta, sigma, T):
    """Delta for a European put option"""
    d1_val = d1(S, K, r, delta, sigma, T)
    return -np.exp(-delta * T) * norm.cdf(-d1_val)


def bs_gamma(S, K, r, delta, sigma, T):
    """Gamma for European call or put (same for both)"""
    d1_val = d1(S, K, r, delta, sigma, T)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.exp(-delta * T) * norm.pdf(d1_val) / (S * sigma * np.sqrt(T))
        return np.where(T > 0, result, 0)


def bs_vega(S, K, r, delta, sigma, T):
    """Vega for European call or put (same for both)"""
    d1_val = d1(S, K, r, delta, sigma, T)
    # Return vega per 1% change in volatility (divide by 100)
    return S * np.exp(-delta * T) * norm.pdf(d1_val) * np.sqrt(T) / 100


# =============================================================================
# Application Class
# =============================================================================

class OptionGreeksApp:
    """Main application class for option greeks visualization"""

    def __init__(self):
        # Default parameters
        self.K = 100.0  # Strike price (fixed for the option)
        self.sigma = 0.30
        self.r = 0.05
        self.delta = 0.02

        # For spot price plots: three maturities
        self.T1 = 0.25
        self.T2 = 0.50
        self.T3 = 1.00

        # For maturity plots: three strikes (as K/S moneyness)
        self.moneyness1 = 0.80
        self.moneyness2 = 1.00
        self.moneyness3 = 1.20

        # Plot mode: 0 = vs Spot Price, 1 = vs Maturity
        self.plot_mode = 0

        # Option type: 0 = Call, 1 = Put
        self.option_type = 0

        # Language: 0 = English, 1 = French
        self.lang = 'en'

        # Create data sources for three lines on each plot
        self.delta_sources = [ColumnDataSource(data={'x': [], 'y': []}) for _ in range(3)]
        self.gamma_sources = [ColumnDataSource(data={'x': [], 'y': []}) for _ in range(3)]
        self.vega_sources = [ColumnDataSource(data={'x': [], 'y': []}) for _ in range(3)]

        # Build UI
        self._create_ui()
        self._update_plots()

    def _t(self, key):
        """Get translation for current language"""
        return TRANSLATIONS[self.lang][key]

    def _create_ui(self):
        """Create all UI components"""

        # Title
        self.title = Div(
            text=self._t('title'),
            width=300, height=50
        )

        # Language selector
        self.language_label = Div(text=self._t('language_label'), width=300, height=20)
        self.language_selector = RadioButtonGroup(
            labels=["English", "Français"], active=0, width=300
        )
        self.language_selector.on_change('active', self._on_language_change)

        # Option type selector
        self.option_type_label = Div(text=self._t('option_type_label'), width=300, height=20)
        self.option_type_selector = RadioButtonGroup(
            labels=[self._t('call'), self._t('put')], active=0, width=300
        )
        self.option_type_selector.on_change('active', self._on_option_type_change)

        # Plot mode selector
        self.mode_label = Div(text=self._t('mode_label'), width=300, height=20)
        self.mode_selector = RadioButtonGroup(
            labels=[self._t('mode_spot'), self._t('mode_maturity')], active=0, width=300
        )
        self.mode_selector.on_change('active', self._on_mode_change)

        # Common parameters section
        self.params_label = Div(text=self._t('params_label'), width=300, height=25)

        self.K_slider = Slider(
            start=50, end=200, value=self.K, step=1,
            title=self._t('strike_price'), width=280
        )
        self.K_slider.on_change('value', self._on_param_change)

        self.sigma_slider = Slider(
            start=0.05, end=0.80, value=self.sigma, step=0.01,
            title=self._t('volatility'), width=280, format="0.00"
        )
        self.sigma_slider.on_change('value', self._on_param_change)

        self.r_slider = Slider(
            start=0.00, end=0.15, value=self.r, step=0.005,
            title=self._t('risk_free_rate'), width=280, format="0.000"
        )
        self.r_slider.on_change('value', self._on_param_change)

        self.delta_slider = Slider(
            start=0.00, end=0.10, value=self.delta, step=0.005,
            title=self._t('dividend_yield'), width=280, format="0.000"
        )
        self.delta_slider.on_change('value', self._on_param_change)

        # Maturity sliders (for plotting vs strike)
        self.maturity_label = Div(text=self._t('maturities_label'), width=300, height=25)

        self.T1_slider = Slider(
            start=0.05, end=1.0, value=self.T1, step=0.05,
            title=self._t('t1_label'), width=280, format="0.00"
        )
        self.T1_slider.on_change('value', self._on_param_change)

        self.T2_slider = Slider(
            start=0.05, end=1.0, value=self.T2, step=0.05,
            title=self._t('t2_label'), width=280, format="0.00"
        )
        self.T2_slider.on_change('value', self._on_param_change)

        self.T3_slider = Slider(
            start=0.05, end=1.0, value=self.T3, step=0.05,
            title=self._t('t3_label'), width=280, format="0.00"
        )
        self.T3_slider.on_change('value', self._on_param_change)

        # Strike sliders (for plotting vs maturity)
        self.strike_label = Div(text=self._t('strikes_label'), width=300, height=25)

        self.K1_slider = Slider(
            start=0.50, end=1.50, value=self.moneyness1, step=0.05,
            title=self._t('k1_label'), width=280, format="0.00"
        )
        self.K1_slider.on_change('value', self._on_param_change)

        self.K2_slider = Slider(
            start=0.50, end=1.50, value=self.moneyness2, step=0.05,
            title=self._t('k2_label'), width=280, format="0.00"
        )
        self.K2_slider.on_change('value', self._on_param_change)

        self.K3_slider = Slider(
            start=0.50, end=1.50, value=self.moneyness3, step=0.05,
            title=self._t('k3_label'), width=280, format="0.00"
        )
        self.K3_slider.on_change('value', self._on_param_change)

        # Create the three plots
        colors = Category10_3
        initial_labels = ["T=0.25", "T=0.50", "T=1.00"]

        # Delta plot (no legend)
        self.delta_plot = figure(
            title="Delta (Δ)", width=350, height=300,
            tools="pan,wheel_zoom,reset", toolbar_location="above"
        )
        for i, (source, color) in enumerate(zip(self.delta_sources, colors)):
            self.delta_plot.line('x', 'y', source=source, line_width=2, color=color)

        # Gamma plot (no legend)
        self.gamma_plot = figure(
            title="Gamma (Γ)", width=350, height=300,
            tools="pan,wheel_zoom,reset", toolbar_location="above"
        )
        for i, (source, color) in enumerate(zip(self.gamma_sources, colors)):
            self.gamma_plot.line('x', 'y', source=source, line_width=2, color=color)

        # Vega plot (no legend)
        self.vega_plot = figure(
            title="Vega (ν) per 1% vol", width=350, height=300,
            tools="pan,wheel_zoom,reset", toolbar_location="above"
        )
        for i, (source, color) in enumerate(zip(self.vega_sources, colors)):
            self.vega_plot.line('x', 'y', source=source, line_width=2, color=color)

        # Create a separate legend panel (invisible plot with just the legend)
        self.legend_panel = figure(
            width=350, height=60,
            toolbar_location=None, tools="",
            outline_line_color=None
        )
        self.legend_panel.axis.visible = False
        self.legend_panel.grid.visible = False

        # Create dummy renderers for the shared legend
        legend_items = []
        for i, (color, label) in enumerate(zip(colors, initial_labels)):
            r = self.legend_panel.line([0], [0], line_width=2, color=color)
            legend_items.append(LegendItem(label=label, renderers=[r]))

        self.shared_legend = Legend(
            items=legend_items,
            orientation="horizontal",
            location="center"
        )
        self.legend_panel.add_layout(self.shared_legend, 'center')

        # Group maturity and strike controls
        self.maturity_controls = column(
            self.maturity_label,
            self.T1_slider, self.T2_slider, self.T3_slider
        )

        self.strike_controls = column(
            self.strike_label,
            self.K1_slider, self.K2_slider, self.K3_slider
        )

        # Build the layout
        self._build_layout()

    def _build_layout(self):
        """Build the application layout based on current mode"""

        # Left panel always has common controls
        left_panel = column(
            self.title,
            self.language_label,
            self.language_selector,
            Div(text="<hr>", width=300, height=10),
            self.option_type_label,
            self.option_type_selector,
            Div(text="<hr>", width=300, height=10),
            self.mode_label,
            self.mode_selector,
            Div(text="<hr>", width=300, height=10),
            self.params_label,
            self.K_slider,
            self.sigma_slider,
            self.r_slider,
            self.delta_slider,
            Div(text="<hr>", width=300, height=10),
        )

        # Add mode-specific controls
        if self.plot_mode == 0:  # vs Strike
            left_panel.children.append(self.maturity_controls)
        else:  # vs Maturity
            left_panel.children.append(self.strike_controls)

        # Right panel with plots and legend underneath middle plot
        from bokeh.layouts import Spacer
        plots_row = row(self.delta_plot, self.gamma_plot, self.vega_plot)
        legend_row = row(Spacer(width=350), self.legend_panel, Spacer(width=350))
        right_panel = column(plots_row, legend_row)

        # Full layout
        self.layout = row(left_panel, right_panel)

    def _on_language_change(self, attr, old, new):
        """Handle language change"""
        self.lang = 'en' if new == 0 else 'fr'
        self._update_ui_text()
        self._update_plots()
        curdoc().title = self._t('doc_title')

    def _update_ui_text(self):
        """Update all UI text elements for current language"""
        # Update labels and titles
        self.title.text = self._t('title')
        self.language_label.text = self._t('language_label')
        self.option_type_label.text = self._t('option_type_label')
        self.option_type_selector.labels = [self._t('call'), self._t('put')]
        self.mode_label.text = self._t('mode_label')
        self.mode_selector.labels = [self._t('mode_spot'), self._t('mode_maturity')]
        self.params_label.text = self._t('params_label')

        # Update slider titles
        self.K_slider.title = self._t('strike_price')
        self.sigma_slider.title = self._t('volatility')
        self.r_slider.title = self._t('risk_free_rate')
        self.delta_slider.title = self._t('dividend_yield')

        # Update maturity/strike labels
        self.maturity_label.text = self._t('maturities_label')
        self.T1_slider.title = self._t('t1_label')
        self.T2_slider.title = self._t('t2_label')
        self.T3_slider.title = self._t('t3_label')

        self.strike_label.text = self._t('strikes_label')
        self.K1_slider.title = self._t('k1_label')
        self.K2_slider.title = self._t('k2_label')
        self.K3_slider.title = self._t('k3_label')

    def _on_mode_change(self, attr, old, new):
        """Handle plot mode change"""
        self.plot_mode = new
        self._build_layout()
        self._update_plots()
        curdoc().clear()
        curdoc().add_root(self.layout)

    def _on_option_type_change(self, attr, old, new):
        """Handle option type change"""
        self.option_type = new
        self._update_plots()

    def _on_param_change(self, attr, old, new):
        """Handle parameter slider changes"""
        self.K = self.K_slider.value
        self.sigma = self.sigma_slider.value
        self.r = self.r_slider.value
        self.delta = self.delta_slider.value
        self.T1 = self.T1_slider.value
        self.T2 = self.T2_slider.value
        self.T3 = self.T3_slider.value
        self.moneyness1 = self.K1_slider.value
        self.moneyness2 = self.K2_slider.value
        self.moneyness3 = self.K3_slider.value
        self._update_plots()

    def _update_plots(self):
        """Update all three plots based on current parameters"""

        K = self.K
        sigma = self.sigma
        r = self.r
        delta = self.delta

        # Always read current mode from the widget to avoid sync issues
        current_mode = self.mode_selector.active

        if current_mode == 0:  # Plot vs Spot Price
            # X-axis: spot prices from 0.5*K to 1.5*K
            S_range = np.linspace(0.5 * K, 1.5 * K, 200)
            maturities = [self.T1, self.T2, self.T3]

            for i, T in enumerate(maturities):
                if self.option_type == 0:  # Call
                    delta_vals = bs_delta_call(S_range, K, r, delta, sigma, T)
                else:  # Put
                    delta_vals = bs_delta_put(S_range, K, r, delta, sigma, T)

                gamma_vals = bs_gamma(S_range, K, r, delta, sigma, T)
                vega_vals = bs_vega(S_range, K, r, delta, sigma, T)

                self.delta_sources[i].data = {'x': S_range, 'y': delta_vals}
                self.gamma_sources[i].data = {'x': S_range, 'y': gamma_vals}
                self.vega_sources[i].data = {'x': S_range, 'y': vega_vals}

            # Update axis labels and ranges
            self.delta_plot.xaxis.axis_label = self._t('spot_axis')
            self.gamma_plot.xaxis.axis_label = self._t('spot_axis')
            self.vega_plot.xaxis.axis_label = self._t('spot_axis')

            # Explicitly set x_range using Range1d
            x_min, x_max = float(S_range[0]), float(S_range[-1])
            self.delta_plot.x_range = Range1d(start=x_min, end=x_max)
            self.gamma_plot.x_range = Range1d(start=x_min, end=x_max)
            self.vega_plot.x_range = Range1d(start=x_min, end=x_max)

            # Update titles
            option_str = self._t('call') if self.option_type == 0 else self._t('put')
            self.delta_plot.title.text = f"{self._t('delta_title')} - {option_str}"
            self.gamma_plot.title.text = self._t('gamma_title')
            self.vega_plot.title.text = self._t('vega_title')

            # Update shared legend labels
            labels = [f"T={T:.2f}" for T in maturities]
            for i, label in enumerate(labels):
                self.shared_legend.items[i].label = {'value': label}

        else:  # Plot vs Maturity
            # X-axis: maturities from 0.05 to 1 year
            T_range = np.linspace(0.05, 1.0, 200)
            # For maturity plots, use moneyness relative to strike K
            # moneyness = K/S, so S = K/moneyness
            spot_prices = [K / self.moneyness1, K / self.moneyness2, K / self.moneyness3]

            for i, S in enumerate(spot_prices):
                if self.option_type == 0:  # Call
                    delta_vals = bs_delta_call(S, K, r, delta, sigma, T_range)
                else:  # Put
                    delta_vals = bs_delta_put(S, K, r, delta, sigma, T_range)

                gamma_vals = bs_gamma(S, K, r, delta, sigma, T_range)
                vega_vals = bs_vega(S, K, r, delta, sigma, T_range)

                self.delta_sources[i].data = {'x': T_range, 'y': delta_vals}
                self.gamma_sources[i].data = {'x': T_range, 'y': gamma_vals}
                self.vega_sources[i].data = {'x': T_range, 'y': vega_vals}

            # Update axis labels and ranges
            self.delta_plot.xaxis.axis_label = self._t('maturity_axis')
            self.gamma_plot.xaxis.axis_label = self._t('maturity_axis')
            self.vega_plot.xaxis.axis_label = self._t('maturity_axis')

            # Explicitly set x_range using Range1d
            x_min, x_max = float(T_range[0]), float(T_range[-1])
            self.delta_plot.x_range = Range1d(start=x_min, end=x_max)
            self.gamma_plot.x_range = Range1d(start=x_min, end=x_max)
            self.vega_plot.x_range = Range1d(start=x_min, end=x_max)

            # Update titles
            option_str = self._t('call') if self.option_type == 0 else self._t('put')
            self.delta_plot.title.text = f"{self._t('delta_title')} - {option_str}"
            self.gamma_plot.title.text = self._t('gamma_title')
            self.vega_plot.title.text = self._t('vega_title')

            # Update shared legend labels
            moneynesses = [self.moneyness1, self.moneyness2, self.moneyness3]
            labels = [f"K/S={m:.2f}" for m in moneynesses]
            for i, label in enumerate(labels):
                self.shared_legend.items[i].label = {'value': label}

        # Update y-axis labels
        self.delta_plot.yaxis.axis_label = "Δ"
        self.gamma_plot.yaxis.axis_label = "Γ"
        self.vega_plot.yaxis.axis_label = "ν"


# =============================================================================
# Main
# =============================================================================

app = OptionGreeksApp()
curdoc().add_root(app.layout)
curdoc().title = app._t('doc_title')
