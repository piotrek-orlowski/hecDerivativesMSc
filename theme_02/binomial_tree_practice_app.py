"""
Interactive Binomial Tree Practice App using Bokeh

A step-by-step educational tool for students to practice binomial tree
construction and option pricing through pen-and-paper calculations.

Run with: bokeh serve --show binomial_tree_practice_app.py
Developer mode: DEV_MODE=1 bokeh serve --show binomial_tree_practice_app.py
"""

import numpy as np
import os
from bokeh.io import curdoc
from bokeh.layouts import column, row, Spacer
from bokeh.models import (
    Button, TextInput, Select, Div, PreText,
    ColumnDataSource, Circle, MultiLine, Text as BokehText
)
from bokeh.plotting import figure
from bintree import ForwardBinomialTree, call_payoff, put_payoff

# Check for developer mode
DEV_MODE = os.environ.get('DEV_MODE', '0') == '1'


class BinomialTreePracticeApp:
    """Main application class for the binomial tree practice tool"""

    def __init__(self):
        # Application state
        self.phase = "input"  # input, params, stock_tree, option_choice, terminal, pricing
        self.tree = None
        self.option_type = None  # 'european' or 'american'
        self.payoff_type = None  # 'call', 'put', 'digital_call', 'digital_put', 'custom'
        self.payoff_func = None
        self.payoff_description = ""
        self.pricing_method = None  # 'risk_neutral', 'replication', 'expectation'

        # Create UI components
        self._create_ui()

        # Initial display
        self._show_input_phase()

    def _create_ui(self):
        """Create all UI components"""

        # Title
        title_text = "<h1>Binomial Tree Practice Tool</h1>"
        if DEV_MODE:
            title_text += "<p style='color:red; font-weight:bold;'>⚠️ DEVELOPER MODE ACTIVE ⚠️</p>"
        self.title = Div(text=title_text, width=1200, height=80 if DEV_MODE else 60)

        # Instructions (left column)
        self.instructions = Div(text="", width=450, height=400)

        # Input Phase Widgets
        self.input_s0 = TextInput(title="Initial Stock Price (S0):", value="100")
        self.input_rf = TextInput(title="Risk-Free Rate (continuously compounded):", value="0.05")
        self.input_div = TextInput(title="Dividend Yield (continuously compounded):", value="0.02")
        self.input_t = TextInput(title="Maturity (years):", value="1.0")
        self.input_n = TextInput(title="Number of Periods (max 4):", value="3")
        self.input_sigma = TextInput(title="Volatility:", value="0.30")
        self.btn_confirm_inputs = Button(label="Confirm Inputs", button_type="success")
        self.btn_confirm_inputs.on_click(self._on_confirm_inputs)

        self.input_widgets = column(
            self.input_s0, self.input_rf, self.input_div,
            self.input_t, self.input_n, self.input_sigma,
            self.btn_confirm_inputs
        )

        # Parameter Calculation Phase
        self.param_h = TextInput(title="Period Length (h):", value="")
        self.param_fwd_ratio = TextInput(title="Forward Ratio:", value="")
        self.param_up = TextInput(title="Up Factor (u):", value="")
        self.param_down = TextInput(title="Down Factor (d):", value="")
        self.param_pstar = TextInput(title="Risk-Neutral Probability (p*):", value="")
        self.btn_verify_params = Button(label="Ready to Verify My Calculations", button_type="primary")
        self.btn_verify_params.on_click(self._on_verify_params)
        self.btn_fill_params = Button(label="[DEV] Fill Correct Answers", button_type="danger", visible=DEV_MODE)
        self.btn_fill_params.on_click(self._fill_params)
        self.btn_proceed_stock = Button(label="Proceed to Stock Price Tree", button_type="success", visible=False)
        self.btn_proceed_stock.on_click(self._on_proceed_to_stock_tree)

        self.param_results = Div(text="", width=1100)

        # Stock Price Tree Calculation Phase
        self.stock_tree_inputs_container = column()  # Will be populated dynamically
        self.stock_price_inputs = {}  # Dictionary to store TextInput widgets by (period, index)
        self.btn_verify_stock_tree = Button(label="Verify Stock Prices", button_type="primary")
        self.btn_verify_stock_tree.on_click(self._on_verify_stock_tree)
        self.btn_fill_stock = Button(label="[DEV] Fill Correct Answers", button_type="danger", visible=DEV_MODE)
        self.btn_fill_stock.on_click(self._fill_stock_prices)
        self.stock_tree_results = Div(text="", width=1100)
        self.btn_proceed_option = Button(label="Proceed to Option Selection", button_type="success", visible=False)
        self.btn_proceed_option.on_click(self._on_proceed_to_option_choice)

        # Option Choice Phase
        self.option_style_select = Select(title="Option Style:",
                                          value="European",
                                          options=["European", "American"])
        self.option_type_select = Select(title="Option Type:",
                                         value="Call",
                                         options=["Call", "Put", "Digital Call", "Digital Put", "Custom"])
        self.option_type_select.on_change('value', self._on_option_type_change)
        self.option_strike_input = TextInput(title="Strike Price (K):", value="100")
        self.option_payout_input = TextInput(title="Digital Payout (for digital options):",
                                            value="100", visible=False)
        self.custom_payoff_input = TextInput(
            title="Custom Payoff Formula (use 'S' for stock price, e.g., 'max(S - 95, 0) - max(S - 105, 0)'):",
            value="", visible=False)
        self.btn_confirm_option = Button(label="Confirm Option Choice", button_type="success")
        self.btn_confirm_option.on_click(self._on_confirm_option)

        self.option_widgets = column(
            self.option_style_select, self.option_type_select,
            self.option_strike_input, self.option_payout_input,
            self.custom_payoff_input, self.btn_confirm_option
        )

        # Terminal Payoff Phase
        self.terminal_inputs_container = row()  # Will hold input fields
        self.terminal_payoff_inputs = {}  # Dictionary to store TextInput widgets by index
        self.btn_verify_terminal = Button(label="Verify Terminal Payoffs", button_type="primary")
        self.btn_verify_terminal.on_click(self._on_verify_terminal)
        self.btn_fill_terminal = Button(label="[DEV] Fill Correct Answers", button_type="danger", visible=DEV_MODE)
        self.btn_fill_terminal.on_click(self._fill_terminal_payoffs)
        self.terminal_results = Div(text="", width=1100)
        self.btn_proceed_pricing = Button(label="Proceed to Pricing", button_type="success", visible=False)
        self.btn_proceed_pricing.on_click(self._on_proceed_to_pricing)

        # Pricing Phase
        self.pricing_method_select = Select(
            title="Choose Pricing Method:",
            value="Backward Induction (Risk-Neutral)",
            options=["Backward Induction (Risk-Neutral)",
                    "Backward Induction (Replication)",
                    "Risk-Neutral Expectation (European only)"]
        )
        self.pricing_method_select.on_change('value', self._on_pricing_method_change)
        self.pricing_inputs_container = column()  # Will hold input fields based on method
        self.pricing_value_inputs = {}  # For risk-neutral values
        self.pricing_delta_inputs = {}  # For delta values
        self.pricing_bond_inputs = {}   # For bond values
        self.pricing_expectation_input = None  # For final expectation value
        self.btn_verify_pricing = Button(label="Verify Pricing", button_type="primary")
        self.btn_verify_pricing.on_click(self._on_verify_pricing)
        self.btn_fill_pricing = Button(label="[DEV] Fill Correct Answers", button_type="danger", visible=DEV_MODE)
        self.btn_fill_pricing.on_click(self._fill_pricing_values)
        self.pricing_results = Div(text="", width=1100)
        self.btn_retry_pricing = Button(label="Try Different Pricing Method", button_type="success", visible=False)
        self.btn_retry_pricing.on_click(self._retry_pricing)
        self.btn_restart = Button(label="Start New Problem", button_type="warning", visible=False)
        self.btn_restart.on_click(self._restart)

        # Main content areas
        self.left_column = column(self.instructions)  # Instructions on left
        self.right_column = column()  # Input fields on right
        self.results_area = column()  # Results below, full width
        self.buttons_area = column()  # Buttons at bottom-right of results

        # Layout: Two columns for instructions/inputs, then results below
        self.layout = column(
            self.title,
            Spacer(height=10),
            row(self.left_column, Spacer(width=20), self.right_column),
            Spacer(height=20),
            self.results_area,
            row(Spacer(width=1), self.buttons_area)  # Right-aligned buttons
        )

    def _show_input_phase(self):
        """Display the initial input phase"""
        self.phase = "input"
        self.instructions.text = """
        <h2>Step 1: Enter Model Parameters</h2>
        <p>Enter the parameters for your binomial tree model. The tree will be constructed
        but hidden from you until you complete the calculations.</p>
        <p><strong>Note:</strong> Maximum 4 periods allowed for practice.</p>
        """
        self.right_column.children = [self.input_widgets]
        self.results_area.children = []
        self.buttons_area.children = []

    def _on_confirm_inputs(self):
        """Handle input confirmation"""
        try:
            # Parse inputs
            s0 = float(self.input_s0.value)
            rf = float(self.input_rf.value)
            div_yld = float(self.input_div.value)
            T = float(self.input_t.value)
            N = int(self.input_n.value)
            sigma = float(self.input_sigma.value)

            # Validate
            if N > 4:
                self.instructions.text = "<p style='color:red;'>Error: Maximum 4 periods allowed!</p>"
                return
            if N < 1:
                self.instructions.text = "<p style='color:red;'>Error: Must have at least 1 period!</p>"
                return

            # Create tree (hidden from student)
            self.tree = ForwardBinomialTree(
                S0=s0, T_maturity=T, N_periods=N,
                sigma=sigma, rf=rf, div_yld=div_yld
            )

            # Move to parameter calculation phase
            self._show_param_phase()

        except ValueError as e:
            self.instructions.text = f"<p style='color:red;'>Error: Invalid input. Please enter valid numbers.</p>"

    def _show_param_phase(self):
        """Show parameter calculation phase"""
        self.phase = "params"
        self.instructions.text = """
        <h2>Step 2: Calculate Tree Parameters</h2>
        <p>Calculate the following parameters for your binomial tree:</p>
        <ul>
        <li><strong>Period Length (h)</strong></li>
        <li><strong>Forward Ratio</strong></li>
        <li><strong>Up Factor (u)</strong></li>
        <li><strong>Down Factor (d)</strong></li>
        <li><strong>Risk-Neutral Probability (p*)</strong></li>
        </ul>
        <p>Enter your calculations below. When ready, click the verify button.</p>
        <p><strong>Required Precision:</strong> 1e-4 (4 decimal places minimum)</p>
        """

        # Reset parameter inputs
        self.param_h.value = ""
        self.param_fwd_ratio.value = ""
        self.param_up.value = ""
        self.param_down.value = ""
        self.param_pstar.value = ""
        self.param_results.text = ""
        self.btn_proceed_stock.visible = False

        # Update layout: inputs on right, results below
        param_inputs = column(
            self.param_h, self.param_fwd_ratio, self.param_up,
            self.param_down, self.param_pstar
        )
        param_buttons = [self.btn_verify_params]
        if DEV_MODE:
            param_buttons.append(self.btn_fill_params)

        self.right_column.children = [param_inputs, row(*param_buttons)]
        self.results_area.children = [self.param_results]
        self.buttons_area.children = [self.btn_proceed_stock]

    def _on_verify_params(self):
        """Verify student's parameter calculations"""
        try:
            student_h = float(self.param_h.value)
            student_fwd = float(self.param_fwd_ratio.value)
            student_up = float(self.param_up.value)
            student_down = float(self.param_down.value)
            student_pstar = float(self.param_pstar.value)

            # Compare with actual values
            correct_h = self.tree.h_period
            correct_fwd = self.tree.fwd_ratio
            correct_up = self.tree.up_factor
            correct_down = self.tree.down_factor
            correct_pstar = self.tree.pstar

            tolerance = 0.0001

            results = "<h3>Verification Results:</h3>"
            results += "<h4>Formulas:</h4><ul>"
            results += "<li><strong>h</strong> = T / N</li>"
            results += "<li><strong>fwd_ratio</strong> = exp((rf - div_yld) × h)</li>"
            results += "<li><strong>u</strong> = fwd_ratio × exp(σ × √h)</li>"
            results += "<li><strong>d</strong> = fwd_ratio × exp(-σ × √h)</li>"
            results += "<li><strong>p*</strong> = (fwd_ratio - d) / (u - d)</li>"
            results += "</ul>"
            results += "<table border='1' style='border-collapse:collapse; padding:5px;'>"
            results += "<tr><th>Parameter</th><th>Your Answer</th><th>Correct Answer</th><th>Status</th></tr>"

            def check(name, student, correct):
                match = abs(student - correct) < tolerance
                status = "✓ Correct" if match else "✗ Incorrect"
                color = "green" if match else "red"
                return f"<tr><td>{name}</td><td>{student:.6f}</td><td>{correct:.6f}</td><td style='color:{color};'>{status}</td></tr>"

            results += check("h (Period Length)", student_h, correct_h)
            results += check("Forward Ratio", student_fwd, correct_fwd)
            results += check("Up Factor", student_up, correct_up)
            results += check("Down Factor", student_down, correct_down)
            results += check("p*", student_pstar, correct_pstar)
            results += "</table>"

            # Check if all correct
            all_correct = (
                abs(student_h - correct_h) < tolerance and
                abs(student_fwd - correct_fwd) < tolerance and
                abs(student_up - correct_up) < tolerance and
                abs(student_down - correct_down) < tolerance and
                abs(student_pstar - correct_pstar) < tolerance
            )

            if all_correct:
                results += "<p style='color:green; font-weight:bold;'>Excellent! All calculations are correct!</p>"
                self.btn_proceed_stock.visible = True
            else:
                results += "<p style='color:orange;'>Review your calculations and try again.</p>"

            self.param_results.text = results

        except ValueError:
            self.param_results.text = "<p style='color:red;'>Error: Please enter valid numbers for all parameters.</p>"

    def _fill_params(self):
        """[DEV] Fill in correct parameter values"""
        self.param_h.value = str(self.tree.h_period)
        self.param_fwd_ratio.value = str(self.tree.fwd_ratio)
        self.param_up.value = str(self.tree.up_factor)
        self.param_down.value = str(self.tree.down_factor)
        self.param_pstar.value = str(self.tree.pstar)

    def _on_proceed_to_stock_tree(self):
        """Proceed to stock price tree calculation"""
        self._show_stock_tree_phase()

    def _show_stock_tree_phase(self):
        """Show stock price tree calculation phase"""
        self.phase = "stock_tree"

        N = self.tree.N_periods
        self.instructions.text = f"""
        <h2>Step 3: Calculate Stock Prices</h2>
        <p><strong>Parameters:</strong> S0 = ${self.tree.S0:.2f}, u = {self.tree.up_factor:.6f}, d = {self.tree.down_factor:.6f}</p>
        <p>Calculate all stock prices in the tree.</p>
        <p>From left to right, the fields represent the nodes from the lowest to the highest prices at each time step.</p>
        <p>Enter your calculated stock prices below.</p>
        <p><strong>Required Precision:</strong> 1e-2 (2 decimal places minimum)</p>
        """

        # Create input fields for each stock price node
        self.stock_price_inputs = {}
        period_rows = []

        for period in range(N + 1):
            # Create label for the period
            period_label = Div(text=f"<strong>Period {period}:</strong>", width=100, height=30)

            # Create inputs for this period (side by side, from lowest to highest)
            inputs_in_period = []
            for index in range(period, -1, -1):  # Reverse order: period down to 0
                n_up = period - index
                n_down = index
                input_field = TextInput(
                    title=f"{n_up}u/{n_down}d",
                    value="",
                    width=120
                )
                self.stock_price_inputs[(period, index)] = input_field
                inputs_in_period.append(input_field)

            # Create row with label and inputs
            period_row = row(period_label, *inputs_in_period)
            period_rows.append(period_row)

        self.stock_tree_inputs_container.children = period_rows
        self.stock_tree_results.text = ""
        self.btn_proceed_option.visible = False

        # Update layout
        stock_buttons = [self.btn_verify_stock_tree]
        if DEV_MODE:
            stock_buttons.append(self.btn_fill_stock)

        self.right_column.children = [
            self.stock_tree_inputs_container,
            row(*stock_buttons)
        ]
        self.results_area.children = [self.stock_tree_results]
        self.buttons_area.children = [self.btn_proceed_option]

    def _on_verify_stock_tree(self):
        """Verify student's stock price calculations"""
        N = self.tree.N_periods
        tolerance = 0.01  # Allow 1 cent tolerance

        try:
            results = "<h3>Verification Results:</h3>"
            results += "<p><strong>Formula:</strong> S(period, index) = S0 × u<sup>(period - index)</sup> × d<sup>index</sup></p>"
            results += "<table border='1' style='border-collapse:collapse; padding:5px;'>"
            results += "<tr><th>Period</th><th>Node</th><th>Your Price</th><th>Correct Price</th><th>Status</th></tr>"

            all_correct = True
            for period in range(N + 1):
                for index in range(period + 1):
                    n_up = period - index
                    n_down = index

                    # Get student's answer
                    student_input = self.stock_price_inputs[(period, index)].value
                    if not student_input:
                        all_correct = False
                        results += f"<tr><td>{period}</td><td>{n_up}u/{n_down}d</td>"
                        results += f"<td style='color:red;'>Not entered</td><td>-</td><td style='color:red;'>✗ Missing</td></tr>"
                        continue

                    student_price = float(student_input)
                    correct_price = self.tree.tree[period][index].stock_price

                    match = abs(student_price - correct_price) < tolerance
                    if not match:
                        all_correct = False

                    status = "✓ Correct" if match else "✗ Incorrect"
                    color = "green" if match else "red"

                    results += f"<tr><td>{period}</td><td>{n_up}u/{n_down}d</td>"
                    results += f"<td>${student_price:.2f}</td><td>${correct_price:.2f}</td>"
                    results += f"<td style='color:{color};'>{status}</td></tr>"

            results += "</table>"

            if all_correct:
                results += "<p style='color:green; font-weight:bold;'>Excellent! All stock prices are correct!</p>"
                self.btn_proceed_option.visible = True
            else:
                results += "<p style='color:orange;'>Review your calculations and try again.</p>"
                self.btn_proceed_option.visible = False

            self.stock_tree_results.text = results

        except ValueError:
            self.stock_tree_results.text = "<p style='color:red;'>Error: Please enter valid numbers for all stock prices.</p>"
            self.btn_proceed_option.visible = False

    def _fill_stock_prices(self):
        """[DEV] Fill in correct stock prices"""
        N = self.tree.N_periods
        for period in range(N + 1):
            for index in range(period + 1):
                correct_price = self.tree.tree[period][index].stock_price
                self.stock_price_inputs[(period, index)].value = f"{correct_price:.2f}"

    def _on_proceed_to_option_choice(self):
        """Proceed to option selection phase"""
        self._show_option_choice_phase()

    def _show_option_choice_phase(self):
        """Show option selection phase"""
        self.phase = "option_choice"
        self.instructions.text = """
        <h2>Step 4: Choose Option to Price</h2>
        <p>Select the type of option you want to price:</p>
        <ul>
        <li><strong>European:</strong> Can only be exercised at maturity</li>
        <li><strong>American:</strong> Can be exercised at any time (after period 0)</li>
        <li><strong>Call/Put:</strong> Standard options with payoffs max(S-K, 0) or max(K-S, 0)</li>
        <li><strong>Digital:</strong> Fixed payout if condition is met</li>
        <li><strong>Custom:</strong> Write your own payoff formula using Python</li>
        </ul>
        """
        self.right_column.children = [self.option_widgets]
        self.results_area.children = []
        self.buttons_area.children = []

    def _on_option_type_change(self, attr, old, new):
        """Handle option type change"""
        if new == "Digital Call" or new == "Digital Put":
            self.option_payout_input.visible = True
        else:
            self.option_payout_input.visible = False

        if new == "Custom":
            self.custom_payoff_input.visible = True
        else:
            self.custom_payoff_input.visible = False

    def _on_confirm_option(self):
        """Confirm option choice and create payoff function"""
        try:
            self.option_type = self.option_style_select.value.lower()
            option_kind = self.option_type_select.value

            if option_kind == "Call":
                K = float(self.option_strike_input.value)
                self.payoff_func = call_payoff(K)
                self.payoff_description = f"max(S - {K}, 0)"
                self.payoff_type = 'call'

            elif option_kind == "Put":
                K = float(self.option_strike_input.value)
                self.payoff_func = put_payoff(K)
                self.payoff_description = f"max({K} - S, 0)"
                self.payoff_type = 'put'

            elif option_kind == "Digital Call":
                K = float(self.option_strike_input.value)
                payout = float(self.option_payout_input.value)
                self.payoff_func = lambda S: payout if S >= K else 0.0
                self.payoff_description = f"${payout} if S >= {K}, else $0"
                self.payoff_type = 'digital_call'

            elif option_kind == "Digital Put":
                K = float(self.option_strike_input.value)
                payout = float(self.option_payout_input.value)
                self.payoff_func = lambda S: payout if S <= K else 0.0
                self.payoff_description = f"${payout} if S <= {K}, else $0"
                self.payoff_type = 'digital_put'

            elif option_kind == "Custom":
                formula = self.custom_payoff_input.value
                # Create safe payoff function
                self.payoff_func = self._create_custom_payoff(formula)
                self.payoff_description = formula
                self.payoff_type = 'custom'

            # Proceed to terminal payoff phase
            self._show_terminal_phase()

        except Exception as e:
            self.instructions.text = f"<p style='color:red;'>Error: {str(e)}</p>"

    def _create_custom_payoff(self, formula):
        """Create a payoff function from user formula"""
        # Basic safety: only allow certain functions
        safe_dict = {
            'max': max,
            'min': min,
            'abs': abs,
            'exp': np.exp,
            'sqrt': np.sqrt,
            '__builtins__': {}
        }

        def payoff(S):
            safe_dict['S'] = S
            try:
                result = eval(formula, safe_dict)
                return float(result)
            except Exception as e:
                raise ValueError(f"Invalid formula: {e}")

        return payoff

    def _show_terminal_phase(self):
        """Show terminal payoff calculation phase"""
        self.phase = "terminal"

        N = self.tree.N_periods

        self.instructions.text = f"""
        <h2>Step 5: Calculate Terminal Payoffs</h2>
        <p>Calculate the payoff at each terminal node (period {N}).</p>
        <p>Enter your calculated payoffs below (from lowest to highest stock price).</p>
        <p><strong>Required Precision:</strong> 1e-2 (2 decimal places minimum)</p>
        """

        # Create input fields for terminal payoffs (from lowest to highest)
        self.terminal_payoff_inputs = {}
        terminal_inputs = []

        for index in range(N, -1, -1):  # Reverse order: N down to 0 (lowest to highest price)
            n_up = N - index
            n_down = index
            stock_price = self.tree.tree[N][index].stock_price

            input_field = TextInput(
                title=f"{n_up}u/{n_down}d\nS=${stock_price:.2f}",
                value="",
                width=120
            )
            self.terminal_payoff_inputs[index] = input_field
            terminal_inputs.append(input_field)

        self.terminal_inputs_container.children = terminal_inputs
        self.terminal_results.text = ""
        self.btn_proceed_pricing.visible = False

        # Update layout
        terminal_buttons = [self.btn_verify_terminal]
        if DEV_MODE:
            terminal_buttons.append(self.btn_fill_terminal)

        self.right_column.children = [
            self.terminal_inputs_container,
            row(*terminal_buttons)
        ]
        self.results_area.children = [self.terminal_results]
        self.buttons_area.children = [self.btn_proceed_pricing]

    def _on_verify_terminal(self):
        """Verify student's terminal payoff calculations"""
        N = self.tree.N_periods
        tolerance = 0.01  # Allow 1 cent tolerance

        try:
            results = "<h3>Verification Results:</h3>"
            results += f"<p><strong>Payoff Formula:</strong> {self.payoff_description}</p>"
            results += "<table border='1' style='border-collapse:collapse; padding:5px;'>"
            results += "<tr><th>Node</th><th>Stock Price</th><th>Your Payoff</th><th>Correct Payoff</th><th>Status</th></tr>"

            all_correct = True
            for index in range(N, -1, -1):  # Iterate in same order as inputs (lowest to highest)
                node = self.tree.tree[N][index]
                n_up = N - index
                n_down = index
                correct_payoff = self.payoff_func(node.stock_price)

                # Get student's answer
                student_input = self.terminal_payoff_inputs[index].value
                if not student_input:
                    all_correct = False
                    results += f"<tr><td>{n_up}u/{n_down}d</td>"
                    results += f"<td>${node.stock_price:.2f}</td>"
                    results += f"<td style='color:red;'>Not entered</td><td>-</td><td style='color:red;'>✗ Missing</td></tr>"
                    continue

                student_payoff = float(student_input)

                match = abs(student_payoff - correct_payoff) < tolerance
                if not match:
                    all_correct = False

                status = "✓ Correct" if match else "✗ Incorrect"
                color = "green" if match else "red"

                results += f"<tr><td>{n_up}u/{n_down}d</td>"
                results += f"<td>${node.stock_price:.2f}</td>"
                results += f"<td>${student_payoff:.2f}</td><td>${correct_payoff:.2f}</td>"
                results += f"<td style='color:{color};'>{status}</td></tr>"

            results += "</table>"

            if all_correct:
                results += "<p style='color:green; font-weight:bold;'>Excellent! All terminal payoffs are correct!</p>"
                self.btn_proceed_pricing.visible = True
            else:
                results += "<p style='color:orange;'>Review your calculations and try again.</p>"
                self.btn_proceed_pricing.visible = False

            self.terminal_results.text = results

        except ValueError:
            self.terminal_results.text = "<p style='color:red;'>Error: Please enter valid numbers for all payoffs.</p>"
            self.btn_proceed_pricing.visible = False

    def _fill_terminal_payoffs(self):
        """[DEV] Fill in correct terminal payoffs"""
        N = self.tree.N_periods
        for index in range(N, -1, -1):
            stock_price = self.tree.tree[N][index].stock_price
            correct_payoff = self.payoff_func(stock_price)
            self.terminal_payoff_inputs[index].value = f"{correct_payoff:.2f}"

    def _on_proceed_to_pricing(self):
        """Proceed to pricing phase"""
        self._show_pricing_phase()

    def _show_pricing_phase(self):
        """Show pricing phase"""
        self.phase = "pricing"

        # Update available methods based on option type
        if self.option_type == "european":
            methods = [
                "Backward Induction (Risk-Neutral)",
                "Backward Induction (Replication)",
                "Risk-Neutral Expectation"
            ]
        else:
            methods = [
                "Backward Induction (Risk-Neutral)",
                "Backward Induction (Replication)"
            ]

        self.pricing_method_select.options = methods
        self.pricing_method_select.value = methods[0]

        # Build terminal payoffs display
        N = self.tree.N_periods
        terminal_display = "<p><strong>Terminal Payoffs (Period " + str(N) + "):</strong> "
        payoffs = []
        for index in range(N, -1, -1):  # Lowest to highest
            n_up = N - index
            n_down = index
            stock_price = self.tree.tree[N][index].stock_price
            payoff = self.payoff_func(stock_price)
            payoffs.append(f"[{n_up}u/{n_down}d]: ${payoff:.2f}")
        terminal_display += " | ".join(payoffs)
        terminal_display += "</p>"

        option_type_display = self.option_type.capitalize()

        self.instructions.text = f"""
        <h2>Step 6: Price the Option</h2>
        <p><strong>Option Type:</strong> {option_type_display}</p>
        <p><strong>Payoff Formula:</strong> {self.payoff_description}</p>
        <p><strong>Risk-Neutral Probability:</strong> p* = {self.tree.pstar:.6f}</p>
        <p><strong>Discount Factor:</strong> exp(-rf × h) = {self.tree.discount_factor:.6f}</p>
        {terminal_display}
        <p>Choose your pricing method and calculate the option value.</p>
        <p>Enter your calculations for each node (backward induction methods) or the final value (expectation method).</p>
        <p><strong>Required Precision:</strong> 1e-3 for option values, deltas, and bonds (3 decimal places minimum)</p>
        """

        # Create inputs for the default method
        self._create_pricing_inputs(methods[0])

        self.pricing_results.text = ""
        self.btn_restart.visible = False
        self.btn_retry_pricing.visible = False

        # Update layout - pricing method selector and inputs on right, results below
        pricing_buttons = [self.btn_verify_pricing]
        if DEV_MODE:
            pricing_buttons.append(self.btn_fill_pricing)

        self.right_column.children = [
            self.pricing_method_select,
            self.pricing_inputs_container,
            row(*pricing_buttons)
        ]
        self.results_area.children = [self.pricing_results]
        self.buttons_area.children = [
            row(self.btn_retry_pricing, Spacer(width=10), self.btn_restart)
        ]

    def _on_pricing_method_change(self, attr, old, new):
        """Handle pricing method change"""
        self._create_pricing_inputs(new)
        self.pricing_results.text = ""

        # Update right column with new inputs
        pricing_buttons = [self.btn_verify_pricing]
        if DEV_MODE:
            pricing_buttons.append(self.btn_fill_pricing)

        self.right_column.children = [
            self.pricing_method_select,
            self.pricing_inputs_container,
            row(*pricing_buttons)
        ]

    def _create_pricing_inputs(self, method):
        """Create input fields based on pricing method"""
        N = self.tree.N_periods

        # Clear previous inputs
        self.pricing_value_inputs = {}
        self.pricing_delta_inputs = {}
        self.pricing_bond_inputs = {}
        self.pricing_expectation_input = None

        if method == "Risk-Neutral Expectation":
            # Just one input for final value
            self.pricing_expectation_input = TextInput(
                title="Option Value (present value of expected payoff):",
                value="",
                width=300
            )
            self.pricing_inputs_container.children = [self.pricing_expectation_input]

        elif method == "Backward Induction (Risk-Neutral)":
            # Create inputs for each node, working backwards from period N-1 to 0
            period_rows = []

            for period in range(N - 1, -1, -1):
                period_label = Div(text=f"<strong>Period {period}:</strong>", width=100, height=30)

                inputs_in_period = []
                for index in range(period, -1, -1):  # Lowest to highest
                    n_up = period - index
                    n_down = index
                    stock_price = self.tree.tree[period][index].stock_price

                    input_field = TextInput(
                        title=f"{n_up}u/{n_down}d\nS=${stock_price:.2f}\nValue:",
                        value="",
                        width=120
                    )
                    self.pricing_value_inputs[(period, index)] = input_field
                    inputs_in_period.append(input_field)

                period_row = row(period_label, *inputs_in_period)
                period_rows.append(period_row)

            self.pricing_inputs_container.children = period_rows

        elif method == "Backward Induction (Replication)":
            # Create inputs for delta, bond, and value at each node
            period_rows = []

            for period in range(N - 1, -1, -1):
                period_label = Div(text=f"<strong>Period {period}:</strong>", width=100, height=30)

                inputs_in_period = []
                for index in range(period, -1, -1):  # Lowest to highest
                    n_up = period - index
                    n_down = index
                    stock_price = self.tree.tree[period][index].stock_price

                    # Create a small column for this node with delta, bond, value
                    delta_input = TextInput(title=f"Δ", value="", width=80)
                    bond_input = TextInput(title=f"B", value="", width=80)
                    value_input = TextInput(title=f"V", value="", width=80)

                    self.pricing_delta_inputs[(period, index)] = delta_input
                    self.pricing_bond_inputs[(period, index)] = bond_input
                    self.pricing_value_inputs[(period, index)] = value_input

                    node_label = Div(text=f"<small>{n_up}u/{n_down}d<br/>S=${stock_price:.2f}</small>",
                                    width=80, height=50)
                    node_inputs = column(node_label, delta_input, bond_input, value_input)
                    inputs_in_period.append(node_inputs)

                period_row = row(period_label, *inputs_in_period)
                period_rows.append(period_row)

            self.pricing_inputs_container.children = period_rows

    def _on_verify_pricing(self):
        """Verify pricing calculations"""
        method = self.pricing_method_select.value

        if method == "Risk-Neutral Expectation":
            self._verify_expectation_pricing()
        elif method == "Backward Induction (Risk-Neutral)":
            self._verify_backward_induction_rn()
        elif method == "Backward Induction (Replication)":
            self._verify_backward_induction_replication()

        self.btn_retry_pricing.visible = True
        self.btn_restart.visible = True

    def _retry_pricing(self):
        """Reset pricing phase to try a different method"""
        self.pricing_results.text = ""
        self.btn_retry_pricing.visible = False
        self.btn_restart.visible = False
        # Re-create inputs for current method (clears all values)
        self._create_pricing_inputs(self.pricing_method_select.value)

        # Update right column with new inputs
        pricing_buttons = [self.btn_verify_pricing]
        if DEV_MODE:
            pricing_buttons.append(self.btn_fill_pricing)

        self.right_column.children = [
            self.pricing_method_select,
            self.pricing_inputs_container,
            row(*pricing_buttons)
        ]

    def _fill_pricing_values(self):
        """[DEV] Fill in correct pricing values based on method"""
        method = self.pricing_method_select.value
        N = self.tree.N_periods

        # First, price the option to populate the tree
        if self.option_type == "european":
            self.tree.price_european(self.payoff_func)
        else:
            self.tree.price_american(self.payoff_func)

        if method == "Risk-Neutral Expectation":
            # Just fill the final value
            option_value = self.tree.tree[0][0].derivative_value
            self.pricing_expectation_input.value = f"{option_value:.8f}"

        elif method == "Backward Induction (Risk-Neutral)":
            # Fill all values from period N-1 down to 0
            for period in range(N - 1, -1, -1):
                for index in range(period + 1):
                    correct_value = self.tree.tree[period][index].derivative_value
                    self.pricing_value_inputs[(period, index)].value = f"{correct_value:.8f}"

        elif method == "Backward Induction (Replication)":
            # Fill delta, bond, and value for all nodes
            for period in range(N - 1, -1, -1):
                for index in range(period + 1):
                    node = self.tree.tree[period][index]
                    self.pricing_delta_inputs[(period, index)].value = f"{node.delta:.8f}"
                    self.pricing_bond_inputs[(period, index)].value = f"{node.bond:.8f}"
                    self.pricing_value_inputs[(period, index)].value = f"{node.derivative_value:.8f}"

    def _verify_expectation_pricing(self):
        """Verify risk-neutral expectation pricing (European only)"""
        if self.option_type != "european":
            self.pricing_results.text = "<p style='color:red;'>Error: Expectation method only for European options!</p>"
            return

        try:
            student_value = float(self.pricing_expectation_input.value)
            correct_value = self.tree.price_european(self.payoff_func)

            # Calculate probabilities for display (lazy evaluation)
            self.tree.calculate_probabilities()

            tolerance = 0.01

            results = "<h3>Verification Results:</h3>"
            results += "<p><strong>Method:</strong> Risk-Neutral Expectation</p>"
            results += f"<p><strong>Formula:</strong> V = exp(-rf × T) × E[Payoff]</p>"

            N = self.tree.N_periods
            discount = self.tree.discount_factor ** N

            results += "<h4>Calculation Steps:</h4>"
            results += "<table border='1' style='border-collapse:collapse; padding:5px;'>"
            results += "<tr><th>Node</th><th>Stock Price</th><th>Payoff</th><th>Probability</th><th>Contribution</th></tr>"

            total_expectation = 0
            for index in range(N, -1, -1):
                node = self.tree.tree[N][index]
                n_up = N - index
                n_down = index
                payoff = self.payoff_func(node.stock_price)
                prob = node.probability
                contribution = payoff * prob
                total_expectation += contribution

                results += f"<tr><td>{n_up}u/{n_down}d</td>"
                results += f"<td>${node.stock_price:.2f}</td>"
                results += f"<td>${payoff:.2f}</td>"
                results += f"<td>{prob:.6f}</td>"
                results += f"<td>${contribution:.4f}</td></tr>"

            results += f"<tr><td colspan='4'><strong>Expected Payoff at T</strong></td>"
            results += f"<td><strong>${total_expectation:.4f}</strong></td></tr>"

            results += f"<tr><td colspan='4'><strong>Discount Factor (exp(-rf × T))</strong></td>"
            results += f"<td><strong>{discount:.6f}</strong></td></tr>"

            results += f"<tr><td colspan='4'><strong>Present Value</strong></td>"
            results += f"<td><strong>${correct_value:.4f}</strong></td></tr>"

            results += "</table>"

            match = abs(student_value - correct_value) < tolerance
            color = "green" if match else "red"
            status = "✓ Correct" if match else "✗ Incorrect"

            results += f"<h4>Your Answer:</h4>"
            results += f"<table border='1' style='border-collapse:collapse; padding:5px;'>"
            results += f"<tr><th>Your Value</th><th>Correct Value</th><th>Status</th></tr>"
            results += f"<tr><td>${student_value:.6f}</td><td>${correct_value:.6f}</td>"
            results += f"<td style='color:{color};'>{status}</td></tr>"
            results += "</table>"

            if match:
                results += "<p style='color:green; font-weight:bold;'>Excellent! Your pricing is correct!</p>"
            else:
                results += "<p style='color:orange;'>Review your calculation and try again.</p>"

            self.pricing_results.text = results

        except ValueError:
            self.pricing_results.text = "<p style='color:red;'>Error: Please enter a valid number.</p>"

    def _verify_backward_induction_rn(self):
        """Verify backward induction using risk-neutral pricing"""
        N = self.tree.N_periods
        tolerance = 0.001  # 1e-3 precision for option values

        # Price the option to get correct values
        if self.option_type == "european":
            option_value = self.tree.price_european(self.payoff_func)
        else:
            option_value = self.tree.price_american(self.payoff_func)

        try:
            results = "<h3>Verification Results:</h3>"
            results += "<p><strong>Method:</strong> Backward Induction (Risk-Neutral)</p>"
            results += f"<p><strong>Formula:</strong> V = exp(-rf × h) × [p* × V<sub>up</sub> + (1 - p*) × V<sub>down</sub>]</p>"
            results += f"<p><em>p* = {self.tree.pstar:.6f}, discount = {self.tree.discount_factor:.6f}</em></p>"

            results += "<h4>Node-by-Node Verification:</h4>"
            results += "<table border='1' style='border-collapse:collapse; padding:5px;'>"
            results += "<tr><th>Period</th><th>Node</th><th>Stock Price</th><th>Your Value</th><th>Correct Value</th><th>Status</th></tr>"

            all_correct = True
            for period in range(N - 1, -1, -1):
                for index in range(period, -1, -1):
                    node = self.tree.tree[period][index]
                    n_up = period - index
                    n_down = index
                    correct_value = node.derivative_value

                    # Get student's answer
                    student_input = self.pricing_value_inputs[(period, index)].value
                    if not student_input:
                        all_correct = False
                        results += f"<tr><td>{period}</td><td>{n_up}u/{n_down}d</td>"
                        results += f"<td>${node.stock_price:.2f}</td>"
                        results += f"<td style='color:red;'>Not entered</td><td>-</td><td style='color:red;'>✗ Missing</td></tr>"
                        continue

                    student_value = float(student_input)

                    match = abs(student_value - correct_value) < tolerance
                    if not match:
                        all_correct = False

                    status = "✓ Correct" if match else "✗ Incorrect"
                    color = "green" if match else "red"

                    results += f"<tr><td>{period}</td><td>{n_up}u/{n_down}d</td>"
                    results += f"<td>${node.stock_price:.2f}</td>"
                    results += f"<td>${student_value:.4f}</td><td>${correct_value:.4f}</td>"
                    results += f"<td style='color:{color};'>{status}</td></tr>"

            results += "</table>"

            results += f"<p><strong>Final Option Value (at node [0,0]): ${option_value:.6f}</strong></p>"

            if all_correct:
                results += "<p style='color:green; font-weight:bold;'>Excellent! All calculations are correct!</p>"
            else:
                results += "<p style='color:orange;'>Review your calculations and try again.</p>"

            self.pricing_results.text = results

        except ValueError:
            self.pricing_results.text = "<p style='color:red;'>Error: Please enter valid numbers for all nodes.</p>"

    def _verify_backward_induction_replication(self):
        """Verify backward induction using replication portfolio"""
        N = self.tree.N_periods
        tolerance = 0.001  # 1e-3 precision for deltas, bonds, and values

        # Price the option to get correct values
        if self.option_type == "european":
            option_value = self.tree.price_european(self.payoff_func)
        else:
            option_value = self.tree.price_american(self.payoff_func)

        try:
            results = "<h3>Verification Results:</h3>"
            results += "<p><strong>Method:</strong> Backward Induction (Replication)</p>"
            results += "<p><strong>Formula:</strong> V = Δ × S + B</p>"

            results += "<h4>Node-by-Node Verification:</h4>"
            results += "<table border='1' style='border-collapse:collapse; padding:5px;'>"
            results += "<tr><th>Period</th><th>Node</th><th>S</th><th>Your Δ</th><th>Correct Δ</th>"
            results += "<th>Your B</th><th>Correct B</th><th>Your V</th><th>Correct V</th><th>Status</th></tr>"

            all_correct = True
            for period in range(N - 1, -1, -1):
                for index in range(period, -1, -1):
                    node = self.tree.tree[period][index]
                    n_up = period - index
                    n_down = index
                    correct_delta = node.delta
                    correct_bond = node.bond
                    correct_value = node.derivative_value

                    # Get student's answers
                    delta_input = self.pricing_delta_inputs[(period, index)].value
                    bond_input = self.pricing_bond_inputs[(period, index)].value
                    value_input = self.pricing_value_inputs[(period, index)].value

                    if not delta_input or not bond_input or not value_input:
                        all_correct = False
                        results += f"<tr><td>{period}</td><td>{n_up}u/{n_down}d</td><td>${node.stock_price:.2f}</td>"
                        results += f"<td colspan='6' style='color:red;'>Incomplete</td><td style='color:red;'>✗ Missing</td></tr>"
                        continue

                    student_delta = float(delta_input)
                    student_bond = float(bond_input)
                    student_value = float(value_input)

                    delta_match = abs(student_delta - correct_delta) < tolerance
                    bond_match = abs(student_bond - correct_bond) < tolerance
                    value_match = abs(student_value - correct_value) < tolerance
                    match = delta_match and bond_match and value_match

                    if not match:
                        all_correct = False

                    status = "✓ Correct" if match else "✗ Incorrect"
                    color = "green" if match else "red"

                    results += f"<tr><td>{period}</td><td>{n_up}u/{n_down}d</td><td>${node.stock_price:.2f}</td>"
                    results += f"<td>{student_delta:.6f}</td><td>{correct_delta:.6f}</td>"
                    results += f"<td>${student_bond:.4f}</td><td>${correct_bond:.4f}</td>"
                    results += f"<td>${student_value:.4f}</td><td>${correct_value:.4f}</td>"
                    results += f"<td style='color:{color};'>{status}</td></tr>"

            results += "</table>"

            results += f"<p><strong>Final Option Value (at node [0,0]): ${option_value:.6f}</strong></p>"

            if all_correct:
                results += "<p style='color:green; font-weight:bold;'>Excellent! All calculations are correct!</p>"
            else:
                results += "<p style='color:orange;'>Review your calculations and try again.</p>"

            self.pricing_results.text = results

        except ValueError:
            self.pricing_results.text = "<p style='color:red;'>Error: Please enter valid numbers for all fields.</p>"


    def _restart(self):
        """Restart the application"""
        self.tree = None
        self.option_type = None
        self.payoff_type = None
        self.payoff_func = None
        self.payoff_description = ""
        self.pricing_method = None
        self._show_input_phase()

    def get_layout(self):
        """Return the Bokeh layout"""
        return self.layout


# Create and run the app
app = BinomialTreePracticeApp()
curdoc().add_root(app.get_layout())
curdoc().title = "Binomial Tree Practice Tool"
