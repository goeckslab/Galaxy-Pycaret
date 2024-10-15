import logging

import numpy as np

from sklearn.metrics import auc, precision_recall_curve
from sklearn.preprocessing import label_binarize


logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


def get_html_template():
    return """
    <html>
    <head>
        <title>Model Training Report</title>
        <style>
          body {
              font-family: Arial, sans-serif;
              margin: 0;
              padding: 20px;
              background-color: #f4f4f4;
          }
          .container {
              max-width: 800px;
              margin: auto;
              background: white;
              padding: 20px;
              box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
          }
          h1 {
              text-align: center;
              color: #333;
          }
          h2 {
              border-bottom: 2px solid #4CAF50;
              color: #4CAF50;
              padding-bottom: 5px;
          }
          table {
              width: 100%;
              border-collapse: collapse;
              margin: 20px 0;
          }
          table, th, td {
              border: 1px solid #ddd;
          }
          th, td {
              padding: 8px;
              text-align: left;
          }
          th {
              background-color: #4CAF50;
              color: white;
          }
          .plot {
              text-align: center;
              margin: 20px 0;
          }
          .plot img {
              max-width: 100%;
              height: auto;
          }
          .tabs {
              display: flex;
              margin-bottom: 20px;
              cursor: pointer;
              justify-content: space-around;
          }
          .tab {
              padding: 10px;
              background-color: #4CAF50;
              color: white;
              border-radius: 5px 5px 0 0;
              flex-grow: 1;
              text-align: center;
              margin: 0 5px;
          }
          .tab.active-tab {
              background-color: #333;
          }
          .tab-content {
              display: none;
              padding: 20px;
              border: 1px solid #ddd;
              border-top: none;
              background-color: white;
          }
          .tab-content.active-content {
              display: block;
          }
      </style>
    </head>
    <body>
    <div class="container">
    """


def get_html_closing():
    return """
        </div>
        <script>
            function openTab(evt, tabName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].style.display = "none";
                }}
                tablinks = document.getElementsByClassName("tab");
                for (i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className =
                        tablinks[i].className.replace(" active-tab", "");
                }}
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active-tab";
            }}
            document.addEventListener("DOMContentLoaded", function() {{
                document.querySelector(".tab").click();
            }});
        </script>
    </body>
    </html>
    """


def customize_figure_layout(fig, margin_dict=None):
    """
    Update the layout of a Plotly figure to reduce margins.

    Parameters:
        fig (plotly.graph_objects.Figure): The Plotly figure to customize.
        margin_dict (dict, optional): A dictionary specifying margin sizes.
            Example: {'l': 10, 'r': 10, 't': 10, 'b': 10}

    Returns:
        plotly.graph_objects.Figure: The updated Plotly figure.
    """
    if margin_dict is None:
        # Set default smaller margins
        margin_dict = {'l': 40, 'r': 40, 't': 40, 'b': 40}

    fig.update_layout(margin=margin_dict)
    return fig


def add_plot_to_html(fig, include_plotlyjs=True):
    custom_margin = {'l': 40, 'r': 40, 't': 60, 'b': 60}
    fig = customize_figure_layout(fig, margin_dict=custom_margin)
    return fig.to_html(full_html=False,
                       default_height=350,
                       include_plotlyjs="cdn" if include_plotlyjs else False)


def add_hr_to_html():
    return "<hr>"


def pr_auc(y_true, y_pred, **kwargs):
    LOG.debug("calculating PR-AUC")
    average = "weighted"
    if len(np.unique(y_true)) == 2:
        # Binary classification
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        return auc(recall, precision)
    else:
        # Multi-class classification
        y_true_binarized = label_binarize(
            y_true, classes=np.unique(y_true))

        if average == 'macro':
            # Macro-average PR-AUC: Calculate PR-AUC
            # for each class, then average
            auc_scores = []
            for i in range(y_true_binarized.shape[1]):
                precision, recall, _ = precision_recall_curve(
                    y_true_binarized[:, i], y_pred[:, i])
                auc_scores.append(auc(recall, precision))
            return np.mean(auc_scores)

        elif average == 'micro':
            # Micro-average PR-AUC: Aggregate true positives
            # and false positives across all classes
            precision, recall, _ = precision_recall_curve(
                y_true_binarized.ravel(), y_pred.ravel())
            return auc(recall, precision)

        elif average == 'weighted':
            # Weighted average: Calculate PR-AUC for each class,
            # weighted by class frequency
            auc_scores = []
            for i in range(y_true_binarized.shape[1]):
                precision, recall, _ = precision_recall_curve(
                    y_true_binarized[:, i], y_pred[:, i])
                auc_scores.append(auc(recall, precision))
            return np.average(auc_scores, weights=np.bincount(y_true))
