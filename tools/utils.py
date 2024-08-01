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
