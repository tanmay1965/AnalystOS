# AnalystOS — Autonomous AI Data Analyst

AnalystOS is an AI-powered data analysis application built with Python and Streamlit. It helps users explore datasets, perform analytical operations, generate visualizations, and maintain investigation history through a structured analysis workflow.

## Key Features

- **Dataset Profiling:** Upload CSV files and inspect dataset structure and data quality.
- **AI-Assisted Analysis:** Use a structured agent workflow to process analytical questions.
- **Workflow Orchestration:** Coordinate analysis steps using LangGraph.
- **Data Processing:** Perform data operations using Python data-analysis tools.
- **Interactive Visualizations:** Generate charts using Plotly.
- **Persistent Memory:** Store investigation history using SQLite.
- **PDF Report Generation:** Export investigation findings and results into PDF reports.
- **Automated Testing:** Validate workflow paths and application behavior using pytest.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Agent Workflow | LangGraph |
| Data Processing | Pandas / configured data-analysis tools |
| Visualization | Plotly |
| Persistent Storage | SQLite |
| Local LLM Integration | Ollama with Llama 3, where enabled |
| Report Generation | PDF export module |
| Testing | pytest |

## Project Structure

```text
AnalystOS/
├── agent/
│   ├── __init__.py
│   ├── graph.py
│   ├── memory.py
│   ├── nodes.py
│   ├── report_export.py
│   ├── state.py
│   └── tools.py
├── data/
│   └── analystos_memory.db
├── tools/
│   ├── __init__.py
│   └── data_profiler.py
├── app.py
├── requirements.txt
└── test_graph_paths.py
```

**Note:** The SQLite database is created or used by the application for persistent memory, depending on its configuration. Do not include private investigation history in a public submission.

## Prerequisites

- Python 3.10 or a compatible version supported by the project's dependencies.
- pip package manager.
- Git (optional).
- Ollama and the configured Llama 3 model if the local LLM integration is enabled.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AnalystOS
```

Alternatively, download and extract the project ZIP, then open the project directory in a terminal.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use Command Prompt instead:

```cmd
venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the local LLM (if enabled)

Install Ollama from its official website:

https://ollama.com/

Then, in a terminal, download the model:

```bash
ollama pull llama3
```

Run the model to verify that it is available:

```bash
ollama run llama3
```

Keep Ollama available while using features that depend on the local model. If the application is configured to use a different model or provider, follow that configuration instead.

## Running AnalystOS

From the project root, with the virtual environment activated, execute:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, typically:

```text
http://localhost:8501
```

Open the displayed URL in your browser.

## Running Tests

Run the automated test suite from the project root:

```bash
python -m pytest -v
```

To run the workflow-path tests specifically:

```bash
python -m pytest -v test_graph_paths.py
```

Review the terminal output to confirm which tests passed or failed. Test results can vary depending on the current code and environment.

## Typical Workflow

1. Launch AnalystOS.
2. Upload a supported CSV dataset.
3. Inspect the dataset profile and available information.
4. Submit an analytical question or investigation.
5. Allow the configured workflow to process the request.
6. Review the returned results and visualizations.
7. Export a PDF report when the report-generation feature is available.
8. Review saved investigation history if persistent memory is enabled.

## Evidence and Reproducibility

For project evaluation, useful supporting evidence includes:

- A screenshot of the running Streamlit application.
- A screenshot of the terminal showing the actual pytest results.
- A diagram of the LangGraph workflow.
- A sample CSV dataset that can be shared safely.
- A generated PDF report containing analytical results and charts.

Use a non-sensitive sample dataset and remove credentials, API keys, private datasets, and personal investigation history before sharing the project.

## Current Scope and Limitations

The available features depend on the implementation and configuration in the submitted source code. Local LLM functionality requires the relevant model and runtime to be configured. Analytical operations, error recovery, persistent memory, chart generation, and PDF export should be validated against the submitted build before being described as fully supported.

## Future Enhancements

- Support for additional structured data formats.
- Improved data-quality auditing and anomaly detection.
- More robust handling of invalid inputs and failed operations.
- Enhanced report layouts and evidence export.
- Expanded automated tests and reproducible evaluation datasets.
- Optional integration with additional LLM providers.

## Author

**Tanmay Challa**

Information Technology | AI Engineering | Data Analytics

Portfolio: https://challa-tanmay-portfolio.netlify.app/

LinkedIn: https://www.linkedin.com/in/tanmay-challa/

---

*AnalystOS — Turning datasets into actionable insights through structured, AI-assisted analysis.*