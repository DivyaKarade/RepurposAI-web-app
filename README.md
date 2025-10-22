# RepurposAI - Drug & Target Discovery Studio

<img src="https://github.com/jchchiu/RepurposAI-web-app/blob/main/docs/images/RepurposAI_LOGO_example.png" width="192">

RepurposAI is a modular, open-source web app for drug repurposing and target discovery.  
It integrates ML, cheminformatics, and the Open Targets Platform.

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the Streamlit app: `streamlit run app/main.py`
3. Explore modules: similarity search, target prediction, pathway mapping

Here’s a suggested project structure:
```
RepurposAI/
├── README.md
├── requirements.txt
├── .gitignore
├── app/
│   ├── main.py                 # Streamlit main app
│   ├── dashboard.py            # Dashboard assembly module
│   ├── visualization.py        # Plotly/Seaborn visualization module
│   └── utils/
│       ├── api_integration.py  # Open Targets & KEGG API helpers
│       ├── similarity.py       # RDKit similarity functions
│       └── ml_model.py         # Placeholder for target prediction ML
├── data/                       # Sample data or placeholder CSVs
├── docs/                       # README, usage guide, hackathon slides
├── tests/                      # Basic test scripts for functions
└── notebooks/                  # Jupyter notebooks for testing ML/API modules
    └── example_notebook.ipynb
```


## ✅ Next Steps for Hackathon Participants

- Fork the repository and pick a task from the GitHub Epic.  
- Implement the placeholder modules in `/app` and `/app/utils`.  
- Add sample Jupyter notebooks in `/notebooks` to test modules.  
- Start building Streamlit pages in `/app/main.py` and `/app/dashboard.py`.  
- Commit your work regularly and create issues/sub-issues for new features or bugs.  
- Tag issues with relevant labels: `backend`, `frontend`, `api`, `ml`, `docs`, `streamlit`, `rdkit`.

