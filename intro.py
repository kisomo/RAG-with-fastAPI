
# python -m venv myfastapi
# myfastapi\Scripts\Activate.ps1
# python.exe -m pip install --upgrade pip
# pip3 install -r requirements.txt
# pip install ipykernel 

# uvicorn main:app --host localhost --port 80

'''
# project structure
rag-with-fastapi/
│
├── main.py
├── chroma_utils.py
├── db_utils.py
├── langchain_utils.py
├── pydantic_models.py
├── requirements.txt
└── chroma_db/  (directory for Chroma persistence)
'''