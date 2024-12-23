from fastapi import FastAPI, File, UploadFile, HTTPException
from mymodels import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
import os     

# Initialize FastAPI app
app = FastAPI()

# This endpoint handles chat interactions. It generates a session ID if not provided, retrieves chat history, 
# invokes the RAG chain to generate a response, logs the interaction, and returns the response.
@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    max_size = 10
    current_size = int(query_input.children)
    excess_no = max_size - current_size
    answer = f"{query_input.model} has {query_input.children} children "
    return QueryResponse(answer=answer, session_id=query_input.session_id, model=query_input.model, max_size=max_size, excess_no=excess_no)

# Document Upload Endpoint:
# This endpoint handles document uploads. It checks for allowed file types, saves the file temporarily, 
# indexes it in Chroma, and updates the document record in the database.
@app.post("/upload-doc", response_model=QueryResponse)
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.html', '.txt']
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")
    temp_file_path = f"temp_{file.filename}"
    max_size = 15
    leng = len(temp_file_path)
    excess_no = max_size - leng
    answer = f"Your document is of length {leng}"
    return QueryResponse(answer=answer, session_id='3', model= "Muthoka", max_size=max_size, excess_no=excess_no)


'''
# List Documents Endpoint:
# This simple endpoint returns a list  of all indexed documents.
@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()


# Delete Document Endpoint:
# This endpoint handles document deletion, removing the document from both Chroma and the database.
@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    chroma_delete_success = delete_doc_from_chroma(request.file_id)
    if chroma_delete_success:
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}

'''


