# FirstRespondersChatbot API Documentation

This document describes the API endpoints available for the FirstRespondersChatbot RAG system. These endpoints can be used by the React frontend to interact with the backend.

## Base URL

All endpoints are relative to the base URL: `http://localhost:8000`

## Endpoints

### Health Check

**Endpoint:** `GET /api/health`

**Description:** Check if the API is running.

**Response:**
```json
{
  "status": "ok"
}
```

### Upload File

**Endpoint:** `POST /api/upload`

**Description:** Upload a file to be indexed by the RAG system.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: The file to upload (PDF, TXT, or MD)

**Response (Success):**
```json
{
  "status": "success",
  "message": "File 'example.pdf' uploaded and indexed successfully",
  "file_path": "uploads/12345_example.pdf"
}
```

**Response (Error):**
```json
{
  "status": "error",
  "message": "Error message"
}
```

### Query

**Endpoint:** `POST /api/query`

**Description:** Send a query to the RAG system and get a response.

**Request:**
- Content-Type: `application/json`
- Body:
```json
{
  "query": "What should I do in case of a heart attack?"
}
```

**Response (Success):**
```json
{
  "status": "success",
  "answer": "The generated answer from the model...",
  "context": [
    {
      "file_name": "first_aid_manual.pdf",
      "snippet": "In case of a heart attack, call emergency services immediately..."
    }
  ],
  "query": "What should I do in case of a heart attack?"
}
```

**Response (Error):**
```json
{
  "status": "error",
  "message": "Error message"
}
```

### Clear Index

**Endpoint:** `POST /api/clear`

**Description:** Clear all indexed documents from the RAG system.

**Request:**
- Content-Type: `application/json`
- Body: Empty

**Response (Success):**
```json
{
  "status": "success",
  "message": "Document index cleared successfully"
}
```

**Response (Error):**
```json
{
  "status": "error",
  "message": "Error message"
}
```

### Get Indexed Files

**Endpoint:** `GET /api/files`

**Description:** Get a list of all indexed files.

**Response (Success):**
```json
{
  "status": "success",
  "files": [
    {
      "name": "example.pdf",
      "path": "uploads/12345_example.pdf",
      "size": 1024,
      "type": "pdf"
    }
  ]
}
```

**Response (Error):**
```json
{
  "status": "error",
  "message": "Error message"
}
```

## Example Usage (JavaScript/React)

```javascript
// Example: Upload a file
const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch('http://localhost:8000/api/upload', {
      method: 'POST',
      body: formData,
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error uploading file:', error);
    return { status: 'error', message: error.message };
  }
};

// Example: Send a query
const sendQuery = async (query) => {
  try {
    const response = await fetch('http://localhost:8000/api/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error sending query:', error);
    return { status: 'error', message: error.message };
  }
};
``` 