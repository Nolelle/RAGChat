"""
Flask server module for the FirstRespondersChatbot RAG system.

This module sets up a Flask server to provide a REST API for the RAG system,
handling file uploads and queries, and returning responses.
"""

import os
import logging
import uuid
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from .rag_system import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RAGServer:
    """Server for the FirstRespondersChatbot RAG system."""

    def __init__(
        self,
        rag_system: RAGSystem = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        debug: bool = True,
        allowed_extensions: set = None,
        max_content_length: int = 16 * 1024 * 1024,  # 16MB
        uploads_dir: str = "uploads",
    ):
        """
        Initialize the RAG server.

        Args:
            rag_system: The RAG system to use
            host: Host to bind the server to
            port: Port to bind the server to
            debug: Whether to run the server in debug mode
            allowed_extensions: Set of allowed file extensions
            max_content_length: Maximum content length for file uploads
            uploads_dir: Directory for file uploads
        """
        self.rag_system = rag_system or RAGSystem()
        self.host = host
        self.port = port
        self.debug = debug
        self.allowed_extensions = allowed_extensions or {
            "pdf",
            "txt",
            "md",
            "docx",
            "html",
        }
        self.max_content_length = max_content_length
        self.uploads_dir = Path(uploads_dir)

        # Ensure uploads directory exists
        os.makedirs(str(self.uploads_dir), exist_ok=True)

        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for all routes

        # Set maximum file size
        self.app.config["MAX_CONTENT_LENGTH"] = self.max_content_length

        # Register routes
        self._register_routes()

        # Initialize request counters and stats
        self.request_count = 0
        self.start_time = time.time()

    def _verify_file_integrity(self, file_data, file_ext) -> tuple:
        """
        Verify file integrity based on file type.

        Args:
            file_data: Binary content of the file
            file_ext: File extension

        Returns:
            tuple: (is_valid, message)
        """
        if file_ext == ".pdf":
            return self._verify_pdf_integrity(file_data)
        elif file_ext == ".docx":
            return self._verify_docx_integrity(file_data)
        elif file_ext in [".txt", ".md", ".html", ".htm"]:
            return self._verify_text_integrity(file_data)

        # Default case for other file types
        return True, f"File verification skipped for {file_ext} files"

    def _verify_pdf_integrity(self, file_data) -> Tuple[bool, str]:
        """
        Verify that a PDF file is valid and readable.

        Args:
            file_data: Binary content of the PDF file

        Returns:
            tuple: (is_valid, message) where is_valid is a boolean and message is a string
        """
        try:
            # Only import PyPDF2 when needed to verify PDFs
            import io
            import PyPDF2

            # Create a file-like object
            pdf_stream = io.BytesIO(file_data)

            # Try to read the PDF
            try:
                pdf_reader = PyPDF2.PdfReader(pdf_stream)
                num_pages = len(pdf_reader.pages)

                # Check if we can access content
                if num_pages > 0:
                    # Try to extract text from the first page
                    first_page = pdf_reader.pages[0]
                    text = first_page.extract_text()

                    # Check if text extraction worked
                    if not text or text.strip() == "":
                        logger.warning(
                            "PDF appears valid but no text could be extracted from the first page"
                        )
                        return (
                            True,
                            "PDF is valid, but may not contain extractable text. Results might be limited.",
                        )

                    # If we have text and it looks like garbled content
                    import re

                    if re.search(r"[^\x00-\x7F]{5,}", text) or re.search(
                        r"([^\w\s]{3,}|([a-zA-Z][^a-zA-Z]){4,})", text
                    ):
                        logger.warning(
                            "PDF contains text but it appears to be corrupted or encoded improperly"
                        )
                        return (
                            True,
                            "PDF contains text but it may be improperly encoded. Results might be affected.",
                        )

                    # If PDF has few pages, ensure we can read them all
                    if num_pages < 5:
                        all_text = ""
                        for i in range(num_pages):
                            page_text = pdf_reader.pages[i].extract_text()
                            all_text += page_text

                        # If total text is very small, warn
                        if len(all_text.strip()) < 100:
                            return (
                                True,
                                "PDF contains very little extractable text. Results might be limited.",
                            )

                return True, f"PDF is valid with {num_pages} pages"
            except Exception as e:
                logger.error(f"Error reading PDF: {str(e)}")
                return False, f"PDF could not be read: {str(e)}"

        except ImportError:
            logger.warning("PyPDF2 not available for PDF verification")
            return True, "PDF verification skipped (PyPDF2 not available)"
        except Exception as e:
            logger.error(f"Error verifying PDF: {str(e)}")
            return False, f"PDF verification failed: {str(e)}"

    def _verify_docx_integrity(self, file_data) -> Tuple[bool, str]:
        """Verify that a DOCX file is valid and readable."""
        try:
            import io
            from docx import Document

            docx_stream = io.BytesIO(file_data)

            try:
                doc = Document(docx_stream)
                paragraphs = len(doc.paragraphs)

                # Check if we can extract any text
                text_content = "\n".join([p.text for p in doc.paragraphs])

                if not text_content or text_content.strip() == "":
                    return (
                        True,
                        "DOCX file is valid but contains no text. Results might be limited.",
                    )

                # If text content is very small, warn
                if len(text_content.strip()) < 100:
                    return (
                        True,
                        "DOCX contains very little text. Results might be limited.",
                    )

                return True, f"DOCX is valid with {paragraphs} paragraphs"
            except Exception as e:
                logger.error(f"Error reading DOCX: {str(e)}")
                return False, f"DOCX could not be read: {str(e)}"

        except ImportError:
            logger.warning("python-docx not available for DOCX verification")
            return True, "DOCX verification skipped (python-docx not available)"
        except Exception as e:
            logger.error(f"Error verifying DOCX: {str(e)}")
            return False, f"DOCX verification failed: {str(e)}"

    def _verify_text_integrity(self, file_data) -> Tuple[bool, str]:
        """Verify that a text file is valid and readable."""
        try:
            # Try to decode the text file with different encodings
            encodings = ["utf-8", "latin-1", "windows-1252", "ascii"]
            decoded = False

            for encoding in encodings:
                try:
                    text = file_data.decode(encoding)
                    decoded = True

                    # Check if the file has reasonable content
                    if not text or text.strip() == "":
                        return True, "Text file is empty. Results might be limited."

                    # If text is very small, warn
                    if len(text.strip()) < 50:
                        return (
                            True,
                            "Text file contains very little content. Results might be limited.",
                        )

                    return True, f"Text file is valid ({len(text)} characters)"
                except UnicodeDecodeError:
                    continue

            if not decoded:
                return (
                    False,
                    "Text file could not be decoded with any standard encoding.",
                )

        except Exception as e:
            logger.error(f"Error verifying text file: {str(e)}")
            return False, f"Text file verification failed: {str(e)}"

    def _register_routes(self):
        """Register routes for the Flask app."""

        # Health check endpoint
        @self.app.route("/api/health", methods=["GET"])
        def health_check():
            """Health check endpoint."""
            uptime = time.time() - self.start_time
            return jsonify(
                {
                    "status": "ok",
                    "uptime": f"{uptime:.2f} seconds",
                    "requests_served": self.request_count,
                    "version": "1.1.0",
                }
            )

        # Upload file endpoint
        @self.app.route("/api/upload", methods=["POST"])
        def upload_file():
            """
            Handle file uploads.

            Returns:
                JSON response with status and message
            """
            self.request_count += 1

            # Check if file part exists in request
            if "file" not in request.files:
                return (
                    jsonify(
                        {"status": "error", "message": "No file part in the request"}
                    ),
                    400,
                )

            file = request.files["file"]

            # Check if file is empty
            if file.filename == "":
                return jsonify({"status": "error", "message": "No file selected"}), 400

            # Get original filename and secure it
            original_filename = file.filename
            filename = secure_filename(original_filename)

            # Check if file has allowed extension
            if not self._allowed_file(filename):
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"File type not allowed. Allowed types: {', '.join(self.allowed_extensions)}",
                        }
                    ),
                    400,
                )

            # Get or create session ID
            session_id = request.form.get("session_id", str(uuid.uuid4()))

            try:
                # Read file data
                file_data = file.read()

                # Get file extension
                file_ext = os.path.splitext(filename)[1].lower()

                # Verify file integrity
                warnings = []
                is_valid, message = self._verify_file_integrity(file_data, file_ext)

                if not is_valid:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": f"Invalid file: {message}",
                            }
                        ),
                        400,
                    )
                elif "may" in message or "might" in message:
                    # Add warning to response if there might be issues
                    warnings.append(message)

                # Save and index file with session ID
                logger.info(
                    f"Saving and indexing file: {filename} for session: {session_id}"
                )

                try:
                    file_path = self.rag_system.save_uploaded_file(
                        file_data, filename, session_id
                    )
                except Exception as e:
                    logger.error(f"Error saving file: {str(e)}")
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": f"Error saving file: {str(e)}",
                            }
                        ),
                        500,
                    )

                # Verify the file was indexed
                doc_count = 0
                for doc in self.rag_system.document_store.filter_documents():
                    if session_id == doc.meta.get("session_id") and str(
                        file_path
                    ) == doc.meta.get("file_path"):
                        doc_count += 1

                logger.info(f"Verified {doc_count} documents indexed from {filename}")

                # If no documents were indexed, add a warning
                if doc_count == 0:
                    warnings.append(
                        "The file could not be properly processed for indexing. It may be corrupted, contain poor text extraction quality, or have security restrictions."
                    )

                response = {
                    "status": "success",
                    "message": f"File '{original_filename}' uploaded successfully",
                    "file_path": file_path,
                    "session_id": session_id,
                    "indexed_documents": doc_count,
                }

                # Add warnings if any were detected
                if warnings:
                    response["warnings"] = warnings
                    if doc_count == 0:
                        response["message"] = (
                            f"File '{original_filename}' uploaded but could not be indexed."
                        )
                        response["status"] = "partial_success"

                return jsonify(response)

            except Exception as e:
                logger.error(f"Error handling file upload: {str(e)}")
                logger.error(traceback.format_exc())

                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"Error processing file: {str(e)}",
                        }
                    ),
                    500,
                )

        # Query endpoint
        @self.app.route("/api/query", methods=["POST"])
        def query():
            """
            Handle queries to the RAG system.

            Returns:
                JSON response with the answer and context
            """
            self.request_count += 1

            # Get query from request
            data = request.json
            if not data or "query" not in data:
                return jsonify({"status": "error", "message": "No query provided"}), 400

            query_text = data["query"]
            # Get session ID (default to None if not provided)
            session_id = data.get("session_id", "default")

            try:
                start_time = time.time()
                # Generate response with session context
                response = self.rag_system.generate_response(
                    query_text, session_id=session_id
                )
                end_time = time.time()

                processing_time = end_time - start_time
                logger.info(f"Query processed in {processing_time:.2f} seconds")

                return jsonify(
                    {
                        "status": "success",
                        "answer": response["answer"],
                        "context": response["context"],
                        "query": response["query"],
                        "session_id": session_id,
                        "processing_time": f"{processing_time:.2f} seconds",
                    }
                )

            except Exception as e:
                logger.error(f"Error handling query: {str(e)}")
                logger.error(traceback.format_exc())

                # Provide a more informative error message
                error_message = str(e)
                if "CUDA out of memory" in error_message:
                    error_message = "The system is experiencing high memory usage. Please try again with a simpler query or wait a moment."
                elif "Connection refused" in error_message:
                    error_message = "The backend services are currently unavailable. Please try again later."

                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"Error processing query: {error_message}",
                        }
                    ),
                    500,
                )

        # Serve uploaded files
        @self.app.route("/api/files/<path:filename>", methods=["GET"])
        def serve_file(filename):
            """Serve uploaded files."""
            try:
                # Add security check to prevent path traversal
                requested_path = os.path.abspath(
                    os.path.join(self.uploads_dir, filename)
                )
                if not requested_path.startswith(os.path.abspath(self.uploads_dir)):
                    return jsonify({"status": "error", "message": "Access denied"}), 403

                # Check if file exists
                if not os.path.exists(requested_path):
                    return (
                        jsonify({"status": "error", "message": "File not found"}),
                        404,
                    )

                # Serve the file
                return send_from_directory(self.uploads_dir, filename)
            except Exception as e:
                logger.error(f"Error serving file: {str(e)}")
                return jsonify({"status": "error", "message": str(e)}), 500

        # Remove file endpoint
        @self.app.route("/api/remove-file", methods=["POST"])
        def remove_file():
            """
            Remove a file from a session.

            Returns:
                JSON response with status and message
            """
            self.request_count += 1

            data = request.json
            if not data or "file_path" not in data:
                return (
                    jsonify({"status": "error", "message": "No file path provided"}),
                    400,
                )

            file_path = data["file_path"]
            session_id = data.get("session_id", "default")

            try:
                success = self.rag_system.remove_file(file_path, session_id)

                if success:
                    return jsonify(
                        {
                            "status": "success",
                            "message": f"File removed from session {session_id} successfully",
                        }
                    )
                else:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": f"File not found in session {session_id}",
                            }
                        ),
                        404,
                    )

            except Exception as e:
                logger.error(f"Error removing file: {str(e)}")
                logger.error(traceback.format_exc())
                return (
                    jsonify(
                        {"status": "error", "message": f"Error removing file: {str(e)}"}
                    ),
                    500,
                )

        # Clear session endpoint
        @self.app.route("/api/clear-session", methods=["POST"])
        def clear_session():
            """
            Clear a specific session.

            Returns:
                JSON response with status and message
            """
            self.request_count += 1

            data = request.json
            session_id = data.get("session_id", "default")

            try:
                success = self.rag_system.clear_session(session_id)

                if success:
                    return jsonify(
                        {
                            "status": "success",
                            "message": f"Session {session_id} cleared successfully",
                        }
                    )
                else:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": f"Session {session_id} not found or already empty",
                            }
                        ),
                        404,
                    )

            except Exception as e:
                logger.error(f"Error clearing session: {str(e)}")
                logger.error(traceback.format_exc())
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"Error clearing session: {str(e)}",
                        }
                    ),
                    500,
                )

        # Clear index endpoint
        @self.app.route("/api/clear", methods=["POST"])
        def clear_index():
            """
            Clear the document index.

            Returns:
                JSON response with status and message
            """
            self.request_count += 1

            try:
                self.rag_system.clear_index()
                return jsonify(
                    {
                        "status": "success",
                        "message": "Document index cleared successfully",
                    }
                )

            except Exception as e:
                logger.error(f"Error clearing index: {str(e)}")
                logger.error(traceback.format_exc())
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"Error clearing index: {str(e)}",
                        }
                    ),
                    500,
                )

        # Get indexed files endpoint
        @self.app.route("/api/files", methods=["GET"])
        def get_indexed_files():
            """
            Get a list of indexed files for a session.

            Returns:
                JSON response with the list of indexed files
            """
            self.request_count += 1

            try:
                # Get session ID from query parameter
                session_id = request.args.get("session_id", "default")

                if session_id in self.rag_system.session_files:
                    files = list(self.rag_system.session_files[session_id])
                    file_info = []

                    for file_path in files:
                        file_path_obj = Path(file_path)
                        if os.path.exists(file_path):
                            file_info.append(
                                {
                                    "name": file_path_obj.name,
                                    "path": file_path,
                                    "size": os.path.getsize(file_path),
                                    "type": file_path_obj.suffix.lower()[
                                        1:
                                    ],  # Remove the dot
                                    "last_modified": os.path.getmtime(file_path),
                                }
                            )

                    return jsonify(
                        {
                            "status": "success",
                            "files": file_info,
                            "session_id": session_id,
                            "count": len(file_info),
                        }
                    )
                else:
                    return jsonify(
                        {
                            "status": "success",
                            "files": [],
                            "session_id": session_id,
                            "count": 0,
                            "message": f"No files found for session {session_id}",
                        }
                    )

            except Exception as e:
                logger.error(f"Error getting indexed files: {str(e)}")
                logger.error(traceback.format_exc())
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"Error getting indexed files: {str(e)}",
                        }
                    ),
                    500,
                )

    def _allowed_file(self, filename: str) -> bool:
        """
        Check if a file has an allowed extension.

        Args:
            filename: The filename to check

        Returns:
            bool: True if the file has an allowed extension, False otherwise
        """
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in self.allowed_extensions
        )

    def run(self):
        """Run the server."""
        logger.info(f"Starting server on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=self.debug)
