"""
Flask server module for the FirstRespondersChatbot RAG system.

This module sets up a Flask server to provide a REST API for the RAG system,
handling file uploads and queries, and returning responses.
"""

import os
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from .rag_system import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGServer:
    """Server for the FirstRespondersChatbot RAG system."""

    def __init__(
        self,
        rag_system: RAGSystem,
        host: str = "0.0.0.0",
        port: int = 8000,
        debug: bool = True,
        allowed_extensions: set = None,
        max_content_length: int = 16 * 1024 * 1024,  # 16MB
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
        """
        self.rag_system = rag_system
        self.host = host
        self.port = port
        self.debug = debug
        self.allowed_extensions = allowed_extensions or {"pdf", "txt", "md"}
        self.max_content_length = max_content_length

        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for all routes

        # Set maximum file size
        self.app.config["MAX_CONTENT_LENGTH"] = self.max_content_length

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register routes for the Flask app."""

        # Health check endpoint
        @self.app.route("/api/health", methods=["GET"])
        def health_check():
            """Health check endpoint."""
            return jsonify({"status": "ok"})

        # Upload file endpoint
        @self.app.route("/api/upload", methods=["POST"])
        def upload_file():
            """
            Handle file uploads.

            Returns:
                JSON response with status and message
            """
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

            # Check if file has allowed extension
            if not self._allowed_file(file.filename):
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
                # Save file with session ID
                filename = secure_filename(file.filename)
                file_data = file.read()
                file_path = self.rag_system.save_uploaded_file(
                    file_data, filename, session_id
                )

                return jsonify(
                    {
                        "status": "success",
                        "message": f"File '{filename}' uploaded and indexed successfully",
                        "file_path": file_path,
                        "session_id": session_id,
                    }
                )

            except Exception as e:
                logger.error(f"Error handling file upload: {str(e)}")
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
            # Get query from request
            data = request.json
            if not data or "query" not in data:
                return jsonify({"status": "error", "message": "No query provided"}), 400

            query_text = data["query"]
            # Get session ID (default to None if not provided)
            session_id = data.get("session_id", "default")

            try:
                # Generate response with session context
                response = self.rag_system.generate_response(
                    query_text, session_id=session_id
                )

                return jsonify(
                    {
                        "status": "success",
                        "answer": response["answer"],
                        "context": response["context"],
                        "query": response["query"],
                        "session_id": session_id,
                    }
                )

            except Exception as e:
                logger.error(f"Error handling query: {str(e)}")
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"Error processing query: {str(e)}",
                        }
                    ),
                    500,
                )

        # Remove file endpoint
        @self.app.route("/api/remove-file", methods=["POST"])
        def remove_file():
            """
            Remove a file from a session.

            Returns:
                JSON response with status and message
            """
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
                                }
                            )

                    return jsonify(
                        {
                            "status": "success",
                            "files": file_info,
                            "session_id": session_id,
                        }
                    )
                else:
                    return jsonify(
                        {
                            "status": "success",
                            "files": [],
                            "session_id": session_id,
                            "message": f"No files found for session {session_id}",
                        }
                    )

            except Exception as e:
                logger.error(f"Error getting indexed files: {str(e)}")
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
