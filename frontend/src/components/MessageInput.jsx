import React, { useState, useRef } from "react";

const MessageInput = ({ addMessage, setIsLoading, handleApiError }) => {
    const [file, setFile] = useState(null);
    const [message, setMessage] = useState('');
    const [uploadStatus, setUploadStatus] = useState(null);
    const inputRef = useRef(null);
    
    const API_URL = 'http://localhost:8000';

    const handleFileUpload = async (event) => {
        const uploadedFile = event.target.files[0];
        if (!uploadedFile) return;
        
        // Check file type
        const allowedTypes = ['application/pdf', 'text/plain', 'text/markdown'];
        if (!allowedTypes.includes(uploadedFile.type)) {
            setUploadStatus('error');
            handleApiError(new Error('Only PDF, TXT, and MD files are allowed'));
            return;
        }
        
        // Check file size (max 10MB)
        if (uploadedFile.size > 10 * 1024 * 1024) {
            setUploadStatus('error');
            handleApiError(new Error('File size must be less than 10MB'));
            return;
        }
        
        setFile(uploadedFile);
        
        // Upload the file to the server
        try {
            setUploadStatus('uploading');
            const formData = new FormData();
            formData.append('file', uploadedFile);
            
            const response = await fetch(`${API_URL}/api/upload`, {
                method: 'POST',
                body: formData,
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.status === 'success') {
                setUploadStatus('success');
                addMessage({
                    text: `File "${uploadedFile.name}" uploaded successfully.`,
                    isUser: false
                });
            } else {
                setUploadStatus('error');
                throw new Error(data.message || 'Error uploading file');
            }
        } catch (error) {
            setUploadStatus('error');
            handleApiError(error);
            setFile(null);
        }
    };

    const handleSendMessage = async () => {
        if (!message.trim()) return;
        
        // Add user message to the chat
        const userMessage = {
            text: message,
            isUser: true
        };
        addMessage(userMessage);
        
        // Clear the input field and focus it
        setMessage('');
        if (inputRef.current) {
            inputRef.current.focus();
        }
        
        // Set loading state
        setIsLoading(true);
        
        try {
            // Send the query to the API
            const response = await fetch(`${API_URL}/api/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: message }),
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Add bot message to the chat
                const botMessage = {
                    text: data.answer,
                    isUser: false,
                    context: data.context
                };
                addMessage(botMessage);
            } else {
                throw new Error(data.message || 'Failed to get a response');
            }
        } catch (error) {
            handleApiError(error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSendMessage();
        }
    };

    const clearFile = () => {
        setFile(null);
        setUploadStatus(null);
    };

    return (
        <div className="p-4 bg-gray-900 border-t border-gray-700 flex items-center shadow-md">
            <div className="flex items-center w-full max-w-4xl mx-auto">
                {/* File upload button */}
                <label className={`w-10 h-10 flex items-center justify-center rounded-full cursor-pointer transition mr-2 ${
                    uploadStatus === 'success' 
                        ? 'bg-green-600' 
                        : uploadStatus === 'error' 
                            ? 'bg-red-600' 
                            : uploadStatus === 'uploading' 
                                ? 'bg-yellow-600' 
                                : 'bg-gray-700 hover:bg-gray-600'
                }`}>
                    <span className="text-lg text-white">{
                        uploadStatus === 'success' 
                            ? '✓' 
                            : uploadStatus === 'error' 
                                ? '✗' 
                                : uploadStatus === 'uploading' 
                                    ? '...' 
                                    : '+'
                    }</span>
                    <input 
                        type="file" 
                        className="hidden" 
                        onChange={handleFileUpload} 
                        accept=".pdf,.txt,.md"
                    />
                </label>

                {/* Message input field */}
                <div className="flex-1 relative">
                    <input 
                        ref={inputRef}
                        type="text"
                        className="w-full p-3 pr-16 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-white placeholder-gray-400"
                        placeholder="Type a message here..."
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyPress={handleKeyPress}
                    />
                    
                    {/* Send button */}
                    <button 
                        className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-blue-600 text-white p-2 rounded-full hover:bg-blue-700 transition"
                        onClick={handleSendMessage}
                        disabled={!message.trim()}
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
                        </svg>
                    </button>
                </div>
            </div>

            {/* Display uploaded file name */}
            {file && (
                <div className="absolute bottom-16 left-0 right-0 bg-gray-800 p-2 border-t border-gray-700 flex items-center justify-between px-4">
                    <div className="flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-400 mr-2" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                        </svg>
                        <span className="text-sm text-gray-300 truncate max-w-xs">
                            {uploadStatus === 'success' ? '✓ ' : ''}{file.name}
                        </span>
                    </div>
                    <button 
                        onClick={clearFile}
                        className="text-gray-400 hover:text-gray-200"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                    </button>
                </div>
            )}
        </div>
    );
};

export default MessageInput;