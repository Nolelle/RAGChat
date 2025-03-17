import React, { useState, useEffect, useRef } from "react";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";
import MessageInput from "./components/MessageInput";

const App = () => {
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const chatEndRef = useRef(null);

    // Add a new message to the chat
    const addMessage = (message) => {
        setMessages(prevMessages => [...prevMessages, message]);
    };

    // Clear chat history
    const clearHistory = () => {
        if (window.confirm("Are you sure you want to clear the chat history?")) {
            setMessages([]);
        }
    };

    // Scroll to bottom of chat when new messages are added
    useEffect(() => {
        if (chatEndRef.current) {
            chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages]);

    // Handle API errors
    const handleApiError = (error) => {
        setError(error.message || "An error occurred");
        setTimeout(() => setError(null), 5000); // Clear error after 5 seconds
    };

    return(
        <div className="flex h-screen bg-gray-900">
            <Sidebar messages={messages} clearHistory={clearHistory} />
            <div className="flex flex-col flex-1 h-full overflow-hidden">
                <ChatWindow 
                    messages={messages} 
                    isLoading={isLoading} 
                    error={error}
                    chatEndRef={chatEndRef}
                />
                <MessageInput 
                    addMessage={addMessage} 
                    setIsLoading={setIsLoading}
                    handleApiError={handleApiError}
                />
            </div>
        </div>
    );
};

export default App;