import React, { useState, useEffect, useRef, useMemo } from "react";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";
import MessageInput from "./components/MessageInput";

const App = () => {
    // Chat sessions state
    const [chatSessions, setChatSessions] = useState(() => {
        const saved = localStorage.getItem('chatSessions');
        return saved ? JSON.parse(saved) : [];
    });
    
    // Current active chat session ID
    const [activeChatId, setActiveChatId] = useState(null);
    
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const chatEndRef = useRef(null);

    // Get current active chat
    const activeChat = chatSessions.find(chat => chat.id === activeChatId) || null;
    
    // Get current messages using useMemo to prevent unnecessary recalculations
    const messages = useMemo(() => activeChat ? activeChat.messages : [], [activeChat]);

    // Save chat sessions to localStorage whenever they change
    useEffect(() => {
        localStorage.setItem('chatSessions', JSON.stringify(chatSessions));
    }, [chatSessions]);

    // Create a new chat session
    const createNewChat = () => {
        const newChatId = Date.now().toString();
        const newChat = {
            id: newChatId,
            title: "New Chat",
            messages: [],
            createdAt: new Date().toISOString(),
            // Add a server session ID for file tracking
            serverSessionId: `session-${newChatId}` 
        };
        
        setChatSessions(prevSessions => [...prevSessions, newChat]);
        setActiveChatId(newChatId);
    };

    // If no active chat, create one
    useEffect(() => {
        if (chatSessions.length === 0) {
            createNewChat();
        } else if (!activeChatId) {
            setActiveChatId(chatSessions[0].id);
        }
    }, [chatSessions, activeChatId]);

    // Add a new message to the active chat
    const addMessage = (message) => {
        setChatSessions(prevSessions => {
            return prevSessions.map(session => {
                if (session.id === activeChatId) {
                    // Update chat title based on first user message if title is still "New Chat"
                    let updatedTitle = session.title;
                    if (message.isUser && session.title === "New Chat" && session.messages.length === 0) {
                        // Limit title length
                        updatedTitle = message.text.length > 30 
                            ? message.text.substring(0, 30) + "..." 
                            : message.text;
                    }
                    
                    return {
                        ...session,
                        title: updatedTitle,
                        messages: [...session.messages, message]
                    };
                }
                return session;
            });
        });
    };

    // Delete a chat session
    const deleteChat = (chatId) => {
        if (window.confirm("Are you sure you want to delete this chat?")) {
            // Get the server session ID before deleting the chat
            const chatToDelete = chatSessions.find(chat => chat.id === chatId);
            const serverSessionId = chatToDelete?.serverSessionId;
            
            // Call API to clear the server session
            if (serverSessionId) {
                clearServerSession(serverSessionId);
            }
            
            setChatSessions(prevSessions => prevSessions.filter(session => session.id !== chatId));
            
            // If we deleted the active chat, set active to the first remaining chat or create a new one
            if (chatId === activeChatId) {
                const remainingSessions = chatSessions.filter(session => session.id !== chatId);
                if (remainingSessions.length > 0) {
                    setActiveChatId(remainingSessions[0].id);
                } else {
                    createNewChat();
                }
            }
        }
    };

    // Clear server session data when deleting a chat
    const clearServerSession = async (serverSessionId) => {
        try {
            const response = await fetch('http://localhost:8000/api/clear-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ session_id: serverSessionId }),
            });
            
            // We don't need to handle the response, just log any errors
            if (!response.ok) {
                console.error('Failed to clear server session:', serverSessionId);
            }
        } catch (error) {
            console.error('Error clearing server session:', error);
        }
    };

    // Clear all chat sessions
    const clearAllChats = () => {
        if (window.confirm("Are you sure you want to clear all chat history?")) {
            // Clear all server sessions
            chatSessions.forEach(chat => {
                if (chat.serverSessionId) {
                    clearServerSession(chat.serverSessionId);
                }
            });
            
            setChatSessions([]);
            createNewChat();
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

    // Get the current server session ID
    const getServerSessionId = () => {
        return activeChat?.serverSessionId || 'default';
    };

    return(
        <div className="flex h-screen bg-gray-900">
            <Sidebar 
                chatSessions={chatSessions}
                activeChatId={activeChatId}
                setActiveChatId={setActiveChatId}
                createNewChat={createNewChat}
                deleteChat={deleteChat}
                clearAllChats={clearAllChats}
            />
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
                    serverSessionId={getServerSessionId()}
                />
            </div>
        </div>
    );
};

export default App;