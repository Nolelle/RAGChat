import React from "react";

const Sidebar = ({ 
    chatSessions, 
    activeChatId, 
    setActiveChatId, 
    createNewChat, 
    deleteChat, 
    clearAllChats 
}) => {
    // Format date to a readable string
    const formatDate = (dateString) => {
        const date = new Date(dateString);
        return date.toLocaleDateString(undefined, { 
            month: 'short', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    // Filter out empty chats (with no messages) from the history display
    const nonEmptyChats = chatSessions.filter(chat => chat.messages.length > 0);

    return (
        <div className="w-64 bg-gray-900 text-white p-4 flex flex-col h-full border-r border-gray-800">
            <h2 className="text-xl font-bold mb-4">First Responders Chatbot</h2>
            
            {/* New Chat Button */}
            <button 
                onClick={createNewChat}
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 mb-4 flex items-center justify-center"
            >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z" clipRule="evenodd" />
                </svg>
                New Chat
            </button>
            
            <div className="mb-4 flex-1 overflow-hidden">
                <div className="flex justify-between items-center mb-2">
                    <h3 className="text-lg">Chat History</h3>
                </div>
                
                {nonEmptyChats.length === 0 ? (
                    <div className="text-gray-400 text-sm">No chat history yet</div>
                ) : (
                    <ul className="text-sm space-y-2 max-h-[calc(100vh-180px)] overflow-y-auto pr-1">
                        {nonEmptyChats.map((chat) => (
                            <li 
                                key={chat.id} 
                                className={`p-2 rounded-lg cursor-pointer hover:bg-gray-800 flex justify-between group ${
                                    chat.id === activeChatId ? 'bg-gray-800 border border-gray-700' : ''
                                }`}
                                onClick={() => setActiveChatId(chat.id)}
                            >
                                <div className="flex-1 min-w-0">
                                    <div className="font-medium truncate">{chat.title}</div>
                                    <div className="text-xs text-gray-400">{formatDate(chat.createdAt)}</div>
                                    <div className="text-xs text-gray-400">{chat.messages.length} messages</div>
                                </div>
                                <button 
                                    className="text-gray-500 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        deleteChat(chat.id);
                                    }}
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                                        <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                                    </svg>
                                </button>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
            
            {chatSessions.length > 0 && (
                <button 
                    onClick={clearAllChats}
                    className="mt-auto bg-red-700 text-white px-4 py-2 rounded hover:bg-red-800 text-sm"
                >
                    Clear All Chats
                </button>
            )}
        </div>
    );
};

export default Sidebar;