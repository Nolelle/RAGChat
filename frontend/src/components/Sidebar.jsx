import React from "react";

const Sidebar = ({ messages, clearHistory }) => {
    return (
        <div className="w-64 bg-gray-900 text-white p-4 flex flex-col h-full border-r border-gray-800">
            <h2 className="text-xl font-bold mb-4">First Responders Chatbot</h2>
            
            <div className="mb-4 flex-1 overflow-hidden">
                <div className="flex justify-between items-center mb-2">
                    <h3 className="text-lg">Chat History</h3>
                </div>
                
                {messages.length === 0 ? (
                    <div className="text-gray-400 text-sm">No chat history yet</div>
                ) : (
                    <ul className="text-sm space-y-2 max-h-[calc(100vh-180px)] overflow-y-auto pr-1">
                        {messages.map((msg, index) => (
                            <li key={index} className={`p-2 rounded-lg ${msg.isUser ? 'bg-blue-800' : 'bg-gray-800'}`}>
                                <div className="font-semibold text-xs mb-1">
                                    {msg.isUser ? 'You' : 'Bot'}
                                </div>
                                <div className="text-xs line-clamp-2 text-gray-300">{msg.text}</div>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
            
            {messages.length > 0 && (
                <button 
                    onClick={clearHistory}
                    className="mt-auto bg-red-700 text-white px-4 py-2 rounded hover:bg-red-800 text-sm"
                >
                    Clear History
                </button>
            )}
        </div>
    );
};

export default Sidebar;