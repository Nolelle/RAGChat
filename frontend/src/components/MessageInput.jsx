import React from "react";

const MessageInput = () => {
    return (
        <div className="p-4 bg-white border-t flex items-center">
            <input 
            type="text"
            className="flex-1 p-2 border rounded-lg"
            placeholder="Type a Message here"
            />
            <button className="ml-2 bg-blue-500 text-white px-4 py-2 rounded-lg">Send</button>
        </div>
    );
};

export default MessageInput;