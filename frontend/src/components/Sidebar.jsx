import React from "react";

const Sidebar = () => {
    return(
        <div className="w-64 h-screen bg-gray-900 text-white p-4">
            <h2 className="text-lg font-bold mb-4">First Responders Chat</h2>
            <ul>
                <li className="p-2 hover:bg-gray-700 rounded">Chats</li>
                <li className="p-2 hover:bg-gray-700 rounded">Library</li>
                <li className="p-2 hover:bg-gray-700 rounded">Apps</li>
            </ul>
        </div>
    );
};

export default Sidebar;