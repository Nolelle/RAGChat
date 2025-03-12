import React from "react";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";
import MessageInput from "./components/MessageInput";

const App = () =>{
    return(
        <div className="flex h-screen">
            <Sidebar />
            <div className="flex flex-col flex-1">
                <ChatWindow />
                <MessageInput />
            </div>
        </div>
    );
};

export default App;