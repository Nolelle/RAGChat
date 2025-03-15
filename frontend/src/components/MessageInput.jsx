import React, { useState } from "react";

const MessageInput = () => {
    const [file, setFile] = useState(null);
    const [message, setMessage] = useState('');

    const handleFileUpload = (event) => {
        const uploadedFile = event.target.files[0];
        if (uploadedFile) {
            setFile(uploadedFile);
            console.log("Uploaded file:", uploadedFile.name);
        }
    };

    return (
        <div className="p-4 bg-white border-t flex items-center">
            
            {/* File upload button */}
            <label className="w-10 h-10 flex items-center justify-center bg-gray-700 rounded-full cursor-pointer hover:bg-gray-600 transition">
                <span className="text-lg">+</span>
                <input type="file" className="hidden" onChange={handleFileUpload} />
            </label>

            {/*Message input field */}
            <input 
            type="text"
            className="flex-1 p-2 border rounded-lg"
            placeholder="Type a Message here"
            />

            {/*Send button*/}
            <button className="ml-2 bg-blue-500 text-white px-4 py-2 rounded-lg">Send</button>

            {/*displays the file name that has been uploaded*/}
            {file && (
                <p className="text-sm text-gray-950 ml-4">
                    Uploaded: {file.name}
                </p>
            )}
        </div>
    );
};

export default MessageInput;